# 替换掉官网的文件kokoro.py,用来v1.1-zh和解决GPU问题
# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from io import BytesIO
from typing import TYPE_CHECKING, Optional

import numpy as np

from ...device_utils import get_available_device, is_device_available

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)

from kokoro import KModel, KPipeline

REPO_ID = "hexgrad/Kokoro-82M-v1.1-zh"

en_pipeline = KPipeline(lang_code="a", repo_id=REPO_ID, model=False)


# from https://hf-mirror.com/hexgrad/Kokoro-82M-v1.1-zh/blob/main/samples/make_zh.py
def en_callable(text):
    if text == "Kokoro":
        return "kˈOkəɹO"
    elif text == "Sol":
        return "sˈOl"
    return next(en_pipeline(text)).phonemes


# HACK: Mitigate rushing caused by lack of training data beyond ~100 tokens
# Simple piecewise linear fn that decreases speed as len_ps increases
def speed_callable(len_ps):
    speed = 0.8
    if len_ps <= 83:
        speed = 1
    elif len_ps < 183:
        speed = 1 - (len_ps - 83) / 500
    return speed * 1.1


class KokoroZHModel:
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "AudioModelFamilyV2",
        device: Optional[str] = None,
        **kwargs,
    ):
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._device = device
        self._model = None
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def load(self):
        if self._device is None:
            self._device = get_available_device()
        else:
            if not is_device_available(self._device):
                raise ValueError(f"Device {self._device} is not available!")

        import os

        config_path = os.path.join(self._model_path, "config.json")
        model_path = os.path.join(self._model_path, "kokoro-v1_1-zh.pth")
        lang_code = self._kwargs.get("lang_code", "z")
        logger.info("Launching Kokoro model with language code: %s", lang_code)

        self._model = KPipeline(
            lang_code=lang_code,
            model=KModel(config=config_path, model=model_path).to(self._device),
            repo_id=REPO_ID,
            en_callable=en_callable,
            device=self._device,
        )

    def speech(
        self,
        input: str,
        voice: str,
        response_format: str = "mp3",
        speed: float = -1.0,
        stream: bool = False,
        **kwargs,
    ):
        import soundfile

        if stream:
            raise Exception("Kokoro does not support stream mode.")
        assert self._model is not None
        if not voice:
            voice = "zf_001"
            logger.info("Auto select speaker: %s", voice)
        elif voice.endswith(".pt"):
            logger.info("Using custom voice pt: %s", voice)
        else:
            logger.info("Using voice: %s", voice)
        logger.info("Speech kwargs: %s", kwargs)
        # hack for speed
        if speed < 0:
            speed = speed_callable
        generator = self._model(text=input, voice=voice, speed=speed, **kwargs)
        results = list(generator)
        audio = np.concatenate([r[2] for r in results])
        # Save the generated audio
        with BytesIO() as out:
            with soundfile.SoundFile(
                out,
                "w",
                24000,
                1,
                format=response_format.upper(),
            ) as f:
                f.write(audio)
            return out.getvalue()
