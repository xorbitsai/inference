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
    from .core import AudioModelFamilyV1

logger = logging.getLogger(__name__)


class KokoroModel:
    # The available voices, should keep sync with https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices
    VOICES = [
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "bf_alice",
        "bf_emma",
        "bf_isabella",
        "bf_lily",
        "bm_daniel",
        "bm_fable",
        "bm_george",
        "bm_lewis",
    ]

    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "AudioModelFamilyV1",
        device: Optional[str] = None,
        **kwargs,
    ):
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

        from kokoro import KPipeline

        config_path = os.path.join(self._model_path, "config.json")
        model_path = os.path.join(self._model_path, "kokoro-v1_0.pth")
        # LANG_CODES = dict(
        #     a='American English',
        #     b='British English',
        # )
        lang_code = self._kwargs.get("lang_code", "a")
        self._model = KPipeline(
            lang_code=lang_code,
            config_path=config_path,
            model_path=model_path,
            device=self._device,
        )

    def speech(
        self,
        input: str,
        voice: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs,
    ):
        import soundfile

        if stream:
            raise Exception("Kokoro does not support stream mode.")
        assert self._model is not None
        if not voice:
            voice = next(iter(self.VOICES))
            logger.info("Auto select speaker: %s", voice)
        elif not voice.endswith(".pt") and voice not in self.VOICES:
            raise ValueError(
                f"Invalid voice: {voice}, available speakers: {self.VOICES}"
            )
        else:
            logger.info("Using custom voice pt: %s", voice)
        logger.info("Speech kwargs: %s", kwargs)
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
