# Copyright 2022-2026 XProbe Inc.
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

from ...device_utils import get_available_device, is_device_available

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


class MeloTTSModel:
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
        import sys

        import nltk

        # English language requires download averaged_perceptron_tagger_eng
        nltk.download("averaged_perceptron_tagger_eng")

        # The yaml config loaded from model has hard-coded the import paths. please refer to: load_hyperpyyaml
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../thirdparty"))

        from melo.api import TTS

        config_path = os.path.join(self._model_path, "config.json")
        ckpt_path = os.path.join(self._model_path, "checkpoint.pth")
        self._model = TTS(
            language=self._model_spec.language,
            device=self._device,
            config_path=config_path,
            ckpt_path=ckpt_path,
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
            raise Exception("MeloTTS does not support stream mode.")
        assert self._model is not None
        speaker_ids = self._model.hps.data.spk2id
        if not voice:
            voice = next(iter(speaker_ids.keys()))
            logger.info("Auto select speaker: %s", voice)
        elif voice not in speaker_ids:
            raise ValueError(
                f"Invalid voice: {voice}, available speakers: {speaker_ids}"
            )
        audio = self._model.tts_to_file(
            text=input, speaker_id=speaker_ids[voice], speed=speed, **kwargs
        )
        # Save the generated audio
        with BytesIO() as out:
            with soundfile.SoundFile(
                out,
                "w",
                self._model.hps.data.sampling_rate,
                1,
                format=response_format.upper(),
            ) as f:
                f.write(audio)
            return out.getvalue()
