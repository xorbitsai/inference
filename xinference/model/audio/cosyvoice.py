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

if TYPE_CHECKING:
    from .core import AudioModelFamilyV1

logger = logging.getLogger(__name__)


class CosyVoiceModel:
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

    def load(self):
        from cosyvoice.cli.cosyvoice import CosyVoice

        self._model = CosyVoice(self._model_path)

    def speech(
        self, input: str, voice: str, response_format: str = "mp3", speed: float = 1.0
    ):
        import torchaudio

        assert self._model is not None
        available_speakers = self._model.list_avaliable_spks()
        if not voice:
            voice = available_speakers[0]
        else:
            assert (
                voice in available_speakers
            ), f"Invalid voice {voice}, CosyVoice available speakers: {available_speakers}"
        output = self._model.inference_sft(input, voice)
        # Save the generated audio
        with BytesIO() as out:
            torchaudio.save(out, output["tts_speech"], 22050, format=response_format)
            return out.getvalue()
