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
import io
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
        self.model_spec = model_spec
        self._device = device
        self._model = None
        self._kwargs = kwargs

    def load(self):
        import os
        import sys

        # The yaml config loaded from model has hard-coded the import paths. please refer to: load_hyperpyyaml
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../thirdparty"))

        from cosyvoice.cli.cosyvoice import CosyVoice

        self._model = CosyVoice(self._model_path)

    def speech(
        self,
        input: str,
        voice: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs,
    ):
        if stream:
            raise Exception("CosyVoiceModel does not support stream.")

        import torchaudio
        from cosyvoice.utils.file_utils import load_wav

        prompt_speech: Optional[bytes] = kwargs.pop("prompt_speech", None)
        prompt_text: Optional[str] = kwargs.pop("prompt_text", None)
        instruct_text: Optional[str] = kwargs.pop("instruct_text", None)

        if "SFT" in self.model_spec.model_name:
            # inference_sft
            assert (
                prompt_speech is None
            ), "CosyVoice SFT model does not support prompt_speech"
            assert (
                prompt_text is None
            ), "CosyVoice SFT model does not support prompt_text"
            assert (
                instruct_text is None
            ), "CosyVoice SFT model does not support instruct_text"
        elif "Instruct" in self.model_spec.model_name:
            # inference_instruct
            assert (
                prompt_speech is None
            ), "CosyVoice Instruct model does not support prompt_speech"
            assert (
                prompt_text is None
            ), "CosyVoice Instruct model does not support prompt_text"
            assert (
                instruct_text is not None
            ), "CosyVoice Instruct model expect a instruct_text"
        else:
            # inference_zero_shot
            # inference_cross_lingual
            assert prompt_speech is not None, "CosyVoice model expect a prompt_speech"
            assert (
                instruct_text is None
            ), "CosyVoice model does not support instruct_text"

        assert self._model is not None
        if prompt_speech:
            assert not voice, "voice can't be set with prompt speech."
            with io.BytesIO(prompt_speech) as prompt_speech_io:
                prompt_speech_16k = load_wav(prompt_speech_io, 16000)
                if prompt_text:
                    logger.info("CosyVoice inference_zero_shot")
                    output = self._model.inference_zero_shot(
                        input, prompt_text, prompt_speech_16k
                    )
                else:
                    logger.info("CosyVoice inference_cross_lingual")
                    output = self._model.inference_cross_lingual(
                        input, prompt_speech_16k
                    )
        else:
            available_speakers = self._model.list_avaliable_spks()
            if not voice:
                voice = available_speakers[0]
            else:
                assert (
                    voice in available_speakers
                ), f"Invalid voice {voice}, CosyVoice available speakers: {available_speakers}"
            if instruct_text:
                logger.info("CosyVoice inference_instruct")
                output = self._model.inference_instruct(
                    input, voice, instruct_text=instruct_text
                )
            else:
                logger.info("CosyVoice inference_sft")
                output = self._model.inference_sft(input, voice)

        # Save the generated audio
        with BytesIO() as out:
            torchaudio.save(out, output["tts_speech"], 22050, format=response_format)
            return out.getvalue()
