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
import io
import logging
from typing import TYPE_CHECKING, Optional

from ..utils import set_all_random_seed

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


class CosyVoiceModel:
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
        self._is_cosyvoice2 = False

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def load(self):
        import os
        import sys

        import torch

        # The yaml config loaded from model has hard-coded the import paths. please refer to: load_hyperpyyaml
        thirdparty_dir = os.path.join(os.path.dirname(__file__), "../../thirdparty")
        sys.path.insert(0, thirdparty_dir)

        kwargs = {}
        if "CosyVoice2" in self._model_spec.model_name:
            from cosyvoice.cli.cosyvoice import CosyVoice2 as CosyVoice

            self._is_cosyvoice2 = True
        else:
            from cosyvoice.cli.cosyvoice import CosyVoice

            self._is_cosyvoice2 = False

        # Unify this configuration name as 'compile' to be compatible with the name 'load_jit'.
        load_jit = self._kwargs.get("load_jit", False) or self._kwargs.get(
            "compile", False
        )
        logger.info("Loading CosyVoice model, compile=%s...", load_jit)
        self._model = CosyVoice(self._model_path, load_jit=load_jit, **kwargs)
        if self._is_cosyvoice2:
            spk2info_file = os.path.join(thirdparty_dir, "cosyvoice/bin/spk2info.pt")
            self._model.frontend.spk2info = torch.load(
                spk2info_file, map_location=self._device
            )

    def _speech_handle(
        self,
        stream,
        input,
        instruct_text,
        prompt_speech,
        prompt_text,
        voice,
        response_format,
    ):
        if prompt_speech:
            from cosyvoice.utils.file_utils import load_wav

            with io.BytesIO(prompt_speech) as prompt_speech_io:
                prompt_speech_16k = load_wav(prompt_speech_io, 16000)

            if prompt_text:
                logger.info("CosyVoice inference_zero_shot")
                output = self._model.inference_zero_shot(
                    input, prompt_text, prompt_speech_16k, stream=stream
                )
            elif instruct_text:
                assert self._is_cosyvoice2
                logger.info("CosyVoice inference_instruct")
                output = self._model.inference_instruct2(
                    input,
                    instruct_text=instruct_text,
                    prompt_speech_16k=prompt_speech_16k,
                    stream=stream,
                )
            else:
                logger.info("CosyVoice inference_cross_lingual")
                output = self._model.inference_cross_lingual(
                    input, prompt_speech_16k, stream=stream
                )
        else:
            available_speakers = self._model.list_available_spks()
            if not voice:
                voice = available_speakers[0]
                logger.info("Auto select speaker: %s", voice)
            else:
                assert (
                    voice in available_speakers
                ), f"Invalid voice {voice}, CosyVoice available speakers: {available_speakers}"
            if instruct_text:
                logger.info("CosyVoice inference_instruct")
                output = self._model.inference_instruct(
                    input, voice, instruct_text=instruct_text, stream=stream
                )
            else:
                logger.info("CosyVoice inference_sft")
                output = self._model.inference_sft(input, voice, stream=stream)

        import torch

        from .utils import audio_stream_generator, audio_to_bytes

        return (
            audio_stream_generator(
                response_format=response_format,
                sample_rate=self._model.sample_rate,
                output_generator=output,
                output_chunk_transformer=lambda c: torch.transpose(
                    c["tts_speech"], 0, 1
                ),
            )
            if stream
            else audio_to_bytes(
                response_format=response_format,
                sample_rate=self._model.sample_rate,
                tensor=torch.cat([o["tts_speech"] for o in output], dim=1),
            )
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
        prompt_speech: Optional[bytes] = kwargs.pop("prompt_speech", None)
        prompt_text: Optional[str] = kwargs.pop("prompt_text", None)
        instruct_text: Optional[str] = kwargs.pop("instruct_text", None)
        seed: Optional[int] = kwargs.pop("seed", 0)

        if "SFT" in self._model_spec.model_name:
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
        elif "Instruct" in self._model_spec.model_name:
            # inference_instruct
            assert (
                prompt_speech is None
            ), "CosyVoice Instruct model does not support prompt_speech"
            assert (
                prompt_text is None
            ), "CosyVoice Instruct model does not support prompt_text"
        elif self._is_cosyvoice2:
            pass
        else:
            # inference_zero_shot
            # inference_cross_lingual
            assert prompt_speech is not None, "CosyVoice model expect a prompt_speech"
            assert (
                instruct_text is None
            ), "CosyVoice model does not support instruct_text"

        assert self._model is not None

        set_all_random_seed(seed)

        return self._speech_handle(
            stream,
            input,
            instruct_text,
            prompt_speech,
            prompt_text,
            voice,
            response_format,
        )
