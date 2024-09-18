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

from ..utils import set_all_random_seed

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

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def load(self):
        import os
        import sys

        # The yaml config loaded from model has hard-coded the import paths. please refer to: load_hyperpyyaml
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../thirdparty"))

        from cosyvoice.cli.cosyvoice import CosyVoice

        self._model = CosyVoice(
            self._model_path, load_jit=self._kwargs.get("load_jit", False)
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
            else:
                logger.info("CosyVoice inference_cross_lingual")
                output = self._model.inference_cross_lingual(
                    input, prompt_speech_16k, stream=stream
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
                    input, voice, instruct_text=instruct_text, stream=stream
                )
            else:
                logger.info("CosyVoice inference_sft")
                output = self._model.inference_sft(input, voice, stream=stream)

        import torch
        import torchaudio

        def _generator_stream():
            with BytesIO() as out:
                writer = torchaudio.io.StreamWriter(out, format=response_format)
                writer.add_audio_stream(sample_rate=22050, num_channels=1)
                i = 0
                last_pos = 0
                with writer.open():
                    for chunk in output:
                        chunk = chunk["tts_speech"]
                        trans_chunk = torch.transpose(chunk, 0, 1)
                        writer.write_audio_chunk(i, trans_chunk)
                        new_last_pos = out.tell()
                        if new_last_pos != last_pos:
                            out.seek(last_pos)
                            encoded_bytes = out.read()
                            yield encoded_bytes
                            last_pos = new_last_pos

        def _generator_block():
            chunks = [o["tts_speech"] for o in output]
            t = torch.cat(chunks, dim=1)
            with BytesIO() as out:
                torchaudio.save(out, t, 22050, format=response_format)
                return out.getvalue()

        return _generator_stream() if stream else _generator_block()

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
