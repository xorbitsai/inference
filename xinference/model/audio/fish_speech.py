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
import os.path
import sys
from io import BytesIO
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from ...device_utils import get_available_device, is_device_available

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


def wav_chunk_header(sample_rate=44100, bit_depth=16, channels=1):
    import wave

    buffer = BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)

    wav_header_bytes = buffer.getvalue()
    buffer.close()
    return wav_header_bytes


class FishSpeechModel:
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
        self._llama_queue = None
        self._model = None
        self._engine = None
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def load(self):
        # There are too many imports from fish_speech.
        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), "../../thirdparty/fish_speech")
        )

        from tools.inference_engine import TTSInferenceEngine
        from tools.llama.generate import launch_thread_safe_queue
        from tools.vqgan.inference import load_model as load_decoder_model

        if self._device is None:
            self._device = get_available_device()
        else:
            if not is_device_available(self._device):
                raise ValueError(f"Device {self._device} is not available!")

        # https://github.com/pytorch/pytorch/issues/129207
        if self._device == "mps":
            logger.warning("The Conv1d has bugs on MPS backend, fallback to CPU.")
            self._device = "cpu"

        enable_compile = self._kwargs.get("compile", False)
        precision = self._kwargs.get("precision", torch.bfloat16)
        logger.info("Loading Llama model, compile=%s...", enable_compile)
        self._llama_queue = launch_thread_safe_queue(
            checkpoint_path=self._model_path,
            device=self._device,
            precision=precision,
            compile=enable_compile,
        )
        logger.info("Llama model loaded, loading VQ-GAN model...")

        checkpoint_path = os.path.join(
            self._model_path,
            "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        )
        self._model = load_decoder_model(
            config_name="firefly_gan_vq",
            checkpoint_path=checkpoint_path,
            device=self._device,
        )

        self._engine = TTSInferenceEngine(
            self._llama_queue, self._model, precision, enable_compile
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
        logger.warning("Fish speech does not support setting voice: %s.", voice)
        if speed != 1.0:
            logger.warning("Fish speech does not support setting speed: %s.", speed)
        from tools.schema import ServeReferenceAudio, ServeTTSRequest

        from .utils import audio_stream_generator, audio_to_bytes

        prompt_speech = kwargs.get("prompt_speech")
        prompt_text = kwargs.get("prompt_text", kwargs.get("reference_text", ""))
        if prompt_speech is not None:
            r = ServeReferenceAudio(audio=prompt_speech, text=prompt_text)
            references = [r]
        else:
            references = []

        assert self._engine is not None
        result = self._engine.inference(
            ServeTTSRequest(
                text=input,
                references=references,
                reference_id=kwargs.get("reference_id"),
                seed=kwargs.get("seed"),
                max_new_tokens=kwargs.get("max_new_tokens", 1024),
                chunk_length=kwargs.get("chunk_length", 200),
                top_p=kwargs.get("top_p", 0.7),
                repetition_penalty=kwargs.get("repetition_penalty", 1.2),
                temperature=kwargs.get("temperature", 0.7),
                streaming=stream,
                format=response_format,
            )
        )

        if stream:

            def _gen_chunk():
                for chunk in result:
                    if chunk.code == "final":
                        continue
                    chunk = chunk.audio[1]
                    if chunk is not None:
                        yield chunk

            return audio_stream_generator(
                response_format=response_format,
                sample_rate=self._model.spec_transform.sample_rate,
                output_generator=_gen_chunk(),
                output_chunk_transformer=lambda c: torch.from_numpy(
                    c.reshape((c.shape[0], 1))
                ),
            )
        else:
            result = list(result)
            sample_rate, audio = result[0].audio
            audio = np.array([audio])
            return audio_to_bytes(
                response_format=response_format,
                sample_rate=sample_rate,
                tensor=torch.from_numpy(audio),
            )
