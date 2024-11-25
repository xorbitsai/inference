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
import gc
import logging
import os.path
import queue
import sys
from io import BytesIO
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from ...device_utils import get_available_device, is_device_available

if TYPE_CHECKING:
    from .core import AudioModelFamilyV1

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
        model_spec: "AudioModelFamilyV1",
        device: Optional[str] = None,
        **kwargs,
    ):
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._device = device
        self._llama_queue = None
        self._model = None
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def load(self):
        # There are too many imports from fish_speech.
        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), "../../thirdparty/fish_speech")
        )

        from tools.llama.generate import launch_thread_safe_queue
        from tools.vqgan.inference import load_model as load_decoder_model

        if self._device is None:
            self._device = get_available_device()
        else:
            if not is_device_available(self._device):
                raise ValueError(f"Device {self._device} is not available!")

        logger.info("Loading Llama model...")
        self._llama_queue = launch_thread_safe_queue(
            checkpoint_path=self._model_path,
            device=self._device,
            precision=torch.bfloat16,
            compile=False,
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

    @torch.inference_mode()
    def _inference(
        self,
        text,
        enable_reference_audio,
        reference_audio,
        reference_text,
        max_new_tokens,
        chunk_length,
        top_p,
        repetition_penalty,
        temperature,
        streaming=False,
    ):
        from fish_speech.utils import autocast_exclude_mps
        from tools.api import decode_vq_tokens, encode_reference
        from tools.llama.generate import (
            GenerateRequest,
            GenerateResponse,
            WrappedGenerateResponse,
        )

        # Parse reference audio aka prompt
        prompt_tokens = encode_reference(
            decoder_model=self._model,
            reference_audio=reference_audio,
            enable_reference_audio=enable_reference_audio,
        )

        # LLAMA Inference
        request = dict(
            device=self._model.device,
            max_new_tokens=max_new_tokens,
            text=text,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            compile=False,
            iterative_prompt=chunk_length > 0,
            chunk_length=chunk_length,
            max_length=2048,
            prompt_tokens=prompt_tokens if enable_reference_audio else None,
            prompt_text=reference_text if enable_reference_audio else None,
        )

        response_queue = queue.Queue()
        self._llama_queue.put(
            GenerateRequest(
                request=request,
                response_queue=response_queue,
            )
        )

        if streaming:
            yield wav_chunk_header(), None, None

        segments = []

        while True:
            result: WrappedGenerateResponse = response_queue.get()  # type: ignore
            if result.status == "error":
                raise Exception(str(result.response))

            result: GenerateResponse = result.response  # type: ignore
            if result.action == "next":
                break

            with autocast_exclude_mps(
                device_type=self._model.device.type, dtype=torch.bfloat16
            ):
                fake_audios = decode_vq_tokens(
                    decoder_model=self._model,
                    codes=result.codes,
                )

            fake_audios = fake_audios.float().cpu().numpy()
            segments.append(fake_audios)

            if streaming:
                yield (fake_audios * 32768).astype(np.int16).tobytes(), None, None

        if len(segments) == 0:
            raise Exception("No audio generated, please check the input text.")

        # No matter streaming or not, we need to return the final audio
        audio = np.concatenate(segments, axis=0)
        yield None, (self._model.spec_transform.sample_rate, audio), None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

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
        if stream is True:
            logger.warning("stream mode is not implemented.")
        import torchaudio

        result = list(
            self._inference(
                text=input,
                enable_reference_audio=False,
                reference_audio=None,
                reference_text=kwargs.get("reference_text", ""),
                max_new_tokens=kwargs.get("max_new_tokens", 1024),
                chunk_length=kwargs.get("chunk_length", 200),
                top_p=kwargs.get("top_p", 0.7),
                repetition_penalty=kwargs.get("repetition_penalty", 1.2),
                temperature=kwargs.get("temperature", 0.7),
            )
        )
        sample_rate, audio = result[0][1]
        audio = np.array([audio])

        # Save the generated audio
        with BytesIO() as out:
            torchaudio.save(
                out, torch.from_numpy(audio), sample_rate, format=response_format
            )
            return out.getvalue()
