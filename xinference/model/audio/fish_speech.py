# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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

FISH_AUDIO_S1_MINI = "FishAudio-S1-mini"
FISH_AUDIO_S2_PRO = "FishAudio-S2-Pro"
_MODERN_FISH_AUDIO_MODELS = {FISH_AUDIO_S1_MINI, FISH_AUDIO_S2_PRO}


def _load_fish_speech_runtime_components(model_name: str):
    if model_name == FISH_AUDIO_S1_MINI:
        from ...thirdparty.fish_speech_s1.fish_speech.inference_engine import (
            TTSInferenceEngine,
        )
        from ...thirdparty.fish_speech_s1.fish_speech.models.dac.inference import (
            load_model as load_decoder_model,
        )
        from ...thirdparty.fish_speech_s1.fish_speech.models.text2semantic.inference import (
            launch_thread_safe_queue,
        )
        from ...thirdparty.fish_speech_s1.fish_speech.utils.schema import (
            ServeReferenceAudio,
            ServeTTSRequest,
        )
    elif model_name == FISH_AUDIO_S2_PRO:
        from ...thirdparty.fish_speech_s2.fish_speech.inference_engine import (
            TTSInferenceEngine,
        )
        from ...thirdparty.fish_speech_s2.fish_speech.models.dac.inference import (
            load_model as load_decoder_model,
        )
        from ...thirdparty.fish_speech_s2.fish_speech.models.text2semantic.inference import (
            launch_thread_safe_queue,
        )
        from ...thirdparty.fish_speech_s2.fish_speech.utils.schema import (
            ServeReferenceAudio,
            ServeTTSRequest,
        )
    else:
        # FishSpeech-1.5 uses the legacy upstream package layout.
        legacy_runtime_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../thirdparty/fish_speech")
        )
        if legacy_runtime_path not in sys.path:
            sys.path.insert(0, legacy_runtime_path)

        from tools.inference_engine import TTSInferenceEngine
        from tools.llama.generate import launch_thread_safe_queue
        from tools.schema import ServeReferenceAudio, ServeTTSRequest
        from tools.vqgan.inference import load_model as load_decoder_model

    return (
        TTSInferenceEngine,
        launch_thread_safe_queue,
        load_decoder_model,
        ServeReferenceAudio,
        ServeTTSRequest,
    )


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
        self._serve_reference_audio = None
        self._serve_tts_request = None
        self._uses_modern_runtime = model_spec.model_name in _MODERN_FISH_AUDIO_MODELS
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def load(self):
        (
            TTSInferenceEngine,
            launch_thread_safe_queue,
            load_decoder_model,
            self._serve_reference_audio,
            self._serve_tts_request,
        ) = _load_fish_speech_runtime_components(self._model_spec.model_name)

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

        if self._uses_modern_runtime:
            decoder_config_name = "modded_dac_vq"
            decoder_filename = "codec.pth"
        else:
            decoder_config_name = "firefly_gan_vq"
            decoder_filename = "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"

        checkpoint_path = os.path.join(self._model_path, decoder_filename)
        self._model = load_decoder_model(
            config_name=decoder_config_name,
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
        from .utils import audio_stream_generator, audio_to_bytes

        if self._serve_reference_audio is None or self._serve_tts_request is None:
            raise RuntimeError("Fish speech model is not loaded")

        prompt_speech = kwargs.get("prompt_speech")
        prompt_text = kwargs.get("prompt_text", kwargs.get("reference_text", ""))
        if prompt_speech is not None:
            r = self._serve_reference_audio(audio=prompt_speech, text=prompt_text)
            references = [r]
        else:
            references = []

        if self._uses_modern_runtime:
            default_top_p = 0.8
            default_repetition_penalty = 1.1
            default_temperature = 0.8
        else:
            default_top_p = 0.7
            default_repetition_penalty = 1.2
            default_temperature = 0.7

        assert self._engine is not None
        result = self._engine.inference(
            self._serve_tts_request(
                text=input,
                references=references,
                reference_id=kwargs.get("reference_id"),
                seed=kwargs.get("seed"),
                use_memory_cache=kwargs.get("use_memory_cache", "off"),
                normalize=kwargs.get("normalize", True),
                max_new_tokens=kwargs.get("max_new_tokens", 1024),
                chunk_length=kwargs.get("chunk_length", 200),
                top_p=kwargs.get("top_p", default_top_p),
                repetition_penalty=kwargs.get(
                    "repetition_penalty", default_repetition_penalty
                ),
                temperature=kwargs.get("temperature", default_temperature),
                streaming=stream,
                format=response_format,
            )
        )

        if stream:

            def _gen_chunk():
                for chunk in result:
                    if chunk.code == "error":
                        raise chunk.error or RuntimeError(
                            "Fish speech inference failed"
                        )
                    if chunk.code != "segment" or chunk.audio is None:
                        continue
                    audio = chunk.audio[1]
                    if audio is not None:
                        yield audio

            return audio_stream_generator(
                response_format=response_format,
                sample_rate=self._get_sample_rate(),
                output_generator=_gen_chunk(),
                output_chunk_transformer=lambda c: torch.from_numpy(
                    np.asarray(c).reshape((-1, 1))
                ),
            )
        else:
            final_audio = None
            for chunk in result:
                if chunk.code == "error":
                    raise chunk.error or RuntimeError("Fish speech inference failed")
                if chunk.code == "final":
                    final_audio = chunk.audio
            if final_audio is None:
                raise RuntimeError("Fish speech inference returned no audio")

            sample_rate, audio = final_audio
            audio = np.asarray(audio)
            if audio.ndim == 1:
                audio = audio[None, :]
            return audio_to_bytes(
                response_format=response_format,
                sample_rate=sample_rate,
                tensor=torch.from_numpy(audio),
            )

    def _get_sample_rate(self) -> int:
        if self._model is None:
            raise RuntimeError("Fish speech model is not loaded")
        if hasattr(self._model, "spec_transform"):
            return int(self._model.spec_transform.sample_rate)
        return int(self._model.sample_rate)
