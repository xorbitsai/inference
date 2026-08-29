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
import math
import os
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

if TYPE_CHECKING:
    import torch

    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)

_DEFAULT_INSTRUCTION = "Speak clearly and naturally."
_DEFAULT_MAX_NEW_TOKENS = 1500
_DEFAULT_MAX_SEQ_LEN = 2048
_DEFAULT_REPETITION_PENALTY = 1.1
_FAST_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "thirdparty"
    / "breeze_tts"
    / "configs"
    / "fast.json"
)


def _load_breeze_runtime_components():
    from ...thirdparty.breeze_tts.breeze_infer.runtime import (
        load_runtime,
        resolve_device,
        set_all_seeds,
        update_generation_config_for_breeze,
    )
    from ...thirdparty.breeze_tts.breeze_infer.templates import (
        get_template,
        prepare_inputs,
    )
    from ...thirdparty.breeze_tts.models.fast_streaming import (
        FastBreezeStreamingRuntime,
        FastStreamingConfig,
    )
    from ...thirdparty.breeze_tts.models.warmup_profile import load_warmup_profile

    return (
        load_runtime,
        resolve_device,
        set_all_seeds,
        update_generation_config_for_breeze,
        get_template,
        prepare_inputs,
        FastBreezeStreamingRuntime,
        FastStreamingConfig,
        load_warmup_profile,
    )


def _validate_cfg_scale(value: Any) -> float:
    try:
        cfg_scale = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"cfg_scale must be greater than 0, got {value!r}") from None
    if not math.isfinite(cfg_scale) or cfg_scale <= 0:
        raise ValueError(f"cfg_scale must be greater than 0, got {value!r}")
    return cfg_scale


def _audio_to_bytes(
    response_format: str, sample_rate: int, audio: "torch.Tensor"
) -> bytes:
    from .utils import audio_to_bytes

    return audio_to_bytes(response_format, sample_rate, audio)


def _audio_stream_generator(
    response_format: str, sample_rate: int, chunks: Iterator[Any]
):
    import numpy as np
    import torch

    from .utils import audio_stream_generator

    return audio_stream_generator(
        response_format=response_format,
        sample_rate=sample_rate,
        output_generator=chunks,
        output_chunk_transformer=lambda chunk: torch.from_numpy(
            np.asarray(chunk.audio, dtype=np.float32)
        ).reshape(-1, 1),
    )


class BreezeTTS2Model:
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
        self._kwargs = kwargs
        self._model = None
        self._tokenizer = None
        self._audio_tokenizer = None
        self._runtime = None
        self._set_all_seeds = None
        self._get_template = None
        self._prepare_inputs = None

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def load(self):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Breeze-TTS-2 requires a CUDA-capable GPU.")
        if self._device is not None and not str(self._device).startswith("cuda"):
            raise ValueError(
                f"Breeze-TTS-2 only supports CUDA devices, got {self._device!r}."
            )

        (
            load_runtime,
            resolve_device,
            self._set_all_seeds,
            update_generation_config_for_breeze,
            self._get_template,
            self._prepare_inputs,
            FastBreezeStreamingRuntime,
            FastStreamingConfig,
            load_warmup_profile,
        ) = _load_breeze_runtime_components()

        device = resolve_device(None if self._device is None else str(self._device))
        logger.info("Loading Breeze-TTS-2 model on %s...", device)
        self._tokenizer, self._model, self._audio_tokenizer = load_runtime(
            Path(self._model_path),
            device=device,
            attn_implementation=self._kwargs.get("attn_implementation", "eager"),
        )
        update_generation_config_for_breeze(self._model)

        runtime_config = FastStreamingConfig(
            max_new_tokens=int(
                self._kwargs.get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS)
            ),
            max_seq_len=int(self._kwargs.get("max_seq_len", _DEFAULT_MAX_SEQ_LEN)),
            collect_timing=bool(self._kwargs.get("collect_timing", False)),
            fast_all=self._kwargs.get("fast_all"),
            fast_text_encoder=bool(self._kwargs.get("fast_text_encoder", False)),
            fast_backbone_prefill=bool(
                self._kwargs.get("fast_backbone_prefill", False)
            ),
            fast_backbone_decode=bool(self._kwargs.get("fast_backbone_decode", False)),
            fast_depth_decoder=bool(self._kwargs.get("fast_depth_decoder", False)),
            fast_codec=bool(self._kwargs.get("fast_codec", False)),
            temperature=self._kwargs.get("temperature"),
            top_k=self._kwargs.get("top_k"),
            top_p=self._kwargs.get("top_p"),
            do_sample=self._kwargs.get("do_sample"),
            repetition_penalty=float(
                self._kwargs.get("repetition_penalty", _DEFAULT_REPETITION_PENALTY)
            ),
        )
        self._runtime = FastBreezeStreamingRuntime(
            self._model,
            self._audio_tokenizer,
            runtime_config,
            tokenizer=self._tokenizer,
        )
        if self._runtime.fast_enabled:
            profile = load_warmup_profile(_FAST_CONFIG)
            profile = replace(
                profile, codec_chunk_frames=self._runtime.codec_chunk_frames
            )
            manifest = self._runtime.warmup_from_profile(profile)
            logger.info(
                "Breeze-TTS-2 fast runtime warmup completed in %.2f ms.",
                manifest["total_elapsed_ms"],
            )

    @staticmethod
    def _save_prompt_audio(prompt_speech: bytes) -> str:
        fd, path = tempfile.mkstemp(prefix="breeze_ref_", suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as prompt_file:
                prompt_file.write(prompt_speech)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(path)
            except OSError:
                pass
            raise
        return path

    @staticmethod
    def _pop_instruction(kwargs: dict) -> Optional[str]:
        instruction = None
        for name in ("instruct", "instruction", "instruct_text"):
            value = kwargs.pop(name, None)
            if instruction is None and value is not None:
                instruction = str(value).strip()
        return instruction

    def _prepare_request(
        self,
        input: str,
        voice: str,
        prompt_speech: Optional[bytes],
        prompt_text: Optional[str],
        instruction: str,
        cfg_scale: float,
        speaker: str,
    ):
        assert self._runtime is not None
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._audio_tokenizer is not None
        assert self._get_template is not None
        assert self._prepare_inputs is not None

        if voice and voice != "S0":
            logger.warning(
                "Breeze-TTS-2 does not use preset OpenAI voices; ignoring voice=%r.",
                voice,
            )

        request_id = f"{self._model_uid}-{uuid.uuid4().hex}"
        request = {
            "id": request_id,
            "text": input,
            "instruction": instruction,
            "speaker": speaker,
        }
        template_name = "tts_instruction"
        prompt_path = None
        try:
            if prompt_speech is not None:
                prompt_path = self._save_prompt_audio(prompt_speech)
                request["ref_audio_path"] = prompt_path
                request["ref_text"] = prompt_text
                template_name = "ref_edit_tata"

            inputs = self._prepare_inputs(
                self._tokenizer,
                self._audio_tokenizer,
                self._model,
                [request],
                self._get_template(template_name),
                guidance_scale=cfg_scale,
                guidance_scale_ref=None,
                guidance_scale_ins=None,
            )
        finally:
            if prompt_path is not None:
                try:
                    os.unlink(prompt_path)
                except OSError:
                    logger.warning("Failed to remove temporary audio %s", prompt_path)
        return request_id, inputs

    def _stream_audio(
        self,
        inputs: dict,
        request_id: str,
        response_format: str,
        seed: int,
    ):
        assert self._runtime is not None
        assert self._set_all_seeds is not None
        self._set_all_seeds(seed)
        chunks = self._runtime.iter_audio_chunks(inputs, request_id=request_id)
        yield from _audio_stream_generator(
            response_format, int(self._runtime.sample_rate), chunks
        )

    def _generate_audio(
        self,
        inputs: dict,
        request_id: str,
        response_format: str,
        seed: int,
    ) -> bytes:
        import numpy as np
        import torch

        assert self._runtime is not None
        assert self._set_all_seeds is not None
        self._set_all_seeds(seed)
        sample_rate = int(self._runtime.sample_rate)
        audio_parts = []
        for chunk in self._runtime.iter_audio_chunks(inputs, request_id=request_id):
            if int(chunk.sample_rate) != sample_rate:
                raise RuntimeError("Breeze-TTS-2 returned inconsistent sample rates.")
            audio_parts.append(np.asarray(chunk.audio, dtype=np.float32).reshape(-1))
        if not audio_parts:
            raise RuntimeError("Breeze-TTS-2 returned no generated audio.")
        audio = torch.from_numpy(np.concatenate(audio_parts)).unsqueeze(0)
        return _audio_to_bytes(response_format, sample_rate, audio)

    def speech(
        self,
        input: str,
        voice: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs,
    ):
        if not isinstance(input, str) or not input.strip():
            raise ValueError("input must be a non-empty string")
        if self._runtime is None:
            raise RuntimeError("Breeze-TTS-2 model is not loaded")

        prompt_speech = kwargs.pop("prompt_speech", None)
        if prompt_speech is not None:
            if not isinstance(prompt_speech, (bytes, bytearray)) or not prompt_speech:
                raise ValueError("prompt_speech must contain reference audio bytes")
            prompt_speech = bytes(prompt_speech)

        prompt_text = kwargs.pop("prompt_text", None)
        if prompt_text is not None:
            prompt_text = str(prompt_text).strip()
        if prompt_speech is not None and not prompt_text:
            raise ValueError(
                "Breeze-TTS-2 requires the reference audio transcript in "
                "`prompt_text` when `prompt_speech` is provided."
            )

        instruction = self._pop_instruction(kwargs)
        if prompt_speech is None and not instruction and prompt_text:
            instruction = prompt_text
        instruction = instruction or _DEFAULT_INSTRUCTION

        cfg_scale = kwargs.pop("cfg_scale", None)
        guidance_scale = kwargs.pop("guidance_scale", None)
        if cfg_scale is None:
            cfg_scale = guidance_scale if guidance_scale is not None else 1.0
        cfg_scale = _validate_cfg_scale(cfg_scale)
        seed = int(kwargs.pop("seed", 42))
        speaker = str(kwargs.pop("speaker", "S0") or "S0")

        if speed is not None and float(speed) != 1.0:
            logger.warning("Breeze-TTS-2 does not support speed; ignoring it.")
        if kwargs:
            logger.warning(
                "Ignoring unsupported Breeze-TTS-2 speech kwargs: %s", kwargs
            )

        request_id, inputs = self._prepare_request(
            input=input,
            voice=voice,
            prompt_speech=prompt_speech,
            prompt_text=prompt_text,
            instruction=instruction,
            cfg_scale=cfg_scale,
            speaker=speaker,
        )
        response_format = response_format or "mp3"
        if stream:
            return self._stream_audio(inputs, request_id, response_format, seed)
        return self._generate_audio(inputs, request_id, response_format, seed)
