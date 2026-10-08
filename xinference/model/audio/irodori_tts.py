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
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)

_IRODORI_VENDOR_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../thirdparty")
)
_DACVAE_VENDOR_ROOT = os.path.join(_IRODORI_VENDOR_ROOT, "dacvae")


def _ensure_vendored_irodori_source() -> None:
    """Expose bundled Irodori-TTS and DACVAE sources before installed copies."""

    source_paths = {
        "Irodori-TTS": os.path.join(_IRODORI_VENDOR_ROOT, "irodori_tts"),
        "DACVAE": os.path.join(_DACVAE_VENDOR_ROOT, "dacvae"),
    }
    for name, package_path in source_paths.items():
        if not os.path.isdir(package_path):
            raise RuntimeError(
                f"{name} vendored source directory is missing: {package_path}"
            )

    for vendor_root in (_IRODORI_VENDOR_ROOT, _DACVAE_VENDOR_ROOT):
        if vendor_root in sys.path:
            sys.path.remove(vendor_root)
        sys.path.insert(0, vendor_root)


def _load_irodori_runtime_components():
    _ensure_vendored_irodori_source()
    from irodori_tts.inference_runtime import (
        InferenceRuntime,
        RuntimeKey,
        SamplingRequest,
        default_runtime_device,
    )

    return InferenceRuntime, RuntimeKey, SamplingRequest, default_runtime_device


def _audio_to_bytes(response_format: str, sample_rate: int, audio: Any) -> bytes:
    from .utils import audio_to_bytes

    return audio_to_bytes(response_format, sample_rate, audio)


class IrodoriTTSModel:
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
        self._runtime = None
        self._sampling_request_cls = None

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def _checkpoint_path(self) -> Path:
        model_file_name = (
            getattr(self._model_spec, "model_file_name", "model.safetensors")
            or "model.safetensors"
        )
        checkpoint = Path(self._model_path) / model_file_name
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Irodori-TTS checkpoint not found: {checkpoint}. "
                "Check that the selected model variant was fully cached."
            )
        return checkpoint

    def load(self):
        try:
            (
                InferenceRuntime,
                RuntimeKey,
                SamplingRequest,
                default_runtime_device,
            ) = _load_irodori_runtime_components()
        except ImportError as exc:
            raise ImportError(
                "Failed to import vendored Irodori-TTS runtime. Install dependencies "
                "declared in its built-in model specification."
            ) from exc

        model_device = str(self._device or default_runtime_device())
        codec_device = str(self._kwargs.get("codec_device") or model_device)
        checkpoint = self._checkpoint_path()
        logger.info("Loading Irodori-TTS model from %s on %s", checkpoint, model_device)
        self._runtime = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=str(checkpoint),
                model_device=model_device,
                codec_repo=str(
                    self._kwargs.get(
                        "codec_repo", "Aratako/Semantic-DACVAE-Japanese-32dim"
                    )
                ),
                model_precision=str(self._kwargs.get("model_precision", "fp32")),
                codec_device=codec_device,
                codec_precision=str(self._kwargs.get("codec_precision", "fp32")),
                codec_deterministic_encode=bool(
                    self._kwargs.get("codec_deterministic_encode", True)
                ),
                codec_deterministic_decode=bool(
                    self._kwargs.get("codec_deterministic_decode", True)
                ),
                compile_model=bool(self._kwargs.get("compile_model", False)),
                compile_dynamic=bool(self._kwargs.get("compile_dynamic", False)),
            )
        )
        self._sampling_request_cls = SamplingRequest

    @staticmethod
    def _write_temporary_file(content: bytes, suffix: str, paths: list[str]) -> str:
        if not isinstance(content, bytes):
            raise ValueError("Irodori-TTS reference input must be bytes.")
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as file:
                file.write(content)
        except Exception:
            os.unlink(path)
            raise
        paths.append(path)
        return path

    def speech(
        self,
        input: str,
        voice: Optional[str],
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs,
    ):
        if stream:
            raise ValueError("Irodori-TTS does not support streaming generation.")
        if self._runtime is None or self._sampling_request_cls is None:
            raise RuntimeError("Irodori-TTS model is not loaded.")
        if speed != 1.0:
            logger.warning("Irodori-TTS does not support speed; ignoring it.")
        if voice:
            logger.warning(
                "Irodori-TTS does not support named voices; ignoring voice=%r.", voice
            )

        prompt_speech = kwargs.pop("prompt_speech", None)
        prompt_latent = kwargs.pop("prompt_latent", None)
        caption = kwargs.pop("caption", None)
        if caption is None:
            caption = kwargs.pop("instruct", kwargs.pop("instruction", None))
        prompt_text = kwargs.pop("prompt_text", None)
        if caption is None:
            caption = prompt_text

        request_fields = (
            "ref_wav",
            "ref_wavs",
            "ref_latent",
            "ref_latents",
            "ref_embed",
            "no_ref",
            "ref_normalize_db",
            "ref_ensure_max",
            "num_candidates",
            "decode_mode",
            "seconds",
            "duration_scale",
            "min_seconds",
            "max_seconds",
            "max_ref_seconds",
            "max_text_len",
            "max_caption_len",
            "num_steps",
            "cfg_scale_text",
            "cfg_scale_caption",
            "cfg_scale_speaker",
            "cfg_guidance_mode",
            "cfg_scale",
            "cfg_min_t",
            "cfg_max_t",
            "truncation_factor",
            "rescale_k",
            "rescale_sigma",
            "context_kv_cache",
            "speaker_kv_scale",
            "speaker_kv_min_t",
            "speaker_kv_max_layers",
            "speaker_uncond_mode",
            "seed",
            "t_schedule_mode",
            "sway_coeff",
            "trim_tail",
            "tail_window_size",
            "tail_std_threshold",
            "tail_mean_threshold",
            "lora_adapter",
        )
        request_kwargs = {
            field: kwargs.pop(field) for field in request_fields if field in kwargs
        }
        if kwargs:
            logger.warning("Ignoring unsupported Irodori-TTS speech kwargs: %s", kwargs)

        temporary_paths: list[str] = []
        try:
            has_file_reference = any(
                request_kwargs.get(field)
                for field in (
                    "ref_wav",
                    "ref_wavs",
                    "ref_latent",
                    "ref_latents",
                    "ref_embed",
                )
            )
            if prompt_speech is not None:
                if has_file_reference:
                    raise ValueError(
                        "prompt_speech cannot be combined with Irodori-TTS reference paths."
                    )
                request_kwargs["ref_wav"] = self._write_temporary_file(
                    prompt_speech, ".wav", temporary_paths
                )
            if prompt_latent is not None:
                if prompt_speech is not None or has_file_reference:
                    raise ValueError(
                        "prompt_latent cannot be combined with other Irodori-TTS reference inputs."
                    )
                request_kwargs["ref_latent"] = self._write_temporary_file(
                    prompt_latent, ".pt", temporary_paths
                )

            has_reference = any(
                request_kwargs.get(field)
                for field in (
                    "ref_wav",
                    "ref_wavs",
                    "ref_latent",
                    "ref_latents",
                    "ref_embed",
                )
            )
            request_kwargs.setdefault("no_ref", not has_reference)
            result = self._runtime.synthesize(
                self._sampling_request_cls(
                    text=input,
                    caption=caption,
                    **request_kwargs,
                )
            )
            return _audio_to_bytes(response_format, result.sample_rate, result.audio)
        finally:
            for path in temporary_paths:
                try:
                    os.unlink(path)
                except OSError:
                    logger.warning(
                        "Failed to remove Irodori-TTS temporary file %s", path
                    )
