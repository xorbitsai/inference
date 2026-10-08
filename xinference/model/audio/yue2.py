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

import io
import os
import sys
from typing import TYPE_CHECKING, Any, Optional

from ...device_utils import get_available_device

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2


_YUE2_VENDOR_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../thirdparty/yue2/src")
)


def _ensure_vendored_source_path() -> None:
    """Expose the pinned YuE2 runtime without installing it from Git."""
    if not os.path.isdir(_YUE2_VENDOR_ROOT):
        raise RuntimeError(
            f"YuE2 vendored source directory is missing: {_YUE2_VENDOR_ROOT}"
        )
    if _YUE2_VENDOR_ROOT in sys.path:
        sys.path.remove(_YUE2_VENDOR_ROOT)
    sys.path.insert(0, _YUE2_VENDOR_ROOT)


class YuE2Model:
    """YuE2 text-to-music model backed by the official Python pipeline."""

    _response_formats = {"flac", "mp3", "ogg", "wav"}
    _initialize_options = {
        "backend",
        "memory_budget_gib",
        "offload_ar",
        "quantization",
        "vae_core_frames",
        "verify_hashes",
    }
    _generation_options = {
        "abc",
        "abc_sampling",
        "cfg_scale",
        "cot",
        "id",
        "semantic_sampling",
    }

    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "AudioModelFamilyV2",
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._device = device
        self._kwargs = kwargs
        self._model = None

    @property
    def model_spec(self) -> "AudioModelFamilyV2":
        return self._model_spec

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def _load_config(self) -> dict[str, Any]:
        config = (self._model_spec.default_model_config or {}).copy()
        config.update(self._kwargs)
        unknown = sorted(set(config) - self._initialize_options)
        if unknown:
            raise ValueError("Unsupported YuE2 load option(s): " + ", ".join(unknown))
        return config

    def _download_vae(self) -> str:
        vae_model_id = getattr(self._model_spec, "vae_model_id", None)
        vae_model_revision = getattr(self._model_spec, "vae_model_revision", None)
        if not vae_model_id:
            raise ValueError("YuE2 model specification is missing `vae_model_id`.")

        if self._model_spec.model_hub == "modelscope":
            from modelscope.hub.snapshot_download import snapshot_download
        else:
            from huggingface_hub import snapshot_download

        from ..utils import retry_snapshot_download

        return retry_snapshot_download(
            snapshot_download,
            vae_model_id,
            None,
            vae_model_id,
            revision=vae_model_revision,
        )

    def load(self) -> None:
        config = self._load_config()
        _ensure_vendored_source_path()
        try:
            import torch
            from yue2 import YuE2Pipeline
        except ImportError as e:
            raise ImportError(
                "YuE2 requires the official yue2 runtime. Enable the model "
                "virtual environment or install the dependencies declared in "
                "its built-in model specification."
            ) from e

        if (
            not torch.cuda.is_available()
            or getattr(torch.version, "hip", None) is not None
            or not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "YuE2 inference requires an NVIDIA CUDA device with BF16 support; "
                "CPU, MPS, and ROCm are not supported."
            )

        device = self._device or get_available_device()
        if torch.device(device).type != "cuda":
            raise ValueError(f"YuE2 requires a CUDA device, but received {device!r}.")
        self._device = str(torch.device(device))

        vae_path = self._download_vae()
        self._model = YuE2Pipeline(
            self._model_path,
            vae_path,
            device=self._device,
            progress=False,
            **config,
        )

    @classmethod
    def _validate_speech_request(
        cls,
        input: str,
        instruct: Any,
        voice: Optional[str],
        response_format: Optional[str],
        speed: Optional[float],
        stream: Optional[bool],
        seed: Any,
        kwargs: dict[str, Any],
    ) -> str:
        if not isinstance(input, str) or not input.strip():
            raise ValueError("YuE2 requires non-empty lyrics in `input`.")
        if not isinstance(instruct, str) or not instruct.strip():
            raise ValueError(
                "YuE2 requires a non-empty music description in `instruct`."
            )
        if voice not in (None, "", "default"):
            raise ValueError(
                "YuE2 only accepts `voice` as null, an empty string, or 'default'."
            )
        if response_format is not None and not isinstance(response_format, str):
            raise ValueError("YuE2 `response_format` must be a string.")
        audio_format = (response_format or "flac").lower()
        if audio_format not in cls._response_formats:
            formats = ", ".join(sorted(cls._response_formats))
            raise ValueError(f"YuE2 supports these response formats: {formats}.")
        if speed != 1.0:
            raise ValueError("YuE2 only supports `speed=1.0`.")
        if stream is not False:
            raise ValueError("YuE2 only supports non-streaming generation.")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("YuE2 `seed` must be an integer in [0, 2**63).")
        unknown = sorted(set(kwargs) - cls._generation_options)
        if unknown:
            raise ValueError(
                "YuE2 does not support speech parameter(s): " + ", ".join(unknown)
            )
        return audio_format

    @staticmethod
    def _audio_to_bytes(audio, sample_rate: int, response_format: str) -> bytes:
        import numpy as np
        import soundfile

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2 or audio.shape[1] != 2:
            raise RuntimeError(
                f"YuE2 returned audio with unexpected shape {audio.shape}; "
                "stereo output was expected."
            )

        subtype = {"flac": "PCM_24", "wav": "FLOAT"}.get(response_format)
        with io.BytesIO() as output:
            soundfile.write(
                output,
                audio,
                sample_rate,
                format=response_format.upper(),
                subtype=subtype,
            )
            return output.getvalue()

    def speech(
        self,
        input: str,
        voice: Optional[str] = None,
        response_format: Optional[str] = "flac",
        speed: Optional[float] = 1.0,
        stream: Optional[bool] = False,
        **kwargs: Any,
    ) -> bytes:
        assert self._model is not None
        kwargs.pop("duration", None)
        instruct = kwargs.pop("instruct", None)
        seed = kwargs.pop("seed", 831001)
        audio_format = self._validate_speech_request(
            input,
            instruct,
            voice,
            response_format,
            speed,
            stream,
            seed,
            kwargs,
        )
        song = self._model(
            style=instruct.strip(),
            lyrics=input,
            seed=seed,
            **kwargs,
        )
        return self._audio_to_bytes(song.audio, song.sample_rate, audio_format)

    def stop(self) -> None:
        if self._model is not None:
            self._model.close()
            self._model = None
