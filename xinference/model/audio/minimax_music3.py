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
import logging
import struct
from typing import TYPE_CHECKING, Any, Optional

from ...device_utils import get_available_device

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


class MiniMaxMusic3Model:
    """Diffusers-backed MiniMax-Music3 text-to-music model."""

    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "AudioModelFamilyV2",
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._device = device
        self._model = None
        self._kwargs = kwargs

    @property
    def model_spec(self) -> "AudioModelFamilyV2":
        return self._model_spec

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    @staticmethod
    def _resolve_dtype(torch_module, dtype):
        if dtype is None:
            return torch_module.bfloat16
        if isinstance(dtype, str):
            try:
                dtype = getattr(torch_module, dtype)
            except AttributeError as e:
                raise ValueError(
                    f"Unsupported MiniMax-Music3 torch_dtype: {dtype}"
                ) from e
        if dtype is not torch_module.bfloat16:
            raise ValueError(
                "MiniMax-Music3 currently supports torch_dtype=bfloat16 only."
            )
        return dtype

    def load(self):
        try:
            import torch
            from diffusers import ComponentsManager, ModularPipeline
        except ImportError as e:
            raise ImportError(
                "MiniMax-Music3 requires Diffusers with ModularPipeline support. "
                "Enable the model virtual environment or install the dependencies "
                "declared in its built-in model specification."
            ) from e

        if (
            not torch.cuda.is_available()
            or getattr(torch.version, "hip", None) is not None
        ):
            raise RuntimeError(
                "MiniMax-Music3 inference requires an NVIDIA CUDA device; "
                "CPU, MPS, and ROCm are not supported."
            )

        device = self._device or get_available_device()
        if torch.device(device).type != "cuda":
            raise ValueError(
                f"MiniMax-Music3 requires a CUDA device, but received {device!r}."
            )
        self._device = str(torch.device(device))

        config = (self._model_spec.default_model_config or {}).copy()
        config.update(self._kwargs)
        torch_dtype = self._resolve_dtype(torch, config.pop("torch_dtype", None))
        cpu_offload = config.pop("cpu_offload", True)
        group_offload = config.pop("group_offload", False)
        quantization = config.pop("quantization", None)
        if quantization not in (None, "none", "bf16"):
            raise ValueError(
                "MiniMax-Music3 quantization is not supported in this initial "
                "Diffusers integration."
            )

        manager = None
        if cpu_offload or group_offload:
            manager = ComponentsManager()
            manager.enable_auto_cpu_offload(device=self._device)

        pipeline_kwargs = {}
        if manager is not None:
            pipeline_kwargs["components_manager"] = manager
        pipeline = ModularPipeline.from_pretrained(
            self._model_path,
            **pipeline_kwargs,
        )
        pipeline.load_components(dtype=torch_dtype)

        if group_offload:
            from diffusers.hooks import apply_group_offloading

            apply_group_offloading(
                pipeline.language_model,
                onload_device=torch.device(self._device),
                offload_device=torch.device("cpu"),
                offload_type="leaf_level",
                use_stream=True,
            )
        elif manager is None:
            pipeline.to(self._device)

        if config:
            logger.warning(
                "Ignoring unsupported MiniMax-Music3 load option(s): %s",
                ", ".join(sorted(config)),
            )
        self._model = pipeline

    @staticmethod
    def _validate_speech_request(
        input: str,
        prompt_text: Optional[str],
        voice: Optional[str],
        response_format: Optional[str],
        speed: Optional[float],
        stream: Optional[bool],
        seed: int,
        duration: float,
        kwargs: dict,
    ) -> None:
        if not isinstance(input, str) or not input.strip():
            raise ValueError("MiniMax-Music3 requires non-empty lyrics in `input`.")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError(
                "MiniMax-Music3 requires a non-empty music description in "
                "`prompt_text`."
            )
        if voice not in (None, "", "default"):
            raise ValueError(
                "MiniMax-Music3 only accepts `voice` as null, an empty string, "
                "or 'default'."
            )
        if response_format is None or response_format.lower() != "wav":
            raise ValueError(
                "MiniMax-Music3 currently supports `response_format=wav` only."
            )
        if speed != 1.0:
            raise ValueError("MiniMax-Music3 only supports `speed=1.0`.")
        if stream is not False:
            raise ValueError("MiniMax-Music3 only supports non-streaming generation.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("MiniMax-Music3 `seed` must be a non-negative integer.")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not 0.04 <= duration <= 360
        ):
            raise ValueError(
                "MiniMax-Music3 `duration` must be a number from 0.04 to 360 seconds."
            )
        if kwargs:
            raise ValueError(
                "MiniMax-Music3 does not support speech parameter(s): "
                + ", ".join(sorted(kwargs))
            )

    @staticmethod
    def _audio_to_native_wav(audio, sample_rate: int) -> bytes:
        import numpy as np

        if hasattr(audio, "detach"):
            audio = audio.detach().float().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2:
            raise RuntimeError(
                f"MiniMax-Music3 returned audio with unexpected shape {audio.shape}."
            )
        if audio.shape[0] == 2:
            audio = audio.T
        elif audio.shape[1] != 2:
            raise RuntimeError(
                f"MiniMax-Music3 returned audio with unexpected shape {audio.shape}; "
                "stereo output was expected."
            )

        audio = np.ascontiguousarray(audio, dtype="<f4")
        channels = audio.shape[1]
        bytes_per_sample = audio.dtype.itemsize
        block_align = channels * bytes_per_sample
        audio_bytes = audio.tobytes(order="C")
        fmt_chunk = struct.pack(
            "<HHIIHH",
            3,  # WAVE_FORMAT_IEEE_FLOAT
            channels,
            sample_rate,
            sample_rate * block_align,
            block_align,
            bytes_per_sample * 8,
        )
        fact_chunk = struct.pack("<I", audio.shape[0])
        riff_size = 4 + 8 + len(fmt_chunk) + 8 + len(fact_chunk) + 8 + len(audio_bytes)
        with io.BytesIO() as output:
            # IEEE-float WAV preserves the pipeline samples. No sample-rate
            # conversion or integer PCM quantization is applied.
            output.write(b"RIFF")
            output.write(struct.pack("<I", riff_size))
            output.write(b"WAVE")
            output.write(b"fmt ")
            output.write(struct.pack("<I", len(fmt_chunk)))
            output.write(fmt_chunk)
            output.write(b"fact")
            output.write(struct.pack("<I", len(fact_chunk)))
            output.write(fact_chunk)
            output.write(b"data")
            output.write(struct.pack("<I", len(audio_bytes)))
            output.write(audio_bytes)
            return output.getvalue()

    def speech(
        self,
        input: str,
        voice: Optional[str] = None,
        response_format: Optional[str] = "wav",
        speed: Optional[float] = 1.0,
        stream: Optional[bool] = False,
        **kwargs: Any,
    ) -> bytes:
        assert self._model is not None
        prompt_text = kwargs.pop("prompt_text", None)
        seed = kwargs.pop("seed", 0)
        duration = kwargs.pop("duration", 60.0)
        self._validate_speech_request(
            input,
            prompt_text,
            voice,
            response_format,
            speed,
            stream,
            seed,
            duration,
            kwargs,
        )

        import torch

        generator = torch.Generator(device=self._device).manual_seed(seed)
        result = self._model(
            prompt=prompt_text,
            lyrics=input,
            audio_duration=float(duration),
            generator=generator,
            output="audios",
        )
        audio = result[0]
        sample_rate = int(self._model.sampling_rate)
        return self._audio_to_native_wav(audio, sample_rate)
