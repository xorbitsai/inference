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

from typing import Any

from ..utils import allow_trust_remote_code
from .stable_diffusion.core import DiffusionModel


class MOSSImageModel(DiffusionModel):
    def load(self) -> None:
        import torch
        from diffusers import DiffusionPipeline

        if self._lora_model or self._gguf_model_path or self._lightning_model_path:
            raise ValueError(
                "MOSS-Image does not support LoRA, GGUF or Lightning models."
            )

        kwargs = self._kwargs.copy()
        dtype = kwargs.pop("torch_dtype", None) or torch.bfloat16
        # The official pipeline samples noise and decodes the FLUX.2 VAE in FP32.
        dtypes = {"default": torch.bfloat16, "vae": torch.float32}
        dtypes.update(dtype if isinstance(dtype, dict) else {"default": dtype})
        dtypes = {
            name: (
                getattr(torch, value.removeprefix("torch."))
                if isinstance(value, str)
                else value
            )
            for name, value in dtypes.items()
        }
        self._torch_dtype = dtypes["default"]
        kwargs["torch_dtype"] = dtypes
        kwargs.setdefault("custom_pipeline", "pipeline_moss_image")
        allowed = allow_trust_remote_code(self.model_family)
        kwargs["trust_remote_code"] = (
            bool(kwargs.get("trust_remote_code", allowed)) and allowed
        )
        if not kwargs["trust_remote_code"]:
            raise ValueError(
                "Loading MOSS-Image executes code shipped in the model repository; "
                "set XINFERENCE_TRUST_REMOTE_CODE=1 to allow custom models, "
                "and remove any explicit trust_remote_code=False setting."
            )
        for option in (
            "cpu_offload",
            "sequential_cpu_offload",
            "attention_slicing",
            "vae_tiling",
            "vae_slicing",
        ):
            kwargs.pop(option, None)

        model: Any = DiffusionPipeline.from_pretrained(self._model_path, **kwargs)
        model.scheduler = type(model.scheduler).from_config(model.scheduler.config)
        self._model = model
        self._load_to_device(model)

    def _need_set_scheduler(self, scheduler: Any) -> bool:
        # MOSSImageScheduler implements the model's shifted, nearest-grid schedule.
        return False

    def _call_model(
        self, response_format: str, model=None, _num_pipeline_calls: int = 1, **kwargs
    ):
        model = model if model is not None else self._model
        progressor = kwargs.pop("progressor", None)
        cancel_event = kwargs.pop("_cancel_event", None)
        if progressor is None and cancel_event is None:
            return super()._call_model(
                response_format,
                model=model,
                _num_pipeline_calls=_num_pipeline_calls,
                **kwargs,
            )

        # The bundled pipeline has no step callback; its progress iterator yields
        # once per denoising step. Restore it even when generation is cancelled.
        original_progress_bar = model.progress_bar

        def progress_bar(timesteps):
            for step, timestep in enumerate(original_progress_bar(timesteps)):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Image generation cancelled")
                yield timestep
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Image generation cancelled")
                if progressor is not None and progressor.request_id:
                    progressor.set_progress((step + 1) / len(timesteps))

        model.progress_bar = progress_bar
        try:
            return super()._call_model(
                response_format,
                model=model,
                _num_pipeline_calls=_num_pipeline_calls,
                **kwargs,
            )
        finally:
            model.progress_bar = original_progress_bar
