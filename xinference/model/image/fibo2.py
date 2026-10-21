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

import os
import re
from typing import Any, List, Optional, Union

from PIL import Image, ImageFilter

from .stable_diffusion.core import DiffusionModel


class Fibo2ImageModel(DiffusionModel):
    def load(self) -> None:
        import torch
        from diffusers import BriaFibo2Pipeline

        assert self._model_path is not None
        if self._lora_model or self._gguf_model_path or self._lightning_model_path:
            raise ValueError(
                "Fibo 2.0 does not support LoRA, GGUF or Lightning models."
            )

        kwargs = self._kwargs.copy()
        dtype = kwargs.pop("torch_dtype", None) or torch.bfloat16
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.removeprefix("torch."))
        self._torch_dtype = dtype
        kwargs["torch_dtype"] = dtype
        for option in (
            "cpu_offload",
            "sequential_cpu_offload",
            "attention_slicing",
            "vae_tiling",
            "vae_slicing",
        ):
            kwargs.pop(option, None)
        self._model = BriaFibo2Pipeline.from_pretrained(self._model_path, **kwargs)
        self._load_to_device(self._model)
        # Both hubs ship the fonts used to render the caption's text images.
        self._fonts = [
            path
            for name in ("Aileron-Regular.otf", "DejaVuSans.ttf")
            if os.path.isfile(path := os.path.join(self._model_path, "fonts", name))
        ]

    def _need_set_scheduler(self, scheduler: Any) -> bool:
        # Keep the checkpoint's image-size-dependent flow-matching schedule.
        return False

    def _call_model(
        self,
        response_format: str,
        model: Any = None,
        _num_pipeline_calls: int = 1,
        **kwargs,
    ):
        kwargs.setdefault("fonts", self._fonts)
        # The OpenAI edit endpoint supplies a whitespace negative prompt;
        # leave the pipeline's trained null caption in effect for that default.
        negative_prompt = kwargs.get("negative_prompt")
        if isinstance(negative_prompt, str) and not negative_prompt.strip():
            kwargs.pop("negative_prompt")
        return super()._call_model(
            response_format, model, _num_pipeline_calls, **kwargs
        )

    async def text_to_image(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        **kwargs,
    ):
        if "text2image" not in self._abilities:
            raise RuntimeError(f"{self._model_uid} does not support text2image")
        return await super().text_to_image(prompt, n, size, response_format, **kwargs)

    def image_to_image(
        self,
        image: Union[Image.Image, List[Image.Image]],
        prompt: Optional[Union[str, List[str]]] = None,
        n: int = 1,
        size: Optional[str] = None,
        response_format: str = "url",
        **kwargs,
    ):
        assert self._model_spec is not None
        generate_kwargs = (self._model_spec.default_generate_config or {}).copy()
        generate_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
        references = generate_kwargs.pop("reference_images", None)
        images = image if isinstance(image, list) else [image]
        if references:
            images = images + (
                references if isinstance(references, list) else [references]
            )
        if size:
            width, height = map(int, re.split(r"[^\d]+", size))
            generate_kwargs.update(width=width, height=height)
        # Without an explicit size, the pipeline chooses a trained resolution
        # from the first image's aspect ratio. Its image processor handles masks.
        return self._call_model(
            response_format,
            image=self._ensure_three_channel_image(images),
            prompt=prompt,
            num_images_per_prompt=n,
            **generate_kwargs,
        )

    def inpainting(
        self,
        image: Image.Image,
        mask_image: Image.Image,
        prompt: Optional[Union[str, List[str]]] = None,
        n: int = 1,
        size: Optional[str] = None,
        response_format: str = "url",
        **kwargs,
    ):
        if mask_blur := kwargs.pop("mask_blur", None):
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(mask_blur))
        return self.image_to_image(
            image, prompt, n, size, response_format, mask=mask_image, **kwargs
        )
