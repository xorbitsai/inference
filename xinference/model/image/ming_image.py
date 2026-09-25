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

"""Adapter for the upstream Ming-Image inference implementation."""

import importlib
import logging
import random
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from PIL import Image as PILImage

from .sdapi import SDAPIDiffusionModelMixin
from .utils import handle_image_result, resolve_image_seed_list

if TYPE_CHECKING:
    from .core import ImageModelFamilyV2

logger = logging.getLogger(__name__)


class MingImageModel(SDAPIDiffusionModelMixin):
    """Expose generation, editing, and layer decomposition through image APIs."""

    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "ImageModelFamilyV2",
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.model_family = self._model_spec = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device or "cuda:0"
        self._kwargs = kwargs
        self._model = None
        self._processor = None
        self._profile = None
        self._infer: Optional[Any] = None
        self._dtype = None

    @property
    def model_spec(self) -> "ImageModelFamilyV2":
        return self._model_spec

    @property
    def model_ability(self) -> List[str]:
        return self._model_spec.model_ability or []

    def load(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Ming-Image requires a CUDA GPU")
        infer = importlib.import_module("xinference.thirdparty.ming_image.infer")
        profile = infer.load_checkpoint_capabilities(self._model_path)
        expected_task = (
            "layer-decompose"
            if self._model_spec.model_name.endswith("-Layer")
            else "text-to-image"
        )
        profile.validate_task(
            expected_task,
            has_reference_image=expected_task == "layer-decompose",
            num_layers=1,
        )

        dtype_name = self._kwargs.get("torch_dtype", "bfloat16")
        if isinstance(dtype_name, torch.dtype):
            dtype_name = str(dtype_name).removeprefix("torch.")
        if dtype_name not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"Unsupported Ming-Image torch_dtype: {dtype_name}")
        device_map = self._kwargs.get("device_map", "none")
        if device_map not in ("none", "auto", "balanced"):
            raise ValueError(f"Unsupported Ming-Image device_map: {device_map}")
        args = SimpleNamespace(
            processor=self._kwargs.get("processor"),
            dtype=dtype_name,
            attn_implementation=self._kwargs.get("attn_implementation", "eager"),
            device=self._device,
            device_map=device_map,
            num_gpus=self._kwargs.get("num_gpus", torch.cuda.device_count()),
        )
        self._model, self._processor = infer.load_model_and_processor(
            Path(self._model_path), args
        )
        self._profile = profile
        self._infer = infer
        self._dtype = infer._dtype(dtype_name)

    def _generate(
        self,
        task: str,
        prompt: Union[str, List[str]],
        image: Optional[PILImage.Image],
        n: int,
        size: Optional[str],
        response_format: str,
        kwargs: Dict[str, Any],
    ):
        import torch

        if self._model is None or self._processor is None or self._infer is None:
            raise RuntimeError("Ming-Image model is not loaded")
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise ValueError("Ming-Image accepts exactly one prompt")
            prompt = prompt[0]
        if not isinstance(prompt, str):
            raise ValueError("Ming-Image requires a string prompt")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        negative_prompt = kwargs.get("negative_prompt")
        if negative_prompt and (
            not isinstance(negative_prompt, str) or negative_prompt.strip()
        ):
            raise ValueError("Ming-Image does not support negative_prompt")

        config = (self._model_spec.default_generate_config or {}).copy()
        config.update(
            {key: value for key, value in kwargs.items() if value is not None}
        )
        if task != "layer-decompose" and config.get("num_layers") is not None:
            raise ValueError(
                "num_layers is only supported by Ming-Image-0.1-Design-Layer"
            )
        resolution = config.get("resolution")
        if size:
            dimensions = [int(value) for value in re.split(r"\D+", size) if value]
            if len(dimensions) != 2 or min(dimensions) <= 0:
                raise ValueError(f"Invalid image size: {size!r}")
            resolution = max(dimensions)
        resolution = self._infer.resolve_task_resolution(task, resolution)
        steps = config.get("num_inference_steps", config.get("steps"))
        cfg = config.get("guidance_scale", config.get("cfg"))
        sampling = self._profile.resolve_sampling_parameters(steps=steps, cfg=cfg)
        num_layers = (
            config.get("num_layers", self._infer.parse_num_layers(prompt))
            if task == "layer-decompose"
            else 1
        )
        if isinstance(num_layers, bool) or not isinstance(num_layers, int):
            raise ValueError("num_layers must be an integer")
        self._profile.validate_task(
            task, has_reference_image=image is not None, num_layers=num_layers
        )
        seeds = resolve_image_seed_list(config.get("seed"), n)
        if seeds is None:
            seed = config.get("seed")
            if seed is not None and (
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or seed < -1
                or seed > 2**31 - 1
            ):
                raise ValueError("seed must be -1 or between 0 and 2147483647")
            seeds = [
                (
                    random.randrange(2**31)
                    if seed is None or seed == -1
                    else (seed + i) % (2**31)
                )
                for i in range(n)
            ]

        # Upstream run_generation reads the reference twice, including an RGBA
        # copy for the four-channel VAE. Keep it as a PNG until inference ends.
        temporary = None
        try:
            if image is not None:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as file:
                    temporary = Path(file.name)
                image.save(temporary, format="PNG")
            images = []
            with torch.inference_mode():
                for seed in seeds:
                    output = self._infer.run_generation(
                        self._model,
                        self._processor,
                        self._profile,
                        task=task,
                        prompt=prompt,
                        input_image=temporary,
                        resolution=resolution,
                        sampling=sampling,
                        seed=seed,
                        num_layers=num_layers,
                        dtype=self._dtype,
                    )
                    images.extend(output[1:] if task == "layer-decompose" else output)
            return handle_image_result(response_format, images)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def text_to_image(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        **kwargs: Any,
    ):
        if "text2image" not in self.model_ability:
            raise RuntimeError(f"{self._model_uid} does not support text2image")
        return self._generate(
            "text-to-image", prompt, None, n, size, response_format, kwargs
        )

    def image_to_image(
        self,
        image: Union[PILImage.Image, List[PILImage.Image]],
        prompt: Optional[Union[str, List[str]]] = None,
        n: int = 1,
        size: Optional[str] = None,
        response_format: str = "url",
        **kwargs: Any,
    ):
        if "image2image" not in self.model_ability:
            raise RuntimeError(f"{self._model_uid} does not support image2image")
        if kwargs.get("reference_images"):
            raise ValueError("Ming-Image accepts exactly one reference image")
        if isinstance(image, list):
            if len(image) != 1:
                raise ValueError("Ming-Image accepts exactly one reference image")
            image = image[0]
        if not isinstance(image, PILImage.Image):
            raise ValueError("Ming-Image requires a PIL reference image")
        task = (
            "layer-decompose"
            if self._model_spec.model_name.endswith("-Layer")
            else "image-edit"
        )
        if prompt is None and task == "layer-decompose":
            prompt = f"Decompose this image into {kwargs.get('num_layers', 5)} layers."
        if prompt is None:
            raise ValueError("Ming-Image image editing requires a prompt")
        return self._generate(task, prompt, image, n, size, response_format, kwargs)
