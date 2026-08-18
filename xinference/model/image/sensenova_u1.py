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

import contextlib
import logging
import os
import re
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import PIL.Image

from .sdapi import SDAPIDiffusionModelMixin
from .utils import handle_image_result

if TYPE_CHECKING:
    from ...types import LoRA
    from .core import ImageModelFamilyV2

logger = logging.getLogger(__name__)

_IMAGE_GRID_FACTOR = 32
_DEFAULT_TARGET_PIXELS = 2048 * 2048
_DEFAULT_INPUT_MAX_PIXELS = 2048 * 2048
_MIN_INPUT_MAX_PIXELS = 512 * 512


class SenseNovaU1Model(SDAPIDiffusionModelMixin):
    """SenseNova-U1 image generation and editing model."""

    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "ImageModelFamilyV2",
        device: Optional[str] = None,
        lora_model: Optional[List["LoRA"]] = None,
        lora_load_kwargs: Optional[Dict[str, Any]] = None,
        lora_fuse_kwargs: Optional[Dict[str, Any]] = None,
        gguf_model_path: Optional[str] = None,
        lightning_model_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._abilities = model_spec.model_ability or []
        self._device = device
        self._lora_model = lora_model
        self._lora_load_kwargs = lora_load_kwargs or {}
        self._lora_fuse_kwargs = lora_fuse_kwargs or {}
        self._gguf_model_path = gguf_model_path
        self._lightning_model_path = lightning_model_path
        self._kwargs = kwargs
        self._model = None
        self._tokenizer = None
        self._prefetch_count = 0
        self._vram_mode = "full"
        self._offload_kwargs: Dict[str, Any] = {}

    @property
    def model_spec(self) -> "ImageModelFamilyV2":
        return self._model_spec

    @property
    def model_ability(self) -> List[str]:
        return self._abilities

    def load(self) -> None:
        thirdparty_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../thirdparty")
        )
        if thirdparty_dir not in sys.path:
            sys.path.insert(0, thirdparty_dir)

        import sensenova_u1
        import torch
        from sensenova_u1.utils import (
            DEFAULT_FAST_ACTIVATION_RESERVE_GIB,
            DEFAULT_FAST_VRAM_FRACTION,
            DEFAULT_FAST_VRAM_HEADROOM_GIB,
            DEFAULT_VRAM_MODE,
            best_available_device,
            load_and_merge_lora_weight_from_safetensors,
            load_model_and_tokenizer,
            vram_mode_to_prefetch_count,
        )

        if self._lightning_model_path:
            raise ValueError("SenseNova-U1 does not support lightning models.")

        kwargs = self._kwargs.copy()
        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype.removeprefix("torch."))

        attn_backend = kwargs.pop("attn_backend", "auto")
        sensenova_u1.set_attn_backend(attn_backend)

        self._vram_mode = kwargs.pop("vram_mode", DEFAULT_VRAM_MODE)
        self._prefetch_count = vram_mode_to_prefetch_count(self._vram_mode)
        self._device = self._device or str(best_available_device())
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)

        self._offload_kwargs = {
            "fast_vram_fraction": kwargs.pop(
                "fast_vram_fraction", DEFAULT_FAST_VRAM_FRACTION
            ),
            "fast_vram_headroom_gib": kwargs.pop(
                "fast_vram_headroom_gib", DEFAULT_FAST_VRAM_HEADROOM_GIB
            ),
            "fast_activation_reserve_gib": kwargs.pop(
                "fast_activation_reserve_gib", DEFAULT_FAST_ACTIVATION_RESERVE_GIB
            ),
            "fast_vram_budget_gib": kwargs.pop("fast_vram_budget_gib", None),
        }
        if kwargs:
            logger.warning("Ignore unsupported SenseNova-U1 load options: %s", kwargs)

        self._model, self._tokenizer = load_model_and_tokenizer(
            self._model_path,
            dtype=torch_dtype,
            device=self._device,
            gguf_checkpoint=self._gguf_model_path,
            device_map=device_map,
            max_memory=max_memory,
            for_offload=self._prefetch_count > 0,
        )

        if self._lora_model:
            if self._lora_load_kwargs or self._lora_fuse_kwargs:
                logger.warning("SenseNova-U1 ignores image LoRA load and fuse options.")
            for lora in self._lora_model:
                self._model = load_and_merge_lora_weight_from_safetensors(
                    self._model, lora.local_path
                )

    def _offload_context(self):
        if self._model is None:
            raise RuntimeError("SenseNova-U1 model is not loaded.")
        if self._prefetch_count == 0:
            return contextlib.nullcontext(self._model)

        from sensenova_u1.utils import (
            make_offload_ctx,
            vram_mode_keeps_generation_resident,
        )

        return make_offload_ctx(
            self._model,
            self._prefetch_count,
            self._device,
            keep_generation_resident=vram_mode_keeps_generation_resident(
                self._vram_mode
            ),
            **self._offload_kwargs,
        )

    @staticmethod
    def _to_pil(batch: Any) -> List[PIL.Image.Image]:
        import numpy as np

        batch = (batch.float() * 0.5 + 0.5).clamp(0, 1)
        array = batch.permute(0, 2, 3, 1).cpu().numpy()
        array = (array * 255.0).round().astype(np.uint8)
        return [PIL.Image.fromarray(image) for image in array]

    @staticmethod
    def _parse_size(size: str) -> Tuple[int, int]:
        dimensions = [int(value) for value in re.split(r"[^\d]+", size) if value]
        if len(dimensions) != 2:
            raise ValueError(f"Invalid image size: {size!r}.")
        width, height = dimensions
        SenseNovaU1Model._validate_size(width, height)
        return width, height

    @staticmethod
    def _validate_size(width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Image width and height must be positive.")
        if width % _IMAGE_GRID_FACTOR or height % _IMAGE_GRID_FACTOR:
            raise ValueError(
                "SenseNova-U1 image width and height must both be multiples of "
                f"{_IMAGE_GRID_FACTOR}, got {width}x{height}."
            )

    def _get_generate_config(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        config = (self._model_spec.default_generate_config or {}).copy()
        config.update(
            {key: value for key, value in kwargs.items() if value is not None}
        )

        if "guidance_scale" in config and "cfg_scale" not in kwargs:
            config["cfg_scale"] = config.pop("guidance_scale")
        else:
            config.pop("guidance_scale", None)
        if "num_inference_steps" in config and "num_steps" not in kwargs:
            config["num_steps"] = config.pop("num_inference_steps")
        else:
            config.pop("num_inference_steps", None)

        for key in (
            "negative_prompt",
            "denoising_strength",
            "progressor",
            "request_id",
        ):
            config.pop(key, None)
        if "cfg_interval" in config:
            config["cfg_interval"] = tuple(config["cfg_interval"])
        return config

    @staticmethod
    def _filter_generate_config(
        config: Dict[str, Any], *, image_to_image: bool
    ) -> Dict[str, Any]:
        supported = {
            "cfg_scale",
            "cfg_norm",
            "timestep_shift",
            "enable_timestep_shift",
            "cfg_interval",
            "num_steps",
            "method",
            "t_eps",
            "think_mode",
            "seed",
        }
        if image_to_image:
            supported.add("img_cfg_scale")
        ignored = sorted(set(config) - supported)
        if ignored:
            logger.warning(
                "Ignore unsupported SenseNova-U1 generation options: %s", ignored
            )
        return {key: value for key, value in config.items() if key in supported}

    def text_to_image(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        **kwargs: Any,
    ):
        if "text2image" not in self._abilities:
            raise RuntimeError(f"{self._model_uid} does not support text2image.")
        if self._tokenizer is None:
            raise RuntimeError("SenseNova-U1 model is not loaded.")
        if not isinstance(prompt, str):
            raise ValueError("SenseNova-U1 text-to-image requires a string prompt.")

        import torch

        width, height = self._parse_size(size)
        generate_config = self._filter_generate_config(
            self._get_generate_config(kwargs), image_to_image=False
        )
        with torch.inference_mode(), self._offload_context() as model:
            output = model.t2i_generate(
                self._tokenizer,
                prompt,
                image_size=(width, height),
                batch_size=n,
                **generate_config,
            )
        if generate_config.get("think_mode"):
            output = output[0]
        return handle_image_result(response_format, self._to_pil(output))

    @staticmethod
    def _convert_to_rgb(image: PIL.Image.Image) -> PIL.Image.Image:
        if image.mode == "RGBA":
            background = PIL.Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.getchannel("A"))
            return background
        return image.convert("RGB") if image.mode != "RGB" else image

    @staticmethod
    def _resolve_input_max_pixels(value: Any, num_images: int) -> Optional[int]:
        if value is None:
            return None
        if value == "auto":
            if num_images <= 2:
                return _DEFAULT_INPUT_MAX_PIXELS
            return max(
                _MIN_INPUT_MAX_PIXELS,
                2 * _DEFAULT_INPUT_MAX_PIXELS // num_images,
            )
        pixels = int(value)
        if pixels < _MIN_INPUT_MAX_PIXELS:
            raise ValueError(
                f"input_max_pixels must be at least {_MIN_INPUT_MAX_PIXELS}."
            )
        return pixels

    @staticmethod
    def _smart_resize(image: PIL.Image.Image, target_pixels: int) -> PIL.Image.Image:
        from sensenova_u1.models.neo_unify.utils import smart_resize

        height, width = smart_resize(
            height=image.height,
            width=image.width,
            factor=_IMAGE_GRID_FACTOR,
            min_pixels=target_pixels,
            max_pixels=target_pixels,
        )
        if (width, height) == image.size:
            return image
        resampling = getattr(PIL.Image, "Resampling", PIL.Image)
        return image.resize((width, height), resampling.LANCZOS)

    @classmethod
    def _prepare_input_images(
        cls,
        images: Sequence[PIL.Image.Image],
        *,
        do_resize: bool,
        input_max_pixels: Optional[int],
    ) -> List[PIL.Image.Image]:
        prepared = [cls._convert_to_rgb(image) for image in images]
        if do_resize and input_max_pixels is not None:
            prepared = [
                cls._smart_resize(image, input_max_pixels) for image in prepared
            ]
        return prepared

    @staticmethod
    def _auto_output_size(
        image: PIL.Image.Image, target_pixels: int
    ) -> Tuple[int, int]:
        from sensenova_u1.models.neo_unify.utils import smart_resize

        height, width = smart_resize(
            height=image.height,
            width=image.width,
            factor=_IMAGE_GRID_FACTOR,
            min_pixels=target_pixels,
            max_pixels=target_pixels,
        )
        return width, height

    def image_to_image(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        prompt: Optional[str] = None,
        n: int = 1,
        size: Optional[str] = None,
        response_format: str = "url",
        **kwargs: Any,
    ):
        if "image2image" not in self._abilities:
            raise RuntimeError(f"{self._model_uid} does not support image2image.")
        if self._tokenizer is None:
            raise RuntimeError("SenseNova-U1 model is not loaded.")
        if prompt is None:
            raise ValueError("SenseNova-U1 image editing requires a prompt.")
        if not isinstance(prompt, str):
            raise ValueError("SenseNova-U1 image editing requires a string prompt.")

        import torch

        images = list(image) if isinstance(image, list) else [image]
        reference_images = kwargs.pop("reference_images", []) or []
        if not isinstance(reference_images, list):
            reference_images = [reference_images]
        images.extend(reference_images)
        if not images:
            raise ValueError("SenseNova-U1 image editing requires at least one image.")

        do_resize = kwargs.pop("do_resize", True)
        input_max_pixels = self._resolve_input_max_pixels(
            kwargs.pop("input_max_pixels", "auto"), len(images)
        )
        target_pixels = int(kwargs.pop("target_pixels", _DEFAULT_TARGET_PIXELS))
        images = self._prepare_input_images(
            images,
            do_resize=do_resize,
            input_max_pixels=input_max_pixels,
        )

        if size and size.lower() != "original":
            width, height = self._parse_size(size)
        else:
            width, height = self._auto_output_size(images[0], target_pixels)
            self._validate_size(width, height)

        generate_config = self._filter_generate_config(
            self._get_generate_config(kwargs), image_to_image=True
        )
        with torch.inference_mode(), self._offload_context() as model:
            output = model.it2i_generate(
                self._tokenizer,
                prompt,
                images,
                image_size=(width, height),
                batch_size=n,
                **generate_config,
            )
        if generate_config.get("think_mode"):
            output = output[0]
        return handle_image_result(response_format, self._to_pil(output))
