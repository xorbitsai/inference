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
import gc
import json
import logging
import os
import random
import re
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import PIL.Image

from .sdapi import SDAPIDiffusionModelMixin
from .utils import handle_image_result, resolve_image_seed_list

if TYPE_CHECKING:
    from ...core.progress_tracker import Progressor
    from .core import ImageModelFamilyV2

logger = logging.getLogger(__name__)

HIDREAM_O1_MODEL_NAMES = {
    "HiDream-O1-Image",
    "HiDream-O1-Image-Dev",
    "HiDream-O1-Image-Dev-2604",
}

HIDREAM_O1_DEV_TIMESTEPS = [
    999,
    987,
    974,
    960,
    945,
    929,
    913,
    895,
    877,
    857,
    836,
    814,
    790,
    764,
    737,
    707,
    675,
    640,
    602,
    560,
    515,
    464,
    409,
    347,
    278,
    199,
    110,
    8,
]


def _add_special_tokens(tokenizer: Any) -> None:
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"


def _get_tokenizer(processor: Any) -> Any:
    from transformers import PreTrainedTokenizerBase

    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer


class HiDreamO1Model(SDAPIDiffusionModelMixin):
    def __init__(
        self,
        model_uid: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        model_spec: Optional["ImageModelFamilyV2"] = None,
        **kwargs: Any,
    ) -> None:
        if model_spec is None:
            raise ValueError("model_spec is required for HiDream-O1-Image")
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._model_spec = model_spec
        self._abilities = model_spec.model_ability or []
        self._kwargs = kwargs
        self._model = None
        self._processor = None

    @property
    def model_ability(self) -> List[str]:
        return self._abilities

    @property
    def _is_dev(self) -> bool:
        return "-Dev" in self._model_spec.model_name

    def load(self) -> None:
        import torch
        from packaging.version import Version

        from ...device_utils import get_available_device

        if not self._model_path:
            raise ValueError("model_path is required for HiDream-O1-Image")

        device = self._device or get_available_device()
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("HiDream-O1-Image requires a CUDA-capable GPU")
        torch_version = Version(torch.__version__.split("+", 1)[0])
        if torch_version < Version("2.10"):
            raise RuntimeError(
                "HiDream-O1-Image requires PyTorch 2.10 or newer, "
                f"got {torch.__version__}"
            )

        from transformers import AutoProcessor

        from ...thirdparty.hidream_o1.qwen3_vl_transformers import (
            Qwen3VLForConditionalGeneration,
        )

        torch_dtype = self._kwargs.get("torch_dtype", "bfloat16")
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": self._kwargs.get("device_map", device),
        }
        for key in (
            "low_cpu_mem_usage",
            "max_memory",
            "offload_folder",
            "offload_state_dict",
        ):
            if key in self._kwargs:
                load_kwargs[key] = self._kwargs[key]

        logger.debug(
            "Loading HiDream-O1-Image from %s with %s",
            self._model_path,
            load_kwargs,
        )
        self._processor = AutoProcessor.from_pretrained(self._model_path)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self._model_path, **load_kwargs
        ).eval()
        _add_special_tokens(_get_tokenizer(self._processor))

    def _prepare_generate_config(
        self, ref_count: int, overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        config = dict(self._model_spec.default_generate_config or {})
        config.update(
            {key: value for key, value in overrides.items() if value is not None}
        )

        if self._is_dev:
            config.setdefault("num_inference_steps", 28)
            config.setdefault("guidance_scale", 0.0)
            config.setdefault("shift", 1.0)
            config.setdefault("timesteps_list", HIDREAM_O1_DEV_TIMESTEPS.copy())
            editing_scheduler = config.pop("editing_scheduler", "flow_match")
            default_scheduler = (
                "flow_match"
                if ref_count == 1 and editing_scheduler == "flow_match"
                else "flash"
            )
            config.setdefault("scheduler_name", default_scheduler)
        else:
            config.setdefault("num_inference_steps", 50)
            config.setdefault("guidance_scale", 5.0)
            config.setdefault("shift", 3.0)
            config.setdefault("timesteps_list", None)
            config.setdefault("scheduler_name", "default")

        config.setdefault("use_flash_attn", self._kwargs.get("use_flash_attn", False))
        return config

    @staticmethod
    def _normalize_references(
        image: Union[PIL.Image.Image, Sequence[PIL.Image.Image]],
        reference_images: Optional[
            Union[PIL.Image.Image, Sequence[PIL.Image.Image]]
        ] = None,
    ) -> List[PIL.Image.Image]:
        images = list(image) if isinstance(image, (list, tuple)) else [image]
        if reference_images is not None:
            images.extend(
                reference_images
                if isinstance(reference_images, (list, tuple))
                else [reference_images]
            )
        if not all(isinstance(item, PIL.Image.Image) for item in images):
            raise TypeError("HiDream-O1 reference images must be PIL images")
        return images

    @staticmethod
    @contextlib.contextmanager
    def _release_after():
        from ...device_utils import empty_cache

        try:
            yield
        finally:
            gc.collect()
            empty_cache()

    def _generate(
        self,
        prompt: Union[str, List[str]],
        n: int,
        width: int,
        height: int,
        response_format: str,
        ref_images: Optional[List[PIL.Image.Image]] = None,
        **kwargs: Any,
    ):
        from ...thirdparty.hidream_o1.pipeline import generate_image

        if self._model is None or self._processor is None:
            raise RuntimeError("HiDream-O1-Image is not loaded")
        if n < 1:
            raise ValueError("n must be greater than 0")
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise ValueError("HiDream-O1-Image supports one prompt per request")
            prompt = prompt[0]

        references = ref_images or []
        progressor: Optional["Progressor"] = kwargs.pop("progressor", None)
        config = self._prepare_generate_config(len(references), kwargs)
        seed = config.pop("seed", 32)
        seeds = resolve_image_seed_list(seed, n)
        if seeds is None and (seed is None or seed == -1):
            seed = random.SystemRandom().randrange(0, 2**31)
        layout_bboxes = config.get("layout_bboxes")
        if layout_bboxes is not None and not isinstance(layout_bboxes, str):
            config["layout_bboxes"] = json.dumps(layout_bboxes)

        allowed_keys = {
            "num_inference_steps",
            "guidance_scale",
            "shift",
            "timesteps_list",
            "scheduler_name",
            "noise_scale_start",
            "noise_scale_end",
            "noise_clip_std",
            "keep_original_aspect",
            "layout_bboxes",
            "use_flash_attn",
        }
        generate_kwargs = {key: config[key] for key in allowed_keys if key in config}
        images: List[PIL.Image.Image] = []

        with tempfile.TemporaryDirectory(prefix="xinference-hidream-o1-") as temp_dir:
            ref_paths: List[str] = []
            for index, image in enumerate(references):
                path = os.path.join(temp_dir, f"reference-{index}.png")
                image.convert("RGB").save(path, format="PNG")
                ref_paths.append(path)

            with self._release_after():
                for image_index in range(n):

                    def report_progress(step: int, total: int, _preview: Any) -> None:
                        if progressor and progressor.request_id:
                            progressor.set_progress(
                                (image_index + (step + 1) / total) / n
                            )

                    images.append(
                        generate_image(
                            model=self._model,
                            processor=self._processor,
                            prompt=prompt,
                            ref_image_paths=ref_paths,
                            height=height,
                            width=width,
                            seed=(
                                seeds[image_index]
                                if seeds is not None
                                else int(seed) + image_index
                            ),
                            callback=report_progress,
                            **generate_kwargs,
                        )
                    )

        return handle_image_result(response_format, images)

    def text_to_image(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        **kwargs: Any,
    ):
        width, height = map(int, re.split(r"[^\d]+", size))
        return self._generate(
            prompt,
            n,
            width,
            height,
            response_format,
            **kwargs,
        )

    def image_to_image(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        prompt: Optional[Union[str, List[str]]] = None,
        n: int = 1,
        size: Optional[str] = None,
        response_format: str = "url",
        **kwargs: Any,
    ):
        if "image2image" not in self._abilities:
            raise RuntimeError(f"{self._model_uid} does not support image2image")
        reference_images = kwargs.pop("reference_images", None)
        references = self._normalize_references(image, reference_images)
        if size:
            width, height = map(int, re.split(r"[^\d]+", size))
        else:
            width, height = references[0].size
        return self._generate(
            prompt or "",
            n,
            width,
            height,
            response_format,
            ref_images=references,
            **kwargs,
        )

    def inpainting(self, **kwargs: Any):
        raise NotImplementedError("HiDream-O1-Image does not support inpainting")

    async def abort_request(self, request_id: str) -> str:
        from ...model.scheduler.core import AbortRequestMessage

        return AbortRequestMessage.NO_OP.name
