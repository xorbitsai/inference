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

"""FLUX 3 Image Klein through the Diffusers modular pipeline."""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

from PIL import Image

from .sdapi import SDAPIDiffusionModelMixin
from .utils import MAX_IMAGE_SEED, handle_image_result, resolve_image_seed_list

if TYPE_CHECKING:
    from .core import ImageModelFamilyV2

logger = logging.getLogger(__name__)


class Flux3Model(SDAPIDiffusionModelMixin):
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
        self._device = device
        self._kwargs = (model_spec.default_model_config or {}).copy()
        self._kwargs.update(kwargs)
        self._model: Any = None

    @property
    def model_ability(self) -> List[str]:
        return self._model_spec.model_ability or []

    def load(self) -> None:
        import torch

        from ...device_utils import get_available_device
        from ..cache_manager import CacheManager
        from ..core import CacheableModelSpec

        try:
            from diffusers import ComponentsManager, Flux3DistilledModularPipeline
        except ImportError as exc:
            raise ImportError(
                "FLUX 3 Image Klein requires Diffusers from "
                "git+https://github.com/huggingface/diffusers"
            ) from exc

        for option in ("lora_model", "gguf_model_path", "lightning_model_path"):
            if self._kwargs.get(option):
                raise ValueError(f"FLUX 3 Image Klein does not support {option}")

        self._device = self._device or get_available_device()
        dtype = self._kwargs.get("torch_dtype", "bfloat16")
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        manager = ComponentsManager()
        pipeline = Flux3DistilledModularPipeline.from_pretrained(
            self._model_path, components_manager=manager
        )

        # Modular indexes can reference other repositories, notably the prompt
        # upsampler. Cache every external component through the selected hub.
        paths = {}
        repo_paths = {self._model_spec.model_id: self._model_path}
        for name in pipeline.pretrained_component_names:
            spec = pipeline.get_component_spec(name)
            repo = spec.pretrained_model_name_or_path
            if not repo:
                raise ValueError(f"Missing repository for FLUX 3 component: {name}")
            if Path(repo).is_dir():
                paths[name] = repo
            else:
                hub = self._model_spec.model_hub
                if repo not in repo_paths:
                    repo_paths[repo] = CacheManager(
                        CacheableModelSpec(
                            model_name=repo.rsplit("/", 1)[-1],
                            model_id=repo,
                            model_hub=hub,
                            model_revision=(
                                "master"
                                if hub == "modelscope"
                                else spec.revision or "main"
                            ),
                        )
                    ).cache()
                paths[name] = repo_paths[repo]
        pipeline.load_components(
            dtype={"vae": torch.float32, "default": dtype},
            pretrained_model_name_or_path=paths,
            revision=None,
            local_files_only=True,
        )
        # Diffusers logs component load failures rather than raising them.
        missing = [
            name
            for name in pipeline.pretrained_component_names
            if getattr(pipeline, name, None) is None
        ]
        if missing:
            raise RuntimeError(
                f"Failed to load FLUX 3 components: {', '.join(missing)}"
            )
        if self._kwargs.get("cpu_offload", False):
            manager.enable_auto_cpu_offload(device=self._device)
        else:
            pipeline.to(self._device)
        self._model = pipeline

    def _generate(
        self,
        prompt: str,
        n: int,
        size: Optional[str],
        response_format: str,
        **kwargs: Any,
    ):
        import torch

        if self._model is None:
            raise RuntimeError("FLUX 3 Image Klein is not loaded")
        if not isinstance(prompt, str):
            raise ValueError("FLUX 3 Image Klein requires a string prompt")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        config = (self._model_spec.default_generate_config or {}).copy()
        config.update(
            {key: value for key, value in kwargs.items() if value is not None}
        )
        if size is not None:
            dimensions = re.fullmatch(r"(\d+)[*xX](\d+)", size)
            if dimensions is None or any(
                int(value) <= 0 for value in dimensions.groups()
            ):
                raise ValueError(f"Invalid image size: {size!r}")
            config["width"], config["height"] = map(int, dimensions.groups())

        seed = config.pop("seed", None)
        seeds = resolve_image_seed_list(seed, n)
        if seeds is not None:
            config["generator"] = [
                torch.Generator().manual_seed(value) for value in seeds
            ]
        elif seed is not None and seed != -1:
            if (
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or not 0 <= seed <= MAX_IMAGE_SEED
            ):
                raise ValueError(
                    f"Image seeds must be -1 or between 0 and {MAX_IMAGE_SEED}"
                )
            config["generator"] = torch.Generator().manual_seed(seed)
        progressor = config.pop("progressor", None)
        return_images = config.pop("_return_images", False)
        allowed = set(self._model.blocks.input_names)
        for key in list(config):
            if key not in allowed:
                logger.warning(
                    "FLUX 3 Image Klein cannot accept `%s`, will ignore it", key
                )
                config.pop(key)
        if progressor is not None:
            progressor.set_progress(0.0)
        config.update(prompt=prompt, num_images_per_prompt=n)
        images = self._model(output="images", **config)
        if progressor is not None:
            progressor.set_progress(1.0)
        return images if return_images else handle_image_result(response_format, images)

    def text_to_image(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        **kwargs: Any,
    ):
        return self._generate(prompt, n, size, response_format, **kwargs)

    def image_to_image(
        self,
        image: Union[Image.Image, List[Image.Image]],
        prompt: str = "",
        n: int = 1,
        size: Optional[str] = None,
        response_format: str = "url",
        **kwargs: Any,
    ):
        images = list(image) if isinstance(image, (list, tuple)) else [image]
        references = kwargs.pop("reference_images", None)
        if references is not None:
            images.extend(
                references if isinstance(references, (list, tuple)) else [references]
            )
        if not 1 <= len(images) <= 10:
            raise ValueError("FLUX 3 Image Klein requires 1 to 10 reference images")
        if not all(isinstance(item, Image.Image) for item in images):
            raise TypeError("FLUX 3 reference images must be PIL images")
        return self._generate(prompt, n, size, response_format, image=images, **kwargs)
