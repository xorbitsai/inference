# Copyright 2022-2026 XProbe Inc.
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

import asyncio
import logging
import random
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ....types import LoRA
from ..utils import handle_image_result

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

logger = logging.getLogger(__name__)

# Image models verified against the sglang-diffusion compatibility matrix:
# https://docs.sglang.io/docs/sglang-diffusion/compatibility_matrix
SGLANG_SUPPORTED_IMAGE_MODELS = (
    "Qwen-Image",
    "Qwen-Image-2512",
    "Z-Image",
    "Z-Image-Turbo",
    "FLUX.1-dev",
)


def _filter_kwargs_by_dataclass_fields(
    kwargs: Dict[str, Any], dataclass_type: Any, purpose: str
) -> Dict[str, Any]:
    import dataclasses

    valid_keys = {f.name for f in dataclasses.fields(dataclass_type)}
    dropped = sorted(set(kwargs) - valid_keys)
    if dropped:
        logger.info("Dropping args unsupported by SGLang %s: %s", purpose, dropped)
    return {k: v for k, v in kwargs.items() if k in valid_keys}


class SGLangDiffusionModel:
    def __init__(
        self,
        model_uid: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        lora_model: Optional[List[LoRA]] = None,
        lora_load_kwargs: Optional[Dict] = None,
        lora_fuse_kwargs: Optional[Dict] = None,
        model_spec: Optional["ImageModelFamilyV2"] = None,
        gguf_model_path: Optional[str] = None,
        lightning_model_path: Optional[str] = None,
        **kwargs,
    ):
        if gguf_model_path:
            raise ValueError(
                "GGUF quantization is not supported by the SGLang image engine, "
                "please use the diffusers engine instead"
            )
        if lightning_model_path:
            raise ValueError(
                "Lightning LoRA acceleration is not supported by the SGLang image "
                "engine, please use the diffusers engine instead"
            )
        if lora_model:
            raise ValueError(
                "LoRA is not supported by the SGLang image engine yet, "
                "please use the diffusers engine instead"
            )
        if kwargs.get("controlnet"):
            raise ValueError(
                "Controlnet is not supported by the SGLang image engine, "
                "please use the diffusers engine instead"
            )
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._model = None
        self._model_spec = model_spec
        self._abilities = model_spec.model_ability or []  # type: ignore
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._abilities

    def load(self):
        try:
            from sglang.multimodal_gen import DiffGenerator
            from sglang.multimodal_gen.runtime.server_args import ServerArgs
        except ImportError:
            error_message = "Failed to import module 'sglang.multimodal_gen'"
            installation_guide = [
                "Please make sure 'sglang' with diffusion support is installed. ",
                "You can install it by `pip install 'sglang[diffusion]'`\n",
            ]
            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        engine_kwargs = _filter_kwargs_by_dataclass_fields(
            self._kwargs, ServerArgs, "server args"
        )
        logger.debug(
            "Loading SGLang diffusion model from %s, kwargs: %s",
            self._model_path,
            engine_kwargs,
        )
        self._model = DiffGenerator.from_pretrained(
            model_path=self._model_path,
            **engine_kwargs,
        )

    def stop(self):
        if self._model is not None:
            try:
                self._model.shutdown()
            except Exception:
                logger.warning(
                    "Failed to shutdown SGLang diffusion model", exc_info=True
                )
            self._model = None

    def _build_sampling_params(
        self,
        prompt: str,
        n: int,
        width: int,
        height: int,
        generate_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams

        seed = generate_config.pop("seed", None)
        if seed is None or seed == -1:
            # SamplingParams uses a fixed default seed; keep the default
            # xinference behavior of generating a new image every call
            seed = random.randint(0, 2**31 - 1)
        params = dict(generate_config)
        params.update(
            prompt=prompt,
            width=width,
            height=height,
            num_outputs_per_prompt=n,
            seed=seed,
            save_output=False,
            return_frames=True,
        )
        return _filter_kwargs_by_dataclass_fields(
            params, SamplingParams, "sampling params"
        )

    @staticmethod
    def _extract_images(result: Any) -> List[Any]:
        import numpy as np
        from PIL import Image

        if result is None:
            raise RuntimeError(
                "SGLang image generation failed, see server logs for details"
            )
        results = result if isinstance(result, list) else [result]
        images = []
        for res in results:
            frames = getattr(res, "frames", None)
            if frames is None:
                continue
            if not isinstance(frames, (list, tuple)):
                frames = [frames]
            for frame in frames:
                if isinstance(frame, Image.Image):
                    images.append(frame)
                elif isinstance(frame, np.ndarray):
                    if frame.ndim == 3 and frame.shape[-1] == 1:
                        frame = frame[..., 0]
                    images.append(Image.fromarray(frame))
                else:
                    raise RuntimeError(
                        f"Unexpected frame type from SGLang: {type(frame)}"
                    )
        if not images:
            raise RuntimeError(
                "SGLang image generation returned no images, "
                "see server logs for details"
            )
        return images

    async def text_to_image(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        **kwargs,
    ):
        assert self._model is not None
        width, height = map(int, re.split(r"[^\d]+", size))
        generate_config: Dict[str, Any] = (
            self._model_spec.default_generate_config or {}  # type: ignore
        ).copy()
        generate_config.update({k: v for k, v in kwargs.items() if v is not None})
        sampling_params_kwargs = self._build_sampling_params(
            prompt, n, width, height, generate_config
        )
        result = await asyncio.to_thread(
            self._model.generate,
            sampling_params_kwargs=sampling_params_kwargs,
        )
        images = self._extract_images(result)
        return handle_image_result(response_format, images)
