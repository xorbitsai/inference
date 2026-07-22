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

# Image models verified against the vllm-omni supported diffusion models:
# https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_image
VLLM_SUPPORTED_IMAGE_MODELS = (
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
        logger.info("Dropping args unsupported by vLLM-Omni %s: %s", purpose, dropped)
    return {k: v for k, v in kwargs.items() if k in valid_keys}


class VLLMDiffusionModel:
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
                "GGUF quantization is not supported by the vLLM image engine, "
                "please use the diffusers engine instead"
            )
        if lightning_model_path:
            raise ValueError(
                "Lightning LoRA acceleration is not supported by the vLLM image "
                "engine, please use the diffusers engine instead"
            )
        if lora_model:
            raise ValueError(
                "LoRA is not supported by the vLLM image engine yet, "
                "please use the diffusers engine instead"
            )
        if kwargs.get("controlnet"):
            raise ValueError(
                "Controlnet is not supported by the vLLM image engine, "
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
            from vllm_omni.entrypoints.omni import Omni
        except ImportError as e:
            error_message = f"Failed to import module 'vllm_omni': {e}"
            installation_guide = [
                "Please make sure 'vllm-omni' is installed and that the installed ",
                "'vllm' shares the same major.minor version (e.g. vllm-omni 0.24.x ",
                "requires vllm 0.24.x). You can install a matching pair by ",
                "`pip install 'vllm-omni==0.24.*' 'vllm==0.24.*'`\n",
            ]
            raise ImportError(
                f"{error_message}\n\n{''.join(installation_guide)}"
            ) from e

        logger.debug(
            "Loading vLLM-Omni diffusion model from %s, kwargs: %s",
            self._model_path,
            self._kwargs,
        )
        self._model = Omni(
            model=self._model_path,
            mode="text-to-image",
            **self._kwargs,
        )

    def stop(self):
        if self._model is not None:
            try:
                self._model.close()
            except Exception:
                logger.warning(
                    "Failed to shutdown vLLM-Omni diffusion model", exc_info=True
                )
            self._model = None

    def _build_sampling_params(
        self,
        n: int,
        width: int,
        height: int,
        generate_config: Dict[str, Any],
    ) -> Any:
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        seed = generate_config.pop("seed", None)
        if seed is None or seed == -1:
            # keep the default xinference behavior of generating a new image
            # every call instead of a fixed default seed
            seed = random.randint(0, 2**31 - 1)
        params = dict(generate_config)
        params.update(
            width=width,
            height=height,
            num_outputs_per_prompt=n,
            seed=seed,
        )
        params = _filter_kwargs_by_dataclass_fields(
            params, OmniDiffusionSamplingParams, "sampling params"
        )
        return OmniDiffusionSamplingParams(**params)

    def _build_prompt(self, prompt: str, width: int, height: int) -> Any:
        # build_text_to_image_prompt adds model-specific prompt structure
        # (e.g. system prompts); fall back to the raw prompt when the running
        # vllm-omni version does not expose these helpers
        try:
            from vllm_omni.model_extras import (
                build_text_to_image_prompt,
                get_model_class_name,
            )
        except ImportError:
            return prompt
        try:
            model_class_name = get_model_class_name(self._model)
            return build_text_to_image_prompt(
                model_class_name=model_class_name,
                prompt=prompt,
                negative_prompt=None,
                height=height,
                width=width,
            )
        except Exception:
            logger.warning(
                "Failed to build vLLM-Omni prompt payload, using raw prompt",
                exc_info=True,
            )
            return prompt

    @staticmethod
    def _extract_images(outputs: Any) -> List[Any]:
        from PIL import Image

        if not outputs:
            raise RuntimeError(
                "vLLM-Omni image generation failed, see server logs for details"
            )
        images = []
        for output in outputs:
            frames = getattr(output, "images", None)
            if not frames:
                request_output = getattr(output, "request_output", None)
                frames = (
                    getattr(request_output, "images", None)
                    if request_output is not None
                    else None
                )
            if not frames:
                continue
            if not isinstance(frames, (list, tuple)):
                frames = [frames]
            for frame in frames:
                if isinstance(frame, Image.Image):
                    images.append(frame)
                else:
                    raise RuntimeError(
                        f"Unexpected image type from vLLM-Omni: {type(frame)}"
                    )
        if not images:
            raise RuntimeError(
                "vLLM-Omni image generation returned no images, "
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
        sampling_params = self._build_sampling_params(n, width, height, generate_config)
        prompt_payload = self._build_prompt(prompt, width, height)
        outputs = await asyncio.to_thread(
            self._model.generate,
            prompt_payload,
            sampling_params_list=[sampling_params],
        )
        images = self._extract_images(outputs)
        return handle_image_result(response_format, images)
