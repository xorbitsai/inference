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

import contextlib
import gc
import logging
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ....types import LoRA
from ..sdapi import SDAPIDiffusionModelMixin
from ..utils import handle_image_result

if TYPE_CHECKING:
    from ....core.progress_tracker import Progressor
    from ..core import ImageModelFamilyV2


logger = logging.getLogger(__name__)


def quantization_predicate(name: str, m) -> bool:
    return hasattr(m, "to_quantized") and m.weight.shape[1] % 512 == 0


def to_latent_size(image_size: Tuple[int, int]):
    h, w = image_size
    h = ((h + 15) // 16) * 16
    w = ((w + 15) // 16) * 16

    if (h, w) != image_size:
        print(
            "Warning: The image dimensions need to be divisible by 16px. "
            f"Changing size to {h}x{w}."
        )

    return (h // 8, w // 8)


class MLXDiffusionModel(SDAPIDiffusionModelMixin):
    def __init__(
        self,
        model_uid: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        lora_model: Optional[List[LoRA]] = None,
        lora_load_kwargs: Optional[Dict] = None,
        lora_fuse_kwargs: Optional[Dict] = None,
        model_spec: Optional["ImageModelFamilyV2"] = None,
        **kwargs,
    ):
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        # model info when loading
        self._model = None
        self._lora_model = lora_model
        self._lora_load_kwargs = lora_load_kwargs or {}
        self._lora_fuse_kwargs = lora_fuse_kwargs or {}
        # info
        self._model_spec = model_spec
        self._abilities = model_spec.model_ability or []  # type: ignore
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._abilities

    @staticmethod
    def support_model(model_name: str) -> bool:
        return "flux" in model_name.lower()

    def load(self):
        try:
            import mlx.nn as nn
        except ImportError:
            error_message = "Failed to import module 'mlx'"
            installation_guide = [
                "Please make sure 'mlx' is installed. ",
                "You can install it by `pip install mlx`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        from ....thirdparty.mlx.flux import FluxPipeline

        logger.debug(
            "Loading model from %s, kwargs: %s", self._model_path, self._kwargs
        )
        flux = self._model = FluxPipeline(
            "flux-" + self._model_spec.model_name.split("-")[1],
            model_path=self._model_path,
            t5_padding=self._kwargs.get("t5_padding", True),
        )
        self._apply_lora()

        quantize = self._kwargs.get("quantize", True)
        if quantize:
            nn.quantize(flux.flow, class_predicate=quantization_predicate)
            nn.quantize(flux.t5, class_predicate=quantization_predicate)
            nn.quantize(flux.clip, class_predicate=quantization_predicate)

    def _apply_lora(self):
        if self._lora_model is not None:
            import mlx.core as mx

            for lora_model in self._lora_model:
                weights, lora_config = mx.load(
                    lora_model.local_path, return_metadata=True
                )
                rank = int(lora_config.get("lora_rank", 8))
                num_blocks = int(lora_config.get("lora_blocks", -1))
                flux = self._model
                flux.linear_to_lora_layers(rank, num_blocks)
                flux.flow.load_weights(list(weights.items()), strict=False)
                flux.fuse_lora_layers()
            logger.info(f"Successfully loaded the LoRA for model {self._model_uid}.")

    @staticmethod
    @contextlib.contextmanager
    def _release_after():
        import mlx.core as mx

        try:
            yield
        finally:
            gc.collect()
            mx.metal.clear_cache()

    def text_to_image(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        **kwargs,
    ):
        import mlx.core as mx

        flux = self._model
        width, height = map(int, re.split(r"[^\d]+", size))

        # Make the generator
        latent_size = to_latent_size((height, width))
        gen_latent_kwargs = {}
        if (num_steps := kwargs.get("num_inference_steps")) is None:
            num_steps = 50 if "dev" in self._model_spec.model_name else 2  # type: ignore
        gen_latent_kwargs["num_steps"] = num_steps
        if guidance := kwargs.get("guidance_scale"):
            gen_latent_kwargs["guidance"] = guidance
        if seed := kwargs.get("seed"):
            gen_latent_kwargs["seed"] = seed

        with self._release_after():
            latents = flux.generate_latents(  # type: ignore
                prompt, n_images=n, latent_size=latent_size, **gen_latent_kwargs
            )

            # First we get and eval the conditioning
            conditioning = next(latents)
            mx.eval(conditioning)
            peak_mem_conditioning = mx.metal.get_peak_memory() / 1024**3
            mx.metal.reset_peak_memory()

            progressor: Progressor = kwargs.pop("progressor", None)
            # Actual denoising loop
            for i, x_t in enumerate(latents):
                mx.eval(x_t)
                progressor.set_progress((i + 1) / num_steps)

            peak_mem_generation = mx.metal.get_peak_memory() / 1024**3
            mx.metal.reset_peak_memory()

            # Decode them into images
            decoded = []
            for i in range(n):
                decoded.append(flux.decode(x_t[i : i + 1], latent_size))  # type: ignore
                mx.eval(decoded[-1])
            peak_mem_decoding = mx.metal.get_peak_memory() / 1024**3
            peak_mem_overall = max(
                peak_mem_conditioning, peak_mem_generation, peak_mem_decoding
            )

            images = []
            x = mx.concatenate(decoded, axis=0)
            x = (x * 255).astype(mx.uint8)
            for i in range(len(x)):
                im = Image.fromarray(np.array(x[i]))
                images.append(im)

        logger.debug(
            f"Peak memory used for the text:       {peak_mem_conditioning:.3f}GB"
        )
        logger.debug(
            f"Peak memory used for the generation: {peak_mem_generation:.3f}GB"
        )
        logger.debug(f"Peak memory used for the decoding:   {peak_mem_decoding:.3f}GB")
        logger.debug(f"Peak memory used overall:            {peak_mem_overall:.3f}GB")

        return handle_image_result(response_format, images)

    def image_to_image(self, **kwargs):
        raise NotImplementedError

    def inpainting(self, **kwargs):
        raise NotImplementedError
