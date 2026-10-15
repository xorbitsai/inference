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

"""Iris-3B and Iris-3B Turbo through the bundled upstream sampling API."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from .sdapi import SDAPIDiffusionModelMixin
from .utils import MAX_IMAGE_SEED, handle_image_result, resolve_image_seed_list

if TYPE_CHECKING:
    from .core import ImageModelFamilyV2


class IrisModel(SDAPIDiffusionModelMixin):
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
        self._device: Any = device
        self._kwargs = (model_spec.default_model_config or {}).copy()
        self._kwargs.update(kwargs)
        self._model: Optional[Any] = None
        self._text_encoder: Optional[Any] = None
        self._config: Optional[Any] = None
        self._sigmas: Optional[List[float]] = None

    @property
    def model_spec(self) -> "ImageModelFamilyV2":
        return self._model_spec

    @property
    def model_ability(self) -> List[str]:
        return self._model_spec.model_ability or []

    def load(self) -> None:
        if sys.version_info < (3, 11):
            raise RuntimeError("Iris-3B requires Python >= 3.11")
        import torch
        from omegaconf import OmegaConf

        from ...constants import XINFERENCE_CACHE_DIR
        from ...thirdparty.iris3b import text  # noqa: F401  (registers text encoders)
        from ...thirdparty.iris3b.config import inference_config
        from ...thirdparty.iris3b.models import IrisDiT
        from ...thirdparty.iris3b.registry import TEXT_ENCODERS
        from ...thirdparty.iris3b.sampling import load_for_inference
        from ..cache_manager import CacheManager
        from ..core import CacheableModelSpec

        self._device = torch.device(
            self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        path = Path(self._model_path)
        subfolder = self._kwargs.get("checkpoint_subfolder")
        # Accept either the whole repository or an already selected Turbo export.
        if subfolder and not (path / "distill.yaml").is_file():
            path = path / subfolder
        distill_path = path / "distill.yaml"
        if subfolder == "turbo" and not distill_path.is_file():
            raise ValueError("Iris-3B Turbo requires turbo/distill.yaml")
        if distill_path.is_file():
            self._sigmas = list(OmegaConf.load(distill_path).dmd.sigmas)

        raw, weights = load_for_inference(path)
        cfg = inference_config(raw)
        text_encoder_path = self._kwargs.get("text_encoder_path")
        if text_encoder_path is not None:
            if not Path(text_encoder_path).is_dir():
                raise ValueError("text_encoder_path must be a local model directory")
            cfg.text_encoder.pretrained = str(text_encoder_path)
        elif not Path(cfg.text_encoder.pretrained).is_dir():
            hub = self._model_spec.model_hub
            if hub not in ("huggingface", "modelscope"):
                raise ValueError(f"Unsupported Iris-3B text encoder hub: {hub}")
            encoder_spec = CacheableModelSpec(
                model_name=cfg.text_encoder.pretrained.rsplit("/", 1)[-1],
                model_id=cfg.text_encoder.pretrained,
                model_hub=hub,
                model_revision="master" if hub == "modelscope" else "main",
            )
            cfg.text_encoder.pretrained = CacheManager(encoder_spec).cache()
        cfg.text_encoder.null_embed_dir = str(
            Path(XINFERENCE_CACHE_DIR) / "iris3b" / "null_embeddings"
        )
        if self._device.type == "cpu":
            cfg.text_encoder.dtype = "float32"

        # Match official inference: FP32 diffusion weights, BF16 CUDA autocast.
        with torch.device("meta"):
            model = IrisDiT(cfg.model)
        model.load_state_dict(weights, strict=True, assign=True)
        model = model.eval().to(device=self._device, dtype=torch.float32)
        self._text_encoder = TEXT_ENCODERS.build(
            cfg.text_encoder.name, cfg.text_encoder, device=self._device
        )
        self._model = model
        self._config = cfg

    def text_to_image(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        **kwargs: Any,
    ):
        import re

        import torch
        from torchvision.transforms.functional import to_pil_image

        from ...thirdparty.iris3b.sampling import generate, generate_few_step

        if self._model is None or self._text_encoder is None or self._config is None:
            raise RuntimeError("Iris-3B model is not loaded")
        if not isinstance(prompt, str):
            raise ValueError("Iris-3B requires a string prompt")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        config = (self._model_spec.default_generate_config or {}).copy()
        config.update(
            {key: value for key, value in kwargs.items() if value is not None}
        )
        dimensions = re.fullmatch(r"(\d+)[*xX](\d+)", size)
        if dimensions is None:
            raise ValueError(f"Invalid image size: {size!r}")
        width, height = map(int, dimensions.groups())
        width, height = config.get("width", width), config.get("height", height)
        patch_size = self._config.model.patch_size
        for dimension in (width, height):
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
                or dimension % patch_size
            ):
                raise ValueError(
                    f"Iris-3B width and height must be positive multiples of {patch_size}"
                )
        negative_prompt = config.get("negative_prompt", "")
        if not isinstance(negative_prompt, str):
            raise ValueError("negative_prompt must be a string")
        sampling = {
            "height": height,
            "width": width,
            "device": self._device,
            "num_train_timesteps": self._config.flow.num_train_timesteps,
            "prediction": self._config.flow.prediction,
        }
        if self._sigmas is not None:
            if config.get("num_inference_steps", len(self._sigmas)) != len(
                self._sigmas
            ):
                raise ValueError(
                    "Iris-3B Turbo uses the fixed schedule in distill.yaml"
                )
            if config.get("guidance_scale", 1.0) != 1.0 or negative_prompt:
                raise ValueError(
                    "Iris-3B Turbo does not support CFG or negative_prompt"
                )
            if any(
                config.get(key) is not None
                for key in ("order", "shift", "cfg_interval", "sigmas")
            ):
                raise ValueError(
                    "Iris-3B Turbo uses the fixed schedule in distill.yaml"
                )
            sampling["sigmas"] = self._sigmas
            sampler = generate_few_step
        else:
            sampling.update(
                steps=config.get("num_inference_steps", self._config.sample.steps),
                cfg_scale=config.get("guidance_scale", self._config.sample.cfg_scale),
                order=config.get("order", self._config.sample.order),
                shift=config.get("shift", self._config.flow.shift),
                cfg_interval=tuple(
                    config.get("cfg_interval", self._config.sample.cfg_interval)
                ),
                negative_prompt=negative_prompt,
            )
            sampler = generate

        seed = config.get("seed")
        seeds = resolve_image_seed_list(seed, n)
        if seeds is None:
            if seed is None or seed == -1:
                seeds = resolve_image_seed_list([], n)
            else:
                # Validate the scalar before deriving independent image seeds.
                resolve_image_seed_list([seed], 1)
                seeds = [(seed + i) % (MAX_IMAGE_SEED + 1) for i in range(n)]
        assert seeds is not None
        device = torch.device(self._device)
        images = []
        with (
            torch.inference_mode(),
            torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ),
        ):
            for index, image_seed in enumerate(seeds):
                output = sampler(
                    self._model,
                    self._text_encoder,
                    [prompt],
                    generator=torch.Generator(device=device).manual_seed(image_seed),
                    **sampling,
                )
                # Official samples are RGB pixels in [-1, 1], without a VAE.
                images.append(
                    to_pil_image(output[0].float().cpu().add(1).div(2).clamp(0, 1))
                )
                progressor = config.get("progressor")
                if progressor is not None:
                    progressor.set_progress((index + 1) / n)
        return handle_image_result(response_format, images)
