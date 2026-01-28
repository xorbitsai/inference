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
import contextlib
import gc
import importlib
import inspect
import itertools
import json
import logging
import math
import os
import re
import sys
import warnings
from glob import glob
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import PIL.Image
import torch
from PIL import ImageOps

from ....device_utils import (
    get_available_device,
    gpu_count,
    move_model_to_available_device,
)
from ....types import LoRA
from ..sdapi import SDAPIDiffusionModelMixin
from ..utils import handle_image_result

if TYPE_CHECKING:
    from ....core.progress_tracker import Progressor
    from ..core import ImageModelFamilyV2

logger = logging.getLogger(__name__)

SAMPLING_METHODS = [
    "default",
    "DPM++ 2M",
    "DPM++ 2M Karras",
    "DPM++ 2M SDE",
    "DPM++ 2M SDE Karras",
    "DPM++ SDE",
    "DPM++ SDE Karras",
    "DPM2",
    "DPM2 Karras",
    "DPM2 a",
    "DPM2 a Karras",
    "Euler",
    "Euler a",
    "Heun",
    "LMS",
    "LMS Karras",
]


def model_accept_param(params: Union[str, List[str]], model: Any) -> bool:
    params = [params] if isinstance(params, str) else params
    # model is diffusers Pipeline
    parameters = inspect.signature(model.__call__).parameters  # type: ignore
    allow_params = False
    for param in parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            # the __call__ can accept **kwargs,
            # we treat it as it can accept any parameters
            allow_params = True
            break
    if not allow_params:
        if all(param in parameters for param in params):
            allow_params = True
    return allow_params


class DiffusionModel(SDAPIDiffusionModelMixin):
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
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        # model info when loading
        self._model = None
        self._lora_model = lora_model
        self._lora_load_kwargs = lora_load_kwargs or {}
        self._lora_fuse_kwargs = lora_fuse_kwargs or {}
        # deepcache
        self._deepcache_helper = None
        # when a model has text2image ability,
        # it will be loaded as AutoPipelineForText2Image
        # for image2image and inpainting,
        # we convert to the corresponding model
        self._torch_dtype = None
        self._ability_to_models: Dict[Tuple[str, Any], Any] = {}
        self._controlnet_models: Dict[str, Any] = {}
        # info
        self._model_spec = model_spec
        self._abilities = model_spec.model_ability or []  # type: ignore
        self._kwargs = kwargs
        self._has_bnb_quantization = False
        # gguf
        self._gguf_model_path = gguf_model_path
        # lightning
        self._lightning_model_path = lightning_model_path

    @property
    def model_ability(self):
        return self._abilities

    def _is_flux2_model(self) -> bool:
        return bool(
            self._model_spec
            and "flux.2" in self._model_spec.model_name.lower()  # type: ignore
        )

    @staticmethod
    def _get_pipeline_type(ability: str) -> type:
        if ability == "text2image":
            from diffusers import AutoPipelineForText2Image as AutoPipelineModel
        elif ability == "image2image":
            from diffusers import AutoPipelineForImage2Image as AutoPipelineModel
        elif ability == "inpainting":
            from diffusers import AutoPipelineForInpainting as AutoPipelineModel
        else:
            raise ValueError(f"Unknown ability: {ability}")
        return AutoPipelineModel

    def _get_controlnet_model(self, name: str, path: str):
        from diffusers import ControlNetModel

        try:
            return self._controlnet_models[name]
        except KeyError:
            logger.debug("Loading controlnet %s, from %s", name, path)
            model = ControlNetModel.from_pretrained(path, torch_dtype=self._torch_dtype)
            self._controlnet_models[name] = model
            return model

    def _get_model(
        self,
        ability: str,
        controlnet_name: Optional[Union[str, List[str]]] = None,
        controlnet_path: Optional[Union[str, List[str]]] = None,
    ):
        try:
            return self._ability_to_models[ability, controlnet_name]
        except KeyError:
            model_type = self._get_pipeline_type(ability)

        assert self._model is not None

        if controlnet_name:
            assert controlnet_path
            if isinstance(controlnet_name, (list, tuple)):
                controlnet = []
                # multiple controlnet
                for name, path in itertools.zip_longest(
                    controlnet_name, controlnet_path
                ):
                    controlnet.append(self._get_controlnet_model(name, path))
            else:
                controlnet = self._get_controlnet_model(
                    controlnet_name, controlnet_path
                )
            model = model_type.from_pipe(self._model, controlnet=controlnet)
        else:
            try:
                from diffusers import (
                    QwenImageImg2ImgPipeline,
                    QwenImageInpaintPipeline,
                    QwenImagePipeline,
                )
            except ImportError:
                QwenImagePipeline = None
                QwenImageImg2ImgPipeline = None
                QwenImageInpaintPipeline = None

            if QwenImagePipeline is not None and isinstance(
                self._model, QwenImagePipeline
            ):
                # special process for Qwen-image
                if ability == "image2image":
                    model = QwenImageImg2ImgPipeline.from_pipe(
                        self._model, torch_dtype=None
                    )
                else:
                    assert ability == "inpainting"
                    model = QwenImageInpaintPipeline.from_pipe(
                        self._model, torch_dtype=None
                    )
            else:
                model = model_type.from_pipe(self._model)
        self._load_to_device(model)

        self._ability_to_models[ability, controlnet_name] = model
        return model

    def _apply_lora(self):
        if self._lora_model is not None:
            logger.info(
                f"Loading the LoRA with load kwargs: {self._lora_load_kwargs}, fuse kwargs: {self._lora_fuse_kwargs}."
            )
            assert self._model is not None
            for lora_model in self._lora_model:
                self._model.load_lora_weights(
                    lora_model.local_path, **self._lora_load_kwargs
                )
            self._model.fuse_lora(**self._lora_fuse_kwargs)
            logger.info(f"Successfully loaded the LoRA for model {self._model_uid}.")

    def _get_layer_cls(self, layer: str):
        with open(os.path.join(self._model_path, "model_index.json")) as f:  # type: ignore
            model_index = json.load(f)
            layer_info = model_index[layer]
            module_name, class_name = layer_info
            module = importlib.import_module(module_name)
            return getattr(module, class_name)

    def load(self):
        if "text2image" in self._abilities or "image2image" in self._abilities:
            from diffusers import AutoPipelineForText2Image as AutoPipelineModel
        elif "inpainting" in self._abilities:
            from diffusers import AutoPipelineForInpainting as AutoPipelineModel
        else:
            raise ValueError(f"Unknown ability: {self._abilities}")

        self._torch_dtype = torch_dtype = self._kwargs.get("torch_dtype")
        if sys.platform != "darwin" and torch_dtype is None:
            # The following params crashes on Mac M2
            self._torch_dtype = self._kwargs["torch_dtype"] = torch.float16
            self._kwargs["use_safetensors"] = any(
                glob(os.path.join(self._model_path, "*/*.safetensors"))
            )
        if isinstance(torch_dtype, str):
            self._torch_dtype = torch_dtype = self._kwargs["torch_dtype"] = getattr(
                torch, torch_dtype
            )

        controlnet = self._kwargs.get("controlnet")
        if controlnet is not None:
            if isinstance(controlnet, tuple):
                self._kwargs["controlnet"] = self._get_controlnet_model(*controlnet)
            else:
                self._kwargs["controlnet"] = [
                    self._get_controlnet_model(*cn) for cn in controlnet
                ]

        # quantizations
        # text_encoder
        quantize_text_encoder = self._kwargs.pop("quantize_text_encoder", None)
        self._quantize_text_encoder(quantize_text_encoder)
        # transformer
        if self._gguf_model_path:
            self._quantize_transformer_gguf()
        else:
            self._quantize_transformer()

        if self._has_bnb_quantization and not self._kwargs.get("device_map"):
            # Ensure bnb-loaded modules are placed explicitly on one device to avoid CPU/GPU mixing
            self._kwargs["device_map"] = get_available_device()

        if (
            (device_count := gpu_count()) > 1
            and "device_map" not in self._kwargs
            and not self._is_flux2_model()
        ):
            logger.debug(
                "Device count (%d) > 1, force to set device_map=balanced", device_count
            )
            self._kwargs["device_map"] = "balanced"

        logger.debug(
            "Loading model from %s, kwargs: %s", self._model_path, self._kwargs
        )
        with self._process_lightning(self._kwargs):
            try:
                self._model = AutoPipelineModel.from_pretrained(
                    self._model_path,
                    **self._kwargs,
                )
            except ValueError:
                model_name_lower = self._model_spec.model_name.lower()
                if "flux.2" in model_name_lower:
                    from diffusers import Flux2Pipeline

                    self._model = Flux2Pipeline.from_pretrained(
                        self._model_path, **self._kwargs
                    )
                elif "flux" in model_name_lower:
                    from diffusers import FluxPipeline

                    self._model = FluxPipeline.from_pretrained(
                        self._model_path, **self._kwargs
                    )
                elif "kontext" in model_name_lower:
                    # TODO: remove this branch when auto pipeline supports
                    # flux.1-kontext-dev
                    from diffusers import FluxKontextPipeline

                    self._model = FluxKontextPipeline.from_pretrained(
                        self._model_path, **self._kwargs
                    )
                elif "qwen" in model_name_lower:
                    # TODO: remove this branch when auto pipeline supports
                    # Qwen-Image
                    from diffusers import DiffusionPipeline

                    self._model = DiffusionPipeline.from_pretrained(
                        self._model_path, **self._kwargs
                    )
                elif "z-image" in model_name_lower or "zimage" in model_name_lower:
                    # TODO: remove this branch when auto pipeline supports Z-Image
                    from diffusers import DiffusionPipeline

                    self._model = DiffusionPipeline.from_pretrained(
                        self._model_path, **self._kwargs
                    )
                else:
                    raise
            self._load_to_device(self._model)
            self._apply_lora()

        if self._kwargs.get("deepcache", False):
            try:
                from DeepCache import DeepCacheSDHelper
            except ImportError:
                error_message = "Failed to import module 'deepcache' when you launch with deepcache=True"
                installation_guide = [
                    "Please make sure 'deepcache' is installed. ",
                    "You can install it by `pip install deepcache`\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
            else:
                self._deepcache_helper = helper = DeepCacheSDHelper()
                helper.set_params(
                    cache_interval=self._kwargs.get("deepcache_cache_interval", 3),
                    cache_branch_id=self._kwargs.get("deepcache_cache_branch_id", 0),
                )

        # Initialize batch scheduler if batching is enabled
        self._image_batch_scheduler = None
        if self._should_use_batching():
            from ..scheduler.flux import FluxBatchScheduler

            self._image_batch_scheduler = FluxBatchScheduler(self)
            # Note: scheduler will be started when first request comes in

    def _should_use_batching(self) -> bool:
        """Check if this model should use batch scheduling for images"""
        from ....constants import XINFERENCE_TEXT_TO_IMAGE_BATCHING_SIZE

        return XINFERENCE_TEXT_TO_IMAGE_BATCHING_SIZE is not None

    def _get_quantize_config(self, method: str, quantization: str, module: str):
        if method == "bnb":
            self._has_bnb_quantization = True
            try:
                import bitsandbytes  # noqa: F401
            except ImportError:
                error_message = "Failed to import module 'bitsandbytes'"
                installation_guide = [
                    "Please make sure 'bitsandbytes' is installed. ",
                    "You can install it by `pip install bitsandbytes`\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

            if module.startswith("diffusers."):
                from diffusers import BitsAndBytesConfig
            else:
                assert module.startswith("transformers.")
                from transformers import BitsAndBytesConfig

            if quantization == "4-bit":
                return BitsAndBytesConfig(load_in_4bit=True)
            elif quantization == "8-bit":
                return BitsAndBytesConfig(load_in_8bit=True)
            elif quantization == "nf4":
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self._torch_dtype,
                )
        elif method == "torchao":
            try:
                import torchao  # noqa: F401
            except ImportError:
                error_message = "Failed to import module 'torchao'"
                installation_guide = [
                    "Please make sure 'torchao' is installed. ",
                    "You can install it by `pip install torchao`\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

            if module.startswith("diffusers."):
                from diffusers import TorchAoConfig
            else:
                assert module.startswith("transformers.")
                from transformers import TorchAoConfig

            return TorchAoConfig(quantization)
        else:
            raise ValueError(f"Unknown quantization method for image model: {method}")

    def _quantize_text_encoder(self, quantize_text_encoder: Optional[str]):
        if self._gguf_model_path:
            # skip quantization when gguf applied to transformer
            return

        if not quantize_text_encoder:
            logger.debug("No text encoder quantization")
            return

        quantization_method = self._kwargs.pop("text_encoder_quantize_method", "bnb")
        quantization = self._kwargs.pop("text_encoder_quantization", "8-bit")

        logger.debug(
            "Quantize text encoder %s with method %s, quantization %s",
            quantize_text_encoder,
            quantization_method,
            quantization,
        )

        torch_dtype = self._torch_dtype
        for text_encoder_name in quantize_text_encoder.split(","):
            quantization_kwargs: Dict[str, Any] = {}
            if torch_dtype:
                quantization_kwargs["torch_dtype"] = torch_dtype
            text_encoder_cls = self._get_layer_cls(text_encoder_name)
            quantization_config = self._get_quantize_config(
                quantization_method, quantization, text_encoder_cls.__module__
            )
            text_encoder = text_encoder_cls.from_pretrained(
                self._model_path,
                subfolder=text_encoder_name,
                quantization_config=quantization_config,
                **quantization_kwargs,
            )
            self._kwargs[text_encoder_name] = text_encoder
        else:
            if not self._kwargs.get("device_map") and not self._is_flux2_model():
                self._kwargs["device_map"] = "balanced"

    def _quantize_transformer(self):
        quantization = None
        nf4 = self._kwargs.pop("transformer_nf4", None)
        if nf4:
            warnings.warn(
                "`transformer_nf4` is deprecated, please use `transformer_quantization=nf4`",
                category=DeprecationWarning,
                stacklevel=2,
            )
            quantization = "nf4"
        method = self._kwargs.pop("transformer_quantize_method", "bnb")
        if not quantization:
            quantization = self._kwargs.pop("transformer_quantization", None)

        if not quantization:
            # skip if no quantization specified
            logger.debug("No transformer quantization")
            return

        logger.debug(
            "Quantize transformer with %s, quantization %s", method, quantization
        )

        torch_dtype = self._torch_dtype
        transformer_cls = self._get_layer_cls("transformer")
        quantization_config = self._get_quantize_config(
            method, quantization, transformer_cls.__module__
        )
        transformer_model = transformer_cls.from_pretrained(
            self._model_path,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
        )
        self._kwargs["transformer"] = transformer_model

    def _quantize_transformer_gguf(self):
        from diffusers import GGUFQuantizationConfig

        # GGUF transformer
        torch_dtype = self._torch_dtype
        logger.debug("Quantize transformer with gguf file %s", self._gguf_model_path)
        self._kwargs["transformer"] = self._get_layer_cls(
            "transformer"
        ).from_single_file(
            self._gguf_model_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
            torch_dtype=torch_dtype,
            config=os.path.join(self._model_path, "transformer"),
        )

    @contextlib.contextmanager
    def _process_lightning(self, kwargs):
        lightning_model_path = self._lightning_model_path
        if not lightning_model_path:
            yield
            return

        from diffusers import FlowMatchEulerDiscreteScheduler

        if "qwen" in self._model_spec.model_name.lower():
            scheduler_config = {
                "base_image_seq_len": 256,
                "base_shift": math.log(3),  # We use shift=3 in distillation
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": math.log(3),  # We use shift=3 in distillation
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": None,  # set shift_terminal to None
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False,
            }
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
            kwargs["scheduler"] = scheduler

            yield

            model = self._model
            logger.debug("Loading lightning lora: %s", self._lightning_model_path)
            model.load_lora_weights(self._lightning_model_path)
        else:
            logger.debug("No lightning applied")
            yield

    def _load_to_device(self, model):
        if self._has_bnb_quantization and not any(
            self._kwargs.get(flag) for flag in ["cpu_offload", "sequential_cpu_offload"]
        ):
            # Bitsandbytes modules do not support manual .to(); they are already on target device
            logger.debug("Skip manual device move for bitsandbytes-quantized model")
            return
        if self._kwargs.get("cpu_offload", False):
            logger.debug("CPU offloading model")
            model.enable_model_cpu_offload()
        elif self._kwargs.get("sequential_cpu_offload", False):
            logger.debug("CPU sequential offloading model")
            model.enable_sequential_cpu_offload()
        elif not self._kwargs.get("device_map"):
            logger.debug("Loading model to available device")
            model = move_model_to_available_device(model)
        if self._kwargs.get("attention_slicing", False):
            model.enable_attention_slicing()
        if self._kwargs.get("vae_tiling", False):
            try:
                model.enable_vae_tiling()
            except AttributeError:
                model.vae.enable_tiling()
        if self._kwargs.get("vae_slicing", False):
            try:
                model.enable_vae_slicing()
            except AttributeError:
                model.vae.enable_slicing()

    def get_max_num_images_for_batching(self):
        return self._kwargs.get("max_num_images", 16)

    @staticmethod
    def _get_scheduler(model: Any, sampler_name: str):
        if not sampler_name or sampler_name == "default":
            return

        assert model is not None

        import diffusers

        kwargs = {}
        if (
            sampler_name.startswith("DPM++")
            and "final_sigmas_type" not in model.scheduler.config
        ):
            # `final_sigmas_type` will be set as `zero` by default which will cause error
            kwargs["final_sigmas_type"] = "sigma_min"

        # see https://github.com/huggingface/diffusers/issues/4167
        # to get A1111 <> Diffusers Scheduler mapping
        if sampler_name == "DPM++ 2M":
            return diffusers.DPMSolverMultistepScheduler.from_config(
                model.scheduler.config, **kwargs
            )
        elif sampler_name == "DPM++ 2M Karras":
            return diffusers.DPMSolverMultistepScheduler.from_config(
                model.scheduler.config, use_karras_sigmas=True, **kwargs
            )
        elif sampler_name == "DPM++ 2M SDE":
            return diffusers.DPMSolverMultistepScheduler.from_config(
                model.scheduler.config, algorithm_type="sde-dpmsolver++", **kwargs
            )
        elif sampler_name == "DPM++ 2M SDE Karras":
            return diffusers.DPMSolverMultistepScheduler.from_config(
                model.scheduler.config,
                algorithm_type="sde-dpmsolver++",
                use_karras_sigmas=True,
                **kwargs,
            )
        elif sampler_name == "DPM++ SDE":
            return diffusers.DPMSolverSinglestepScheduler.from_config(
                model.scheduler.config, **kwargs
            )
        elif sampler_name == "DPM++ SDE Karras":
            return diffusers.DPMSolverSinglestepScheduler.from_config(
                model.scheduler.config, use_karras_sigmas=True, **kwargs
            )
        elif sampler_name == "DPM2":
            return diffusers.KDPM2DiscreteScheduler.from_config(
                model.scheduler.config, **kwargs
            )
        elif sampler_name == "DPM2 Karras":
            return diffusers.KDPM2DiscreteScheduler.from_config(
                model.scheduler.config, use_karras_sigmas=True, **kwargs
            )
        elif sampler_name == "DPM2 a":
            return diffusers.KDPM2AncestralDiscreteScheduler.from_config(
                model.scheduler.config, **kwargs
            )
        elif sampler_name == "DPM2 a Karras":
            return diffusers.KDPM2AncestralDiscreteScheduler.from_config(
                model.scheduler.config, use_karras_sigmas=True, **kwargs
            )
        elif sampler_name == "Euler":
            return diffusers.EulerDiscreteScheduler.from_config(
                model.scheduler.config, **kwargs
            )
        elif sampler_name == "Euler a":
            return diffusers.EulerAncestralDiscreteScheduler.from_config(
                model.scheduler.config, **kwargs
            )
        elif sampler_name == "Heun":
            return diffusers.HeunDiscreteScheduler.from_config(
                model.scheduler.config, **kwargs
            )
        elif sampler_name == "LMS":
            return diffusers.LMSDiscreteScheduler.from_config(
                model.scheduler.config, **kwargs
            )
        elif sampler_name == "LMS Karras":
            return diffusers.LMSDiscreteScheduler.from_config(
                model.scheduler.config, use_karras_sigmas=True, **kwargs
            )
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

    def _need_set_scheduler(self, scheduler: Any) -> bool:
        """Determine whether it is necessary to set up a scheduler"""
        if self._model_spec is None:
            return False
        if scheduler is None:
            return False
        if "FLUX" in self._model_spec.model_name:
            logger.warning("FLUX model, skipping scheduler setup")
            return False
        return True

    @contextlib.contextmanager
    def _reset_when_done(self, model: Any, sampler_name: str):
        scheduler = DiffusionModel._get_scheduler(model, sampler_name)
        if self._need_set_scheduler(scheduler):
            logger.debug("Use scheduler %s", scheduler)
            default_scheduler = model.scheduler
            model.scheduler = scheduler
            try:
                yield
            finally:
                model.scheduler = default_scheduler
        else:
            yield

    @staticmethod
    @contextlib.contextmanager
    def _release_after():
        from ....device_utils import empty_cache

        try:
            yield
        finally:
            gc.collect()
            empty_cache()

    @contextlib.contextmanager
    def _wrap_deepcache(self, model: Any):
        if self._deepcache_helper:
            self._deepcache_helper.pipe = model
            self._deepcache_helper.enable()
        try:
            yield
        finally:
            if self._deepcache_helper:
                self._deepcache_helper.disable()
                self._deepcache_helper.pipe = None

    @staticmethod
    def _process_progressor(kwargs: dict):
        import diffusers

        progressor: Progressor = kwargs.pop("progressor", None)

        def report_status_callback(
            pipe: diffusers.DiffusionPipeline,
            step: int,
            timestep: int,
            callback_kwargs: dict,
        ):
            num_steps = pipe.num_timesteps
            progressor.set_progress((step + 1) / num_steps)

            return callback_kwargs

        if progressor and progressor.request_id:
            kwargs["callback_on_step_end"] = report_status_callback

    def _call_model(
        self,
        response_format: str,
        model=None,
        **kwargs,
    ):
        model = model if model is not None else self._model
        is_padded = kwargs.pop("is_padded", None)
        origin_size = kwargs.pop("origin_size", None)
        seed = kwargs.pop("seed", None)
        return_images = kwargs.pop("_return_images", None)
        if seed is not None and seed != -1:
            kwargs["generator"] = generator = torch.Generator(device=get_available_device())  # type: ignore
            if seed != -1:
                kwargs["generator"] = generator.manual_seed(seed)
        sampler_name = kwargs.pop("sampler_name", None)
        self._process_progressor(kwargs)
        assert callable(model)
        with self._reset_when_done(
            model, sampler_name
        ), self._release_after(), self._wrap_deepcache(model):
            logger.debug("stable diffusion args: %s, model: %s", kwargs, model)
            # Some pipelines (e.g., Z-Image img2img) can't handle guidance_scale=None.
            if kwargs.get("guidance_scale", "unset") is None:
                kwargs.pop("guidance_scale", None)
            self._filter_kwargs(model, kwargs)
            images = model(**kwargs).images

        if images and isinstance(images[0], (list, tuple)):
            images = list(itertools.chain.from_iterable(images))

        # revert padding if padded
        if is_padded and origin_size:
            new_images = []
            x, y = origin_size
            for img in images:
                new_images.append(img.crop((0, 0, x, y)))
            images = new_images

        if return_images:
            return images

        return handle_image_result(response_format, images)

    @classmethod
    def _filter_kwargs(cls, model, kwargs: dict):
        for arg in ["negative_prompt", "num_inference_steps"]:
            if not kwargs.get(arg):
                kwargs.pop(arg, None)

        for key in list(kwargs):
            allow_key = model_accept_param(key, model)
            if not allow_key:
                logger.warning(f"{type(model)} cannot accept `{key}`, will ignore it")
                kwargs.pop(key)

    async def text_to_image(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        **kwargs,
    ):
        """Text to image method that handles both batching and non-batching"""
        if self._image_batch_scheduler:
            await self._ensure_scheduler_started()
            # Use batching path
            from concurrent.futures import Future as ConcurrentFuture

            future: ConcurrentFuture = ConcurrentFuture()
            await self._image_batch_scheduler.add_request(
                prompt, future, n, size, response_format, **kwargs
            )

            fut = asyncio.wrap_future(future)
            return await fut
        else:
            # Use direct path
            return await self._direct_text_to_image(
                prompt, n, size, response_format, **kwargs
            )

    async def _ensure_scheduler_started(self):
        """Ensure the image batch scheduler is started"""
        if self._image_batch_scheduler and not self._image_batch_scheduler._running:
            await self._image_batch_scheduler.start()

    def _gen_config_for_lightning(self, kwargs):
        if (
            not kwargs.get("num_inference_steps")
            and self._lightning_model_path is not None
        ):
            is_4_steps = "4steps" in self._lightning_model_path
            if is_4_steps:
                kwargs["num_inference_steps"] = 4
            else:
                assert "8steps" in self._lightning_model_path
                kwargs["num_inference_steps"] = 8

    async def _direct_text_to_image(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        **kwargs,
    ):
        width, height = map(int, re.split(r"[^\d]+", size))
        generate_kwargs = self._model_spec.default_generate_config.copy()  # type: ignore
        generate_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
        generate_kwargs["width"], generate_kwargs["height"] = width, height
        self._gen_config_for_lightning(generate_kwargs)

        return await asyncio.to_thread(
            self._call_model,
            prompt=prompt,  # type: ignore
            num_images_per_prompt=n,  # type: ignore
            response_format=response_format,
            **generate_kwargs,
        )

    async def abort_request(self, request_id: str) -> str:
        """Abort a running request."""
        from ....model.scheduler.core import AbortRequestMessage

        # Check if we have a cancel callback for this request
        if hasattr(self, "_cancel_callbacks") and request_id in self._cancel_callbacks:
            cancel_callback = self._cancel_callbacks.pop(request_id)
            cancel_callback()
            return AbortRequestMessage.DONE.name

        return AbortRequestMessage.NO_OP.name

    @staticmethod
    def pad_to_multiple(image, multiple=8):
        x, y = image.size
        padding_x = (multiple - x % multiple) % multiple
        padding_y = (multiple - y % multiple) % multiple
        padding = (0, 0, padding_x, padding_y)
        return ImageOps.expand(image, padding)

    @staticmethod
    def _model_expects_four_channel_input(model: Any) -> bool:
        vae = getattr(model, "vae", None)
        input_channels = getattr(getattr(vae, "config", None), "input_channels", None)
        return input_channels == 4

    @staticmethod
    def _ensure_four_channel_image(image: Any, model: Any):
        image_processor = getattr(model, "image_processor", None)
        if (
            image_processor is not None
            and getattr(image_processor, "config", None) is not None
        ):
            image_processor.config.do_convert_rgb = False

        if isinstance(image, list):
            if not image:
                return image
            if isinstance(image[0], PIL.Image.Image):
                return [
                    img.convert("RGBA") if img.mode != "RGBA" else img for img in image
                ]
            return image
        if isinstance(image, PIL.Image.Image):
            return image.convert("RGBA") if image.mode != "RGBA" else image
        return image

    @staticmethod
    def _ensure_three_channel_image(image: Any):
        if isinstance(image, list):
            if not image:
                return image
            if isinstance(image[0], PIL.Image.Image):
                return [
                    img.convert("RGB") if img.mode != "RGB" else img for img in image
                ]
            return image
        if isinstance(image, PIL.Image.Image):
            return image.convert("RGB") if image.mode != "RGB" else image
        return image

    def image_to_image(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        prompt: Optional[Union[str, List[str]]] = None,
        n: int = 1,
        size: Optional[str] = None,
        response_format: str = "url",
        **kwargs,
    ):
        if self._kwargs.get("controlnet") or self._model_spec.model_ability == [  # type: ignore
            "image2image"
        ]:
            model = self._model
        else:
            ability = "image2image"
            if ability not in self._abilities:
                raise RuntimeError(f"{self._model_uid} does not support image2image")
            model = self._get_model(ability)

        if padding_image_to_multiple := kwargs.pop("padding_image_to_multiple", None):
            # Model like SD3 image to image requires image's height and width is times of 16
            # padding the image if specified
            if isinstance(image, list):
                origin_x, origin_y = image[0].size
            else:
                origin_x, origin_y = image.size
            kwargs["origin_size"] = (origin_x, origin_y)
            kwargs["is_padded"] = True
            image = self.pad_to_multiple(image, multiple=int(padding_image_to_multiple))

        if size:
            width, height = map(int, re.split(r"[^\d]+", size))
            if padding_image_to_multiple:
                if isinstance(image, list):
                    width, height = image[0].size
                else:
                    width, height = image.size
            kwargs["width"] = width
            kwargs["height"] = height
        else:
            # SD3 image2image cannot accept width and height
            allow_width_height = model_accept_param(["width", "height"], model)
            if allow_width_height:
                if isinstance(image, list):
                    kwargs["width"], kwargs["height"] = image[0].size
                else:
                    kwargs["width"], kwargs["height"] = image.size

        if self._model_expects_four_channel_input(model):
            image = self._ensure_four_channel_image(image, model)
        else:
            image = self._ensure_three_channel_image(image)

        # generate config for lightning
        self._gen_config_for_lightning(kwargs)

        return self._call_model(
            image=image,
            prompt=prompt,
            num_images_per_prompt=n,
            response_format=response_format,
            model=model,
            **kwargs,
        )

    def inpainting(
        self,
        image: PIL.Image,
        mask_image: PIL.Image,
        prompt: Optional[Union[str, List[str]]] = None,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        **kwargs,
    ):
        ability = "inpainting"
        if ability not in self._abilities:
            raise RuntimeError(f"{self._model_uid} does not support inpainting")

        if (
            "text2image" in self._abilities or "image2image" in self._abilities
        ) and self._model is not None:
            model = self._get_model(ability)
        else:
            model = self._model

        if mask_blur := kwargs.pop("mask_blur", None):
            logger.debug("Process mask image with mask_blur: %s", mask_blur)
            mask_image = model.mask_processor.blur(mask_image, blur_factor=mask_blur)  # type: ignore

        if "width" not in kwargs:
            kwargs["width"], kwargs["height"] = map(int, re.split(r"[^\d]+", size))

        if padding_image_to_multiple := kwargs.pop("padding_image_to_multiple", None):
            # Model like SD3 inpainting requires image's height and width is times of 16
            # padding the image if specified
            origin_x, origin_y = image.size
            kwargs["origin_size"] = (origin_x, origin_y)
            kwargs["is_padded"] = True
            image = self.pad_to_multiple(image, multiple=int(padding_image_to_multiple))
            mask_image = self.pad_to_multiple(
                mask_image, multiple=int(padding_image_to_multiple)
            )
            # calculate actual image size after padding
            kwargs["width"], kwargs["height"] = image.size

        # generate config for lightning
        self._gen_config_for_lightning(kwargs)

        return self._call_model(
            image=image,
            mask_image=mask_image,
            prompt=prompt,
            num_images_per_prompt=n,
            response_format=response_format,
            model=model,
            **kwargs,
        )
