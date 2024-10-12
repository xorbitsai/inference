# Copyright 2022-2023 XProbe Inc.
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

import base64
import contextlib
import gc
import inspect
import itertools
import logging
import os
import re
import sys
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import PIL.Image
import torch
from PIL import ImageOps

from ....constants import XINFERENCE_IMAGE_DIR
from ....device_utils import get_available_device, move_model_to_available_device
from ....types import Image, ImageList, LoRA
from ..sdapi import SDAPIDiffusionModelMixin

if TYPE_CHECKING:
    from ....core.progress_tracker import Progressor
    from ..core import ImageModelFamilyV1

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
        model_spec: Optional["ImageModelFamilyV1"] = None,
        **kwargs,
    ):
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

    @property
    def model_ability(self):
        return self._abilities

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
            self._kwargs["variant"] = "fp16"
            self._kwargs["use_safetensors"] = True
        if isinstance(torch_dtype, str):
            self._kwargs["torch_dtype"] = getattr(torch, torch_dtype)

        controlnet = self._kwargs.get("controlnet")
        if controlnet is not None:
            if isinstance(controlnet, tuple):
                self._kwargs["controlnet"] = self._get_controlnet_model(*controlnet)
            else:
                self._kwargs["controlnet"] = [
                    self._get_controlnet_model(*cn) for cn in controlnet
                ]

        quantize_text_encoder = self._kwargs.pop("quantize_text_encoder", None)
        if quantize_text_encoder:
            try:
                from transformers import BitsAndBytesConfig, T5EncoderModel
            except ImportError:
                error_message = "Failed to import module 'transformers'"
                installation_guide = [
                    "Please make sure 'transformers' is installed. ",
                    "You can install it by `pip install transformers`\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

            try:
                import bitsandbytes  # noqa: F401
            except ImportError:
                error_message = "Failed to import module 'bitsandbytes'"
                installation_guide = [
                    "Please make sure 'bitsandbytes' is installed. ",
                    "You can install it by `pip install bitsandbytes`\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

            for text_encoder_name in quantize_text_encoder.split(","):
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                quantization_kwargs = {}
                if torch_dtype:
                    quantization_kwargs["torch_dtype"] = torch_dtype
                text_encoder = T5EncoderModel.from_pretrained(
                    self._model_path,
                    subfolder=text_encoder_name,
                    quantization_config=quantization_config,
                    **quantization_kwargs,
                )
                self._kwargs[text_encoder_name] = text_encoder
                self._kwargs["device_map"] = "balanced"

        logger.debug(
            "Loading model from %s, kwargs: %s", self._model_path, self._kwargs
        )
        self._model = AutoPipelineModel.from_pretrained(
            self._model_path,
            **self._kwargs,
        )
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

    def _load_to_device(self, model):
        if self._kwargs.get("cpu_offload", False):
            logger.debug("CPU offloading model")
            model.enable_model_cpu_offload()
        elif self._kwargs.get("sequential_cpu_offload", False):
            logger.debug("CPU sequential offloading model")
            model.enable_sequential_cpu_offload()
        elif not self._kwargs.get("device_map"):
            logger.debug("Loading model to available device")
            model = move_model_to_available_device(self._model)
        # Recommended if your computer has < 64 GB of RAM
        if self._kwargs.get("attention_slicing", True):
            model.enable_attention_slicing()
        if self._kwargs.get("vae_tiling", False):
            model.enable_vae_tiling()

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

    @staticmethod
    @contextlib.contextmanager
    def _reset_when_done(model: Any, sampler_name: str):
        assert model is not None
        scheduler = DiffusionModel._get_scheduler(model, sampler_name)
        if scheduler:
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
            self._filter_kwargs(model, kwargs)
            images = model(**kwargs).images

        # revert padding if padded
        if is_padded and origin_size:
            new_images = []
            x, y = origin_size
            for img in images:
                new_images.append(img.crop((0, 0, x, y)))
            images = new_images

        if return_images:
            return images

        if response_format == "url":
            os.makedirs(XINFERENCE_IMAGE_DIR, exist_ok=True)
            image_list = []
            with ThreadPoolExecutor() as executor:
                for img in images:
                    path = os.path.join(XINFERENCE_IMAGE_DIR, uuid.uuid4().hex + ".jpg")
                    image_list.append(Image(url=path, b64_json=None))
                    executor.submit(img.save, path, "jpeg")
            return ImageList(created=int(time.time()), data=image_list)
        elif response_format == "b64_json":

            def _gen_base64_image(_img):
                buffered = BytesIO()
                _img.save(buffered, format="jpeg")
                return base64.b64encode(buffered.getvalue()).decode()

            with ThreadPoolExecutor() as executor:
                results = list(map(partial(executor.submit, _gen_base64_image), images))  # type: ignore
                image_list = [Image(url=None, b64_json=s.result()) for s in results]  # type: ignore
            return ImageList(created=int(time.time()), data=image_list)
        else:
            raise ValueError(f"Unsupported response format: {response_format}")

    @classmethod
    def _filter_kwargs(cls, model, kwargs: dict):
        for arg in ["negative_prompt", "num_inference_steps"]:
            if not kwargs.get(arg):
                kwargs.pop(arg, None)

        for key in list(kwargs):
            allow_key = model_accept_param(key, model)
            if not allow_key:
                warnings.warn(f"{type(model)} cannot accept `{key}`, will ignore it")
                kwargs.pop(key)

    def text_to_image(
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

        return self._call_model(
            prompt=prompt,
            num_images_per_prompt=n,
            response_format=response_format,
            **generate_kwargs,
        )

    @staticmethod
    def pad_to_multiple(image, multiple=8):
        x, y = image.size
        padding_x = (multiple - x % multiple) % multiple
        padding_y = (multiple - y % multiple) % multiple
        padding = (0, 0, padding_x, padding_y)
        return ImageOps.expand(image, padding)

    def image_to_image(
        self,
        image: PIL.Image,
        prompt: Optional[Union[str, List[str]]] = None,
        n: int = 1,
        size: Optional[str] = None,
        response_format: str = "url",
        **kwargs,
    ):
        if self._kwargs.get("controlnet"):
            model = self._model
        else:
            ability = "image2image"
            if ability not in self._abilities:
                raise RuntimeError(f"{self._model_uid} does not support image2image")
            model = self._get_model(ability)

        if padding_image_to_multiple := kwargs.pop("padding_image_to_multiple", None):
            # Model like SD3 image to image requires image's height and width is times of 16
            # padding the image if specified
            origin_x, origin_y = image.size
            kwargs["origin_size"] = (origin_x, origin_y)
            kwargs["is_padded"] = True
            image = self.pad_to_multiple(image, multiple=int(padding_image_to_multiple))

        if size:
            width, height = map(int, re.split(r"[^\d]+", size))
            if padding_image_to_multiple:
                width, height = image.size
            kwargs["width"] = width
            kwargs["height"] = height
        else:
            # SD3 image2image cannot accept width and height
            allow_width_height = model_accept_param(["width", "height"], model)
            if allow_width_height:
                kwargs["width"], kwargs["height"] = image.size

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

        return self._call_model(
            image=image,
            mask_image=mask_image,
            prompt=prompt,
            num_images_per_prompt=n,
            response_format=response_format,
            model=model,
            **kwargs,
        )
