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
import logging
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from typing import Dict, List, Optional, Union

from ....constants import XINFERENCE_IMAGE_DIR
from ....device_utils import move_model_to_available_device
from ....types import Image, ImageList, LoRA

logger = logging.getLogger(__name__)


class DiffusionModel:
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        device: Optional[str] = None,
        lora_model: Optional[List[LoRA]] = None,
        lora_load_kwargs: Optional[Dict] = None,
        lora_fuse_kwargs: Optional[Dict] = None,
        ability: Optional[str] = None,
        **kwargs,
    ):
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._model = None
        self._lora_model = lora_model
        self._lora_load_kwargs = lora_load_kwargs or {}
        self._lora_fuse_kwargs = lora_fuse_kwargs or {}
        self._ability = ability
        self._kwargs = kwargs

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
        import torch

        if self._ability in [None, "text2image", "image2image"]:
            from diffusers import AutoPipelineForText2Image as AutoPipelineModel
        elif self._ability == "inpainting":
            from diffusers import AutoPipelineForInpainting as AutoPipelineModel
        else:
            raise ValueError(f"Unknown ability: {self._ability}")

        controlnet = self._kwargs.get("controlnet")
        if controlnet is not None:
            from diffusers import ControlNetModel

            logger.debug("Loading controlnet %s", controlnet)
            self._kwargs["controlnet"] = ControlNetModel.from_pretrained(controlnet)

        torch_dtype = self._kwargs.get("torch_dtype")
        if sys.platform != "darwin" and torch_dtype is None:
            # The following params crashes on Mac M2
            self._kwargs["torch_dtype"] = torch.float16
            self._kwargs["use_safetensors"] = True

        logger.debug("Loading model %s", AutoPipelineModel)
        self._model = AutoPipelineModel.from_pretrained(
            self._model_path,
            **self._kwargs,
        )
        self._model = move_model_to_available_device(self._model)
        # Recommended if your computer has < 64 GB of RAM
        self._model.enable_attention_slicing()
        self._apply_lora()

    def _call_model(
        self,
        height: int,
        width: int,
        num_images_per_prompt: int,
        response_format: str,
        **kwargs,
    ):
        logger.debug(
            "stable diffusion args: %s",
            dict(
                kwargs,
                height=height,
                width=width,
                num_images_per_prompt=num_images_per_prompt,
            ),
        )
        assert callable(self._model)
        images = self._model(
            height=height,
            width=width,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs,
        ).images
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
                results = list(map(partial(executor.submit, _gen_base64_image), images))
                image_list = [Image(url=None, b64_json=s.result()) for s in results]
            return ImageList(created=int(time.time()), data=image_list)
        else:
            raise ValueError(f"Unsupported response format: {response_format}")

    def text_to_image(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        **kwargs,
    ):
        # References:
        # https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet_sdxl
        width, height = map(int, re.split(r"[^\d]+", size))
        return self._call_model(
            prompt=prompt,
            height=height,
            width=width,
            num_images_per_prompt=n,
            response_format=response_format,
            **kwargs,
        )

    def image_to_image(
        self,
        image: bytes,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        **kwargs,
    ):
        width, height = map(int, re.split(r"[^\d]+", size))
        return self._call_model(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_images_per_prompt=n,
            response_format=response_format,
            **kwargs,
        )

    def inpainting(
        self,
        image: bytes,
        mask_image: bytes,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        **kwargs,
    ):
        width, height = map(int, re.split(r"[^\d]+", size))
        return self._call_model(
            image=image,
            mask_image=mask_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_images_per_prompt=n,
            response_format=response_format,
            **kwargs,
        )
