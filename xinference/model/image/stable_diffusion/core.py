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
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from typing import List, Optional, Union

from ....constants import XINFERENCE_IMAGE_DIR
from ....types import Image, ImageList

logger = logging.getLogger(__name__)


class DiffusionModel:
    def __init__(
        self, model_uid: str, model_path: str, device: Optional[str] = None, **kwargs
    ):
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._model = None
        self._kwargs = kwargs

    def load(self):
        import torch
        from diffusers import AutoPipelineForText2Image

        controlnet = self._kwargs.get("controlnet")
        if controlnet is not None:
            from diffusers import ControlNetModel

            logger.debug("Loading controlnet %s", controlnet)
            self._kwargs["controlnet"] = ControlNetModel.from_pretrained(controlnet)

        self._model = AutoPipelineForText2Image.from_pretrained(
            self._model_path,
            **self._kwargs,
            # The following params crashes on Mac M2
            # torch_dtype=torch.float16,
            # use_safetensors=True,
        )
        if torch.cuda.is_available():
            self._model = self._model.to("cuda")
        elif torch.backends.mps.is_available():
            self._model = self._model.to("mps")
        # Recommended if your computer has < 64 GB of RAM
        self._model.enable_attention_slicing()

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
