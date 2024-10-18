# Copyright 2022-2024 XProbe Inc.
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
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING, Optional

from ...constants import XINFERENCE_IMAGE_DIR
from ...types import Image, ImageList

if TYPE_CHECKING:
    from .core import ImageModelFamilyV1


def get_model_version(
    image_model: "ImageModelFamilyV1", controlnet: Optional["ImageModelFamilyV1"]
) -> str:
    return (
        image_model.model_name
        if controlnet is None
        else f"{image_model.model_name}--{controlnet.model_name}"
    )


def handle_image_result(response_format: str, images) -> ImageList:
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
