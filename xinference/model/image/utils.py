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
    from .core import ImageModelFamilyV2


def get_model_version(
    image_model: "ImageModelFamilyV2", controlnet: Optional["ImageModelFamilyV2"]
) -> str:
    return (
        image_model.model_name
        if controlnet is None
        else f"{image_model.model_name}--{controlnet.model_name}"
    )


def _flatten_images(images):
    if images and isinstance(images[0], (list, tuple)):
        flat_images = []
        for group in images:
            if isinstance(group, (list, tuple)):
                flat_images.extend(group)
            else:
                flat_images.append(group)
        return flat_images
    return images


def _needs_png(image) -> bool:
    if image.mode in ("RGBA", "LA"):
        return True
    if image.mode == "P" and "transparency" in image.info:
        return True
    return False


def handle_image_result(response_format: str, images) -> ImageList:
    images = _flatten_images(images)
    if response_format == "url":
        os.makedirs(XINFERENCE_IMAGE_DIR, exist_ok=True)
        image_list = []
        with ThreadPoolExecutor() as executor:
            for img in images:
                use_png = _needs_png(img)
                suffix = ".png" if use_png else ".jpg"
                path = os.path.join(XINFERENCE_IMAGE_DIR, uuid.uuid4().hex + suffix)
                image_list.append(Image(url=path, b64_json=None))
                fmt = "png" if use_png else "jpeg"
                executor.submit(img.save, path, fmt)
        return ImageList(created=int(time.time()), data=image_list)
    elif response_format == "b64_json":

        def _gen_base64_image(_img):
            buffered = BytesIO()
            fmt = "png" if _needs_png(_img) else "jpeg"
            _img.save(buffered, format=fmt)
            return base64.b64encode(buffered.getvalue()).decode()

        with ThreadPoolExecutor() as executor:
            results = list(map(partial(executor.submit, _gen_base64_image), images))  # type: ignore
            image_list = [Image(url=None, b64_json=s.result()) for s in results]  # type: ignore
        return ImageList(created=int(time.time()), data=image_list)
    else:
        raise ValueError(f"Unsupported response format: {response_format}")
