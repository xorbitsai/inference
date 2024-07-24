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
from io import BytesIO

import PIL.Image
import PIL.ImageOps

from ....types import Image
from ..core import FlexibleModel, FlexibleModelSpec


class ImageRemoveBackgroundModel(FlexibleModel):
    def infer(self, **kwargs):
        invert = kwargs.get("invert", False)
        b64_image: str = kwargs.get("image")  # type: ignore
        only_mask = kwargs.pop("only_mask", True)
        image_format = kwargs.pop("image_format", "PNG")
        if not b64_image:
            raise ValueError("No image found to remove background")
        image = base64.b64decode(b64_image)

        try:
            from rembg import remove
        except ImportError:
            error_message = "Failed to import module 'rembg'"
            installation_guide = [
                "Please make sure 'rembg' is installed. ",
                "You can install it by visiting the installation section of the git repo:\n",
                "https://github.com/danielgatis/rembg?tab=readme-ov-file#installation",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        im = PIL.Image.open(BytesIO(image))
        om = remove(im, only_mask=only_mask, **kwargs)
        if invert:
            om = PIL.ImageOps.invert(om)

        buffered = BytesIO()
        om.save(buffered, format=image_format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return Image(url=None, b64_json=img_str)


def launcher(model_uid: str, model_spec: FlexibleModelSpec, **kwargs) -> FlexibleModel:
    task = kwargs.get("task")
    device = kwargs.get("device")

    if task == "remove_background":
        return ImageRemoveBackgroundModel(
            model_uid=model_uid,
            model_path=model_spec.model_uri,  # type: ignore
            device=device,
            config=kwargs,
        )
    else:
        raise ValueError(f"Unknown Task for image processing: {task}")
