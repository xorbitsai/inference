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

from enum import Enum
from typing import List, Tuple, Union

import torch
from PIL import Image

from .._compat import LANCZOS, NEAREST
from .latent import LatentUpscaler, latent_upscale_modes
from .realesrgan import RealESRGANmodel


class HiResUpscaler(str, Enum):
    none = "None"
    Latent = "Latent"
    LatentAntialiased = "Latent (antialiased)"
    LatentBicubic = "Latent (bicubic)"
    LatentBicubicAntialiased = "Latent (bicubic antialiased)"
    LatentNearest = "Latent (nearest)"
    LatentNearestExact = "Latent (nearest-exact)"
    Lanczos = "Lanczos"
    Nearest = "Nearest"
    ESRGAN_4x = "R-ESRGAN 4x+"
    ESRGAN_4x_ANIME = "R-ESRGAN 4x+ Anime6B"
    LDSR = "LDSR"
    ScuNET_GAN = "ScuNET GAN"
    ScuNET_PSNR = "ScuNET PSNR"
    SwinIR_4x = "SwinIR 4x"


SCALER_INFO = {
    HiResUpscaler.ESRGAN_4x: {"model": RealESRGANmodel, "scale": 4},
    HiResUpscaler.ESRGAN_4x_ANIME: {
        "model": RealESRGANmodel,
        "scale": 4,
    },
}


def upscaler_preprocess(upscaler: HiResUpscaler, kwargs):
    # for latent-based upscaler
    # tell text_to_image return latent instead of image
    if upscaler.value.startswith("Latent"):
        kwargs["output_type"] = "latent"


def _get_size(img: Union[Image.Image, torch.Tensor]) -> tuple:
    if isinstance(img, torch.Tensor):
        return img.shape
    else:
        return img.size


def _get_upscaler_model(upscaler: HiResUpscaler, **kwargs):
    try:
        return SCALER_INFO[upscaler]["model"](upscaler.value, **kwargs)  # type: ignore
    except KeyError:
        if upscaler.value.startswith("Latent"):
            try:
                _ = latent_upscale_modes[upscaler]
                return LatentUpscaler(upscaler.value, **kwargs)
            except KeyError:
                pass
        raise NotImplementedError(f"scaler({str(upscaler)}) not implemented yet")


def upscale(
    upscaler: HiResUpscaler,
    images: List[Image.Image],
    scale: int,
    size: Tuple[int, int],
    **kwargs,
) -> List[Image.Image]:
    assert (
        len({_get_size(img) for img in images}) == 1
    ), f"image must have same size, got {[_get_size(img) for img in images]}"

    if upscaler in (HiResUpscaler.Lanczos, HiResUpscaler.Nearest):
        resample = LANCZOS if upscaler == HiResUpscaler.Lanczos else NEAREST
        return [image.resize(size, resample) for image in images]

    model = _get_upscaler_model(upscaler, **kwargs)
    model.load()

    if hasattr(model, "batch_upscale"):
        return model.batch_upscale(images, size)  # type:ignore
    else:
        dest_w, dest_h = size

        result_images = []
        for image in images:
            for i in range(3):
                if (
                    image.width >= dest_w
                    and image.height >= dest_h
                    and (i > 0 or scale != 1)
                ):
                    break

                shape = (image.width, image.height)

                image = model.upscale(image, (dest_w, dest_h))

                if shape == (image.width, image.height):
                    break

            if image.width != dest_w or image.height != dest_h:
                image = image.resize((int(dest_w), int(dest_h)), resample=LANCZOS)
            result_images.append(image)
        return result_images
