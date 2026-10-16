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

import os

import numpy as np
from PIL import Image
from torch.hub import download_url_to_file

from ....constants import XINFERENCE_CACHE_DIR
from ....device_utils import get_available_device
from ..utils import make_valid_filename


class RealESRGANmodel:
    def __init__(self, model_name: str, **kwargs):
        self._model_name = model_name
        self._kwargs = kwargs
        # params about model
        self._upsampler = None

    @staticmethod
    def _ensure_model_downloaded(model_name: str, model_url: str) -> str:
        cache_name = make_valid_filename(model_name)
        cache_dir = os.path.realpath(os.path.join(XINFERENCE_CACHE_DIR, cache_name))
        os.makedirs(cache_dir, exist_ok=True)

        file_name = model_url.rsplit("/", 1)[-1]
        file_path = os.path.join(cache_dir, file_name)
        if os.path.exists(file_path):
            return file_path

        # download
        download_url_to_file(model_url, file_path)
        return file_path

    def load(self):
        # Spandrel loads the same RealESRGAN weights without BasicSR's dependency
        # on torchvision.transforms.functional_tensor (removed in torchvision).
        try:
            from spandrel import ImageModelDescriptor, ModelLoader
        except ImportError as exc:
            raise ImportError("RealESRGAN requires `pip install spandrel`") from exc

        urls = {
            "R-ESRGAN 4x+": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "R-ESRGAN 4x+ Anime6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        }
        if self._model_name not in urls:
            raise ValueError(f"Unknown upscaler: {self._model_name}")
        path = self._ensure_model_downloaded(self._model_name, urls[self._model_name])
        device = self._kwargs.get("device") or get_available_device()
        descriptor = ModelLoader(device=device).load_from_file(path)
        if not isinstance(descriptor, ImageModelDescriptor):
            raise ValueError("Expected an image upscaling model")
        self._upsampler = descriptor.eval()
        if (
            str(device).startswith("cuda")
            and not self._kwargs.get("fp32", False)
            and descriptor.supports_half
        ):
            descriptor.half()

    def upscale(self, img: Image.Image, *_) -> Image.Image:
        import torch

        model = self._upsampler
        if model is None:
            raise RuntimeError("Upscaler must be loaded before inference")
        pixels = (
            torch.from_numpy(np.array(img.convert("RGB"), dtype=np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        height, width = pixels.shape[-2:]
        scale = model.scale
        tile = int(self._kwargs.get("ESRGAN_tile", 192)) or max(height, width)
        padding = int(self._kwargs.get("ESRGAN_tile_overlap", 8))
        if tile < 1 or padding < 0:
            raise ValueError("Invalid upscaler tile dimensions")
        result = torch.empty((1, 3, height * scale, width * scale))
        with torch.inference_mode():
            for y in range(0, height, tile):
                for x in range(0, width, tile):
                    y1, x1 = min(y + tile, height), min(x + tile, width)
                    top, left = max(y - padding, 0), max(x - padding, 0)
                    bottom, right = min(y1 + padding, height), min(x1 + padding, width)
                    patch = pixels[:, :, top:bottom, left:right].to(
                        device=model.device, dtype=model.dtype
                    )
                    output = model(patch).float().cpu()
                    result[:, :, y * scale : y1 * scale, x * scale : x1 * scale] = (
                        output[
                            :,
                            :,
                            (y - top) * scale : (y1 - top) * scale,
                            (x - left) * scale : (x1 - left) * scale,
                        ]
                    )
        array = (
            result.squeeze(0)
            .permute(1, 2, 0)
            .clamp(0, 1)
            .mul(255)
            .round()
            .byte()
            .numpy()
        )
        return Image.fromarray(array)
