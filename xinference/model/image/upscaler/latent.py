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

from typing import Tuple

import torch

from ..constants import DOWNSAMPLE_FACTOR

latent_upscale_modes = {
    "Latent": {"mode": "bilinear", "antialias": False},
    "Latent (antialiased)": {"mode": "bilinear", "antialias": True},
    "Latent (bicubic)": {"mode": "bicubic", "antialias": False},
    "Latent (bicubic antialiased)": {"mode": "bicubic", "antialias": True},
    "Latent (nearest)": {"mode": "nearest", "antialias": False},
    "Latent (nearest-exact)": {"mode": "nearest-exact", "antialias": False},
}


class LatentUpscaler:
    def __init__(self, model_name: str, **kwargs):
        self._model_name = model_name
        self._kwargs = kwargs

    def load(self):
        # no model required for latent upscaler
        pass

    def batch_upscale(
        self, images: torch.Tensor, size: Tuple[int, int]
    ) -> torch.Tensor:
        target_width, target_height = size
        mode = latent_upscale_modes[self._model_name]
        return torch.nn.functional.interpolate(
            images,
            size=(
                target_height
                // self._kwargs.get("downsample_factor", DOWNSAMPLE_FACTOR),
                target_width
                // self._kwargs.get("downsample_factor", DOWNSAMPLE_FACTOR),
            ),
            mode=mode["mode"],
            antialias=mode["antialias"],
        )
