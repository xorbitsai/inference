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

from typing import List, Optional, Tuple

import torch

from ...device_utils import get_available_device

# originally implemented in stable-diffusion-webui


def randn(
    seed: int,
    shape: Tuple[int, ...],
    generator: torch.Generator = None,
    dtype: torch.dtype = None,
    device=None,
) -> torch.Tensor:
    """Generate a tensor with random numbers from a normal distribution using seed.

    Uses the seed parameter to set the global torch seed; to generate more with that seed, use randn_like/randn_without_seed.
    """

    generator = generator or create_generator(seed, device)
    device = str(device or get_available_device())
    if device == "mps":
        return torch.randn(shape, device="cpu", dtype=dtype, generator=generator).to(
            device
        )
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def randn_without_seed(
    shape: Tuple[int, ...],
    generator: torch.Generator = None,
    dtype: torch.dtype = None,
    device=None,
) -> torch.Tensor:
    """Generate a tensor with random numbers from a normal distribution using the previously initialized generator.

    Use either randn() or manual_seed() to initialize the generator."""

    device = str(device or get_available_device())
    if device == "mps":
        return torch.randn(shape, device="cpu", dtype=dtype, generator=generator).to(
            device
        )
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def create_generator(seed: int, device=None) -> torch.Generator:
    device = str(device or get_available_device())
    device = "cpu" if device == "mps" else device
    generator = torch.Generator(device).manual_seed(int(seed))
    return generator


# from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
def slerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm * high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * (1 - val) + high * val

    omega = torch.acos(dot.clamp(-1, 1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (
        torch.sin(val * omega) / so
    ).unsqueeze(1) * high
    return res


class ImageRNG:
    def __init__(
        self,
        shape: Tuple[int, ...],
        seeds: List[int],
        subseeds: Optional[List[int]] = None,
        subseed_strength: float = 0.0,
        seed_resize_from_h: int = 0,
        seed_resize_from_w: int = 0,
        override_settings: Optional[dict] = None,
        dtype: torch.dtype = None,
        device=None,
    ):
        self.device = device or get_available_device()
        self.shape = tuple(map(int, shape))
        self.seeds = seeds
        self.subseeds = subseeds
        self.subseed_strength = subseed_strength
        self.seed_resize_from_h = seed_resize_from_h
        self.seed_resize_from_w = seed_resize_from_w
        self.override_settings = override_settings or {}
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        self.generators = [create_generator(seed, self.device) for seed in seeds]

        self.is_first = True

    def first(self) -> torch.Tensor:
        noise_shape = (
            self.shape
            if self.seed_resize_from_h <= 0 or self.seed_resize_from_w <= 0
            else (
                self.shape[0],
                int(self.seed_resize_from_h) // 8,
                int(self.seed_resize_from_w // 8),
            )
        )

        xs = []

        for i, (seed, generator) in enumerate(zip(self.seeds, self.generators)):
            subnoise = None
            if self.subseeds is not None and self.subseed_strength != 0:
                subseed = 0 if i >= len(self.subseeds) else self.subseeds[i]
                subnoise = randn(
                    subseed, noise_shape, dtype=self.dtype, device=self.device
                )

            if noise_shape != self.shape:
                noise = randn(seed, noise_shape, dtype=self.dtype, device=self.device)
            else:
                noise = randn(
                    seed,
                    self.shape,
                    generator=generator,
                    dtype=self.dtype,
                    device=self.device,
                )

            if subnoise is not None:
                noise = slerp(self.subseed_strength, noise, subnoise)

            if noise_shape != self.shape:
                x = randn(
                    seed,
                    self.shape,
                    generator=generator,
                    dtype=self.dtype,
                    device=self.device,
                )
                dx = (self.shape[2] - noise_shape[2]) // 2
                dy = (self.shape[1] - noise_shape[1]) // 2
                w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
                h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
                tx = 0 if dx < 0 else dx
                ty = 0 if dy < 0 else dy
                dx = max(-dx, 0)
                dy = max(-dy, 0)

                x[:, ty : ty + h, tx : tx + w] = noise[:, dy : dy + h, dx : dx + w]
                noise = x

            xs.append(noise)

        eta_noise_seed_delta = self.override_settings.get("eta_noise_seed_delta") or 0
        if eta_noise_seed_delta:
            self.generators = [
                create_generator(seed + eta_noise_seed_delta, self.device)
                for seed in self.seeds
            ]

        return torch.stack(xs).to(self.device)

    def next(self) -> torch.Tensor:
        if self.is_first:
            self.is_first = False
            return self.first()

        xs = []
        for generator in self.generators:
            x = randn_without_seed(
                self.shape, generator=generator, dtype=self.dtype, device=self.device
            )
            xs.append(x)

        return torch.stack(xs).to(self.device)
