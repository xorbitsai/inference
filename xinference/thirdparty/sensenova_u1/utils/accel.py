"""Accelerator-agnostic helpers for CUDA / XPU.

The codebase used to hard-code ``torch.cuda.*`` for device-availability checks,
cache management, and stream APIs. This module centralises the small amount of
namespace switching needed so the same code paths work on both CUDA (incl.
ROCm via the ``cuda`` namespace) and Intel XPU.

CPU / MPS are intentionally out of scope: the layer-offload / inference path
needs pinned host memory and dedicated transfer streams, which neither of
those backends exposes uniformly. Targets PyTorch >= 2.5 where ``torch.xpu``
is in-tree.
"""

from __future__ import annotations

import torch

SUPPORTED_DEVICE_TYPES: tuple[str, ...] = ("cuda", "xpu")


def accel_module(device: torch.device):
    """Return ``torch.cuda`` or ``torch.xpu`` matching ``device.type``."""
    if device.type == "cuda":
        return torch.cuda
    if device.type == "xpu":
        return torch.xpu
    raise NotImplementedError(f"No accelerator namespace for device {device!r}; supported: {SUPPORTED_DEVICE_TYPES}.")


def require_accelerator(device: torch.device) -> None:
    """Raise unless ``device`` is a backend the inference path actually supports."""
    if device.type not in SUPPORTED_DEVICE_TYPES:
        raise NotImplementedError(
            f"Inference requires a CUDA or XPU device (got {device!r}). "
            "CPU / MPS lack the pinned-memory and stream primitives used here."
        )


def is_available(device_type: str) -> bool:
    """``True`` if the backend module reports an available device."""
    if device_type == "cuda":
        return torch.cuda.is_available()
    if device_type == "xpu":
        return torch.xpu.is_available()
    return False


def best_available_device() -> torch.device:
    """Pick the best available accelerator, preferring CUDA over XPU over CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def empty_cache(device: torch.device | None = None) -> None:
    """Release the accelerator's caching-allocator blocks.

    With ``device=None`` runs the equivalent of "for every backend that has a
    device available, drop its cache" — used at teardown where we just want
    the global state cleaned up regardless of which backend we ran on.
    """
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.xpu.is_available():
            torch.xpu.empty_cache()
        return
    if device.type in SUPPORTED_DEVICE_TYPES and is_available(device.type):
        accel_module(device).empty_cache()


def synchronize(device: torch.device | None = None) -> None:
    """Block until pending work on the accelerator completes."""
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if torch.xpu.is_available():
            torch.xpu.synchronize()
        return
    if device.type in SUPPORTED_DEVICE_TYPES and is_available(device.type):
        accel_module(device).synchronize(device)


def manual_seed_all(seed: int) -> None:
    """Seed all available accelerator devices (CUDA + XPU)."""
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
