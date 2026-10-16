"""Memory-aware device placement for the MLLM inference frontend."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


DEFAULT_GPU0_RESERVED_LAYER_EQUIVALENTS = 4


class MLLMDeviceMapError(ValueError):
    """Raised when a requested MLLM layout is incomplete or unsafe."""


@dataclass(frozen=True)
class MLLMDevicePlan:
    num_hidden_layers: int
    n_gpu: int
    gpu0_reserved_layer_equivalents: int
    layer_counts: tuple[int, ...]
    layer_devices: tuple[int, ...]
    device_map: dict[str, int]


def load_mllm_num_hidden_layers(model_directory: str | Path) -> int:
    config_path = Path(model_directory) / "config.json"
    try:
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MLLMDeviceMapError(f"cannot read MLLM config {config_path}: {exc}") from None

    llm_config = config.get("llm_config")
    nested = llm_config.get("num_hidden_layers") if isinstance(llm_config, dict) else None
    root = config.get("num_hidden_layers")
    if nested is not None and root is not None and nested != root:
        raise MLLMDeviceMapError(
            "ambiguous decoder depth: "
            f"llm_config.num_hidden_layers={nested!r}, num_hidden_layers={root!r}"
        )
    value = nested if nested is not None else root
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MLLMDeviceMapError(
            f"missing or invalid MLLM num_hidden_layers in {config_path}: {value!r}"
        )
    return value


def allocate_mllm_layer_counts(
    num_hidden_layers: int,
    n_gpu: int,
    *,
    gpu0_reserved_layer_equivalents: int = DEFAULT_GPU0_RESERVED_LAYER_EQUIVALENTS,
) -> tuple[int, ...]:
    if (
        isinstance(n_gpu, bool)
        or not isinstance(n_gpu, int)
        or n_gpu <= 0
    ):
        raise MLLMDeviceMapError(f"n_gpu must be a positive integer, got {n_gpu!r}")
    if isinstance(num_hidden_layers, bool) or not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
        raise MLLMDeviceMapError(f"num_hidden_layers must be positive, got {num_hidden_layers!r}")
    if (
        isinstance(gpu0_reserved_layer_equivalents, bool)
        or not isinstance(gpu0_reserved_layer_equivalents, int)
        or gpu0_reserved_layer_equivalents < 0
    ):
        raise MLLMDeviceMapError(
            "gpu0_reserved_layer_equivalents must be a non-negative integer"
        )

    if n_gpu == 1:
        # Single-GPU plan: every decoder layer and every fixed image module
        # shares logical device 0, so nothing is kept sharded.
        return (num_hidden_layers,)

    effective_target = max(
        gpu0_reserved_layer_equivalents,
        math.ceil((num_hidden_layers + gpu0_reserved_layer_equivalents) / n_gpu),
    )
    gpu0_layers = min(
        num_hidden_layers,
        max(0, effective_target - gpu0_reserved_layer_equivalents),
    )
    base, remainder = divmod(num_hidden_layers - gpu0_layers, n_gpu - 1)
    other_counts = [base] * (n_gpu - 1)
    for index in range(len(other_counts) - remainder, len(other_counts)):
        other_counts[index] += 1
    counts = (gpu0_layers, *other_counts)
    if sum(counts) != num_hidden_layers:
        raise AssertionError(f"invalid internal MLLM allocation: {counts}")
    return counts


def build_mllm_device_plan(
    num_hidden_layers: int,
    n_gpu: int,
    *,
    gpu0_reserved_layer_equivalents: int = DEFAULT_GPU0_RESERVED_LAYER_EQUIVALENTS,
) -> MLLMDevicePlan:
    counts = allocate_mllm_layer_counts(
        num_hidden_layers,
        n_gpu,
        gpu0_reserved_layer_equivalents=gpu0_reserved_layer_equivalents,
    )
    layer_devices = tuple(
        device for device, count in enumerate(counts) for _ in range(count)
    )
    device_map = {
        f"model.model.layers.{layer_index}": device
        for layer_index, device in enumerate(layer_devices)
    }
    # Image conditioning and diffusion modules are attached after the base
    # checkpoint load and are intentionally placed on logical CUDA device 0.
    device_map.update(
        {
            "vision": 0,
            "linear_proj": 0,
            "model.model.word_embeddings.weight": 0,
            "model.model.norm.weight": 0,
            "model.lm_head.weight": 0,
            "model.model.norm": 0,
        }
    )
    return MLLMDevicePlan(
        num_hidden_layers=num_hidden_layers,
        n_gpu=n_gpu,
        gpu0_reserved_layer_equivalents=gpu0_reserved_layer_equivalents,
        layer_counts=counts,
        layer_devices=layer_devices,
        device_map=device_map,
    )


def validate_loaded_layer_devices(
    actual_devices: list[int], plan: MLLMDevicePlan
) -> None:
    actual = tuple(actual_devices)
    if actual != plan.layer_devices:
        raise MLLMDeviceMapError(
            "loaded MLLM layer placement disagrees with the requested plan: "
            f"actual={actual}, expected={plan.layer_devices}"
        )
