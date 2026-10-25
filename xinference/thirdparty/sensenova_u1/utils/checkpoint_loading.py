"""Shared model + tokenizer loader for SenseNova-U1.

Centralises the ``AutoConfig`` / ``AutoTokenizer`` / ``AutoModel`` calls used
by the example scripts and the ComfyUI app, and adds an optional GGUF
checkpoint override.

Usage:

    from sensenova_u1.utils import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(
        model_path="sensenova/SenseNova-U1-8B-MoT",
        dtype=torch.bfloat16,
        # device=None auto-picks CUDA > XPU > CPU. Pass an explicit
        # "cuda" / "cuda:0" / "xpu" / "xpu:0" to override.
    )

    # GGUF override (config / tokenizer still come from `model_path`):
    model, tokenizer = load_model_and_tokenizer(
        model_path="sensenova/SenseNova-U1-8B-MoT",
        dtype=torch.bfloat16,
        gguf_checkpoint="/path/to/SenseNova-U1-8B-MoT-Q5_K_M.gguf",
    )
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import accel

LOGGER = logging.getLogger(__name__)


def _default_device() -> torch.device:
    """Pick CUDA, then XPU, then CPU. Used as the default ``device`` for loaders."""
    return accel.best_available_device()


def add_offload_args(parser: argparse.ArgumentParser) -> None:
    """Add Transformers/Accelerate device-map and layer-offload flags to an example CLI."""
    from .offload import (
        DEFAULT_FAST_ACTIVATION_RESERVE_GIB,
        DEFAULT_FAST_VRAM_FRACTION,
        DEFAULT_FAST_VRAM_HEADROOM_GIB,
        DEFAULT_VRAM_MODE,
        VRAM_MODE_OPTIONS,
    )

    def fraction(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or not 0 < parsed <= 1:
            raise argparse.ArgumentTypeError("must satisfy 0 < value <= 1")
        return parsed

    def nonnegative_gib(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0:
            raise argparse.ArgumentTypeError("must be >= 0")
        return parsed

    def positive_gib(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0:
            raise argparse.ArgumentTypeError("must be > 0")
        return parsed

    parser.add_argument(
        "--device_map",
        default=None,
        help=(
            "Optional Transformers device_map, e.g. 'auto', 'balanced', "
            "'balanced_low_0', or 'sequential'. When set, the model is loaded "
            "with Accelerate dispatch and is not moved again with .to(device). "
            "Use this for multi-GPU split; for low-VRAM single-card, prefer --vram_mode."
        ),
    )
    parser.add_argument(
        "--max_memory",
        default=None,
        help=(
            "Optional per-device memory limits for --device_map, either JSON "
            "or comma-separated KEY=VALUE pairs, e.g. '0=20GiB,1=20GiB'."
        ),
    )
    parser.add_argument(
        "--vram_mode",
        choices=list(VRAM_MODE_OPTIONS),
        default=DEFAULT_VRAM_MODE,
        help=(
            "Single-GPU layer-offload mode. "
            "'full' = no offload, whole model on GPU, fastest (default). "
            "'fast' = async prefetch, then retain generation layers within the GPU memory budget. "
            "'low' = synchronous per-layer CPU<->GPU swap, smallest weight footprint. "
            "'balanced' = async prefetch, overlaps H2D with compute, faster than 'low'. "
            "Mutually exclusive with --device_map (layer offload requires the model on CPU)."
        ),
    )
    parser.add_argument(
        "--fast_vram_fraction",
        type=fraction,
        default=DEFAULT_FAST_VRAM_FRACTION,
        help="Fast-mode automatic VRAM budget as a fraction of physical memory (default: 0.90).",
    )
    parser.add_argument(
        "--fast_vram_headroom_gib",
        type=nonnegative_gib,
        default=DEFAULT_FAST_VRAM_HEADROOM_GIB,
        help="Fast-mode reusable VRAM headroom reserved after projected activation growth (default: 2 GiB).",
    )
    parser.add_argument(
        "--fast_activation_reserve_gib",
        type=nonnegative_gib,
        default=DEFAULT_FAST_ACTIVATION_RESERVE_GIB,
        help="Fast-mode allowance for activations allocated after decoder layers (default: 4 GiB).",
    )
    parser.add_argument(
        "--fast_vram_budget_gib",
        type=positive_gib,
        default=None,
        help="Optional absolute fast-mode VRAM budget in GiB; overrides --fast_vram_fraction.",
    )


def infer_input_device(model: nn.Module, fallback: str | torch.device | None = None) -> torch.device:
    """Pick a usable device for tensors passed into a dispatched model.

    When ``fallback`` is ``None`` (the default), auto-detects the best
    accelerator (CUDA > XPU > CPU).
    """
    for param in model.parameters():
        if param.device.type not in {"cpu", "meta"}:
            return param.device
    if fallback is None:
        return _default_device()
    return torch.device(fallback) if isinstance(fallback, str) else fallback


def _resolve_local_model_path(model_path: str) -> str:
    """Resolve a HF id to a complete cached snapshot directory when offline.

    Mirrors transformers' fall-back behaviour but skips the up-front HEAD
    request that times out on offline machines. A partially cached snapshot
    must not shadow the Hub id, because Transformers would then treat it as a
    local directory and lose the opportunity to fetch its missing weights.
    """
    if Path(model_path).exists():
        return model_path
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(model_path, local_files_only=True))
    except Exception:
        return model_path
    if _has_complete_model_weights(snapshot):
        return str(snapshot)
    return model_path


def _has_complete_model_weights(snapshot: Path) -> bool:
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = snapshot / index_name
        if not index_path.is_file():
            continue
        try:
            shard_names = set(json.loads(index_path.read_text())["weight_map"].values())
        except (KeyError, TypeError, OSError, json.JSONDecodeError):
            return False
        return bool(shard_names) and all((snapshot / shard_name).is_file() for shard_name in shard_names)
    return (snapshot / "model.safetensors").is_file() or (snapshot / "pytorch_model.bin").is_file()


def load_model_and_tokenizer(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: str | torch.device | None = None,
    gguf_checkpoint: str | None = None,
    device_map: str | None = None,
    max_memory: str | dict[int | str, str] | None = None,
    for_offload: bool = False,
) -> tuple[nn.Module, Any]:
    """Build a SenseNova-U1 model + tokenizer pair.

    ``model_path`` always provides the config and tokenizer (HF id or local
    directory containing ``config.json``).

    Weight loading branches on ``gguf_checkpoint``:

    - ``None``: standard ``AutoModel.from_pretrained(model_path, ...)``.
      The ``device_map`` / ``max_memory`` accelerate kwargs apply on this
      path; when ``device_map`` is ``None`` the model is ``.to(device)``
      after loading.
    - ``"*.gguf"``: build a meta-init model from the config and inject
      dequantizing weights from the GGUF file via the diffusers quantizer.
      The accelerate kwargs are ignored on this path.

    When ``for_offload=True`` the loaded model stays on CPU (no ``.to(device)``)
    so a downstream layer-offload wrapper can manage CPU<->GPU movement
    itself. ``device_map`` is forced to ``None`` in this mode (with a warning)
    because accelerate's static placement is incompatible with dynamic offload.
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from .. import check_checkpoint_compatibility
    from ..models.neo_unify.transformers_compat import pretrained_dtype_kwargs

    if for_offload and device_map:
        LOGGER.warning(
            "for_offload=True overrides device_map=%r (accelerate placement is incompatible with layer offload).",
            device_map,
        )
        device_map = None

    if device is None and not device_map and not for_offload:
        device = _default_device()

    model_path = _resolve_local_model_path(model_path)
    config = AutoConfig.from_pretrained(model_path)
    check_checkpoint_compatibility(config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if gguf_checkpoint is not None:
        gguf_device = torch.device("cpu") if for_offload else device
        model = _load_from_gguf(config, gguf_checkpoint, dtype=dtype, device=gguf_device)
    else:
        model_kwargs: dict[str, Any] = {"config": config, **pretrained_dtype_kwargs(dtype)}
        if device_map:
            model_kwargs["device_map"] = device_map
            parsed_max_memory = _normalize_max_memory(max_memory)
            if parsed_max_memory:
                model_kwargs["max_memory"] = parsed_max_memory

        model = AutoModel.from_pretrained(model_path, **model_kwargs).eval()
        if not device_map and device is not None and not for_offload:
            model = model.to(device)

    return model, tokenizer


def _normalize_max_memory(value: str | dict | None) -> dict[int | str, str]:
    """Accept a parsed mapping, JSON object, or comma-separated CLI form ``"0=20GiB,cpu=64GiB"``."""
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return {_coerce_memory_key(k): str(v) for k, v in value.items()}
    stripped = value.strip()
    if stripped.startswith("{"):
        raw = json.loads(stripped)
        if not isinstance(raw, dict):
            raise RuntimeError("max_memory JSON must be an object")
        return {_coerce_memory_key(k): str(v) for k, v in raw.items()}
    result: dict[int | str, str] = {}
    for item in stripped.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise RuntimeError("max_memory entries must look like 0=20GiB,cpu=64GiB.")
        key, memory = item.split("=", 1)
        key = key.strip()
        memory = memory.strip()
        if not key or not memory:
            raise RuntimeError("max_memory entries must include both device and memory.")
        result[_coerce_memory_key(key)] = memory
    return result


def _coerce_memory_key(key: object) -> int | str:
    if isinstance(key, int):
        return key
    key_str = str(key)
    return int(key_str) if key_str.isdigit() else key_str


parse_max_memory = _normalize_max_memory


def _load_from_gguf(
    config,
    gguf_checkpoint: str,
    *,
    dtype: torch.dtype,
    device: str | torch.device | None,
) -> nn.Module:
    try:
        from accelerate import init_empty_weights
    except ImportError as exc:
        raise RuntimeError("GGUF loading requires `accelerate`; install it in your environment.") from exc

    from transformers import AutoModel

    from .gguf_loader import load_gguf_checkpoint, set_gguf2meta_model

    print(f"[gguf] loading quantized checkpoint from {gguf_checkpoint}")
    with init_empty_weights():
        model = AutoModel.from_config(config)

    state_dict = load_gguf_checkpoint(gguf_checkpoint)
    print(f"[gguf] parsed {len(state_dict)} tensors")
    target_device = torch.device(device) if isinstance(device, str) else device
    # set_gguf2meta_model places weights on `target_device` while injecting;
    # callers that ultimately want a different device can `.to()` afterwards.
    set_gguf2meta_model(model, state_dict, dtype, target_device)

    n_gguf_linear = sum(1 for m in model.modules() if type(m).__name__ == "GGUFLinear")
    print(f"[gguf] {n_gguf_linear} GGUFLinear modules active (dequantized at forward time)")
    if n_gguf_linear == 0:
        print("[gguf] WARNING: no GGUFLinear modules found — quantizer hook did not run as expected")

    del state_dict
    gc.collect()
    accel.empty_cache()
    return model.eval()
