from __future__ import annotations

import importlib.metadata
import json
from collections.abc import Mapping
from typing import Any

import torch

QUANTIZATION_METADATA_KEY = "irodori_quantization_json"
QUANTIZATION_FORMAT_VERSION = 1
QUANTIZATION_BACKEND = "torchao"
QUANTIZATION_PROFILES = ("core", "all-linear")
QUANTIZATION_TYPE_INT8_WEIGHT_ONLY = "int8_weight_only"
QUANTIZATION_TYPE_INT8_DYNAMIC = "int8_dynamic_activation_int8_weight"
QUANTIZATION_TYPE_INT4_WEIGHT_ONLY = "int4_weight_only"
QUANTIZATION_TYPE_FLOAT8_WEIGHT_ONLY = "float8_weight_only"
QUANTIZATION_TYPE_FLOAT8_DYNAMIC = "float8_dynamic_activation_float8_weight"
INT4_GROUP_SIZES = (32, 64, 128, 256)
DEFAULT_INT4_GROUP_SIZE = 128
INT4_PACKING_FORMAT = "tile_packed_to_4d"
FLOAT8_DYNAMIC_ACTIVATION_VALUE_LB = 1e-12
QUANTIZATION_TYPES = (
    QUANTIZATION_TYPE_INT8_WEIGHT_ONLY,
    QUANTIZATION_TYPE_INT8_DYNAMIC,
    QUANTIZATION_TYPE_INT4_WEIGHT_ONLY,
    QUANTIZATION_TYPE_FLOAT8_WEIGHT_ONLY,
    QUANTIZATION_TYPE_FLOAT8_DYNAMIC,
)
QUANTIZATION_CLI_CHOICES = (
    "int8-weight-only",
    "int8-dynamic",
    "int4-weight-only",
    "float8-weight-only",
    "float8-dynamic",
)
_CLI_TO_QUANTIZATION_TYPE = dict(zip(QUANTIZATION_CLI_CHOICES, QUANTIZATION_TYPES, strict=True))
_QUANTIZATION_TYPE_TO_CLI = {value: key for key, value in _CLI_TO_QUANTIZATION_TYPE.items()}


def _require_torchao_safetensors() -> tuple[Any, Any]:
    try:
        from torchao.prototype.safetensors.safetensors_support import (
            flatten_tensor_state_dict,
            unflatten_tensor_state_dict,
        )
    except ImportError as exc:
        raise RuntimeError(
            "This checkpoint uses torchao quantization. Install the project with a "
            "supported backend extra or run `pip install torchao>=0.16,<0.17`."
        ) from exc
    return flatten_tensor_state_dict, unflatten_tensor_state_dict


def normalize_quantization_type(value: str) -> str:
    normalized = str(value).strip().lower()
    normalized = _CLI_TO_QUANTIZATION_TYPE.get(normalized, normalized)
    if normalized not in QUANTIZATION_TYPES:
        expected = ", ".join(QUANTIZATION_CLI_CHOICES)
        raise ValueError(f"Unsupported quantization type={value!r}. Expected one of: {expected}.")
    return normalized


def quantization_cli_name(value: str) -> str:
    return _QUANTIZATION_TYPE_TO_CLI[normalize_quantization_type(value)]


def _build_torchao_config(
    quantization_type: str,
    *,
    int4_group_size: int = DEFAULT_INT4_GROUP_SIZE,
) -> tuple[Any, Any]:
    try:
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            Float8WeightOnlyConfig,
            Int4WeightOnlyConfig,
            Int8DynamicActivationInt8WeightConfig,
            Int8WeightOnlyConfig,
            quantize_,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Quantization requires torchao. Install the project with a supported backend extra "
            "or run `pip install torchao>=0.16,<0.17`."
        ) from exc

    normalized = normalize_quantization_type(quantization_type)
    if normalized == QUANTIZATION_TYPE_INT4_WEIGHT_ONLY:
        if int4_group_size not in INT4_GROUP_SIZES:
            raise ValueError(
                f"Unsupported INT4 group size={int4_group_size}. "
                f"Expected one of: {', '.join(map(str, INT4_GROUP_SIZES))}."
            )
        return quantize_, Int4WeightOnlyConfig(
            group_size=int4_group_size,
            int4_packing_format=INT4_PACKING_FORMAT,
            version=2,
        )
    config_classes = {
        QUANTIZATION_TYPE_INT8_WEIGHT_ONLY: Int8WeightOnlyConfig,
        QUANTIZATION_TYPE_INT8_DYNAMIC: Int8DynamicActivationInt8WeightConfig,
        QUANTIZATION_TYPE_FLOAT8_WEIGHT_ONLY: Float8WeightOnlyConfig,
    }
    if normalized == QUANTIZATION_TYPE_FLOAT8_DYNAMIC:
        return quantize_, Float8DynamicActivationFloat8WeightConfig(
            activation_value_lb=FLOAT8_DYNAMIC_ACTIVATION_VALUE_LB,
            version=2,
        )
    return quantize_, config_classes[normalized](version=2)


def parse_quantization_metadata(metadata: Mapping[str, str]) -> dict[str, Any] | None:
    raw = metadata.get(QUANTIZATION_METADATA_KEY)
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid Irodori quantization metadata.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Irodori quantization metadata must be a JSON object.")
    if payload.get("format_version") != QUANTIZATION_FORMAT_VERSION:
        raise ValueError(
            "Unsupported Irodori quantization format_version="
            f"{payload.get('format_version')!r}."
        )
    if payload.get("backend") != QUANTIZATION_BACKEND:
        raise ValueError(f"Unsupported quantization backend={payload.get('backend')!r}.")
    quantization_type = payload.get("quantization_type")
    if quantization_type not in QUANTIZATION_TYPES:
        raise ValueError(
            f"Unsupported quantization_type={quantization_type!r}."
        )
    return payload


def is_torchao_quantized_state_dict(state_dict: Mapping[str, torch.Tensor]) -> bool:
    return any(type(tensor).__module__.startswith("torchao.") for tensor in state_dict.values())


def _core_quantization_filter(module: torch.nn.Module, fqn: str) -> bool:
    if not isinstance(module, torch.nn.Linear):
        return False
    if fqn.startswith("pretrained_text_backbone.backbone.layers."):
        return True
    transformer_prefixes = (
        "text_encoder.blocks.",
        "caption_encoder.blocks.",
        "speaker_encoder.blocks.",
        "blocks.",
    )
    if not fqn.startswith(transformer_prefixes):
        return False
    return ".attention." in fqn or ".attn." in fqn or ".mlp." in fqn


def _all_linear_filter(module: torch.nn.Module, _fqn: str) -> bool:
    return isinstance(module, torch.nn.Linear)


def quantize_model(
    model: torch.nn.Module,
    *,
    quantization_type: str = QUANTIZATION_TYPE_INT8_WEIGHT_ONLY,
    profile: str = "core",
    int4_group_size: int = DEFAULT_INT4_GROUP_SIZE,
) -> list[str]:
    normalized_profile = str(profile).strip().lower()
    if normalized_profile not in QUANTIZATION_PROFILES:
        raise ValueError(
            f"Unsupported quantization profile={profile!r}. "
            f"Expected one of: {', '.join(QUANTIZATION_PROFILES)}."
        )

    if normalized_profile == "core":
        filter_fn = _core_quantization_filter
    else:
        filter_fn = _all_linear_filter

    selected = [
        fqn
        for fqn, module in model.named_modules()
        if fqn and filter_fn(module, fqn)
    ]
    if not selected:
        raise ValueError(f"Quantization profile {normalized_profile!r} selected no modules.")

    quantize_, quantization_config = _build_torchao_config(
        quantization_type,
        int4_group_size=int4_group_size,
    )
    quantize_(
        model,
        quantization_config,
        filter_fn=filter_fn,
    )
    quantized = [
        fqn
        for fqn, module in model.named_modules()
        if fqn
        and isinstance(module, torch.nn.Linear)
        and type(module.weight).__module__.startswith("torchao.")
    ]
    if not quantized:
        raise ValueError(
            f"Quantization type {normalize_quantization_type(quantization_type)!r} "
            "did not quantize any selected modules."
        )
    return quantized


def quantize_model_int8_weight_only(
    model: torch.nn.Module,
    *,
    profile: str = "core",
) -> list[str]:
    return quantize_model(
        model,
        quantization_type=QUANTIZATION_TYPE_INT8_WEIGHT_ONLY,
        profile=profile,
    )


def flatten_quantized_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    base_metadata: Mapping[str, str],
    quantization_type: str = QUANTIZATION_TYPE_INT8_WEIGHT_ONLY,
    profile: str,
    compute_dtype: torch.dtype,
    quantized_modules: int,
    int4_group_size: int = DEFAULT_INT4_GROUP_SIZE,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    if not is_torchao_quantized_state_dict(state_dict):
        raise ValueError("State dictionary does not contain torchao quantized tensors.")
    normalized_type = normalize_quantization_type(quantization_type)
    flatten_tensor_state_dict, _ = _require_torchao_safetensors()
    flattened, torchao_metadata = flatten_tensor_state_dict(dict(state_dict))
    payload = {
        "format_version": QUANTIZATION_FORMAT_VERSION,
        "backend": QUANTIZATION_BACKEND,
        "backend_version": importlib.metadata.version("torchao"),
        "quantization_type": normalized_type,
        "compute_dtype": str(compute_dtype).removeprefix("torch."),
        "profile": str(profile),
        "quantized_modules": int(quantized_modules),
    }
    if normalized_type == QUANTIZATION_TYPE_INT4_WEIGHT_ONLY:
        if int4_group_size not in INT4_GROUP_SIZES:
            raise ValueError(
                f"Unsupported INT4 group size={int4_group_size}. "
                f"Expected one of: {', '.join(map(str, INT4_GROUP_SIZES))}."
            )
        payload["group_size"] = int4_group_size
        payload["packing_format"] = INT4_PACKING_FORMAT
    elif normalized_type == QUANTIZATION_TYPE_FLOAT8_DYNAMIC:
        payload["activation_value_lb"] = FLOAT8_DYNAMIC_ACTIVATION_VALUE_LB
    metadata = dict(base_metadata)
    metadata.update(torchao_metadata)
    metadata[QUANTIZATION_METADATA_KEY] = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return flattened, metadata


def unflatten_quantized_state_dict(
    flattened: Mapping[str, torch.Tensor],
    *,
    metadata: Mapping[str, str],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = parse_quantization_metadata(metadata)
    if payload is None:
        raise ValueError("Checkpoint has no Irodori quantization metadata.")
    _, unflatten_tensor_state_dict = _require_torchao_safetensors()
    state_dict, leftover = unflatten_tensor_state_dict(dict(flattened), dict(metadata))
    if leftover:
        raise ValueError(
            "TorchAO safetensors deserialization left unexpected tensors: "
            f"{sorted(leftover)[:8]}"
        )
    if not is_torchao_quantized_state_dict(state_dict):
        raise ValueError("Quantized checkpoint metadata did not reconstruct quantized tensors.")
    return state_dict, payload
