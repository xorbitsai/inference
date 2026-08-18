"""GGUF checkpoint loader for transformers/diffusers-style models.

Public API:
    load_gguf_checkpoint(path) -> dict[str, Tensor | GGUFParameter]
    set_gguf2meta_model(meta_model, state_dict, dtype, device) -> nn.Module
    match_state_dict(meta_model, state_dict, show_num=10) -> dict  # debug helper
"""

from __future__ import annotations

import gc

import torch
from torch import nn


def load_gguf_checkpoint(gguf_checkpoint_path: str) -> dict:
    """Parse a .gguf file into a state-dict-compatible mapping.

    F32 / F16 tensors come back as plain torch tensors; everything else is
    wrapped in ``GGUFParameter`` so the diffusers quantizer can dequantize on
    the fly during forward.
    """
    from diffusers.utils import is_gguf_available, is_torch_available

    if not (is_gguf_available() and is_torch_available()):
        raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint.")

    import gguf
    from diffusers.quantizers.gguf.utils import SUPPORTED_GGUF_QUANT_TYPES, GGUFParameter
    from gguf import GGUFReader

    reader = GGUFReader(gguf_checkpoint_path)
    parsed: dict = {}
    for tensor in reader.tensors:
        name = tensor.name
        quant_type = tensor.tensor_type
        is_quant = quant_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
        if is_quant and quant_type not in SUPPORTED_GGUF_QUANT_TYPES:
            supported = "\n".join(str(t) for t in SUPPORTED_GGUF_QUANT_TYPES)
            raise ValueError(f"{name} has unsupported quant type {quant_type}.\nSupported:\n{supported}")
        weights = torch.from_numpy(tensor.data.copy())
        parsed[name] = GGUFParameter(weights, quant_type=quant_type) if is_quant else weights
        del tensor, weights
    del reader
    gc.collect()
    return parsed


def set_gguf2meta_model(
    meta_model: nn.Module,
    model_state_dict: dict,
    dtype: torch.dtype,
    device: torch.device | None,
) -> nn.Module:
    """Inject GGUF weights into a meta-initialized model.

    The model **must** have been built with ``accelerate.init_empty_weights()``
    so its parameters live on the meta device. This function:
      1. Replaces ``nn.Linear`` modules with ``GGUFLinear`` (via the quantizer hook).
      2. Loads the parsed state-dict into those modules.
      3. Returns the model cast to ``dtype`` (non-quant params only).
    """
    from diffusers import GGUFQuantizationConfig
    from diffusers.models.model_loading_utils import load_model_dict_into_meta
    from diffusers.quantizers.gguf import GGUFQuantizer

    g_config = GGUFQuantizationConfig(compute_dtype=dtype or torch.bfloat16)
    hf_quantizer = GGUFQuantizer(quantization_config=g_config)
    hf_quantizer.pre_quantized = True  # required: weights are already quantized

    device_map = {"": device} if device is not None else None
    hf_quantizer._process_model_before_weight_loading(meta_model, device_map=device_map, state_dict=model_state_dict)
    load_model_dict_into_meta(
        meta_model,
        model_state_dict,
        hf_quantizer=hf_quantizer,
        device_map=device_map,
        dtype=dtype,
    )
    hf_quantizer._process_model_after_weight_loading(meta_model)

    del model_state_dict
    gc.collect()
    return meta_model.to(dtype=dtype)


def match_state_dict(meta_model: nn.Module, sd: dict, show_num: int = 10) -> dict:
    """Debug helper: report how well a parsed state-dict matches a model.

    Returns a dict with counts/sets for programmatic checks.
    """
    model_keys = set(meta_model.state_dict().keys())
    sd_keys = set(sd.keys())
    matching = model_keys & sd_keys
    extra = sd_keys - model_keys
    missing = model_keys - sd_keys

    print(f"[gguf] matching keys: {len(matching)}")
    if extra:
        print(f"[gguf] extra in state_dict (not in model): {len(extra)}")
        for k in list(extra)[:show_num]:
            print(f"  + {k}")
    if missing:
        print(f"[gguf] missing in state_dict (in model only): {len(missing)}")
        for k in list(missing)[:show_num]:
            print(f"  - {k}")
    print(f"[gguf] sample matches: {list(matching)[:5]}")

    return {
        "matching": len(matching),
        "extra": extra,
        "missing": missing,
    }
