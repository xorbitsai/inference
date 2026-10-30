# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Apply a declared YaRN recipe to a loaded MLX language model."""

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional


def configured_yarn_parameters(
    model_path: str, overrides: Dict[str, Any], native_context: Optional[int]
) -> Optional[Dict[str, Any]]:
    """Honor explicit/checkpoint YaRN even when no automatic override is needed."""
    try:
        checkpoint = json.loads((Path(model_path) / "config.json").read_text())
    except (OSError, ValueError):
        checkpoint = {}
    text_config = checkpoint.get("text_config", checkpoint)
    candidate = None
    found = False
    for config in (overrides, checkpoint):
        for language in (config.get("text_config"), config):
            if not isinstance(language, dict):
                continue
            for key in ("rope_parameters", "rope_scaling"):
                if key in language:
                    candidate = language[key]
                    found = True
                    break
            if found:
                break
        if found:
            break
    if (
        not isinstance(candidate, dict)
        or candidate.get("rope_type", candidate.get("type")) != "yarn"
    ):
        return None
    result = dict(candidate)
    result.setdefault(
        "original_max_position_embeddings",
        text_config.get("max_position_embeddings") or native_context,
    )
    result.setdefault(
        "rope_theta",
        text_config.get(
            "rope_theta",
            (text_config.get("rope_parameters") or {}).get("rope_theta", 10000.0),
        ),
    )
    if (
        type(result.get("factor")) not in (int, float)
        or not math.isfinite(result["factor"])
        or result["factor"] < 1
    ):
        raise ValueError("MLX YaRN factor must be a number greater than or equal to 1")
    original = result["original_max_position_embeddings"]
    if type(original) is not int or original <= 0:
        raise ValueError("MLX YaRN original window must be a positive integer")
    return result


class _YarnMRoPE:
    """Retain the model's multimodal position layout and scale its frequencies."""

    def __init__(self, rotary: Any, yarn: Any):
        import mlx.core as mx

        self._rotary = rotary
        self._mscale = yarn.mscale
        if hasattr(rotary, "attention_scaling"):
            rotary.attention_scaling = 1.0
        frequencies = 1.0 / yarn._freqs
        descriptor = getattr(type(rotary), "inv_freq", None)
        if isinstance(descriptor, property) and descriptor.fset is None:
            rotary._inv_freq = frequencies
        else:
            rotary.inv_freq = frequencies
        mx.eval(frequencies)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._rotary, name)

    def __call__(self, x: Any, position_ids: Any) -> Any:
        cos, sin = self._rotary(x, position_ids)
        return cos * self._mscale, sin * self._mscale

    def apply_rotary(
        self,
        q: Any,
        k: Any,
        position_ids: Any,
        *,
        unsqueeze_dim: int = 1,
        cast_output: bool = True,
    ) -> Any:
        # New mlx-vlm versions fuse rotation in apply_rotary. Use the scaled
        # cos/sin path here as well, including the attention magnitude correction.
        import mlx.core as mx

        cos, sin = self(k if cast_output else k.astype(mx.float32), position_ids)
        cos, sin = mx.expand_dims(cos, unsqueeze_dim), mx.expand_dims(
            sin, unsqueeze_dim
        )
        dims = cos.shape[-1]

        def rotate(x):
            prefix, tail = x[..., :dims], x[..., dims:]
            half = dims // 2
            rotated = mx.concatenate([-prefix[..., half:], prefix[..., :half]], axis=-1)
            return mx.concatenate([prefix * cos + rotated * sin, tail], axis=-1)

        return rotate(q), rotate(k)


def apply_auto_yarn(model: Any, parameters: Dict[str, Any]) -> None:
    """Change model instances only; never patch upstream classes or checkpoints."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.rope_utils import YarnRoPE

    language_model = getattr(model, "language_model", model)
    count = 0
    for layer in list(language_model.modules()):
        for name in ("rope", "rotary_emb"):
            rotary = getattr(layer, name, None)
            if rotary is None:
                continue
            if isinstance(rotary, _YarnMRoPE) or getattr(
                rotary, "_xinference_auto_yarn", False
            ):
                count += 1
                continue
            dims = getattr(rotary, "dims", getattr(rotary, "dim", None))
            if not isinstance(dims, int) or dims <= 0:
                continue
            is_mrope = hasattr(rotary, "mrope_section") and hasattr(rotary, "inv_freq")
            # Avoid replacing unrelated/custom rotary implementations silently.
            if not is_mrope and not isinstance(rotary, nn.RoPE):
                continue
            original = parameters["original_max_position_embeddings"]
            yarn = YarnRoPE(
                dims=dims,
                traditional=getattr(rotary, "traditional", False),
                base=parameters.get("rope_theta", getattr(rotary, "base", 10000.0)),
                scaling_factor=parameters["factor"],
                max_position_embeddings=int(original * parameters["factor"]),
                original_max_position_embeddings=original,
                **{
                    key: parameters[key]
                    for key in ("beta_fast", "beta_slow", "mscale", "mscale_all_dim")
                    if key in parameters
                },
            )
            if "attention_factor" in parameters:
                yarn.mscale = parameters["attention_factor"]
            mx.eval(yarn._freqs)
            yarn._xinference_auto_yarn = True
            setattr(layer, name, _YarnMRoPE(rotary, yarn) if is_mrope else yarn)
            count += 1
    if not count:
        raise ValueError("Automatic YaRN found no compatible MLX RoPE layers")
    # Cache contracts must describe the effective position encoding.
    args = getattr(language_model, "args", None)
    args = getattr(args, "text_config", args)
    for field in ("rope_parameters", "rope_scaling"):
        if args is not None and hasattr(args, field):
            setattr(args, field, dict(parameters))
