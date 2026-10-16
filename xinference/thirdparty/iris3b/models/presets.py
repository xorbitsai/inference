# Modified for Xinference: use package-relative imports.
"""Named model presets.

"iris-3b" is the Iris-3B architecture and equals the ``ModelConfig``
defaults: a hybrid trunk of 8 dual-stream MM-DiT blocks followed by 16
single-stream blocks (width 2560, 20 query / 5 KV heads, sigmoid attention
gate, sandwich RMSNorm, shared-bias adaLN with adaLN-zero init), a
layerwise-attention text adapter over twelve Qwen3-VL-4B layers, and a 4-block
post-modulation PiT pixel stage at patch size 16.
"""

from dataclasses import replace

from ..config import ModelConfig

PRESETS: dict[str, ModelConfig] = {"iris-3b": ModelConfig()}


def get_preset(name: str, **overrides) -> ModelConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset '{name}' (known: {sorted(PRESETS)})")
    return replace(PRESETS[name], **overrides)
