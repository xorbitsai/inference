"""Opt-in experimental FP8 AR linear layers; NAR always restores exact BF16.

No quantized quality or speed claim is implied by enabling this module.
FP8 kernels require NVIDIA compute capability 8.9 or newer. Original weights
remain in CPU memory, outside the registered module tree, for exact restore.
"""
from __future__ import annotations

import re
import torch
from torch import nn
from torch.nn import functional as F

AR_LINEAR = re.compile(r"model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)$")


def quantize_tensor(value):
    source = value.detach().float()
    limit = torch.finfo(torch.float8_e4m3fn).max
    scale = source.abs().amax().clamp_min(1e-12) / limit
    return (source / scale).clamp(-limit, limit).to(torch.float8_e4m3fn), scale.float().reshape(1)


class FP8Linear(nn.Module):
    """Per-tensor E4M3 weights and dynamic activations, BF16/FP16 output."""
    def __init__(self, linear, device):
        super().__init__()
        weight, scale = quantize_tensor(linear.weight)
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.register_buffer("weight", weight.to(device))
        self.register_buffer("weight_scale", scale.to(device))
        self.register_buffer("bias", linear.bias.detach().clone().to(device) if linear.bias is not None else None)

    def forward(self, value):
        if value.device.type != "cuda":
            raise RuntimeError("FP8 execution requires CUDA; restore_ar(model) before CPU/MPS use")
        if value.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("FP8 AR expects BF16/FP16 input")
        shape = value.shape
        rows = value.reshape(-1, self.in_features)
        # cuBLAS FP8 accepts aligned GEMM rows; one-token decode is padded here.
        count = rows.shape[0]
        padded = F.pad(rows, (0, 0, 0, (-count) % 16))
        activation, scale = quantize_tensor(padded)
        output = torch._scaled_mm(activation, self.weight.t(), scale_a=scale,
                                  scale_b=self.weight_scale, out_dtype=value.dtype,
                                  bias=self.bias, use_fast_accum=False)
        if isinstance(output, tuple):
            output = output[0]
        return output[:count].reshape(*shape[:-1], self.out_features)

    def dequantized_weight(self):
        """Diagnostic/reference helper; normal inference uses scaled_mm."""
        return self.weight.float() * self.weight_scale


def _replace(model, name, replacement):
    parent_name, child = name.rsplit(".", 1)
    setattr(model.get_submodule(parent_name), child, replacement)


def prepare_fp8_ar(model, device):
    """Quantize only AR projections/MLPs once, preserving CPU BF16 originals."""
    if model is None:
        raise ValueError("Load the MoT model before preparing FP8")
    device = torch.device(device)
    if getattr(model, "_yue2_fp8_originals", None):
        return quantization_status(model)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Experimental FP8 AR requires CUDA compute capability >=8.9; use quantization='none'")
    if torch.cuda.get_device_capability(device) < (8, 9):
        raise RuntimeError("Experimental FP8 AR requires CUDA compute capability >=8.9")
    selected = [(name, module) for name, module in model.named_modules() if AR_LINEAR.fullmatch(name)]
    if not selected or any(not isinstance(module, nn.Linear) for _, module in selected):
        raise ValueError("Expected unquantized YuE2 AR Linear layers")
    if any(module.weight.dtype != torch.bfloat16 for _, module in selected):
        raise ValueError("Experimental FP8 preparation requires the original BF16 AR weights")
    if any(module.in_features % 16 or module.out_features % 16 for _, module in selected):
        raise ValueError("FP8 matrix dimensions must be multiples of 16")
    originals = {}
    # A plain attribute dictionary is intentionally not an nn.ModuleDict:
    # model.to(cuda) must not move these reference weights back onto the GPU.
    object.__setattr__(model, "_yue2_fp8_originals", originals)
    try:
        for name, original in selected:
            replacement = FP8Linear(original, device)
            originals[name] = original.to("cpu")
            _replace(model, name, replacement)
    except BaseException:
        restore_ar(model)
        raise
    return quantization_status(model)


def restore_ar(model):
    """Restore exact original tensors before NAR prefill, save, or CPU use."""
    if model is None:
        return
    originals = getattr(model, "_yue2_fp8_originals", None)
    if not originals:
        return
    for name, original in originals.items():
        replacement = model.get_submodule(name)
        device = replacement.weight.device
        _replace(model, name, original.to(device))
    object.__setattr__(model, "_yue2_fp8_originals", {})


def quantization_status(model):
    originals = getattr(model, "_yue2_fp8_originals", {}) if model is not None else {}
    return {"mode": "fp8" if originals else "none", "active_ar_linears": len(originals),
            "weight_format": "float8_e4m3fn" if originals else None,
            "original_weights": "cpu_bfloat16" if originals else None,
            "quality_validation": "unvalidated", "performance_validation": "unvalidated"}
