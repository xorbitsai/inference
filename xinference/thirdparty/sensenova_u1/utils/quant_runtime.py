"""Load v2 FP8/INT8 checkpoints; quantized linears never use floating GEMM."""

import json

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


class QuantLinear(nn.Module):
    def __init__(self, weight, scale, fmt, group=0, bias=None):
        super().__init__()
        self.format, self.group, self.calls = fmt, group, 0
        # Prepare INT8's contiguous transpose once, not on every forward.
        wt = weight.t().contiguous() if fmt == "int8-convrot" else weight.t()
        self.register_buffer("weight", wt)
        self.register_buffer("scale", scale.reshape(1, -1).float())
        self.register_buffer("bias", bias)
        self.register_buffer("one", torch.ones((), device=weight.device, dtype=torch.float32))
        if group:
            from comfy_kitchen.tensor.int8_utils import _build_hadamard

            self.register_buffer("rotation", _build_hadamard(group, weight.device, torch.bfloat16))

    def forward(self, x):
        self.calls += 1
        shape = x.shape
        if self.group:
            x = (x.reshape(-1, shape[-1] // self.group, self.group) @ self.rotation).reshape(shape)
        x = x.reshape(-1, shape[-1]).float()
        if self.format == "fp8":
            activation_scale = x.abs().amax().clamp_min(1e-8) / 448.0
            qx = (x / activation_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            # FP32 output permits per-channel scaling without BF16 overflow.
            y = torch._scaled_mm(qx, self.weight, scale_a=activation_scale, scale_b=self.one, out_dtype=torch.float32)
            y = y * self.scale
        else:
            activation_scale = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
            qx = (x / activation_scale).round().clamp(-127, 127).to(torch.int8)
            rows = qx.shape[0]
            padded = F.pad(qx, (0, 0, 0, max(0, 32 - rows)))
            y = torch._int_mm(padded, self.weight)[:rows].float()
            y = y * activation_scale * self.scale
        if self.bias is not None:
            y = y + self.bias.float()
        return y.reshape(*shape[:-1], -1).to(torch.bfloat16)


def load_quant(checkpoint, device):
    from pathlib import Path

    resources = str(Path(checkpoint).parent)
    with safe_open(checkpoint, framework="pt", device="cpu") as reader:
        spec = json.loads(reader.metadata()["quantization"])
        if spec["schema"] != "sensenova-w8a8-v2":
            raise ValueError("Expected a v2 quantized checkpoint")
        fmt, layers = spec["format"], spec["layers"]
        with init_empty_weights():
            model = AutoModel.from_config(AutoConfig.from_pretrained(resources))
        skip = set()
        keys = set(reader.keys())
        for name, cfg in layers.items():
            old = model.get_submodule(name)
            if not isinstance(old, nn.Linear):
                raise TypeError(name)
            w, s, b = name + ".weight", name + ".weight_scale", name + ".bias"
            weight, scale = reader.get_tensor(w).to(device), reader.get_tensor(s).to(device)
            expected_dtype = torch.float8_e4m3fn if fmt == "fp8" else torch.int8
            if weight.dtype != expected_dtype or scale.shape != (old.out_features, 1):
                raise ValueError(f"Invalid quantized weight/scale: {name}")
            bias = reader.get_tensor(b).to(device) if b in keys else None
            layer = QuantLinear(weight, scale, fmt, cfg.get("group_size", 0), bias)
            parent, attr = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), attr, layer)
            skip.update([w, s, b])
        for name in sorted(keys - skip):
            set_module_tensor_to_device(model, name, device, value=reader.get_tensor(name))
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        raise RuntimeError("Uninitialized model tensors")
    return model.eval(), AutoTokenizer.from_pretrained(resources)
