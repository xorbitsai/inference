# Modified for Xinference: use package-relative imports.
"""Attention layers with a pluggable kernel backend."""

from functools import cache
from importlib import import_module

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .norms import RMSNorm
from .rope import apply_rope

ATTENTION_BACKENDS = ("sdpa", "torch_flash", "torch_cudnn", "fa3", "fa4")

_EXTERNAL_MODULES = {
    "fa3": "flash_attn_3.flash_attn_interface",
    "fa4": "flash_attn.cute",
}


@cache
def _external_attention_func(backend: str):
    module_name = _EXTERNAL_MODULES[backend]
    try:
        return import_module(module_name).flash_attn_func
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"attention backend '{backend}' is unavailable: cannot import {module_name}"
        ) from exc


def _validate_external_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    backend: str,
    attn_mask: torch.Tensor | None,
    enable_gqa: bool,
) -> None:
    if attn_mask is not None:
        raise ValueError(f"attention backend '{backend}' does not support arbitrary attention masks")
    if not (q.dtype == k.dtype == v.dtype and q.dtype in (torch.float16, torch.bfloat16)):
        raise ValueError(f"attention backend '{backend}' requires matching float16 or bfloat16 Q/K/V")
    if k.shape[-3] != v.shape[-3]:
        raise ValueError("K and V must have the same number of heads")
    if enable_gqa:
        if q.shape[-3] % k.shape[-3]:
            raise ValueError("GQA requires the number of Q heads to be divisible by the number of KV heads")
    elif q.shape[-3] != k.shape[-3]:
        raise ValueError("unequal Q/KV heads require enable_gqa=True")
    if not (q.device == k.device == v.device and q.device.type == "cuda"):
        raise ValueError(f"attention backend '{backend}' requires Q/K/V on one CUDA device")
    major, _ = torch.cuda.get_device_capability(q.device)
    if backend == "fa3" and major != 9:
        raise ValueError("attention backend 'fa3' requires a Hopper (SM90) GPU")
    if backend == "fa4" and major not in (9, 10):
        raise ValueError("attention backend 'fa4' requires a Hopper or Blackwell GPU")


def scaled_dot_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    backend: str = "sdpa",
    attn_mask: torch.Tensor | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Attention over ``[B, heads, N, head_dim]`` tensors."""
    if backend not in ATTENTION_BACKENDS:
        raise ValueError(f"unknown attention backend '{backend}'")
    if backend == "sdpa":
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa
        )
    if backend in ("torch_flash", "torch_cudnn"):
        if attn_mask is not None:
            # both kernels are selected explicitly here, and neither consumes an
            # arbitrary mask: flash is disqualified by check_for_attn_mask and
            # the call silently degrades or errors deep in dispatch. Refuse it
            # loudly so a future packing mask cannot produce quiet garbage.
            raise ValueError(f"attention backend '{backend}' does not support arbitrary attention masks")
        selected = (
            SDPBackend.FLASH_ATTENTION
            if backend == "torch_flash"
            else SDPBackend.CUDNN_ATTENTION
        )
        with sdpa_kernel(selected):
            return F.scaled_dot_product_attention(q, k, v, enable_gqa=enable_gqa)

    _validate_external_attention(q, k, v, backend, attn_mask, enable_gqa)
    func = _external_attention_func(backend)
    kwargs = {"pack_gqa": True} if enable_gqa else {}
    out = func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), **kwargs)
    if backend == "fa4":
        out, _ = out
    return out.transpose(1, 2)


class SelfAttention(nn.Module):
    """Multi-head self-attention with per-head QK RMSNorm and RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        norm_eps: float = 1e-6,
        backend: str = "sdpa",
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.backend = backend
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``attn_mask``: optional bool mask broadcastable to [B, heads, N, N],
        True = attend. Default None keeps the unmasked (and flash-eligible) path."""
        batch, n, dim = x.shape
        q, k, v = self.qkv(x).reshape(batch, n, 3, self.num_heads, self.head_dim).unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = apply_rope(q, k, rope)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = scaled_dot_product(q, k, v, backend=self.backend, attn_mask=attn_mask)
        return self.proj(out.transpose(1, 2).reshape(batch, n, dim))


class JointAttention(nn.Module):
    """MM-DiT joint attention: separate stream projections, one attention op.

    Text and image keep their own QKV / output projections and QK norms;
    sequences are concatenated text-first for a single softmax over both.
    Optional grouped-query attention shrinks both streams' KV projections,
    which unfuses QKV into separate Q/K/V linears exactly as the single-stream
    block does. Per-stream sigmoid gates are supplied by the caller and applied
    to the attention output before the output projection, so the gate weights
    stay owned by the block. ``text_out=False`` drops the text output
    projection: text still supplies queries, keys and values to the joint
    softmax, but only the image half leaves the layer.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        norm_eps: float = 1e-6,
        backend: str = "sdpa",
        num_kv_heads: int | None = None,
        text_out: bool = True,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by heads {num_heads}")
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if num_kv_heads <= 0 or num_heads % num_kv_heads:
            raise ValueError(f"query heads {num_heads} must be divisible by positive KV heads {num_kv_heads}")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.backend = backend
        if num_kv_heads == num_heads:
            self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.qkv_y = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.q_proj_x = self.k_proj_x = self.v_proj_x = None
            self.q_proj_y = self.k_proj_y = self.v_proj_y = None
        else:
            kv_dim = num_kv_heads * self.head_dim
            self.qkv_x = self.qkv_y = None
            self.q_proj_x = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj_x = nn.Linear(dim, kv_dim, bias=qkv_bias)
            self.v_proj_x = nn.Linear(dim, kv_dim, bias=qkv_bias)
            self.q_proj_y = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj_y = nn.Linear(dim, kv_dim, bias=qkv_bias)
            self.v_proj_y = nn.Linear(dim, kv_dim, bias=qkv_bias)
        self.q_norm_x = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.k_norm_x = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.q_norm_y = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.k_norm_y = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.proj_x = nn.Linear(dim, dim, bias=True)
        self.proj_y = nn.Linear(dim, dim, bias=True) if text_out else None

    def _project(self, tokens: torch.Tensor, stream: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, n_tokens, _ = tokens.shape
        fused = self.qkv_x if stream == "x" else self.qkv_y
        if fused is not None:
            return fused(tokens).reshape(batch, n_tokens, 3, self.num_heads, self.head_dim).unbind(2)
        if stream == "x":
            q_proj, k_proj, v_proj = self.q_proj_x, self.k_proj_x, self.v_proj_x
        else:
            q_proj, k_proj, v_proj = self.q_proj_y, self.k_proj_y, self.v_proj_y
        q = q_proj(tokens).reshape(batch, n_tokens, self.num_heads, self.head_dim)
        k = k_proj(tokens).reshape(batch, n_tokens, self.num_kv_heads, self.head_dim)
        v = v_proj(tokens).reshape(batch, n_tokens, self.num_kv_heads, self.head_dim)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        rope_img: torch.Tensor,
        rope_txt: torch.Tensor | None = None,
        gate_x: torch.Tensor | None = None,
        gate_y: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``gate_x``/``gate_y``: multiplicative [B, N, D] gates already squashed
        by the caller, applied to each stream's attention output before its
        output projection. The text output is ``None`` under ``text_out=False``,
        which also rejects a ``gate_y`` it could not apply."""
        batch, n_img, dim = x.shape
        n_txt = y.shape[1]
        qx, kx, vx = self._project(x, "x")
        qy, ky, vy = self._project(y, "y")
        qx, kx = self.q_norm_x(qx), self.k_norm_x(kx)
        qy, ky = self.q_norm_y(qy), self.k_norm_y(ky)
        qx, kx = apply_rope(qx, kx, rope_img)
        if rope_txt is not None:
            qy, ky = apply_rope(qy, ky, rope_txt)
        q = torch.cat([qy, qx], dim=1).transpose(1, 2)
        k = torch.cat([ky, kx], dim=1).transpose(1, 2)
        v = torch.cat([vy, vx], dim=1).transpose(1, 2)
        out = scaled_dot_product(
            q, k, v, backend=self.backend, enable_gqa=self.num_kv_heads != self.num_heads
        )
        out = out.transpose(1, 2).reshape(batch, n_txt + n_img, dim)
        out_x = out[:, n_txt:]
        if gate_x is not None:
            out_x = out_x * gate_x
        if self.proj_y is None:
            if gate_y is not None:
                raise ValueError("gate_y was passed to a JointAttention built with text_out=False")
            return self.proj_x(out_x), None
        out_y = out[:, :n_txt]
        if gate_y is not None:
            out_y = out_y * gate_y
        return self.proj_x(out_x), self.proj_y(out_y)
