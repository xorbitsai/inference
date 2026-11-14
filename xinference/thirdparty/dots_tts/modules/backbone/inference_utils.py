from __future__ import annotations

from functools import partial
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules.backbone.layers import (
    _compiled_block_mask_flex_attention,
    rotate_half,
)

_COMPILE_OPTIONS = {
    "mode": "reduce-overhead",
    "fullgraph": True,
    "dynamic": False,
}


def _compile(raw: Callable[..., Any]) -> Callable[..., Any]:
    return torch.compile(raw, **_COMPILE_OPTIONS)


def compile_module_forward(module: nn.Module) -> Callable[..., Any]:
    return partial(_compile(getattr(type(module), "forward")), module)


def compile_bound_method(
    owner: Any,
    method_name: str,
    *,
    unwrap: bool = False,
) -> Callable[..., Any]:
    raw = getattr(type(owner), method_name)
    if unwrap:
        raw = getattr(raw, "__wrapped__", raw)
    return partial(_compile(raw), owner)


def build_fused_qkv_projection(attn: nn.Module) -> nn.Linear:
    hidden_size = int(attn.num_heads) * int(attn.head_dim)
    has_bias = attn.q_proj.bias is not None
    projs = (attn.q_proj, attn.k_proj, attn.v_proj)
    qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=has_bias)
    qkv_proj.to(device=attn.q_proj.weight.device, dtype=attn.q_proj.weight.dtype)
    with torch.no_grad():
        qkv_proj.weight.copy_(torch.cat([proj.weight for proj in projs], dim=0))
        if has_bias:
            qkv_proj.bias.copy_(torch.cat([proj.bias for proj in projs], dim=0))
    return qkv_proj


def fuse_qkv_projection(attn: nn.Module) -> None:
    attn.qkv_proj = build_fused_qkv_projection(attn)


def project_attention(
    attn: nn.Module,
    x: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    rotary_cos: torch.Tensor | None = None,
    rotary_sin: torch.Tensor | None = None,
    key_prefix: torch.Tensor | None = None,
    value_prefix: torch.Tensor | None = None,
    cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    cache_positions: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    block_mask: Any | None = None,
    dropout_p: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq_len, _hidden = x.shape
    num_heads = int(num_heads)
    head_dim = int(head_dim)
    qkv = attn.qkv_proj(x).view(batch, seq_len, 3, num_heads, head_dim)
    q, k, v = qkv.permute(2, 0, 3, 1, 4)

    q = attn.q_norm(q)
    k = attn.k_norm(k)
    if rotary_cos is not None:
        cos, sin = rotary_cos, rotary_sin
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif cos.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        q = (q * cos + rotate_half(q) * sin).to(dtype=q.dtype)
        k = (k * cos + rotate_half(k) * sin).to(dtype=k.dtype)

    if cache is not None:
        if cache_positions is None:
            raise ValueError("cache_positions is required when cache is provided.")
        key, value = cache
        start = int(cache_positions[0].item())
        end = start + int(cache_positions.numel())
        expected = torch.arange(
            start,
            end,
            device=cache_positions.device,
            dtype=cache_positions.dtype,
        )
        if bool(torch.equal(cache_positions, expected)):
            for target, source in ((key, k), (value, v)):
                target[:, :, start:end, :].copy_(source)
        else:
            for target, source in ((key, k), (value, v)):
                target.index_copy_(2, cache_positions, source)
    elif key_prefix is not None or value_prefix is not None:
        if key_prefix is None or value_prefix is None:
            raise ValueError("key_prefix and value_prefix must be provided together.")
        key = torch.cat([key_prefix, k], dim=2)
        value = torch.cat([value_prefix, v], dim=2)
    else:
        key, value = k, v

    if block_mask is not None:
        out = _compiled_block_mask_flex_attention(q, key, value, block_mask)
    else:
        if dropout_p is None:
            dropout_p = attn.attn_drop.p if attn.training else 0.0
        out = F.scaled_dot_product_attention(
            q,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=float(dropout_p),
            is_causal=bool(is_causal),
        )

    out = out.permute(0, 2, 1, 3).reshape(batch, seq_len, -1)
    return attn.o_dropout(attn.o_proj(out)), k, v


def build_rotary_cos_sin(
    rotary: Callable[[torch.Tensor], Any],
    *,
    start_pos: int,
    seq_len: int,
    device: torch.device,
    batched: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.tensor(
        int(start_pos), device=device, dtype=torch.float32
    ) + torch.arange(
        int(seq_len),
        device=device,
        dtype=torch.float32,
    )
    if batched:
        positions = positions.reshape(1, int(seq_len))
    rotary_embedding = rotary(positions)
    return rotary_embedding.cos(), rotary_embedding.sin()


def build_causal_update_mask(
    *,
    capacity_tokens: int,
    valid_persistent_tokens: int,
    prev_len: int,
    current_len: int,
    device: torch.device,
) -> torch.Tensor:
    q_len = int(prev_len) + int(current_len)
    kv_idx = torch.arange(int(capacity_tokens) + q_len, device=device)
    q_idx = torch.arange(q_len, device=device).reshape(q_len, 1)
    tail_idx = kv_idx.reshape(1, int(capacity_tokens) + q_len) - int(capacity_tokens)
    past = (kv_idx < int(valid_persistent_tokens)).reshape(1, -1)
    prev_query = q_idx < int(prev_len)
    prev_causal = (tail_idx >= 0) & (tail_idx < int(prev_len)) & (tail_idx <= q_idx)
    current_full = tail_idx >= 0
    return (past | torch.where(prev_query, prev_causal, current_full)).reshape(
        1,
        1,
        q_len,
        int(capacity_tokens) + q_len,
    )
