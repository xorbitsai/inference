# Copyright © 2024 Apple Inc.

import math
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


def _rope(pos: mx.array, dim: int, theta: float):
    scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    omega = 1.0 / (theta**scale)
    x = pos[..., None] * omega
    cosx = mx.cos(x)
    sinx = mx.sin(x)
    pe = mx.stack([cosx, -sinx, sinx, cosx], axis=-1)
    pe = pe.reshape(*pe.shape[:-1], 2, 2)

    return pe


@partial(mx.compile, shapeless=True)
def _ab_plus_cd(a, b, c, d):
    return a * b + c * d


def _apply_rope(x, pe):
    s = x.shape
    x = x.reshape(*s[:-1], -1, 1, 2)
    x = _ab_plus_cd(x[..., 0], pe[..., 0], x[..., 1], pe[..., 1])
    return x.reshape(s)


def _attention(q: mx.array, k: mx.array, v: mx.array, pe: mx.array):
    B, H, L, D = q.shape

    q = _apply_rope(q, pe)
    k = _apply_rope(k, pe)
    x = mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** (-0.5))

    return x.transpose(0, 2, 1, 3).reshape(B, L, -1)


def timestep_embedding(
    t: mx.array, dim: int, max_period: int = 10000, time_factor: float = 1000.0
):
    half = dim // 2
    freqs = mx.arange(0, half, dtype=mx.float32) / half
    freqs = freqs * (-math.log(max_period))
    freqs = mx.exp(freqs)

    x = (time_factor * t)[:, None] * freqs[None]
    x = mx.concatenate([mx.cos(x), mx.sin(x)], axis=-1)

    return x.astype(t.dtype)


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()

        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: mx.array):
        n_axes = ids.shape[-1]
        pe = mx.concatenate(
            [_rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            axis=-3,
        )

        return pe[:, None]


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out_layer(nn.silu(self.in_layer(x)))


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = nn.RMSNorm(dim)
        self.key_norm = nn.RMSNorm(dim)

    def __call__(self, q: mx.array, k: mx.array) -> tuple[mx.array, mx.array]:
        return self.query_norm(q), self.key_norm(k)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, pe: mx.array) -> mx.array:
        H = self.num_heads
        B, L, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        q, k = self.norm(q, k)
        x = _attention(q, k, v, pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: mx.array
    scale: mx.array
    gate: mx.array


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def __call__(self, x: mx.array) -> Tuple[ModulationOut, Optional[ModulationOut]]:
        x = self.lin(nn.silu(x))
        xs = mx.split(x[:, None, :], self.multiplier, axis=-1)

        mod1 = ModulationOut(*xs[:3])
        mod2 = ModulationOut(*xs[3:]) if self.is_double else None

        return mod1, mod2


class DoubleStreamBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.img_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )

        self.img_norm2 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approx="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )

        self.txt_norm2 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approx="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def __call__(
        self, img: mx.array, txt: mx.array, vec: mx.array, pe: mx.array
    ) -> Tuple[mx.array, mx.array]:
        B, L, _ = img.shape
        _, S, _ = txt.shape
        H = self.num_heads

        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = mx.split(img_qkv, 3, axis=-1)
        img_q = img_q.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        img_k = img_k.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        img_v = img_v.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        img_q, img_k = self.img_attn.norm(img_q, img_k)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = mx.split(txt_qkv, 3, axis=-1)
        txt_q = txt_q.reshape(B, S, H, -1).transpose(0, 2, 1, 3)
        txt_k = txt_k.reshape(B, S, H, -1).transpose(0, 2, 1, 3)
        txt_v = txt_v.reshape(B, S, H, -1).transpose(0, 2, 1, 3)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k)

        # run actual attention
        q = mx.concatenate([txt_q, img_q], axis=2)
        k = mx.concatenate([txt_k, img_k], axis=2)
        v = mx.concatenate([txt_v, img_v], axis=2)

        attn = _attention(q, k, v, pe)
        txt_attn, img_attn = mx.split(attn, [S], axis=1)

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(
            (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        )

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp(
            (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
        )

        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: Optional[float] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approx="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def __call__(self, x: mx.array, vec: mx.array, pe: mx.array):
        B, L, _ = x.shape
        H = self.num_heads

        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift

        q, k, v, mlp = mx.split(
            self.linear1(x_mod),
            [self.hidden_size, 2 * self.hidden_size, 3 * self.hidden_size],
            axis=-1,
        )
        q = q.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        q, k = self.norm(q, k)

        # compute attention
        y = _attention(q, k, v, pe)

        # compute activation in mlp stream, cat again and run second linear layer
        y = self.linear2(mx.concatenate([y, self.mlp_act(mlp)], axis=2))
        return x + mod.gate * y


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def __call__(self, x: mx.array, vec: mx.array):
        shift, scale = mx.split(self.adaLN_modulation(vec), 2, axis=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
