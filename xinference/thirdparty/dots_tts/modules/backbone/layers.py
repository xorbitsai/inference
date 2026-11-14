import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


def _block_mask_flex_attention(q, k, v, block_mask):
    backend = os.environ.get("DOTS_TTS_FLEX_ATTENTION_BACKEND")
    kernel_options = None if not backend else {"BACKEND": backend.upper()}
    return flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
        kernel_options=kernel_options,
    )


_compiled_block_mask_flex_attention = torch.compile(
    _block_mask_flex_attention,
    fullgraph=True,
    dynamic=False,
    mode="max-autotune-no-cudagraphs",
)

_compiled_create_block_mask = torch.compile(
    create_block_mask,
    fullgraph=True,
    dynamic=False,
)


class Dropout(nn.Module):
    def __init__(
        self, p: float = 0.5, inplace: bool = False, force_drop: bool = False, **_kwargs
    ):
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError(
                f"dropout probability has to be between 0 and 1, but got {p}"
            )
        self.p = p
        self.inplace = inplace
        self.force_drop = force_drop

    def forward(self, x, **_kwargs):
        return F.dropout(
            x,
            p=self.p,
            training=True if self.force_drop else self.training,
            inplace=self.inplace,
        )


class Conv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        bias: bool = True,
        padding=None,
        causal: bool = False,
        **_kwargs,
    ):
        self.causal = causal
        if padding is None:
            if causal:
                padding = 0
                self.left_padding = dilation * (kernel_size - 1)
            else:
                padding = int((kernel_size * dilation - dilation) / 2)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias,
        )

        self.in_channels = in_channels

    def forward(self, x):
        if self.causal:
            x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)
        return super().forward(x)


class ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding=None,
        padding_mode: str = "zeros",
        causal: bool = False,
        **_kwargs,
    ):
        if padding is None:
            padding = 0 if causal else (kernel_size - stride) // 2
        if causal:
            assert padding == 0, "padding is not allowed in causal ConvTranspose1d."
            assert kernel_size == 2 * stride, (
                "kernel_size must be equal to 2*stride in Causal ConvTranspose1d."
            )

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
        )

        self.causal = causal
        self.stride = stride

    def forward(self, x):
        x = super().forward(x)
        if self.causal:
            x = x[:, :, : -self.stride]
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        hidden_size,
        ffn_hidden_size=4096,
        act_layer=nn.GELU,
        dropout=0.0,
        **_kwargs,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size)
        self.act = act_layer()
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size)
        self.drop = Dropout(dropout)

    def forward(self, x, _mask=None, **_kwargs):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.autocast(enabled=False, device_type="cuda")
def apply_rotary_pos_emb(pos, t):
    if pos.dim() == 3:
        pos = pos.unsqueeze(1)
    input_dtype = t.dtype
    return (t * pos.cos() + rotate_half(t) * pos.sin()).to(dtype=input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=50000):
        super().__init__()
        self.dim = int(dim)
        self._theta = float(theta)
        self.register_buffer(
            "inv_freq",
            self._build_inv_freq(device=torch.device("cpu")),
            persistent=False,
        )

    def _build_inv_freq(self, device: torch.device) -> torch.Tensor:
        return 1.0 / (
            self._theta
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _apply(self, fn):
        super()._apply(fn)
        self.inv_freq = self._build_inv_freq(device=self.inv_freq.device)
        return self

    @torch.autocast(enabled=False, device_type="cuda")
    def forward(self, t):
        inv_freq = self.inv_freq
        if inv_freq.device != t.device or inv_freq.dtype != torch.float32:
            self.inv_freq = self._build_inv_freq(device=t.device)
            inv_freq = self.inv_freq
        t = t.to(dtype=inv_freq.dtype)
        if t.dim() == 1:
            freqs = torch.einsum("i , j -> i j", t, inv_freq)
        else:
            freqs = torch.einsum("bi, j -> bij", t, inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        dropout: float = 0.0,
        norm_layer: str = "LayerNorm",
        rotary_bias: bool = False,
        rotary_theta: float | None = 50000,
        attn_backend: str = "sdpa",
        **_kwargs,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, (
            "hidden_size should be divisible by num_heads"
        )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.rotary_bias = rotary_bias
        if attn_backend not in {"sdpa", "flex"}:
            raise ValueError(f"Unsupported attention backend: {attn_backend!r}.")
        self.attn_backend = attn_backend

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        norm_layer = getattr(nn, norm_layer)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = Dropout(attn_drop)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.o_dropout = Dropout(dropout)

        if self.rotary_bias:
            self.rotary = RotaryEmbedding(self.head_dim, theta=rotary_theta)

    def forward(self, q, k=None, v=None, mask=None, pos_ids=None, **_kwargs):
        k = k or q
        v = v or q
        B, L, _ = q.shape
        _, S, _ = v.shape
        if mask is not None:
            if mask.ndim == 2:  # [B, L]
                assert L == S
                mask = rearrange(mask, "b j -> b 1 1 j")
                mask = mask.expand(-1, self.num_heads, L, -1)
            elif mask.ndim == 3:  # [B, L, S]
                assert mask.size(1) == L and mask.size(2) == S
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        q, k = self.q_norm(q), self.k_norm(k)

        # Apply rotary
        if self.rotary_bias:
            if L == S:
                if pos_ids is None:
                    rotary_emb = self.rotary(torch.arange(L, device=q.device))
                else:
                    rotary_emb = self.rotary(pos_ids)
                q, k = (apply_rotary_pos_emb(rotary_emb, tensor) for tensor in (q, k))
            else:
                q_rotary_emb = self.rotary(torch.arange(L, device=q.device))
                k_rotary_emb = self.rotary(torch.arange(S, device=k.device))
                q = apply_rotary_pos_emb(q_rotary_emb, q)
                k = apply_rotary_pos_emb(k_rotary_emb, k)

        attn_bias = torch.zeros(B, self.num_heads, L, S, dtype=q.dtype, device=q.device)

        if mask is not None:
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.o_dropout(self.o_proj(out))
