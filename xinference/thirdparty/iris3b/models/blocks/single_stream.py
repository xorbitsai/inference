# Modified for Xinference: use package-relative imports.
"""Single-stream block: text and image share one set of weights.

Text and image tokens are concatenated and share attention, MLP, norms, and
adaLN weights. Optional grouped-query attention reduces KV projections,
sigmoid gating controls the content-dependent attention branch, and sandwich
RMSNorm normalizes each branch before it enters the residual stream.
"""

import torch
from torch import nn

from ...nn.attention import scaled_dot_product
from ...nn.mlp import SwiGLU
from ...nn.modulation import ModulationBuilder, modulate
from ...nn.norms import RMSNorm
from ...nn.rope import apply_rope
from ...registry import BLOCKS


@BLOCKS.register("single_stream")
class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        norm_eps: float = 1e-6,
        attn_backend: str = "sdpa",
        modulation: ModulationBuilder | None = None,
        num_kv_heads: int | None = None,
        gated_attention: bool = False,
        sandwich_norm: bool = False,
        text_out: bool = True,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by query heads {num_heads}")
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if num_kv_heads <= 0 or num_heads % num_kv_heads:
            raise ValueError(f"query heads {num_heads} must be divisible by positive KV heads {num_kv_heads}")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.backend = attn_backend
        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)
        if num_kv_heads == num_heads:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.q_proj = self.k_proj = self.v_proj = None
        else:
            kv_dim = num_kv_heads * self.head_dim
            self.qkv = None
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, kv_dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, kv_dim, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.attn_gate = nn.Linear(dim, dim, bias=False) if gated_attention else None
        self.attn_proj = nn.Linear(dim, dim, bias=True)
        self.attn_post_norm = RMSNorm(dim, eps=norm_eps) if sandwich_norm else nn.Identity()
        self.mlp = SwiGLU(dim, mlp_ratio)
        self.mlp_post_norm = RMSNorm(dim, eps=norm_eps) if sandwich_norm else nn.Identity()
        self.adaln = (modulation or ModulationBuilder(dim))("shared")
        # text and image share every weight here, so dropping the text output
        # removes no parameters; it only skips the shared output projection and
        # MLP over the text rows, whose results the model discards anyway.
        self.text_out = text_out

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        cond: torch.Tensor,
        rope_img: torch.Tensor,
        rope_txt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``cond``: [B, 1, D]. The returned text stream is the untouched input
        when ``text_out`` is False."""
        batch, n_img, dim = x.shape
        n_txt = y.shape[1]
        (s1, c1, g1, s2, c2, g2) = self.adaln(cond).chunk(6, dim=-1)

        tokens = torch.cat([y, x], dim=1)
        h = modulate(self.norm1(tokens), s1, c1)
        n_tokens = n_txt + n_img
        if self.qkv is not None:
            q, k, v = self.qkv(h).reshape(batch, n_tokens, 3, self.num_heads, self.head_dim).unbind(2)
        else:
            q = self.q_proj(h).reshape(batch, n_tokens, self.num_heads, self.head_dim)
            k = self.k_proj(h).reshape(batch, n_tokens, self.num_kv_heads, self.head_dim)
            v = self.v_proj(h).reshape(batch, n_tokens, self.num_kv_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        qx, kx = apply_rope(q[:, n_txt:], k[:, n_txt:], rope_img)
        qy, ky = q[:, :n_txt], k[:, :n_txt]
        if rope_txt is not None:
            qy, ky = apply_rope(qy, ky, rope_txt)
        q = torch.cat([qy, qx], dim=1).transpose(1, 2)
        k = torch.cat([ky, kx], dim=1).transpose(1, 2)
        v = v.transpose(1, 2)
        attn = scaled_dot_product(
            q,
            k,
            v,
            backend=self.backend,
            enable_gqa=self.num_kv_heads != self.num_heads,
        )
        attn = attn.transpose(1, 2).reshape(batch, n_tokens, dim)
        if not self.text_out:
            # text still supplied keys and values to the attention above; from
            # here on only the image rows are computed
            attn, h, tokens = attn[:, n_txt:], h[:, n_txt:], x
        if self.attn_gate is not None:
            attn = attn * torch.sigmoid(self.attn_gate(h))
        attn = self.attn_post_norm(self.attn_proj(attn))
        tokens = tokens + g1 * attn
        mlp = self.mlp(modulate(self.norm2(tokens), s2, c2))
        tokens = tokens + g2 * self.mlp_post_norm(mlp)
        if not self.text_out:
            return tokens, y
        return tokens[:, n_txt:], tokens[:, :n_txt]
