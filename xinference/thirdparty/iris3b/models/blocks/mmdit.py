# Modified for Xinference: use package-relative imports.
"""Dual-stream MM-DiT block: separate text and image weights, joint attention.

Carries the same optional attention variants as the single-stream block --
grouped-query attention, a sigmoid content gate, and sandwich RMSNorm -- so
those axes can be measured without also changing the stream topology. Each
variant is instantiated per stream, because dual-stream keeps separate weights.
"""

import torch
from torch import nn

from ...nn.attention import JointAttention
from ...nn.mlp import SwiGLU
from ...nn.modulation import ModulationBuilder, modulate
from ...nn.norms import RMSNorm
from ...registry import BLOCKS


@BLOCKS.register("mmdit")
class MMDiTBlock(nn.Module):
    """Text and image streams with separate weights, fused by joint attention.

    Each stream carries its own 6-parameter adaLN projection driven by the
    (already SiLU-activated) conditioning vector. SiLU is applied once by the
    caller, not inside the projections. ``modulation`` supplies those
    projections; the default builder returns a per-block
    ``Linear(dim, 6*dim)``, and the shared-core builders swap in a shared core
    plus a per-block residual behind the same interface.

    ``text_out=False`` builds the block without its text output path: the text
    stream is still normalized, modulated and projected into the joint
    attention, but nothing after that attention exists on the text side and the
    input text tokens are returned unchanged. Used for the final block, whose
    text outputs are discarded by the model anyway.
    """

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
        self.norm_x1 = RMSNorm(dim, eps=norm_eps)
        self.norm_x2 = RMSNorm(dim, eps=norm_eps)
        self.norm_y1 = RMSNorm(dim, eps=norm_eps)
        self.norm_y2 = RMSNorm(dim, eps=norm_eps) if text_out else None
        self.attn = JointAttention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
            backend=attn_backend,
            num_kv_heads=num_kv_heads,
            text_out=text_out,
        )
        self.attn_gate_x = nn.Linear(dim, dim, bias=False) if gated_attention else None
        self.attn_gate_y = nn.Linear(dim, dim, bias=False) if gated_attention and text_out else None
        self.attn_post_norm_x = RMSNorm(dim, eps=norm_eps) if sandwich_norm else nn.Identity()
        self.attn_post_norm_y = (
            (RMSNorm(dim, eps=norm_eps) if sandwich_norm else nn.Identity()) if text_out else None
        )
        self.mlp_x = SwiGLU(dim, mlp_ratio)
        self.mlp_y = SwiGLU(dim, mlp_ratio) if text_out else None
        self.mlp_post_norm_x = RMSNorm(dim, eps=norm_eps) if sandwich_norm else nn.Identity()
        self.mlp_post_norm_y = (
            (RMSNorm(dim, eps=norm_eps) if sandwich_norm else nn.Identity()) if text_out else None
        )
        build = modulation or ModulationBuilder(dim)
        self.adaln_img = build("img")
        self.adaln_txt = build("txt")
        self.text_out = text_out

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        cond: torch.Tensor,
        rope_img: torch.Tensor,
        rope_txt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``x``: [B, L, D] image tokens; ``y``: [B, T, D] text tokens.

        ``cond``: [B, 1, D] conditioning shared by both streams. The returned
        text stream is the untouched input when the block was built with
        ``text_out=False``.
        """
        (xs1, xc1, xg1, xs2, xc2, xg2) = self.adaln_img(cond).chunk(6, dim=-1)
        (ys1, yc1, yg1, ys2, yc2, yg2) = self.adaln_txt(cond).chunk(6, dim=-1)
        hx = modulate(self.norm_x1(x), xs1, xc1)
        hy = modulate(self.norm_y1(y), ys1, yc1)
        attn_x, attn_y = self.attn(
            hx,
            hy,
            rope_img,
            rope_txt,
            gate_x=None if self.attn_gate_x is None else torch.sigmoid(self.attn_gate_x(hx)),
            gate_y=None if self.attn_gate_y is None else torch.sigmoid(self.attn_gate_y(hy)),
        )
        x = x + xg1 * self.attn_post_norm_x(attn_x)
        x = x + xg2 * self.mlp_post_norm_x(self.mlp_x(modulate(self.norm_x2(x), xs2, xc2)))
        if not self.text_out:
            return x, y
        y = y + yg1 * self.attn_post_norm_y(attn_y)
        y = y + yg2 * self.mlp_post_norm_y(self.mlp_y(modulate(self.norm_y2(y), ys2, yc2)))
        return x, y
