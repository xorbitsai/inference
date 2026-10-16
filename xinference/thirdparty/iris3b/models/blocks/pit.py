# Modified for Xinference: use package-relative imports.
"""Pixel-level transformer (PiT) block.

Operates on per-patch pixel sequences ``[B*L, p*p, d_pix]`` conditioned on
that patch's semantic token ``[B*L, D]`` (timestep-fused by the caller).

Three ingredients:
- pixel-wise adaLN: one linear expands the patch token into a *distinct*
  modulation tuple for every pixel of the patch;
- pixel token compaction: the patch's pixels are flattened and linearly
  compressed to a single token so global attention runs at patch
  granularity (with 2D RoPE over the patch grid), then expanded back;
- a per-pixel GELU MLP.

Two modulation layouts:
- "pre" (default): DiT-style input modulation + output gates, 6 tuples;
- "post": affine (scale, shift) on branch outputs, no gates, 4 tuples -
  used as a loss-spike mitigation. Note the chunk order is
  (scale, shift) per branch, opposite of the pre-mod naming order, and the
  block is NOT identity at zero-init.
"""

import torch
from torch import nn

from ...nn.attention import SelfAttention
from ...nn.mlp import GeluMLP
from ...nn.modulation import modulate
from ...nn.norms import RMSNorm


class PiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        cond_dim: int,
        pixels_per_patch: int,
        attn_hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        modulation: str = "pre",
        qk_norm: bool = True,
        norm_eps: float = 1e-6,
        attn_backend: str = "sdpa",
    ):
        super().__init__()
        if modulation not in ("pre", "post"):
            raise ValueError(f"unknown PiT modulation '{modulation}'")
        self.modulation = modulation
        self.pixels_per_patch = pixels_per_patch
        n_mod = 6 if modulation == "pre" else 4
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)
        self.adaln = nn.Linear(cond_dim, n_mod * hidden_size * pixels_per_patch, bias=True)
        self.compress = nn.Linear(pixels_per_patch * hidden_size, attn_hidden_size, bias=True)
        self.expand = nn.Linear(attn_hidden_size, pixels_per_patch * hidden_size, bias=True)
        self.attn = SelfAttention(
            attn_hidden_size,
            num_heads,
            qkv_bias=False,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
            backend=attn_backend,
        )
        self.mlp = GeluMLP(hidden_size, mlp_ratio)

    def _global_attn(self, pixels: torch.Tensor, rope: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        """Compact each patch to one token, attend over the patch grid, expand back."""
        n_patches = grid[0] * grid[1]
        batch = pixels.shape[0] // n_patches
        compact = self.compress(pixels.reshape(batch * n_patches, -1))
        attn = self.attn(compact.reshape(batch, n_patches, -1), rope=rope)
        return self.expand(attn.reshape(batch * n_patches, -1)).reshape_as(pixels)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        rope: torch.Tensor,
        grid: tuple[int, int],
    ) -> torch.Tensor:
        """``x``: [B*L, p*p, d_pix]; ``cond``: [B*L, cond_dim]; ``grid``: (H/p, W/p)."""
        mods = self.adaln(cond).reshape(x.shape[0], self.pixels_per_patch, -1)
        if self.modulation == "pre":
            shift1, scale1, gate1, shift2, scale2, gate2 = mods.chunk(6, dim=-1)
            h = modulate(self.norm1(x), shift1, scale1)
            x = x + gate1 * self._global_attn(h, rope, grid)
            x = x + gate2 * self.mlp(modulate(self.norm2(x), shift2, scale2))
        else:
            scale1, shift1, scale2, shift2 = mods.chunk(4, dim=-1)
            x = x + modulate(self._global_attn(self.norm1(x), rope, grid), shift1, scale1)
            x = x + modulate(self.mlp(self.norm2(x)), shift2, scale2)
        return x
