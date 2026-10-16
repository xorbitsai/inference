# Modified for Xinference: use package-relative imports.
"""Input embedders: timestep, patch tokens, text tokens, pixel tokens."""

import math

import torch
from torch import nn

from .attention import SelfAttention
from .mlp import SwiGLU
from .norms import RMSNorm


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep features + 2-layer SiLU MLP.

    ``max_period`` defaults to 10 (not the classic 10000): model time is the
    shifted flow level scaled to [0, 1000], and the small period keeps the
    sinusoid bank resolving that range.
    """

    def __init__(self, hidden_size: int, freq_dim: int = 256, max_period: float = 10.0):
        super().__init__()
        if freq_dim % 2:
            raise ValueError(f"freq_dim must be even, got {freq_dim}")
        self.freq_dim = freq_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        n = self.freq_dim // 2
        k = torch.arange(n, dtype=torch.float32, device=t.device)
        phase = torch.outer(t.float(), torch.exp(-math.log(self.max_period) * k / n))
        return torch.cat([phase.cos(), phase.sin()], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """``t``: [B] (or [B*k]) in model-time units -> [B, 1, hidden]."""
        batch = t.shape[0]
        emb = self.timestep_embedding(t.reshape(-1))
        emb = self.mlp(emb.to(next(self.mlp.parameters()).dtype))
        return emb.reshape(batch, -1, emb.shape[-1])


class PatchEmbedder(nn.Module):
    """Linear projection of unfolded patch vectors, with optional norm."""

    def __init__(self, in_dim: int, hidden_size: int, norm: bool = False, norm_eps: float = 1e-6):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_size, bias=True)
        self.norm = RMSNorm(hidden_size, eps=norm_eps) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class TextEmbedder(nn.Module):
    """Project frozen text-encoder states into model width: Linear + RMSNorm."""

    def __init__(self, text_dim: int, hidden_size: int, norm_eps: float = 1e-6):
        super().__init__()
        self.proj = nn.Linear(text_dim, hidden_size, bias=True)
        self.norm = RMSNorm(hidden_size, eps=norm_eps)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(y))


class TextAdapterBlock(nn.Module):
    """Unconditioned pre-norm transformer block: ``x + attn(norm(x))`` then
    ``x + swiglu(norm(x))``.

    No adaLN, no timestep, no gates - the adapter refines frozen text features
    and never sees the diffusion state.
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
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(
            dim, num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, norm_eps=norm_eps, backend=attn_backend
        )
        self.norm2 = RMSNorm(dim, eps=norm_eps)
        self.mlp = SwiGLU(dim, mlp_ratio)

    def forward(self, y: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        y = y + self.attn(self.norm1(y), attn_mask=attn_mask)
        return y + self.mlp(self.norm2(y))


class LayerwiseAttentionBlock(nn.Module):
    """Pre-norm transformer block over one token's encoder-layer states."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 1.3,
        norm_eps: float = 1e-6,
        attn_backend: str = "sdpa",
    ):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(
            dim,
            num_heads,
            qkv_bias=False,
            qk_norm=False,
            norm_eps=norm_eps,
            backend=attn_backend,
        )
        self.norm2 = RMSNorm(dim, eps=norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden, bias=True),
            nn.SiLU(),
            nn.Linear(hidden, dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TransformerTextEmbedder(nn.Module):
    """Text adapter with capacity: ``Linear`` -> N unconditioned blocks -> RMSNorm.

    The blocks run with **no RoPE**: text positions come from the learned
    ``y_pos_embedding`` table added by ``IrisDiT.forward`` *after* this module
    and from the text RoPE inside the trunk blocks downstream; neither is
    duplicated here.

    Padding **is** masked here: text states arrive padded to ``text_len`` and
    this adapter masks pad *keys* inside its own blocks, so its outputs at real
    tokens never depend on the padding. The trunk's joint attention downstream
    is left as it is. The mask is required, not optional.

    ``mask`` is the encoder's ``[B, T]`` attention mask (1 = real token). Only
    keys are masked; every query keeps its own position (the mask is OR'd with
    the diagonal) so an all-pad row degenerates to identity attention instead of
    an all -inf softmax row, i.e. NaN-free without a host sync. Real-token
    outputs are unaffected by that diagonal term - a real query already attends
    itself.
    """

    def __init__(
        self,
        text_dim: int,
        hidden_size: int,
        num_blocks: int = 2,
        num_heads: int = 24,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        norm_eps: float = 1e-6,
        attn_backend: str = "sdpa",
    ):
        super().__init__()
        # same names as TextEmbedder: the projection and the output norm stay
        # 1:1 across variants for checkpoint conversion
        self.proj = nn.Linear(text_dim, hidden_size, bias=True)
        self.blocks = nn.ModuleList(
            [
                TextAdapterBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    norm_eps=norm_eps,
                    attn_backend=attn_backend,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=norm_eps)

    def forward(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.shape != y.shape[:2]:
            raise ValueError(f"text mask must be {tuple(y.shape[:2])}, got {tuple(mask.shape)}")
        keep = mask.bool()[:, None, None, :]  # [B, 1, 1, T]: broadcast over heads and queries
        eye = torch.eye(y.shape[1], dtype=torch.bool, device=y.device)
        attn_mask = keep | eye
        y = self.proj(y)
        for block in self.blocks:
            y = block(y, attn_mask=attn_mask)
        return self.norm(y)


class LayerwiseTextEmbedder(nn.Module):
    """Aggregate frozen encoder layers per token, then refine across tokens."""

    def __init__(
        self,
        text_dim: int,
        hidden_size: int,
        num_layers: int,
        layer_num_heads: int,
        layer_mlp_ratio: float = 1.3,
        refiner_num_heads: int = 24,
        refiner_mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        norm_eps: float = 1e-6,
        attn_backend: str = "sdpa",
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"text_lap_num_layers must be positive, got {num_layers}")
        self.text_dim = text_dim
        self.num_layers = num_layers
        self.layer_blocks = nn.ModuleList(
            [
                LayerwiseAttentionBlock(
                    text_dim,
                    layer_num_heads,
                    mlp_ratio=layer_mlp_ratio,
                    norm_eps=norm_eps,
                    attn_backend=attn_backend,
                )
                for _ in range(2)
            ]
        )
        self.layer_pool = nn.Linear(num_layers, 1, bias=True)
        self.refiner = TransformerTextEmbedder(
            text_dim,
            hidden_size,
            num_blocks=2,
            num_heads=refiner_num_heads,
            mlp_ratio=refiner_mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
            attn_backend=attn_backend,
        )

    def forward(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expected = (*mask.shape, self.num_layers, self.text_dim)
        if y.shape != expected:
            raise ValueError(f"layerwise text states must be {expected}, got {tuple(y.shape)}")
        batch, tokens, layers, dim = y.shape
        y = y.reshape(batch * tokens, layers, dim)
        for block in self.layer_blocks:
            y = block(y)
        y = self.layer_pool(y.transpose(1, 2)).squeeze(-1)
        y = y.reshape(batch, tokens, dim)
        return self.refiner(y, mask)


def _sincos_1d(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """[M] positions -> [M, embed_dim] with (sin | cos) halves."""
    omega = torch.arange(embed_dim // 2, dtype=torch.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000.0**omega
    out = torch.outer(pos.reshape(-1).double(), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def sincos_pos_embed_2d(embed_dim: int, height: int, width: int) -> torch.Tensor:
    """Fixed 2D sincos table ``[height*width, embed_dim]`` (float32).

    Follows the MAE/DiT grid convention: meshgrid built width-first, first
    half of channels from the first grid axis, second half from the second.
    """
    grid_h = torch.arange(height, dtype=torch.float32)
    grid_w = torch.arange(width, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # each [H, W]
    grid = torch.stack(grid, dim=0).reshape(2, 1, height, width)
    emb = torch.cat([_sincos_1d(embed_dim // 2, grid[0]), _sincos_1d(embed_dim // 2, grid[1])], dim=1)
    return emb.float()


class PixelEmbedder(nn.Module):
    """Per-pixel linear embedding grouped into per-patch sequences.

    ``[B, C, H, W] -> [B * (H/p) * (W/p), p*p, hidden]`` with pixels row-major
    inside each patch. Optionally adds a fixed full-resolution 2D sincos
    positional embedding before grouping (cached per (H, W)).
    """

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int, abs_pos_embed: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.abs_pos_embed = abs_pos_embed
        self.proj = nn.Linear(in_channels, hidden_size, bias=True)
        # keyed by (H, W, device, dtype): the table is not a buffer, so without
        # the device in the key every forward re-issues a host->device copy of
        # H*W*hidden floats (16 MiB at 512px, 64 MiB at 1024px)
        self._pos_cache: dict[tuple[int, int, str, torch.dtype], torch.Tensor] = {}

    def _pos(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (height, width, str(device), dtype)
        table = self._pos_cache.get(key)
        if table is None:
            table = sincos_pos_embed_2d(self.proj.out_features, height, width)
            table = table.reshape(height, width, -1).to(device=device, dtype=dtype)
            self._pos_cache[key] = table
        return table

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        p = self.patch_size
        h_patches, w_patches = height // p, width // p
        tokens = self.proj(x.permute(0, 2, 3, 1))  # [B, H, W, hidden]
        if self.abs_pos_embed:
            tokens = tokens + self._pos(height, width, tokens.device, tokens.dtype)
        tokens = tokens.reshape(batch, h_patches, p, w_patches, p, -1)
        tokens = tokens.permute(0, 1, 3, 2, 4, 5)  # [B, Hp, Wp, p, p, hidden]
        return tokens.reshape(batch * h_patches * w_patches, p * p, -1)
