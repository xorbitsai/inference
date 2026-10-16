# Modified for Xinference: use package-relative imports.
"""Dual-level pixel-space diffusion transformer, the Iris model.

Level 1 (patch stage): text-conditioned transformer over p x p patch tokens.
Level 2 (pixel stage): shallow per-pixel refinement conditioned on the patch
stage's output tokens (timestep-fused). No VAE anywhere - input and output
are RGB velocities in pixel space.
"""

import inspect
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ModelConfig
from . import ac
from .blocks.pit import PiTBlock
from ..nn.embeddings import (
    LayerwiseTextEmbedder,
    PatchEmbedder,
    PixelEmbedder,
    TextEmbedder,
    TimestepEmbedder,
    TransformerTextEmbedder,
)
from ..nn.modulation import ModulationBuilder
from ..nn.norms import RMSNorm
from ..nn.rope import rope_1d, rope_2d
from ..registry import BLOCKS


def _caller_site() -> str:
    """Where the current forward() was called from: first frame outside torch
    and this module. Error path only - inspect.stack() is expensive."""
    for frame in inspect.stack()[2:]:
        if f"{os.sep}torch{os.sep}" in frame.filename or frame.filename == __file__:
            continue
        return f"{frame.filename}:{frame.lineno} in {frame.function}()"
    return "<unknown call site>"


@dataclass
class IrisOutput:
    x: torch.Tensor  # predicted velocity, [B, C, H, W]
    # patch tokens captured after the requested 1-based block indices
    features: dict[int, torch.Tensor]


class FinalLayer(nn.Module):
    """Unmodulated head: RMSNorm + zero-initialized Linear."""

    def __init__(self, hidden_size: int, out_dim: int, norm_eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


class IrisDiT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        cfg.validate()
        # activation-checkpointing policy, set by the trainer from
        # train.activation_checkpointing / train.ac_selective_every
        self.activation_checkpointing = "none"
        self.ac_selective_every = 2
        p = cfg.patch_size
        dim = cfg.hidden_size

        self.s_embedder = PatchEmbedder(p * p * cfg.in_channels, dim)
        self.t_embedder = TimestepEmbedder(dim, max_period=cfg.timestep_max_period)
        if cfg.text_adapter == "linear":
            self.y_embedder = TextEmbedder(cfg.text_dim, dim, norm_eps=cfg.norm_eps)
        elif cfg.text_adapter == "blocks2":
            self.y_embedder = TransformerTextEmbedder(
                cfg.text_dim,
                dim,
                num_blocks=2,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                qk_norm=cfg.qk_norm,
                norm_eps=cfg.norm_eps,
                attn_backend=cfg.attn_backend,
            )
        elif cfg.text_adapter == "lap_blocks2":
            self.y_embedder = LayerwiseTextEmbedder(
                cfg.text_dim,
                dim,
                num_layers=cfg.text_lap_num_layers,
                layer_num_heads=cfg.text_lap_num_heads,
                layer_mlp_ratio=cfg.text_lap_mlp_ratio,
                refiner_num_heads=cfg.num_heads,
                refiner_mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                qk_norm=cfg.qk_norm,
                norm_eps=cfg.norm_eps,
                attn_backend=cfg.attn_backend,
            )
        else:
            raise ValueError(
                f"unknown model.text_adapter '{cfg.text_adapter}' (linear | blocks2 | lap_blocks2)"
            )
        # the masked adapter must never run without a mask; see forward()
        self._adapter_needs_mask = cfg.text_adapter != "linear"
        self.y_pos_embedding = (
            nn.Parameter(torch.randn(1, cfg.text_len, dim)) if cfg.text_abs_pos_embed else None
        )

        # Shared modulation cores (``model.modulation`` = shared_*), one per
        # stream, owned here and referenced by every patch block; empty under
        # per_block. A hybrid trunk points its single-stream half at the dual
        # half's image core rather than allocating a third core.
        self.modulation_cores = nn.ModuleDict()
        build_modulation = ModulationBuilder(
            dim,
            mode=cfg.modulation,
            rank=cfg.modulation_rank,
            cores=self.modulation_cores,
            stream_aliases={"shared": "img"} if cfg.dual_depth else None,
        )

        # a hybrid trunk is dual-stream for the first dual_depth blocks, then
        # cfg.block: dual costs 2x the parameters of single at the same FLOPs
        dual_cls, tail_cls = BLOCKS.get("mmdit"), BLOCKS.get(cfg.block)
        # both block families carry the same attention variants, so GQA, the
        # sigmoid content gate and sandwich norm can be measured independently
        # of the stream topology
        block_options = {
            "num_kv_heads": cfg.num_kv_heads,
            "gated_attention": cfg.gated_attention,
            "sandwich_norm": cfg.sandwich_norm,
        }
        final_text_out = cfg.final_block_text == "keep"
        self.blocks = nn.ModuleList(
            [
                (dual_cls if i < cfg.dual_depth else tail_cls)(
                    dim,
                    cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    qkv_bias=cfg.qkv_bias,
                    qk_norm=cfg.qk_norm,
                    norm_eps=cfg.norm_eps,
                    attn_backend=cfg.attn_backend,
                    modulation=build_modulation,
                    text_out=final_text_out or i < cfg.depth - 1,
                    **block_options,
                )
                for i in range(cfg.depth)
            ]
        )

        if not cfg.pixel.enabled:
            # patch-only: DiT-style unpatchify head
            self.pixel_embedder = None
            self.pixel_blocks = None
            self.final_layer = FinalLayer(dim, p * p * cfg.in_channels, norm_eps=cfg.norm_eps)
        else:
            self.pixel_embedder = PixelEmbedder(
                cfg.in_channels, cfg.pixel.hidden_size, p, abs_pos_embed=cfg.pixel.abs_pos_embed
            )
            self.pixel_blocks = nn.ModuleList(
                [
                    PiTBlock(
                        cfg.pixel.hidden_size,
                        cond_dim=dim,
                        pixels_per_patch=p * p,
                        attn_hidden_size=cfg.pixel.attn_hidden_size,
                        num_heads=cfg.pixel.num_heads,
                        mlp_ratio=cfg.pixel.mlp_ratio,
                        modulation=cfg.pixel.modulation,
                        qk_norm=cfg.qk_norm,
                        norm_eps=cfg.norm_eps,
                        attn_backend=cfg.attn_backend,
                    )
                    for _ in range(cfg.pixel.depth)
                ]
            )
            self.final_layer = FinalLayer(cfg.pixel.hidden_size, cfg.in_channels, norm_eps=cfg.norm_eps)

        # keyed by (shape key, device str): tables live where they are used,
        # so the hot loop never re-issues H2D transfers
        self._rope_img: dict[tuple, torch.Tensor] = {}
        self._rope_txt: dict[tuple, torch.Tensor] = {}
        self._rope_pix: dict[tuple, torch.Tensor] = {}
        self.initialize_weights()

    # -- weight init ----------------------------------------------------------
    def initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.s_embedder.proj.weight)
        nn.init.zeros_(self.s_embedder.proj.bias)
        for linear in (self.t_embedder.mlp[0], self.t_embedder.mlp[2]):
            nn.init.normal_(linear.weight, std=0.02)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        # adaLN projections keep PyTorch default init unless adaln_zero_init is
        # set. Every modulation output projection is named ``adaln*`` so this
        # scan catches it: the per-block Linear, the shared cores, and under
        # shared_lowrank each block's bias-free residual U. A residual's V
        # (``down``) is intentionally left alone -- zeroing both factors would
        # pin the residual's gradient at zero forever.
        if self.cfg.adaln_zero_init:
            for module in self.modules():
                for name, child in module.named_children():
                    if name.startswith("adaln") and isinstance(child, nn.Linear):
                        nn.init.zeros_(child.weight)
                        if child.bias is not None:
                            nn.init.zeros_(child.bias)

    # -- rope caches -----------------------------------------------------------
    def _fetch_rope_img(self, grid: tuple[int, int], device: torch.device) -> torch.Tensor:
        key = (grid, str(device))
        if key not in self._rope_img:
            head_dim = self.cfg.hidden_size // self.cfg.num_heads
            self._rope_img[key] = rope_2d(
                head_dim,
                grid[0],
                grid[1],
                theta=self.cfg.rope_theta,
                scale=self.cfg.rope_scale,
                aspect=self.cfg.rope_aspect,
                frame_pairs=self.cfg.rope_frame_pairs,
                frame_theta=self.cfg.rope_frame_theta,
            ).to(device)
        return self._rope_img[key]

    def _fetch_rope_txt(self, length: int, device: torch.device) -> torch.Tensor | None:
        if not self.cfg.text_rope:
            return None
        key = (length, str(device))
        if key not in self._rope_txt:
            head_dim = self.cfg.hidden_size // self.cfg.num_heads
            self._rope_txt[key] = rope_1d(head_dim, length, theta=self.cfg.text_rope_theta).to(device)
        return self._rope_txt[key]

    def _fetch_rope_pix(self, grid: tuple[int, int], device: torch.device) -> torch.Tensor:
        key = (grid, str(device))
        if key not in self._rope_pix:
            head_dim = self.cfg.pixel.attn_hidden_size // self.cfg.pixel.num_heads
            # no frame axis here: in-context reference images would carry patch
            # tokens only and never enter the per-pixel pathway
            self._rope_pix[key] = rope_2d(
                head_dim,
                grid[0],
                grid[1],
                theta=self.cfg.rope_theta,
                scale=self.cfg.rope_scale,
                aspect=self.cfg.rope_aspect,
            ).to(device)
        return self._rope_pix[key]

    # -- forward ---------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        capture: tuple[int, ...] = (),
        y_mask: torch.Tensor | None = None,
    ) -> IrisOutput:
        """Predict flow velocity.

        Args:
            x: noisy image, [B, C, H, W], H and W divisible by patch_size.
            t: model time in [0, 1000] (shifted flow level x 1000), [B].
            y: text encoder states, [B, T, text_dim], or selected layer states
                [B, T, L, text_dim] for ``text_adapter="lap_blocks2"``.
            capture: 1-based block indices whose output patch tokens to return.
            y_mask: encoder attention mask, [B, T] (1 = real token). Ignored by
                ``text_adapter="linear"``; REQUIRED by the masked transformer
                adapters, which raise rather than run maskless. The trunk's
                joint attention never consumes it.
        """
        cfg = self.cfg
        p = cfg.patch_size
        batch, _, height, width = x.shape
        if height % p or width % p:
            # unfold and fold both floor, so they agree and the DiT-unpatchify
            # head would silently emit a zero band on the remainder rows
            raise ValueError(f"input {height}x{width} is not divisible by patch_size {p}")
        grid = (height // p, width // p)
        n_patches = grid[0] * grid[1]

        patches = F.unfold(x, kernel_size=p, stride=p).transpose(1, 2)  # [B, L, p*p*C]
        s = self.s_embedder(patches)
        t_emb = self.t_embedder(t)  # [B, 1, D]
        cond = F.silu(t_emb)

        y = y[:, : cfg.text_len]
        if self._adapter_needs_mask:
            if y_mask is None:
                raise ValueError(
                    f"model.text_adapter='{cfg.text_adapter}' masks pad text positions and therefore "
                    f"requires y_mask (the encoder's [B, T] attention mask); it was not passed at "
                    f"{_caller_site()}. Silently running maskless would train and sample under "
                    "different attention than every other call site."
                )
            y = self.y_embedder(y, y_mask[:, : cfg.text_len])
        else:
            y = self.y_embedder(y)  # linear adapter is mask-agnostic by design
        if self.y_pos_embedding is not None:
            y = y + self.y_pos_embedding[:, : y.shape[1]].to(y.dtype)

        rope_img = self._fetch_rope_img(grid, x.device)
        rope_txt = self._fetch_rope_txt(y.shape[1], x.device)

        # eval and sampling never build a backward graph, so no region is entered
        policy = self.activation_checkpointing if self.training else "none"
        every = self.ac_selective_every

        features: dict[int, torch.Tensor] = {}
        for i, block in enumerate(self.blocks):
            s, y = ac.run(block, (s, y, cond, rope_img, rope_txt), policy, i, every)
            if i + 1 in capture:
                features[i + 1] = s

        s = F.silu(t_emb + s)  # timestep re-fused into every patch token

        if self.pixel_blocks is None:
            out = self.final_layer(s)  # [B, L, p*p*C]
            folded = out.transpose(1, 2)
        else:
            s_cond = s.reshape(batch * n_patches, -1)
            pixels = self.pixel_embedder(x)  # [B*L, p*p, d_pix]
            rope_pix = self._fetch_rope_pix(grid, x.device)
            for i, block in enumerate(self.pixel_blocks):
                pixels = ac.run(block, (pixels, s_cond, rope_pix, grid), policy, i, every)
            out = self.final_layer(pixels)  # [B*L, p*p, C]
            folded = out.reshape(batch, n_patches, p * p, -1).permute(0, 3, 2, 1)
            folded = folded.reshape(batch, -1, n_patches)

        x_out = F.fold(folded, output_size=(height, width), kernel_size=p, stride=p)
        return IrisOutput(x=x_out, features=features)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
