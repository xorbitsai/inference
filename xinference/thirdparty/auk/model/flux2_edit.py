"""
Flux2Edit backbone — Flux2Audio variant using pre-encoded LLM text embeddings.

Two-phase architecture:
  1. Double-stream MMDiTBlock (joint text-audio attention)
  2. Single-stream DiTBlock (concatenated text+audio)

ein notation:
b - batch
n - sequence
nt - text sequence
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, fields

import torch
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from auk.model.modules import (
    AdaLayerNorm_Final,
    ConvPositionEmbedding,
    DiTBlock,
    MMDiTBlock,
    TimestepEmbedding,
)


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class AudioPromptEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(out_dim)

    def _embed(self, x: float["b n d"], mask: bool["b n"] | None = None):
        x = self.linear(x)
        x = self.conv_pos_embed(x, mask=mask) + x
        return x

    def forward(
        self,
        x: float["b n d"],
        ref: float["b np d"] | None = None,
        drop_audio_cond: bool = False,
        mask: bool["b n"] | None = None,
        ref_mask: bool["b np"] | None = None,
    ):
        x_emb = self._embed(x, mask=mask)
        if ref is None:
            return x_emb
        if drop_audio_cond:
            ref = torch.zeros_like(ref)
        ref_emb = self._embed(ref, mask=ref_mask)
        return x_emb, ref_emb


@dataclass
class Flux2EditConfig:
    dim: int = 1024
    depth: int = 8
    heads: int = 16
    dim_head: int = 64
    dropout: float = 0.1
    ff_mult: float = 2.0
    text_hidden_dim: int = 2048
    checkpoint_activations: bool = False
    checkpoint_every_n_layers: int = 1  # 1=every layer, n>1=every n-th layer (only used when checkpoint_activations)
    attn_backend: str = "torch"  # torch | flash_attn
    attn_mask_enabled: bool = False
    num_layers: int = 8  # double-stream (MMDiT) block count
    num_single_layers: int = 24  # single-stream (DiT) block count

    def __post_init__(self):
        logger.info("Flux2Edit Config:")
        for f in fields(self):
            logger.info(f"  {f.name}: {getattr(self, f.name)}")

    @classmethod
    def from_dict(cls, config_dict: dict):
        valid_fields = {f.name for f in fields(cls)}
        valid_params = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**valid_params)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        return hasattr(self, key)


class Flux2Edit(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        latent_dim=100,
        text_hidden_dim=2048,
        checkpoint_activations=False,
        checkpoint_every_n_layers=1,
        attn_backend="torch",
        attn_mask_enabled=False,
        num_layers=8,
        num_single_layers=24,
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth

        self.time_embed = TimestepEmbedding(dim)

        # text projection: Linear(text_hidden_dim, dim) -> RMSNorm(dim)
        self.txt_norm = nn.RMSNorm(dim, elementwise_affine=True)
        self.txt_proj = nn.Linear(text_hidden_dim, dim)

        self.audio_embed = AudioPromptEmbedding(latent_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)

        # Double Stream Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    ff_mult=ff_mult,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                )
                for i in range(num_layers)
            ]
        )

        # Single Stream Transformer Blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, latent_dim)

        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_every_n_layers = max(1, checkpoint_every_n_layers)

        # text cache (mirrors Flux2Audio interface)
        self.text_cond, self.text_uncond = None, None

        self.initialize_weights()

    def initialize_weights(self):
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm_x.linear.weight, 0)
            nn.init.constant_(block.attn_norm_x.linear.bias, 0)
            nn.init.constant_(block.attn_norm_c.linear.weight, 0)
            nn.init.constant_(block.attn_norm_c.linear.bias, 0)

        for block in self.single_transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            return module(*inputs)

        return ckpt_forward

    def project_text(self, text: float["b nt h"], drop_text: bool = False):
        """Project LLM hidden states to model dim: Linear -> RMSNorm."""
        c = self.txt_norm(self.txt_proj(text))
        if drop_text:
            c = torch.zeros_like(c)
        return c

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def _embed_audio(
        self,
        x: float["b n d"],
        ref: float["b np d"] | None,
        drop_audio_cond: bool,
        mask: bool["b n"] | None,
        ref_mask: bool["b np"] | None,
    ):
        if ref is not None and ref.shape[1] == 0:
            ref = None
        out = self.audio_embed(
            x,
            ref=ref,
            drop_audio_cond=drop_audio_cond,
            mask=mask,
            ref_mask=ref_mask,
        )
        if isinstance(out, tuple):
            x_emb, ref_emb = out
            prompt_len = ref_emb.shape[1]
            audio = torch.cat([ref_emb, x_emb], dim=1)  # TODO：ref_emb 和 noised latent拼接在一起
            if mask is not None or ref_mask is not None:
                B, N = x_emb.shape[:2]
                if mask is None:
                    mask = torch.ones(B, N, dtype=torch.bool, device=x_emb.device)
                if ref_mask is None:
                    ref_mask = torch.ones(B, prompt_len, dtype=torch.bool, device=ref_emb.device)
                audio_mask = torch.cat([ref_mask, mask], dim=1)
            else:
                audio_mask = None
            return audio, audio_mask, prompt_len
        # ref is None — allow target-only forward for symmetry
        return out, mask, 0

    def forward(
        self,
        x: float["b n d"],  # noised input audio
        text: float["b nt h"] = None,  # pre-encoded text embeddings from LLM
        time: float["b"] | float[""] = None,  # time step
        mask: bool["b n"] | None = None,
        c_mask: bool["b n"] | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
        ref: float["b np d"] | None = None,  # prompt audio latent
        ref_mask: bool["b np"] | None = None,
    ):
        # print(x.shape)
        batch = x.shape[0]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)  # B * D

        # text mask: padding positions are all-zero in LLM output
        if c_mask is None:
            c_mask = text.abs().sum(-1) > 0  # [B, nt], True = valid

        if cfg_infer:
            # cond branch
            if cache and self.text_cond is not None:
                c_cond = self.text_cond
            else:
                c_cond = self.project_text(text, drop_text=False)
                if cache:
                    self.text_cond = c_cond
            x_cond, a_mask_cond, prompt_len = self._embed_audio(x, ref, drop_audio_cond=False, mask=mask, ref_mask=ref_mask)

            # uncond branch
            if cache and self.text_uncond is not None:
                c_uncond = self.text_uncond
            else:
                c_uncond = self.project_text(text, drop_text=True)
                if cache:
                    self.text_uncond = c_uncond
            x_uncond, a_mask_uncond, _ = self._embed_audio(x, ref, drop_audio_cond=True, mask=mask, ref_mask=ref_mask)

            x = torch.cat((x_cond, x_uncond), dim=0)
            c = torch.cat((c_cond, c_uncond), dim=0)
            t = torch.cat((t, t), dim=0)

            if a_mask_cond is not None and a_mask_uncond is not None:
                audio_mask = torch.cat((a_mask_cond, a_mask_uncond), dim=0)
            else:
                audio_mask = None
            c_mask = torch.cat((c_mask, c_mask), dim=0)
        else:
            c = self.project_text(text, drop_text=drop_text)  # b, seq, d
            x, audio_mask, prompt_len = self._embed_audio(x, ref, drop_audio_cond=drop_audio_cond, mask=mask, ref_mask=ref_mask)

        seq_len = x.shape[1]  # reference audio prompt | noised audio latent
        text_len = c.shape[1]  # nt

        rope_audio = self.rotary_embed.forward_from_seq_len(seq_len)
        rope_text = self.rotary_embed.forward_from_seq_len(text_len)

        # Phase 1: Double-stream MMDiTBlock
        for i, block in enumerate(self.transformer_blocks):
            if self.checkpoint_activations and i % self.checkpoint_every_n_layers == 0:
                c, x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block),
                    x,
                    c,
                    t,
                    audio_mask,
                    rope_audio,
                    rope_text,
                    c_mask,
                    use_reentrant=False,
                )
            else:
                c, x = block(x, c, t, mask=audio_mask, rope=rope_audio, c_rope=rope_text, c_mask=c_mask)

        # Phase 2: Concatenate text+audio -> single-stream DiTBlock
        x = torch.cat([c, x], dim=1)
        rope = self.rotary_embed.forward_from_seq_len(text_len + seq_len)

        if audio_mask is not None:
            single_mask = torch.cat([c_mask, audio_mask], dim=1)
        else:
            single_mask = None

        for i, block in enumerate(self.single_transformer_blocks):
            if self.checkpoint_activations and i % self.checkpoint_every_n_layers == 0:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, single_mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=single_mask, rope=rope)

        # extract noised-target portion: drop text prefix and (seq_prepend) prompt prefix
        x = x[:, text_len + prompt_len :]
        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
