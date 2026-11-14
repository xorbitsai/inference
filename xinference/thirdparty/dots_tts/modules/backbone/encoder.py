from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from ...modules.backbone.layers import Conv1d, Mlp, MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads=16,
        ffn_hidden_size=4096,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        norm_layer="LayerNorm",
        **kwargs,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(
            hidden_size,
            num_heads,
            attn_drop=attn_dropout,
            norm_layer=norm_layer,
            **kwargs,
        )
        norm_cls = getattr(nn, norm_layer)
        self.attn_norm = norm_cls(hidden_size)
        self.ffn = Mlp(
            hidden_size, ffn_hidden_size, dropout=ffn_dropout, act_layer=nn.SiLU
        )
        self.ffn_norm = norm_cls(hidden_size)
        self.hidden_size = hidden_size

    def _build_causal_mask(self, T: int, device):
        return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

    def _build_padding_mask(self, x_lens, max_len: int, device):
        B = x_lens.size(0)
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
        return positions < x_lens.unsqueeze(1)

    def _fuse_attn_mask(self, causal_mask, padding_mask):
        if causal_mask is None and padding_mask is None:
            return None
        if causal_mask is None:
            row = padding_mask.unsqueeze(2)
            col = padding_mask.unsqueeze(1)
            return row & col
        if padding_mask is None:
            return causal_mask.unsqueeze(0)

        _B, _T = padding_mask.shape
        causal = causal_mask.unsqueeze(0)
        row = padding_mask.unsqueeze(2)
        col = padding_mask.unsqueeze(1)
        pad_2d = row & col
        return causal & pad_2d

    def forward(
        self,
        x,
        x_lens=None,
        causal=True,
    ):
        _B, T, C = x.shape
        assert self.hidden_size == C
        device = x.device

        causal_mask = self._build_causal_mask(T, device) if causal else None
        if x_lens is not None:
            padding_mask = self._build_padding_mask(x_lens, T, device)
        else:
            padding_mask = None
        fused_mask = self._fuse_attn_mask(causal_mask, padding_mask)

        h = self.attn_norm(x)
        h = self.attn(
            q=h,
            mask=fused_mask,
        )
        x = x + h

        h = self.ffn_norm(x)
        h = self.ffn(h)
        return x + h


class SuperviseEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.get("hidden_size", 1024)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    hidden_size=self.hidden_size,
                    num_heads=config.get("num_heads", 16),
                    ffn_hidden_size=config.get("ffn_hidden_size", 4096),
                    norm_layer=config.get("norm_layer", "LayerNorm"),
                )
                for _ in range(config.get("num_layers", 6))
            ]
        )
        self.causal = config.get("causal", False)

    def forward(self, x, x_lens=None):
        batch_size, seq_len, _ = x.shape
        if x_lens is None:
            x_lens = torch.full(
                (batch_size,), seq_len, device=x.device, dtype=torch.long
            )
        for layer in self.layers:
            x = layer(x, x_lens=x_lens, causal=self.causal)
        return x


class VAESemanticEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, config):
        super().__init__()
        self.expects_normalized_input = False
        in_ds_rate = 2
        self.patch_size = int(config.patch_size)
        self.in_ds_rate = in_ds_rate
        self.ds_proj = Conv1d(
            in_dim, in_dim, kernel_size=in_ds_rate, stride=in_ds_rate, causal=True
        )
        self.in_proj = nn.Linear(in_dim, config.PatchEncoder.hidden_size)
        self.encoder = SuperviseEncoder(config.PatchEncoder)
        self.out_ds_rate = self.patch_size // in_ds_rate
        self.out_proj = nn.Linear(
            config.PatchEncoder.hidden_size * self.out_ds_rate, out_dim
        )

    @property
    def input_dtype(self) -> torch.dtype:
        return self.in_proj.weight.dtype

    def forward(self, x, x_lens=None):
        x = self._downsample(x)
        x = self.in_proj(x)
        z = self.encoder(x, x_lens=x_lens)
        return self._project_embeddings(z)

    def _downsample(self, x):
        return self.ds_proj(x.transpose(1, 2)).transpose(1, 2)

    def _project_embeddings(self, z):
        if self.out_ds_rate > 1:
            z = rearrange(z, "b (s d) h -> b s (d h)", d=self.out_ds_rate)
        return self.out_proj(z)
