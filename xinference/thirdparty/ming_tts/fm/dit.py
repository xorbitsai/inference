import torch
import torch.nn as nn
import numpy as np
import math
from .modules import FinalLayer, DiTBlock
from x_transformers.x_transformers import RotaryEmbedding


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TimestepEmbedder(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # b d
        return time


class CondEmbedder(nn.Module):
    def __init__(self, input_feature_size, hidden_size, dropout_prob):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.cond_embedder = nn.Linear(input_feature_size, hidden_size)

    def cond_drop(self, llm_cond):
        bsz = llm_cond.shape[0]
        drop_latent_mask = torch.rand(bsz) < self.dropout_prob
        drop_latent_mask = drop_latent_mask.unsqueeze(-1).unsqueeze(-1).to(llm_cond.dtype).to(llm_cond.device)
        fake_latent = torch.zeros(llm_cond.shape).to(llm_cond.device)
        llm_cond  = drop_latent_mask * fake_latent + (1 - drop_latent_mask) * llm_cond

        return llm_cond

    def forward(self, llm_cond, train):
        use_dropout = self.dropout_prob > 0
        if train and use_dropout:
            llm_cond = self.cond_drop(llm_cond)

        llm_cond = self.cond_embedder(llm_cond)

        return llm_cond


class DiT(nn.Module):
    def __init__(
        self,
        in_channels=4,
        hidden_size=1024,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        llm_cond_dim=896,
        cfg_dropout_prob=0.1,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.c_embedder = CondEmbedder(llm_cond_dim, hidden_size, cfg_dropout_prob)
        self.hidden_size = hidden_size
        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)

    def forward(self, x, t, c, latent_history, mask=None):
        t = self.t_embedder(t).unsqueeze(1)
        x_now = self.x_embedder(x)
        x_history = self.x_embedder(latent_history)
        x = torch.cat([x_history, x_now], dim=1)
        c = self.c_embedder(c, self.training)
        y = t + c
        x = torch.cat([y, x], dim=1)
        rope = self.rotary_embed.forward_from_seq_len(x.shape[1])

        if mask is not None:
            mask_pad = mask.clone().detach()[:, :1].expand(-1, x_history.shape[1] + c.shape[1])
            mask = torch.cat([mask_pad, mask], dim=-1)
        for block in self.blocks:
            x = block(x, mask, rope)
        x = self.final_layer(x)
        return x

    def forward_with_cfg(self, x, t, c, latent_history, patch_size):
        """Run the conditional and unconditional branches as one batch.

        Both halves are always computed, whatever the guidance scale: the
        sampler combines them as ``pred + (pred - null_pred) * cfg``, so a
        scale of 1 still needs the unconditional prediction.
        """
        x = torch.cat([x, x], dim=0)
        latent_history = torch.cat([latent_history, latent_history], dim=0)
        c = torch.cat([c, torch.zeros_like(c)], dim=0)
        if t.ndim == 0:
            t = t.repeat(x.shape[0])
        model_out = self.forward(x, t, c, latent_history)
        return model_out[:, -patch_size:, :]


class Aggregator(nn.Module):
    def __init__(
        self,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        llm_input_dim=896,
        **kwargs,
    ):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.word_embedder = nn.Embedding(1, hidden_size)
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.hidden_size = hidden_size

        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, llm_input_dim)

    def forward(self, x, mask=None):
        x = self.x_embedder(x)
        cls_embed = self.word_embedder(
            torch.zeros((x.shape[0], 1), dtype=torch.long, device=x.device))
        x = torch.cat([cls_embed, x], dim=1)

        rope = self.rotary_embed.forward_from_seq_len(x.shape[1])
        if mask is not None:
            mask_pad = mask.clone().detach()[:, :1]
            mask = torch.cat([mask_pad, mask], dim=-1)
        for block in self.blocks:
            x = block(x, mask, rope)
        x = self.final_layer(x)
        x = x[:, :1, :]
        return x
