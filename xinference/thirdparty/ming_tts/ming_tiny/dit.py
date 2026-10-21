# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from .modules import FinalLayer, DiTBlock
from x_transformers.x_transformers import RotaryEmbedding



#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################



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
    """
    Embeds class labels into vector representations. Also handles llm hidden dropout for classifier-free guidance.
    """
    def __init__(self, input_feature_size, hidden_size, dropout_prob):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.cond_embedder = nn.Linear(input_feature_size, hidden_size)

    def cond_drop(self, llm_cond):
        """
        Drops llm hidden to enable classifier-free guidance.
        """
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
    """
    Diffusion model with a Transformer backbone.
    """
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
        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)


        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w_x = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w_x.view([w_x.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize label embedding table:
        w_c = self.c_embedder.cond_embedder.weight.data
        nn.init.xavier_uniform_(w_c.view([w_c.shape[0], -1]))
        nn.init.constant_(self.c_embedder.cond_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.time_mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.time_mlp[2].weight, std=0.02)


        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)



    def forward(self, x, t, c, latent_history, mask=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        t = self.t_embedder(t).unsqueeze(1)          # (N, D)
        x_now = self.x_embedder(x)  #转换x_noise的维度
        x_history = self.x_embedder(latent_history) #转换target history的维度
        x = torch.cat([x_history, x_now], dim=1) # 拼接patch history和当前patch
        c = self.c_embedder(c, self.training)  # (N, 1, 896) -> (N, 1, D)，同时带有CFG替换操作
        y = t + c
        x = torch.cat([y, x], dim=1) # # (N, 1 + patch_size *2, D)
        rope = self.rotary_embed.forward_from_seq_len(x.shape[1])

        if mask is not None:
            mask_pad = mask.clone().detach()[:, :1].expand(-1, x_history.shape[1] + c.shape[1])
            mask = torch.cat([mask_pad, mask], dim=-1)
        for block in self.blocks:
            x = block(x, mask, rope)                      # (N, T, D)
        x = self.final_layer(x)               # (N, T, patch_size ** 2 * out_channels)
        #x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, c, cfg_scale, latent_history, patch_size):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        if not cfg_scale == 1:
            x = torch.cat([x, x], dim=0)
            latent_history = torch.cat([latent_history, latent_history], dim=0)
            fake_latent = torch.zeros(c.shape).to(c.device)
            c = torch.cat([c, fake_latent], dim=0)
        if t.ndim == 0:
            t = t.repeat(x.shape[0])
        model_out = self.forward(x, t, c, latent_history)
        return model_out[:, -patch_size:, :]
