# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Vector quantizer."""
from typing import List
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from dacvae.nn.layers import NormConv1d

class VAEBottleneck(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        codebook_size: int = 0,
        codebook_dim: Union[int, list] = 512,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.in_proj = NormConv1d(input_dim, codebook_dim*2, kernel_size=1)
        self.out_proj = NormConv1d(codebook_dim, input_dim, kernel_size=1)
        self.dummy_codebook_loss = torch.tensor(0.0)


    def forward(self, z, n_quantizers: int = None):
        mean, scale = self.in_proj(z).chunk(2, dim=1)
        z_q, kl = self._vae_sample(mean, scale)
        z_q = self.out_proj(z_q)
        return z_q, torch.zeros(z_q.size()), z_q, kl, self.dummy_codebook_loss

    def _vae_sample(self, mean, scale):
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * stdev + mean
        kl = (mean * mean + var - logvar - 1).sum(1).mean()
        return latents, kl
