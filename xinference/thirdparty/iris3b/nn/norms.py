"""Normalization layers."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMS normalization computed in fp32, learned gain applied in input dtype."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * normed.to(x.dtype)
