"""Feed-forward variants."""

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    """Gated SiLU feed-forward with the 2/3 width rule, bias-free.

    Effective hidden width is ``int(2 * int(dim * mlp_ratio) / 3)`` so the
    three projections cost about the same as a classic 4x GELU MLP.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(2 * int(dim * mlp_ratio) / 3)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GeluMLP(nn.Module):
    """Classic 2-layer GELU MLP with biases (used by the per-pixel branch)."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))
