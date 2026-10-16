import math
import torch
import torch.nn.functional as F
from torch import nn
from fireredtts3.llm.modules import (
    RotaryEmbedding,
    TimestepEmbedder,
    Attention,
    FeedForward,
    RMSNorm,
)


class Transpose(torch.nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor):
        x = torch.transpose(x, self.dim0, self.dim1)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.block = torch.nn.Sequential(
            nn.Conv1d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                padding=(kernel_size-1)//2
            ),
            nn.Mish(),
            nn.Conv1d(
                out_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                padding=(kernel_size-1)//2
            ),
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: shape (b, t, c)
            mask: shape (b, t, 1), default to None
        """
        if mask is not None: x = x * mask
        x = x.transpose(1, 2)
        x = self.block(x)
        x = x.transpose(1, 2)
        if mask is not None: x = x * mask
        return x


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTBlock(nn.Module):
    def __init__(
        self, 
        hidden_size, 
        num_heads, 
        mlp_ratio=4.0, 
        dropout=0.1, 
        **kwargs
    ):
        super().__init__()
        # Attn
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            dropout=dropout
        )
        # Conv
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.conv = ConvBlock(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3)
        # FFN 
        self.norm3 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(dim=hidden_size, mult=mlp_ratio, dropout=dropout, approximate="tanh")
        # Time AdaLN condition
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, mask: torch.Tensor, rope: torch.Tensor):
        """
        Args:
            x(torch.Tensor): shape (b, t, c)
            c(torch.Tensor): shae (b, 1, c), time condition
            mask(torch.Tensor): default to None, DO NOT USE THIS ARG.
            rope(torch.Tensor): positional embedding.
        """
        # AdaLN for t
        (
            shift_msa, scale_msa, gate_msa, 
            shift_mlp, scale_mlp, gate_mlp, 
            shift_conv, scale_conv, gate_conv
        ) = self.adaLN_modulation(c).chunk(9, dim=-1)
        # Attn
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            mask=mask, 
            rope=rope
        )
        # Conv
        x = x + gate_conv * self.conv(
            modulate(self.norm2(x), shift_conv, scale_conv), 
            mask=mask,
        )
        # FFN
        x = x + gate_mlp * self.mlp(
            modulate(self.norm3(x), shift_mlp, scale_mlp)
        )
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_ratio: float = 4.0,
        depth: int = 28,
        num_heads: int = 8,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_proj = nn.Linear(in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=0.0) 
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)

    def forward(self, x:torch.Tensor, t:torch.Tensor):
        """Full attention DiT head.

        Args:
            x: shape (b, t, c), including xt and other conditions
            t: shape (b,).
        Returns:
            pred: shape (b, t, c)
        """
        # time
        t = self.t_embedder(t.view(-1)).unsqueeze(1)  # (b, 1, c)
        x = self.in_proj(x)
        rope = self.rotary_embed.forward_from_seq_len(x.shape[1])
        for block in self.blocks:
            x = block(x, t, mask=None, rope=rope)
        x = self.final_layer(x, t)
        return x

