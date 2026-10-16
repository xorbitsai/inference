# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Keep compatibility with older PyTorch versions
# where weight_norm is in a different place.
# try:
#     from torch.nn.utils.parametrizations import weight_norm
# except ImportError:
from torch.nn.utils import weight_norm


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


def activation(act: str, **act_params):
    if act == "ELU":
        return nn.ELU(**act_params)
    elif act == "Snake":
        return Snake1d(**act_params)
    elif act == "Tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation: {act}")


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


def apply_parametrization_norm(module: nn.Module, norm: str = "none"):
    assert norm in ["none", "weight_norm"]
    if norm == "weight_norm":
        return weight_norm(module)
    else:
        return module

class NormConv1d(nn.Conv1d):
    """1D Causal Convolution with padding"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        norm: str = "weight_norm",  # Normalization method, value: "none", "weight_norm", "spectral_norm"
        causal: bool = False,
        pad_mode: str = "none",  # Padding mode, value: "none", "auto"
        **kwargs
    ):
        if pad_mode == "none":
            pad = (kernel_size - stride) * dilation // 2
        else:
            pad = 0

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            **kwargs
        )

        apply_parametrization_norm(self, norm)

        self.causal = causal
        self.pad_mode = pad_mode

    def pad(self, x: torch.Tensor):
        if self.pad_mode == "none":
            return x

        length = x.shape[-1]
        kernel_size = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        dilation = self.dilation if isinstance(self.dilation, int) else self.dilation[0]
        stride = self.stride if isinstance(self.stride, int) else self.stride[0]

        effective_kernel_size = (kernel_size - 1) * dilation + 1
        padding_total = (effective_kernel_size - stride)
        n_frames = (length - effective_kernel_size + padding_total) / stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
        extra_padding = ideal_length - length

        if self.causal:
            pad_x = F.pad(x, (padding_total, extra_padding))
        else:
            padding_right = extra_padding // 2
            padding_left = padding_total - padding_right
            pad_x = F.pad(x, (padding_left, padding_right + extra_padding))

        return pad_x

    def forward(self, x: torch.Tensor):
        x = self.pad(x)
        return super().forward(x)

class NormConvTranspose1d(nn.ConvTranspose1d):
    """1D Transposed Convolution with padding"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        norm: str = "weight_norm",  # Normalization method, value: "none", "weight_norm", "spectral_norm"
        causal: bool = False,
        pad_mode: str = "none",  # Padding mode, value: "none", "auto"
        **kwargs
    ):
        if pad_mode == "none":
            padding = (stride + 1) // 2
            output_padding = 1 if stride % 2 else 0
        else:
            padding = 0
            output_padding = 0

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            output_padding=output_padding,
            **kwargs
        )

        self = apply_parametrization_norm(self, norm)
        self.causal = causal
        self.pad_mode = pad_mode

    def unpad(self, x: torch.Tensor):
        if self.pad_mode == "none":
            return x
        length = x.shape[-1]
        kernel_size = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        stride = self.stride if isinstance(self.stride, int) else self.stride[0]

        padding_total = kernel_size - stride
        if self.causal:
            padding_left = 0
            end = length - padding_total
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            end = length - padding_right

        x = x[..., padding_left:end]
        return x

    def forward(self, x):
        y = super().forward(x)
        return self.unpad(y)


class MsgProcessor(torch.nn.Module):
    """
    Apply the secret message to the encoder output.
    Args:
        nbits: Number of bits used to generate the message. Must be non-zero
        hidden_size: Dimension of the encoder output
    """

    def __init__(self, nbits: int, hidden_size: int):
        super().__init__()
        assert nbits > 0, "MsgProcessor should not be built in 0bit watermarking"
        self.nbits = nbits
        self.hidden_size = hidden_size
        self.msg_processor = nn.Embedding(2 * nbits, hidden_size)

    def forward(self, hidden: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        """
        Build the embedding map: 2 x k -> k x h, then sum on the first dim
        Args:
            hidden: The encoder output, size: batch x hidden x frames
            msg: The secret message, size: batch x k
        """
        # create indices to take from embedding layer
        # k: 0 2 4 ... 2k
        indices = 2 * torch.arange(msg.shape[-1]).to(hidden.device)
        indices = indices.repeat(msg.shape[0], 1)  # b x k
        indices = (indices + msg).long()
        msg_aux = self.msg_processor(indices)  # b x k -> b x k x h
        msg_aux = msg_aux.sum(dim=-2)  # b x k x h -> b x h
        msg_aux = msg_aux.unsqueeze(-1).repeat(
            1, 1, hidden.shape[2]
        )  # b x h -> b x h x t/f
        hidden = hidden + msg_aux  # -> b x h x t/f
        return hidden
