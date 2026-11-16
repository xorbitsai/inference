# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

import torch.nn as nn
from torch.nn import functional as F

from .alias_free_filter import LowPassFilter1d, kaiser_sinc_filter1d


class UpSample1d(nn.Module):
    def __init__(
        self, ratio=2, kernel_size=None, channels=None, causal=True, fixed_filter=False
    ):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.stride = ratio
        self.channels = channels
        self.causal = causal
        self.fixed_filter = fixed_filter
        if causal:
            self.pad = 0
        else:
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = (
                self.pad * self.stride + (self.kernel_size - self.stride) // 2
            )
            self.pad_right = (
                self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
            )
        filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
        )
        if self.fixed_filter:
            self.register_buffer("filter", filter)
        else:
            self.filter = nn.Parameter(filter.expand(channels, -1, -1).clone())

    # x: [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        if self.fixed_filter:
            x = self.ratio * F.conv_transpose1d(
                x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C
            )
        else:
            x = self.ratio * F.conv_transpose1d(
                x, self.filter, stride=self.stride, groups=C
            )
        if self.causal:
            x = x[..., : -(self.kernel_size - self.stride)]
        else:
            x = x[..., self.pad_left : -self.pad_right]

        return x


class DownSample1d(nn.Module):
    def __init__(
        self, ratio=2, kernel_size=None, channels=None, causal=True, fixed_filter=False
    ):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
            channels=channels,
            causal=causal,
            fixed_filter=fixed_filter,
        )

    def forward(self, x):
        return self.lowpass(x)
