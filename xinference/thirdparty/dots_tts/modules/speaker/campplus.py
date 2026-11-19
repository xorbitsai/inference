# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from ...modules.speaker.campplus_layers import (
    BasicResBlock,
    CAMDenseTDNNBlock,
    DenseLayer,
    StatsPool,
    TDNNLayer,
    TransitLayer,
    get_nonlinear,
)
from ...modules.speaker.fbank import _SPEAKER_FBANK_N_MELS


class FCM(nn.Module):
    def __init__(
        self,
        block=BasicResBlock,
        num_blocks=(2, 2),
        m_channels=32,
        feat_dim=_SPEAKER_FBANK_N_MELS,
    ):
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[1], stride=2)

        self.conv2 = nn.Conv2d(
            m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        return out.reshape(shape[0], shape[1] * shape[2], shape[3])


class CAMPPlus(nn.Module):
    _TDNN_KERNEL_SIZE = 5
    _TDNN_STRIDE = 2
    _TDNN_PADDING = 2

    def __init__(
        self,
        feat_dim=_SPEAKER_FBANK_N_MELS,
        embedding_size=512,
        growth_rate=32,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
        memory_efficient=True,
    ):
        super().__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels

        self.xvector = nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(
                            channels,
                            init_channels,
                            self._TDNN_KERNEL_SIZE,
                            stride=self._TDNN_STRIDE,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    ),
                ]
            )
        )
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2), strict=True)
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector.add_module(f"block{i + 1}", block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                f"transit{i + 1}",
                TransitLayer(
                    channels, channels // 2, bias=False, config_str=config_str
                ),
            )
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))

        self.xvector.add_module("stats", StatsPool())
        self.xvector.add_module(
            "dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_")
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _conv_output_lengths(lengths, kernel_size, stride=1, padding=0, dilation=1):
        return (
            torch.div(
                lengths + 2 * padding - dilation * (kernel_size - 1) - 1,
                stride,
                rounding_mode="floor",
            )
            + 1
        )

    @staticmethod
    def _make_length_mask(lengths, max_len, device):
        lengths = lengths.to(device=device, dtype=torch.long).clamp(min=0, max=max_len)
        return torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)

    def _masked_stats_pooling(self, x, lengths, unbiased=True, eps=1e-2):
        lengths = lengths.to(device=x.device, dtype=torch.long).clamp(
            min=1, max=x.size(-1)
        )
        mask = self._make_length_mask(lengths, x.size(-1), x.device).unsqueeze(1)
        mask = mask.to(dtype=x.dtype)

        denom = lengths.to(dtype=x.dtype).view(-1, 1).clamp_min(1.0)
        mean = (x * mask).sum(dim=-1) / denom

        centered = (x - mean.unsqueeze(-1)) * mask
        var_denom = (
            (lengths - 1).clamp_min(1).to(dtype=x.dtype).view(-1, 1)
            if unbiased
            else denom
        )
        var = centered.pow(2).sum(dim=-1) / var_denom
        std = torch.sqrt(var.clamp_min(eps))
        return torch.cat([mean, std], dim=1)

    def forward(self, x, lengths=None):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        if lengths is not None:
            lengths = lengths.to(device=x.device, dtype=torch.long).clamp(min=1)

        for name, module in self.xvector.named_children():
            if name == "stats":
                x = (
                    self._masked_stats_pooling(x, lengths)
                    if lengths is not None
                    else module(x)
                )
                continue

            x = module(x)
            if name == "tdnn" and lengths is not None:
                lengths = self._conv_output_lengths(
                    lengths,
                    kernel_size=self._TDNN_KERNEL_SIZE,
                    stride=self._TDNN_STRIDE,
                    padding=self._TDNN_PADDING,
                )

        return x
