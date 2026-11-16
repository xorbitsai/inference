# Copyright (c) 2022 NVIDIA CORPORATION.
#   Licensed under the MIT license.
from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from ...modules.backbone.layers import Conv1d, ConvTranspose1d
from ...modules.vocoder.alias_free_act import Activation1d, Snake, SnakeBeta
from ...modules.vocoder.config import AudioVAEConfig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class Conv1d_S(nn.Module):
    "Conv1d for spectral normalisation and orthogonal initialisation"

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        groups=1,
        causal=False,
    ):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.causal = causal
        pad = 0 if causal else dilation * (kernel_size - 1) // 2
        self.causal_pad = dilation * (kernel_size - 1) if causal else 0

        self.layer = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                dilation=dilation,
                groups=groups,
            )
        )

    def forward(self, inputs):
        if self.causal and self.causal_pad > 0:
            inputs = F.pad(inputs, (self.causal_pad, 0))
        return self.layer(inputs)


class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """

    def __init__(
        self,
        dimension: int,
        num_layers: int = 2,
        skip: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.skip = skip
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=dimension,
            hidden_size=dimension,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        if self.bidirectional:
            self.proj_out = nn.Linear(dimension * 2, dimension)

    def forward(self, x):
        y, _ = self.lstm(x)
        if self.bidirectional:
            y = self.proj_out(y)
        if self.skip:
            y = y + x
        return y


class ResStack(nn.Module):
    def __init__(self, channel, kernel_size=3, base=3, nums=4, causal=False):
        super().__init__()

        self.layers = nn.ModuleList([])
        for i in range(nums):
            dil = base**i
            pad1 = dil * (kernel_size - 1) if causal else dil
            pad2 = (kernel_size - 1) if causal else 1
            block = [
                nn.LeakyReLU(),
            ]
            if causal and pad1 > 0:
                block.append(nn.ConstantPad1d((pad1, 0), 0.0))
            block.append(
                nn.utils.weight_norm(
                    nn.Conv1d(
                        channel,
                        channel,
                        kernel_size=kernel_size,
                        dilation=dil,
                        padding=0 if causal else pad1,
                    )
                )
            )
            block.append(nn.LeakyReLU())
            if causal and pad2 > 0:
                block.append(nn.ConstantPad1d((pad2, 0), 0.0))
            block.append(
                nn.utils.weight_norm(
                    nn.Conv1d(
                        channel,
                        channel,
                        kernel_size=kernel_size,
                        dilation=1,
                        padding=0 if causal else pad2,
                    )
                )
            )
            self.layers.append(nn.Sequential(*block))

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=100,
        base_channels=12,
        proj_kernel_size=3,
        stack_kernel_size=3,
        stack_dilation_base=2,
        stacks=6,
        channels=(12, 24, 48, 96, 192, 384, 768),
        down_sample_factors=(2, 2, 2, 2, 4, 4),
        causal=False,
        lookahead=0,
    ):
        super().__init__()

        act_slope = 0.2
        layers = []
        # pre proj_layer
        layers += [
            Conv1d_S(
                in_channels,
                base_channels,
                kernel_size=proj_kernel_size,
                stride=1,
                causal=causal,
            ),
            nn.LeakyReLU(act_slope, True),
        ]

        # channels: [512, 256, 128, 64], upsample_factors: [5, 2, 2]
        for (in_c, out_c), down_f in zip(
            itertools.pairwise(channels), down_sample_factors, strict=True
        ):
            layers += [
                Conv1d_S(
                    in_c, out_c, kernel_size=down_f * 2, stride=down_f, causal=causal
                ),
                ResStack(
                    out_c, stack_kernel_size, stack_dilation_base, stacks, causal=causal
                ),
                nn.LeakyReLU(act_slope, True),
            ]

        # post layers
        if lookahead > 0:
            layers += [
                Conv1d_S(
                    channels[-1],
                    out_channels,
                    kernel_size=lookahead * 2 + 1,
                    stride=1,
                    causal=False,
                ),
            ]
        else:
            layers += [
                Conv1d_S(
                    channels[-1],
                    out_channels,
                    kernel_size=proj_kernel_size,
                    stride=1,
                    causal=causal,
                ),
            ]
        self.generator = nn.Sequential(*layers)

    def forward(self, conditions, _z_inputs=None):
        return self.generator(conditions)


class AMPBlock1(torch.nn.Module):
    def __init__(
        self,
        h,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
        activation=None,
        causal=True,
    ):
        super().__init__()
        self.h = h

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        causal=causal,
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        causal=causal,
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        causal=causal,
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels, channels, kernel_size, 1, dilation=1, causal=causal
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels, channels, kernel_size, 1, dilation=1, causal=causal
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels, channels, kernel_size, 1, dilation=1, causal=causal
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # total number of conv layers

        if (
            activation == "snake"
        ):  # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=Snake(channels, alpha_logscale=h.snake_logscale),
                        causal=causal,
                        fixed_filter=True,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif (
            activation == "snakebeta"
        ):  # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale),
                        causal=causal,
                        fixed_filter=True,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2, strict=True):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class AMPBlock2(torch.nn.Module):
    def __init__(
        self, h, channels, kernel_size=3, dilation=(1, 3), activation=None, causal=True
    ):
        super().__init__()
        self.h = h

        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        causal=causal,
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        causal=causal,
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs)  # total number of conv layers

        if (
            activation == "snake"
        ):  # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=Snake(channels, alpha_logscale=h.snake_logscale),
                        causal=causal,
                        fixed_filter=True,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif (
            activation == "snakebeta"
        ):  # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale),
                        causal=causal,
                        fixed_filter=True,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        for c, a in zip(self.convs, self.activations, strict=True):
            xt = a(x)
            xt = c(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for layer in self.convs:
            remove_weight_norm(layer)


class Decoder(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        causal = h.causal

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        num_decoder_lookahead = h.get("num_decoder_lookahead", 2)
        # pre conv
        self.conv_pre = weight_norm(
            Conv1d(
                h.latent_dim,
                h.upsample_initial_channel,
                kernel_size=2 * num_decoder_lookahead + 1,
                stride=1,
                causal=False,
            )
        )

        # define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        resblock = AMPBlock1 if h.resblock == "1" else AMPBlock2

        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(h.upsample_rates, h.upsample_kernel_sizes, strict=True)
        ):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2**i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                causal=causal,
                            )
                        )
                    ]
                )
            )

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(
                h.resblock_kernel_sizes, h.resblock_dilation_sizes, strict=True
            ):
                self.resblocks.append(
                    resblock(h, ch, k, d, activation=h.activation, causal=causal)
                )

        # post conv
        if (
            h.activation == "snake"
        ):  # periodic nonlinearity with snake function and anti-aliasing
            activation_post = Snake(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(
                activation=activation_post, causal=causal, fixed_filter=False
            )
        elif (
            h.activation == "snakebeta"
        ):  # periodic nonlinearity with snakebeta function and anti-aliasing
            activation_post = SnakeBeta(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(
                activation=activation_post, causal=causal, fixed_filter=False
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, causal=causal, bias=h.get("use_bias_at_final", True))
        )

        # weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, z):
        # pre conv
        x = self.conv_pre(z)

        for i in range(self.num_upsamples):
            # upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        if self.h.get("use_tanh_at_final", True):
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]

        return x

    def remove_weight_norm(self):
        for upsample_layers in self.ups:
            for upsample_layer in upsample_layers:
                remove_weight_norm(upsample_layer)
        for resblock in self.resblocks:
            resblock.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class AudioVAE(nn.Module):
    def __init__(self, h: AudioVAEConfig):
        super().__init__()
        self.config = h

        self.h = h
        self.hop_size = int(np.prod(h.downsample_rates))
        self.sample_rate = h.sample_rate
        self.decoder_lookahead = int(h.get("num_decoder_lookahead", 2))

        self.audio_encoder = Encoder(
            out_channels=h.latent_dim,
            down_sample_factors=h.downsample_rates,
            channels=h.downsample_channels,
            causal=h.causal_encoder,
            lookahead=h.get("num_encoder_lookahead", 2),
        )

        intermediate_size = h.latent_dim * 4
        self.enc_mi_layer = nn.Sequential(
            nn.Linear(h.latent_dim, intermediate_size),
            SLSTM(intermediate_size, num_layers=h.mi_num_layers),
            nn.Linear(intermediate_size, h.latent_dim),
        )
        self.dec_mi_layer = nn.Sequential(
            nn.Linear(h.latent_dim, intermediate_size),
            SLSTM(intermediate_size, num_layers=h.mi_num_layers),
            nn.Linear(intermediate_size, h.latent_dim),
        )
        self.pre_proj = Conv1d(
            in_channels=h.latent_dim,
            out_channels=h.latent_dim * 2,
            kernel_size=1,
            stride=1,
        )
        self.post_proj = Conv1d(
            in_channels=h.latent_dim, out_channels=h.latent_dim, kernel_size=1, stride=1
        )

        self.decoder = Decoder(h)

    def inference(self, data):
        latents = self.extract_latents(data["sample"])
        return {"sample": self.inference_from_latents(latents)}

    @torch.autocast(enabled=False, device_type="cuda")
    def extract_latents(self, x, do_sample=False):
        x = x.float()
        x = self.audio_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.enc_mi_layer(x)
        x = x.permute(0, 2, 1)
        x = self.pre_proj(x)
        if do_sample:
            m_q, logs_q = torch.split(x, self.h.latent_dim, dim=1)
            x = m_q + torch.randn_like(m_q) * torch.exp(logs_q)
        return x

    def inference_from_latents(self, x, do_sample=True, noise_scale=1.0):
        if do_sample:
            assert x.size(1) == self.h.latent_dim * 2, (
                f"Input must be like [B, D, H], got {x.shape}"
            )
            m_q, logs_q = torch.split(x, self.h.latent_dim, dim=1)
            x = m_q + torch.randn_like(m_q) * torch.exp(logs_q) * noise_scale
        else:
            assert x.size(1) == self.h.latent_dim, (
                f"Input must be like [B, D, H], got {x.shape}"
            )
        x = self.post_proj(x)
        x = x.permute(0, 2, 1)
        x = self.dec_mi_layer(x)
        x = x.permute(0, 2, 1)
        return self.decoder(x)

    def remove_weight_norm(self):
        self.decoder.remove_weight_norm()
