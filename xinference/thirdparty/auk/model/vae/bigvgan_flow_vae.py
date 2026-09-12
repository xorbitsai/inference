# Copyright (c) 2022 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import logging
import math
import os
import sys
from dataclasses import dataclass, field, fields

import torch
import torch.distributions as D
import torch.nn as nn
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from auk.model.vae.modules.bigvgan import activations
from auk.model.vae.modules.bigvgan.alias_free_torch import *
from auk.model.vae.modules.commons.layers import Conv1d, ConvTranspose1d
from auk.model.vae.modules.commons.ops import init_weights
from auk.model.vae.modules.vits.flows import ResidualCouplingBlock


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

LRELU_SLOPE = 0.1


@dataclass
class BigVGANFlowVAEConfig:
    upsample_rates: list = field(default_factory=lambda: [5, 4, 3, 2, 2, 2])
    upsample_kernel_sizes: list = field(default_factory=lambda: [10, 8, 6, 4, 4, 4])
    upsample_initial_channel: int = 1536
    resblock_kernel_sizes: list = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    downsample_rates: list = field(default_factory=lambda: [2, 2, 2, 3, 4, 5])
    downsample_channels: list = field(default_factory=lambda: [12, 24, 48, 96, 192, 384, 768])
    snake_logscale: bool = True
    latent_dim: int = 64
    use_vae: bool = True
    causal: bool = True
    flow_hidden_channels: int = 256
    act_causal: bool = True

    def __post_init__(self):
        logger.info("BigVGANFlowVAE Config:")
        for f in fields(self):
            logger.info(f"  {f.name}: {getattr(self, f.name)}")

    @classmethod
    def from_dict(cls, config_dict: dict):
        valid_fields = {f.name for f in fields(cls)}
        valid_params = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**valid_params)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        return hasattr(self, key)


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
        norm_type="weight_norm",
        init_type=None,
    ):
        super(Conv1d_S, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        pad = dilation * (kernel_size - 1) // 2

        self.layer = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
        )
        if init_type == "orthogonal":
            nn.init.orthogonal_(self.layer.weight)
        elif init_type == "normal":
            nn.init.normal_(self.layer.weight, mean=0.0, std=0.01)

        if norm_type == "weight_norm":
            self.layer = weight_norm(self.layer)
        elif norm_type == "spectral_norm":
            self.layer = spectral_norm(self.layer)

    def forward(self, inputs):
        return self.layer(inputs)


class ResStack(nn.Module):
    def __init__(self, channel, kernel_size=3, base=3, nums=4):
        super(ResStack, self).__init__()

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LeakyReLU(),
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            channel,
                            channel,
                            kernel_size=kernel_size,
                            dilation=base**i,
                            padding=base**i,
                        )
                    ),
                    nn.LeakyReLU(),
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            channel,
                            channel,
                            kernel_size=kernel_size,
                            dilation=1,
                            padding=1,
                        )
                    ),
                )
                for i in range(nums)
            ]
        )

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
        channels=[12, 24, 48, 96, 192, 384, 768],
        down_sample_factors=[2, 2, 2, 2, 4, 4],
        use_vae=False,
    ):
        super(Encoder, self).__init__()

        act_slope = 0.2
        if use_vae:
            out_channels = out_channels * 2
        layers = []
        # pre proj_layer
        layers += [
            Conv1d_S(in_channels, base_channels, kernel_size=proj_kernel_size, stride=1),
            nn.LeakyReLU(act_slope, True),
        ]

        # channels: [512, 256, 128, 64], upsample_factors: [5, 2, 2]
        for (in_c, out_c), down_f in zip(zip(channels[:-1], channels[1:]), down_sample_factors):
            layers += [
                Conv1d_S(in_c, out_c, kernel_size=down_f * 2, stride=down_f),
                ResStack(out_c, stack_kernel_size, stack_dilation_base, stacks),
                nn.LeakyReLU(act_slope, True),
            ]

        # post layers
        layers += [
            Conv1d_S(channels[-1], out_channels, proj_kernel_size, stride=1),
            # nn.Tanh() TODO
        ]
        self.generator = nn.Sequential(*layers)

    def forward(self, conditions, z_inputs=None):
        return self.generator(conditions)


class AMPBlock1(torch.nn.Module):
    def __init__(
        self,
        h,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
        causal=True,
        act_causal=False,
    ):
        super(AMPBlock1, self).__init__()
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
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, causal=causal)),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, causal=causal)),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, causal=causal)),
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2)  # total number of conv layers

        # periodic nonlinearity with snakebeta function and anti-aliasing
        self.activations = nn.ModuleList(
            [
                Activation1d(
                    activation=activations.SnakeBeta(channels, alpha_logscale=h.snake_logscale),
                    causal=act_causal,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class BigVGANFlowVAE(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        causal = h.causal
        act_causal = h.get("act_causal", False)
        self.hop_size = math.prod(h.downsample_rates)

        self.register_buffer("global_mean", torch.zeros(h.latent_dim, dtype=torch.float32))
        self.register_buffer("global_log_std", torch.ones(h.latent_dim, dtype=torch.float32))

        self.audio_encoder = Encoder(
            out_channels=h.latent_dim,
            use_vae=h.use_vae,
            down_sample_factors=h.downsample_rates,
            channels=h.downsample_channels,
        )

        self.flow = ResidualCouplingBlock(h.latent_dim, h.flow_hidden_channels, 5, 1, 4, gin_channels=0, causal=causal)

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # pre conv
        self.conv_pre = weight_norm(Conv1d(h.latent_dim, h.upsample_initial_channel, 7, 1, causal=False))

        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
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
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(
                    AMPBlock1(
                        h,
                        ch,
                        k,
                        d,
                        causal=causal,
                        act_causal=act_causal,
                    )
                )

        # post conv: periodic nonlinearity with snakebeta function and anti-aliasing
        activation_post = activations.SnakeBeta(ch, alpha_logscale=h.snake_logscale)
        self.activation_post = Activation1d(activation=activation_post, causal=act_causal)
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, causal=causal, bias=False))

        # weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, data):
        x = data["sample"]
        outputs = {}

        x = self.audio_encoder(x)
        assert self.h.use_vae

        m_q, logs_q = torch.split(x, self.h.latent_dim, dim=1)
        z = m_q + torch.randn_like(m_q) * torch.exp(logs_q)

        # Flow
        mask = torch.ones([z.size(0), 1, z.size(-1)]).to(z.device)
        z_p = self.flow(z, mask)

        p_z = D.Normal(torch.zeros_like(m_q), torch.ones_like(logs_q))

        q_z = D.Normal(z_p, torch.exp(logs_q))
        kl_div = D.kl_divergence(q_z, p_z).mean()
        outputs["kl_div"] = kl_div

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
        x = torch.clamp(x, min=-1.0, max=1.0)
        outputs["sample"] = x

        return outputs

    @torch.autocast(enabled=False, device_type="cuda")
    def encoding_and_normalization(self, sample, sample_lengths=None):
        latent_stats = self.audio_encoder(sample)
        if sample_lengths is None:
            sample_lengths = torch.LongTensor([sample.size(-1)] * sample.size(0)).to(sample.device)
        latent_lens = sample_lengths // self.hop_size
        mean, log_std = latent_stats.chunk(2, 1)  #  b, d, t
        latents = mean + torch.randn_like(mean) * torch.exp(log_std)
        latents = latents.transpose(1, 2).float()  # b, t, d
        latents = (latents - self.global_mean.float()) / torch.sqrt(self.global_log_std.float())
        latent_lens = torch.clamp(latent_lens, max=latents.size(1))  # clamp to avoid out of range
        return latents, latent_lens

    def denormalize(self, latents):
        latents = latents.float()
        return latents * torch.sqrt(self.global_log_std.float()) + self.global_mean.float()

    def inference_from_latents(self, x):
        assert x.size(1) == self.h.latent_dim, f"Input must be like [B, D, H], got {x.shape}"

        # pre conv
        x = self.conv_pre(x)

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
        x = torch.clamp(x, min=-1.0, max=1.0)
        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            for l_i in l:
                remove_weight_norm(l_i)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
