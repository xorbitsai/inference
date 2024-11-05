# Copyright © 2024 Apple Inc.

from dataclasses import dataclass
from typing import List

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.upsample import upsample_nearest


@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: List[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(
            num_groups=32,
            dims=in_channels,
            eps=1e-6,
            affine=True,
            pytorch_compatible=True,
        )
        self.q = nn.Linear(in_channels, in_channels)
        self.k = nn.Linear(in_channels, in_channels)
        self.v = nn.Linear(in_channels, in_channels)
        self.proj_out = nn.Linear(in_channels, in_channels)

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape

        y = x.reshape(B, 1, -1, C)
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=C ** (-0.5))
        y = self.proj_out(y)

        return x + y.reshape(B, H, W, C)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(
            num_groups=32,
            dims=in_channels,
            eps=1e-6,
            affine=True,
            pytorch_compatible=True,
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.GroupNorm(
            num_groups=32,
            dims=out_channels,
            eps=1e-6,
            affine=True,
            pytorch_compatible=True,
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Linear(in_channels, out_channels)

    def __call__(self, x):
        h = x
        h = self.norm1(h)
        h = nn.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nn.silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def __call__(self, x: mx.array):
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def __call__(self, x: mx.array):
        x = upsample_nearest(x, (2, 2))
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = []
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = []
            attn = []  # TODO: Remove the attn, nobody appends anything to it
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = {}
            down["block"] = block
            down["attn"] = attn
            if i_level != self.num_resolutions - 1:
                down["downsample"] = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = {}
        self.mid["block_1"] = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid["attn_1"] = AttnBlock(block_in)
        self.mid["block_2"] = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(
            num_groups=32, dims=block_in, eps=1e-6, affine=True, pytorch_compatible=True
        )
        self.conv_out = nn.Conv2d(
            block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1
        )

    def __call__(self, x: mx.array):
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level]["block"][i_block](hs[-1])

                # TODO: Remove the attn
                if len(self.down[i_level]["attn"]) > 0:
                    h = self.down[i_level]["attn"][i_block](h)

                hs.append(h)

            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level]["downsample"](hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid["block_1"](h)
        h = self.mid["attn_1"](h)
        h = self.mid["block_2"](h)

        # end
        h = self.norm_out(h)
        h = nn.silu(h)
        h = self.conv_out(h)

        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = {}
        self.mid["block_1"] = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid["attn_1"] = AttnBlock(block_in)
        self.mid["block_2"] = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = []
        for i_level in reversed(range(self.num_resolutions)):
            block = []
            attn = []  # TODO: Remove the attn, nobody appends anything to it

            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = {}
            up["block"] = block
            up["attn"] = attn
            if i_level != 0:
                up["upsample"] = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(
            num_groups=32, dims=block_in, eps=1e-6, affine=True, pytorch_compatible=True
        )
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def __call__(self, z: mx.array):
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid["block_1"](h)
        h = self.mid["attn_1"](h)
        h = self.mid["block_2"](h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level]["block"][i_block](h)

                # TODO: Remove the attn
                if len(self.up[i_level]["attn"]) > 0:
                    h = self.up[i_level]["attn"][i_block](h)

            if i_level != 0:
                h = self.up[i_level]["upsample"](h)

        # end
        h = self.norm_out(h)
        h = nn.silu(h)
        h = self.conv_out(h)

        return h


class DiagonalGaussian(nn.Module):
    def __call__(self, z: mx.array):
        mean, logvar = mx.split(z, 2, axis=-1)
        if self.training:
            std = mx.exp(0.5 * logvar)
            eps = mx.random.normal(shape=z.shape, dtype=z.dtype)
            return mean + std * eps
        else:
            return mean


class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.reg = DiagonalGaussian()

        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def sanitize(self, weights):
        new_weights = {}
        for k, w in weights.items():
            if w.ndim == 4:
                w = w.transpose(0, 2, 3, 1)
                w = w.reshape(-1).reshape(w.shape)
                if w.shape[1:3] == (1, 1):
                    w = w.squeeze((1, 2))
            new_weights[k] = w
        return new_weights

    def encode(self, x: mx.array):
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: mx.array):
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def __call__(self, x: mx.array):
        return self.decode(self.encode(x))
