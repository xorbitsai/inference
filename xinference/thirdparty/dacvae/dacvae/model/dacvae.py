# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import math
from typing import List, Optional
from typing import Union

import numpy as np
import os
import torch
from audiotools.ml import BaseModel
from torch import nn

from .base import CodecMixin
from dacvae.nn.layers import MsgProcessor, NormConv1d, NormConvTranspose1d, Snake1d, activation
from dacvae.nn.quantize import ResidualVectorQuantize
from dacvae.nn.bottleneck import VAEBottleneck
from huggingface_hub import hf_hub_download

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(
        self,
        dim: int = 16,
        kernel: int = 7,
        dilation: int = 1,
        act: str = "Snake",
        stride: int = 1,
        compress: int = 1,
        pad_mode: str = "none",
        causal: bool = False,
        norm: str = "weight_norm",
        true_skip: bool = False,
    ):
        super().__init__()
        kernels = [kernel, 1]
        dilations = [dilation, 1]

        hidden = dim // compress

        if act == "Snake":
            act_params = {"channels": dim}
        elif act == "ELU":
            act_params = {"alpha": 1.0}
        else:
            raise ValueError(f"Unsupported activation: {act}")

        layers = []
        for i, (kernel_size, dilation) in enumerate(zip(kernels, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernels) - 1 else hidden

            layers += [
                activation(act=act, **act_params),
                NormConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    norm=norm,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]

        self.block = nn.Sequential(*layers)
        self.true_skip = true_skip

    def shortcut(self, x: torch.Tensor, y: torch.Tensor):
        if self.true_skip:
            return x
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x

    def forward(self, x):
        y = self.block(x)
        return y + self.shortcut(x, y)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, kernel=7, dilation=1),
            ResidualUnit(dim // 2, kernel=7, dilation=3),
            ResidualUnit(dim // 2, kernel=7, dilation=9),
            Snake1d(dim // 2),
            NormConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                pad_mode="none",
            ),
        )

    def forward(self, x):
        return self.block(x)


class LSTMBlock(nn.Module):
    def __init__(self, *args, skip: bool = True, **kwargs):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(*args, **kwargs)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        return y.permute(1, 2, 0)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        layers = [NormConv1d(1, d_model, kernel_size=7, pad_mode="none")]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            layers += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        layers += [
            Snake1d(d_model),
            NormConv1d(d_model, d_latent, kernel_size=3, pad_mode="none"),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*layers)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


default_decoder_convtr_kwargs = {
    "acts": ["Snake", "ELU"],
    "pad_mode": ["none", "auto"],
    "norm": ["weight_norm", "none"],
}


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        stride_wm: int = 1,
        acts: Optional[List[str]] = None,
        pad_modes: Optional[List[str]] = None,
        norms: Optional[List[str]] = None,
        downsampling_factor: int = 3,
        last_kernel_size: Optional[int] = None,
    ):
        super().__init__()

        # Set up activation sequences beween convolution
        if acts is None:
            acts = default_decoder_convtr_kwargs["acts"]
        if pad_modes is None:
            pad_modes = default_decoder_convtr_kwargs["pad_mode"]
        if norms is None:
            norms = default_decoder_convtr_kwargs["norm"]

        conv_strides = [stride, stride_wm]
        conv_in_dim = input_dim
        conv_out_dim = output_dim
        layers = []
        for act, norm, pad_mode, conv_stride in zip(acts, norms, pad_modes, conv_strides):
            if act == "Snake":
                act_params = {"channels": input_dim}
                causal = False
            else:  # ELU
                act_params = {"alpha": 1.0}
                causal = True
                conv_in_dim //= downsampling_factor
                conv_out_dim //= downsampling_factor
            layers += [
                activation(act=act, **act_params),
                NormConvTranspose1d(
                    conv_in_dim,
                    conv_out_dim,
                    kernel_size=2 * conv_stride,
                    stride=conv_stride,
                    causal=causal,
                    pad_mode=pad_mode,
                    norm=norm,
                ),
            ]

        layers += [
            ResidualUnit(output_dim, dilation=1, act="Snake", compress=1, causal=False, pad_mode="none", norm="weight_norm", true_skip=False),
            ResidualUnit(output_dim, dilation=3, act="Snake", compress=1, causal=False, pad_mode="none", norm="weight_norm", true_skip=False),
            ResidualUnit(output_dim // downsampling_factor, kernel=3, act="ELU", compress=2, causal=True, pad_mode="auto", norm="none", true_skip=True),
            ResidualUnit(output_dim // downsampling_factor, kernel=3, act="ELU", compress=2, causal=True, pad_mode="auto", norm="none", true_skip=True),
            ResidualUnit(output_dim, dilation=9, act="Snake", compress=1, causal=False, pad_mode="none", norm="weight_norm", true_skip=False),
        ]

        if last_kernel_size is not None:
            layers += [ResidualUnit(output_dim, kernel=last_kernel_size, act="Snake", pad_mode="none", norm="weight_norm", causal=False, true_skip=True)]
        else:
            layers += [nn.Identity()]

        layers += [
            nn.ELU(alpha=1.0),
            NormConv1d(conv_out_dim, conv_in_dim, kernel_size=2 * stride_wm, stride=stride_wm, causal=True, pad_mode="auto", norm="none"),
        ]

        self.block = nn.ModuleList(layers)
        self._chunk_size = len(acts)

    def forward(self, x):
        layer_cnt = len(self.block)
        chunks = [self.block[i:i + self._chunk_size] for i in range(0, layer_cnt, self._chunk_size)]
        group = [layer for j, chunk in enumerate(chunks) if j % self._chunk_size == 0 for layer in chunk]
        group = nn.Sequential(*group)
        return group(x)

    def upsample_group(self):
        layer_cnt = len(self.block)
        chunks = [self.block[i:i + self._chunk_size] for i in range(0, layer_cnt, self._chunk_size)]
        group = [layer for j, chunk in enumerate(chunks) if j % self._chunk_size != 0 for layer in chunk]
        return nn.Sequential(*group[(len(group) // 2):])

    def downsample_group(self):
        layer_cnt = len(self.block)
        chunks = [self.block[i:i + self._chunk_size] for i in range(0, layer_cnt, self._chunk_size)]
        group = [layer for j, chunk in enumerate(chunks) if j % self._chunk_size != 0 for layer in chunk]
        return nn.Sequential(*group[:(len(group) // 2)])


default_wm_encoder_kwargs = {
    "acts": ["Snake", "Tanh"],
    "pad_mode": ["none", "auto"],
    "norm": ["weight_norm", "none"],
}


class WatermarkEncoderBlock(nn.Module):
    def __init__(
        self,
        in_dim: int = 96,
        out_dim: int = 128,
        wm_channels: int = 32,
        hidden: int = 512,
        lstm_layers: Optional[int] = None,
        acts: Optional[List[str]] = None,
        pad_modes: Optional[List[str]] = None,
        norms: Optional[List[str]] = None,
    ):
        super().__init__()

        if acts is None:
            acts = default_wm_encoder_kwargs["acts"]
        if pad_modes is None:
            pad_modes = default_wm_encoder_kwargs["pad_mode"]
        if norms is None:
            norms = default_wm_encoder_kwargs["norm"]

        pre_layers = []
        for i, (act, norm, pad_mode) in enumerate(zip(acts, norms, pad_modes)):
            input_dim = in_dim if i == 0 else 1
            output_dim = 1 if i == 0 else wm_channels
            if act == "Snake":
                act_params = {"channels": in_dim}
                causal = False
            else:  # Tanh
                act_params = {}
                causal = True
            pre_layers += [
                activation(act=act, **act_params),
                NormConv1d(
                    input_dim,
                    output_dim,
                    kernel_size=7,
                    causal=causal,
                    pad_mode=pad_mode,
                    norm=norm,
                ),
            ]
        self.pre = nn.Sequential(*pre_layers)

        if lstm_layers is not None:
            post_layers = [
                LSTMBlock(hidden, hidden, lstm_layers),
            ]
        else:
            post_layers = []

        post_layers += [
            nn.ELU(alpha=1.0),
            NormConv1d(hidden, out_dim, kernel_size=7, causal=True, norm="none", pad_mode="auto"),
        ]

        self.post = nn.Sequential(*post_layers)

    def forward(self, x):
        return self.pre(x)

    def forward_conv(self, x):
        _conv = self.pre[-1]
        try:
            torch.nn.utils.remove_weight_norm(self.pre[-1].conv)
            x = self.pre(x)
            return x
        finally:
            self.pre[-1] = _conv

    def forward_no_conv(self, x):
        _conv = self.pre[-1]
        try:
            self.pre[-1] = nn.Identity()
            return self.pre(x)
        finally:
            self.pre[-1] = _conv

    def post_process(self, x):
        return self.post(x)



class WatermarkDecoderBlock(nn.Module):
    def __init__(
        self,
        in_dim: int = 128,
        out_dim: int = 1,
        channels: int = 32,
        hidden: int = 512,
        lstm_layers: Optional[int] = None,
    ):
        super().__init__()

        pre_layers = [
            NormConv1d(in_dim, hidden, kernel_size=7, causal=True, norm="none", pad_mode="auto"),
        ]

        if lstm_layers is not None:
            pre_layers += [
                LSTMBlock(hidden, hidden, lstm_layers),
            ]

        self.pre = nn.Sequential(*pre_layers)

        post_layers = [
            nn.ELU(alpha=1.0),
            NormConv1d(channels, out_dim, kernel_size=7, causal=True, norm="none", pad_mode="auto"),
        ]

        self.post = nn.Sequential(*post_layers)

    def forward(self, x):
        x = self.pre(x)
        return x

    def forward_no_conv(self, x):
        _conv = self.pre[-1]
        try:
            self.pre[-1] = nn.Identity()
            x = self.pre(x)
        finally:
            self.pre[-1] = _conv

    def post_process(self, x):
        return self.post(x)


class Watermarker(nn.Module):
    def __init__(
        self,
        dim: int,
        d_out: int = 1,
        d_latent: int = 128,
        channels: int = 32,
        hidden: int = 512,
        nbits: int = 16,
        lstm_layers: Optional[int] = None,
    ):
        super().__init__()

        self.encoder_block = WatermarkEncoderBlock(dim, d_latent, channels, hidden=hidden, lstm_layers=lstm_layers)
        self.msg_processor = MsgProcessor(nbits, d_latent)
        self.decoder_block = WatermarkDecoderBlock(d_latent, d_out, channels, hidden=hidden, lstm_layers=lstm_layers)

    def random_message(self, bsz: int):
        if self.msg_processor is not None:
            nbits: int = self.msg_processor.nbits  # type: ignore
        else:
            nbits = 16
        return torch.randint(0, 2, (bsz, nbits), dtype=torch.float32)  # type: ignore

    def forward(self, x: torch.Tensor, msg: torch.Tensor):
        x = self.encoder_block(x)
        x = self.msg_processor(x, msg)
        x = self.decoder_block(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        wm_rates,
        wm_channels: int = 32,
        nbits: int = 16,
        d_out: int = 1,
        d_wm_out: int = 128,
        blending: str = "linear",  # "linear" or "conv"
    ):
        super().__init__()

        # Add first conv layer
        layers = [NormConv1d(input_channel, channels, kernel_size=7, stride=1)]

        # Add upsampling + MRF blocks
        for i, (stride, wm_stride) in enumerate(zip(rates, wm_rates)):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride, wm_stride)]

        self.model = nn.ModuleList(layers)

        # Add watermarking
        self.wm_model = Watermarker(
            output_dim, d_out, d_wm_out, wm_channels, hidden=512, nbits=nbits, lstm_layers=2,  # type: ignore
        )
        self.alpha = wm_channels / d_wm_out
        self.blending = blending

    def forward(self, x, message: Optional[torch.Tensor] = None):
        for layer in self.model:
            x = layer(x)
        return self.watermark(x, message)


    def watermark(self, x, message: Optional[torch.Tensor] = None):
        if self.alpha == 0.0:
            return x
        h = self.wm_model.encoder_block(x)
        upsampler = map(lambda x: x.upsample_group(), self.model[1:])
        upsampler = list(upsampler)[::-1]
        for layer in upsampler:
            h = layer(h)
        h = self.wm_model.encoder_block.post_process(h)
        if message is None:
            bsz = x.shape[0]
            message = self.wm_model.random_message(bsz)

        message = message.to(x.device)
        h = self.wm_model.msg_processor(h, message)
        h = self.wm_model.decoder_block(h)

        downsampler = map(lambda x: x.downsample_group(), self.model[1:])
        for layer in downsampler:
            h = layer(h)
        h = self.wm_model.decoder_block.post_process(h)

        if self.blending == "conv":
            return self.wm_model.encoder_block.forward(x) + self.alpha * h
        else:
            return self.wm_model.encoder_block.forward_no_conv(x) + self.alpha * h


class DAC(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: Optional[int] = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        wm_rates: Optional[List[int]] = None,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        assert isinstance(latent_dim, int)

        if wm_rates is None:
            wm_rates = [8, 5, 4, 2]

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            wm_rates,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.delay = self.get_delay()

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: Optional[int] = None,
    ):
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        n_quantizers : int, optional
            Number of quantizers to use, by default None
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
        """
        z = self.encoder(audio_data)
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            z, n_quantizers
        )

        return z, codes, latents, commitment_loss, codebook_loss

    def decode(self, z: torch.Tensor, message: Optional[torch.Tensor] = None):
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input
        length : int, optional
            Number of samples in output audio, by default None
        message : Tensor[B x nbits], optional
            Message to embed in the audio, by default None

        Returns
        -------
        dict
            A dictionary with the following keys:
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        return self.decoder(z, message=message)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: Optional[int] = None,
        n_quantizers: Optional[int] = None,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z, codes, latents, commitment_loss, codebook_loss = self.encode(
            audio_data, n_quantizers
        )

        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }


class DACVAE(DAC):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: Optional[int] = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
    ):
        super().__init__(
            encoder_dim=encoder_dim,
            encoder_rates=encoder_rates,
            latent_dim=latent_dim,
            decoder_dim=decoder_dim,
            decoder_rates=decoder_rates,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
            sample_rate=sample_rate,
        )
        self.quantizer = VAEBottleneck(
            input_dim=self.latent_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )

    @classmethod
    def load(cls, path):
        if not os.path.exists(path) and path.startswith("facebook/"):
            path = hf_hub_download(repo_id=path, filename="weights.pth")
        return super().load(path)

    def _pad(self, wavs):
        length = wavs.size(-1)
        if length % self.hop_length:
            p1d = (0, self.hop_length - (length % self.hop_length))
            return torch.nn.functional.pad(wavs, p1d, "reflect")
        else:
            return wavs

    def encode(
        self,
        audio_data: torch.Tensor,
    ):
        z = self.encoder(self._pad(audio_data))
        mean, scale = self.quantizer.in_proj(z).chunk(2, dim=1)
        encoded_frames, _ = self.quantizer._vae_sample(mean, scale)
        return encoded_frames

    def decode(self, encoded_frames: torch.Tensor):
        emb = self.quantizer.out_proj(encoded_frames)
        return self.decoder(emb)
