# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.nn import Conv1d
import numpy as np


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class Upsample(nn.Module):
    def __init__(self, mult, r):
        super(Upsample, self).__init__()
        self.r = r
        self.upsample = nn.Sequential(nn.Upsample(mode="nearest", scale_factor=r),
                                      nn.LeakyReLU(0.2),
                                      nn.ReflectionPad1d(3),
                                      nn.utils.weight_norm(nn.Conv1d(mult, mult // 2, kernel_size=7, stride=1))
                                      )
        r_kernel = r if r >= 5 else 5
        self.trans_upsample = nn.Sequential(nn.LeakyReLU(0.2),
                                            nn.utils.weight_norm(nn.ConvTranspose1d(mult, mult // 2,
                                                                                    kernel_size=r_kernel * 2, stride=r,
                                                                                    padding=r_kernel - r // 2,
                                                                                    output_padding=r % 2)
                                                                 ))

    def forward(self, x):
        x = torch.sin(x) + x
        out1 = self.upsample(x)
        out2 = self.trans_upsample(x)
        return out1 + out2


class Downsample(nn.Module):
    def __init__(self, mult, r):
        super(Downsample, self).__init__()
        self.r = r
        r_kernel = r if r >= 5 else 5
        self.trans_downsample = nn.Sequential(nn.LeakyReLU(0.2),
                                              nn.utils.weight_norm(nn.Conv1d(mult, mult * 2,
                                                                             kernel_size=r_kernel * 2, stride=r,
                                                                             padding=r_kernel - r // 2)
                                                                   ))

    def forward(self, x):
        out = self.trans_downsample(x)
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_zero_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class Audio2Mel(nn.Module):
    def __init__(
            self,
            hop_length=300,
            sampling_rate=24000,
            n_mel_channels=80,
            mel_fmin=0.,
            mel_fmax=None,
            frame_size=0.05,
            device='cpu'
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################

        self.n_fft = int(np.power(2., np.ceil(np.log(sampling_rate * frame_size) / np.log(2))))
        window = torch.hann_window(int(sampling_rate * frame_size)).float()
        mel_basis = librosa_mel_fn(
            sampling_rate, self.n_fft, n_mel_channels, mel_fmin, mel_fmax
        )  # Mel filter (by librosa)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)

        self.hop_length = hop_length
        self.win_length = int(sampling_rate * frame_size)
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        fft = torch.stft(
            audio.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(torch.clamp(real_part ** 2 + imag_part ** 2, min=1e-5))
        mel_output = torch.matmul(self.mel_basis, magnitude)

        log_mel_spec = 20 * torch.log10(torch.clamp(mel_output, min=1e-5)) - 20
        norm_mel = (log_mel_spec + 115.) / 115.
        mel_comp = torch.clamp(norm_mel * 8. - 4., -4., 4.)

        return mel_comp


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, dim_in=None):
        super().__init__()
        if dim_in is None:
            dim_in = dim

        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim_in, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim_in, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


'''
参照hifigan（https://arxiv.org/pdf/2010.05646.pdf）v2结构
多尺度主要是kernel_size不同，3组并行卷积模块，每个卷积模块内部采用不同的串行dilation size，且中间交叉正常无dilation卷积层
'''


class ResBlockMRFV2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlockMRFV2, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.2)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.2)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlockMRFV2Inter(torch.nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResBlockMRFV2Inter, self).__init__()
        self.block1 = ResBlockMRFV2(channels)
        self.block2 = ResBlockMRFV2(channels, 7)
        self.block3 = ResBlockMRFV2(channels, 11)

    def forward(self, x):
        xs = self.block1(x)
        xs += self.block2(x)
        xs += self.block3(x)
        x = xs / 3
        return x


class Generator(nn.Module):
    def __init__(self, input_size_, ngf, n_residual_layers, num_band, args, ratios=[5, 5, 4, 3], onnx_export=False,
                 device='cpu'):
        super().__init__()
        self.hop_length = args.frame_shift
        self.args = args
        self.onnx_export = onnx_export

        # ------------- Define upsample layers ----------------
        mult = int(2 ** len(ratios))
        model_up = []
        input_size = input_size_
        model_up += [
            nn.ReflectionPad1d(3),
            WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0),
        ]

        # Upsample to raw audio scale
        for i, r in enumerate(ratios):
            model_up += [Upsample(mult * ngf, r)]
            model_up += [ResBlockMRFV2Inter(mult * ngf // 2)]
            mult //= 2

        model_up += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(ngf, num_band, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        if not args.use_tanh:
            model_up[-1] = nn.Conv1d(num_band, num_band, 1)
        model_up[-2].apply(weights_zero_init)

        self.model_up = nn.Sequential(*model_up)

        self.apply(weights_init)

    def forward(self, mel, step=None):
        # mel input: (batch_size, seq_num, 80)
        if self.onnx_export:
            mel = mel.transpose(1, 2)
            # on onnx, for engineering, mel input: (batch_size, 80, seq_num)

        # Between Down and up
        x = mel

        # Upsample pipline
        cnt_after_upsample = 0

        for i, m in enumerate(self.model_up):
            x = m(x)

            if type(m) == Upsample:
                cnt_after_upsample += 1

        return x