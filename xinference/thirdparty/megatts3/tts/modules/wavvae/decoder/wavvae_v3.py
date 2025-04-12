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

import argparse
import torch
from torch import nn
import torch.nn.functional as F

from tts.modules.wavvae.decoder.seanet_encoder import Encoder
from tts.modules.wavvae.decoder.diag_gaussian import DiagonalGaussianDistribution
from tts.modules.wavvae.decoder.hifigan_modules import Generator, Upsample


class WavVAE_V3(nn.Module):
    def __init__(self, hparams=None):
        super().__init__()
        self.encoder = Encoder(dowmsamples=[6, 5, 4, 4, 2])
        self.proj_to_z = nn.Linear(512, 64)
        self.proj_to_decoder = nn.Linear(32, 320)

        config_path = hparams['melgan_config']
        args = argparse.Namespace()
        args.__dict__.update(config_path)
        self.latent_upsampler = Upsample(320, 4)
        self.decoder = Generator(
            input_size_=160, ngf=128, n_residual_layers=4,
            num_band=1, args=args, ratios=[5,4,4,3])

    ''' encode waveform into 25 hz latent representation '''
    def encode_latent(self, audio):
        posterior = self.encode(audio)
        latent = posterior.sample().permute(0, 2, 1)  # (b,t,latent_channel)
        return latent

    def encode(self, audio):
        x = self.encoder(audio).permute(0, 2, 1)
        x = self.proj_to_z(x).permute(0, 2, 1)
        poseterior = DiagonalGaussianDistribution(x)
        return poseterior

    def decode(self, latent):
        latent = self.proj_to_decoder(latent).permute(0, 2, 1)
        return self.decoder(self.latent_upsampler(latent))

    def forward(self, audio):
        posterior = self.encode(audio)
        latent = posterior.sample().permute(0, 2, 1)  # (b, t, latent_channel)
        recon_wav = self.decode(latent)
        return recon_wav, posterior