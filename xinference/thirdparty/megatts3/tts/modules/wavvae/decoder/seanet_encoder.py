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

from typing import List

import torch
from torch import nn
from tts.modules.wavvae.encoder.common_modules.seanet import SEANetEncoder

class Encoder(nn.Module):
    def __init__(
        self,
        dowmsamples: List[int] = [6, 5, 5, 4, 2],
    ):
        super().__init__()

        # breakpoint()
        self.frame_rate = 25  # not use
        self.encoder = SEANetEncoder(causal=False, n_residual_layers=1, norm='weight_norm', pad_mode='reflect', lstm=2,
                                dimension=512, channels=1, n_filters=32, ratios=dowmsamples, activation='ELU',
                                kernel_size=7, residual_kernel_size=3, last_kernel_size=7, dilation_base=2,
                                true_skip=False, compress=2)

    def forward(self, audio: torch.Tensor):
        audio = audio.unsqueeze(1)                  # audio(16,24000)
        emb = self.encoder(audio)
        return emb
