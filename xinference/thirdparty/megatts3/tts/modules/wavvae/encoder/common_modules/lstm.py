# MIT License

# Copyright (c) Meta Platforms, Inc. and affiliates.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (c) [2023] [Meta Platforms, Inc. and affiliates.] 
# Copyright (c) [2025] [Ziyue Jiang] 
# SPDX-License-Identifier: MIT
# This file has been modified by Ziyue Jiang on 2025/03/19
# Original file was released under MIT, with the full license text # available at https://github.com/facebookresearch/encodec/blob/gh-pages/LICENSE.
# This modified file is released under the same license.

"""LSTM layers module."""
from torch import nn


class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    # 修改transpose顺序
    def forward(self, x):
        x1 = x.permute(2, 0, 1)
        y, _ = self.lstm(x1)
        y = y.permute(1, 2, 0)
        if self.skip:
            y = y + x
        return y
