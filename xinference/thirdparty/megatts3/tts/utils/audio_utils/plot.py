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

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

LINE_COLORS = ['w', 'r', 'orange', 'k', 'cyan', 'm', 'b', 'lime', 'g', 'brown', 'navy']


def spec_to_figure(spec, vmin=None, vmax=None, title='', f0s=None, dur_info=None, figsize=(12, 6)):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    H = spec.shape[1] // 2
    fig = plt.figure(figsize=figsize)
    plt.title(title)
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)

    if dur_info is not None:
        assert isinstance(dur_info, dict)
        txt = dur_info['txt']
        dur_gt = dur_info['dur_gt']
        if isinstance(dur_gt, torch.Tensor):
            dur_gt = dur_gt.cpu().numpy()
        dur_gt = np.cumsum(dur_gt).astype(int)
        for i in range(len(dur_gt)):
            shift = (i % 8) + 1
            plt.text(dur_gt[i], shift * 4, txt[i])
            plt.vlines(dur_gt[i], 0, H // 2, colors='b')  # blue is gt
        plt.xlim(0, dur_gt[-1])
        if 'dur_pred' in dur_info:
            dur_pred = dur_info['dur_pred']
            if isinstance(dur_pred, torch.Tensor):
                dur_pred = dur_pred.cpu().numpy()
            dur_pred = np.cumsum(dur_pred).astype(int)
            for i in range(len(dur_pred)):
                shift = (i % 8) + 1
                plt.text(dur_pred[i], H + shift * 4, txt[i])
                plt.vlines(dur_pred[i], H, H * 1.5, colors='r')  # red is pred
            plt.xlim(0, max(dur_gt[-1], dur_pred[-1]))
    if f0s is not None:
        ax = plt.gca()
        ax2 = ax.twinx()
        # ax.set_xticks()

        if not isinstance(f0s, dict):
            f0s = {'f0': f0s}
        for i, (k, f0) in enumerate(f0s.items()):
            if f0 is not None:
                if isinstance(f0, torch.Tensor):
                    f0 = f0.cpu().numpy()
                ax2.plot(
                    np.arange(len(f0)) + 0.5, f0, label=k, c=LINE_COLORS[i], linewidth=1, alpha=0.5)
        ax2.set_ylim(0, 1000)
        ax2.legend()
    return fig


def align_to_figure(align, dur_info):
    if isinstance(align, torch.Tensor):
        align = align.cpu().numpy()
    H = align.shape[1]
    fig = plt.figure(figsize=(12, 6))
    plt.pcolor(align.T, vmin=0, vmax=1)
    if dur_info is not None:
        assert isinstance(dur_info, dict)
        txt = dur_info['txt']
        dur_gt = dur_info['dur_gt']
        if isinstance(dur_gt, torch.Tensor):
            dur_gt = dur_gt.cpu().numpy()
        dur_gt = np.cumsum(dur_gt).astype(int) // 2
        for i in range(len(dur_gt)):
            plt.text(dur_gt[i], i, txt[i], color='red')
            plt.vlines(dur_gt[i], 0, H, colors='b')  # blue is gt
        # plt.xlim(0, dur_gt[-1])
    return fig
