# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, encoder_dim, llm_dim, downsample_rate=2):
        super().__init__()
        self.ds = downsample_rate
        self.linear1 = nn.Linear(encoder_dim * downsample_rate, llm_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x, x_lens):
        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.ds
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(
            batch_size, seq_len // self.ds, feat_dim * self.ds
        )

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        new_x_lens = torch.clamp(x_lens, max=seq_len) // self.ds
        return x, new_x_lens
