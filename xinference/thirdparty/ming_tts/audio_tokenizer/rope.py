# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of the torchtune source tree:
# https://github.com/pytorch/torchtune
#
# Vendored verbatim from ``torchtune.modules.RotaryPositionalEmbeddings`` so
# that inference does not pull in the torchtune/torchao training stack for a
# single module. Both buffers are non-persistent, so vendoring does not change
# the checkpoint's state dict.

from typing import Optional

import torch
from torch import nn


class RotaryPositionalEmbeddings(nn.Module):
    """Rotary Positional Embeddings (https://arxiv.org/abs/2104.09864).

    Args:
        dim: Head dimension, i.e. ``embed_dim // num_heads``.
        max_seq_len: Positions are cached up to this length and recomputed
            beyond it.
        base: Base of the geometric progression of rotation angles.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )
        # [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply RoPE to ``x`` of shape ``[b, s, n_h, h_d]``.

        ``input_pos`` carries the absolute position of each token, which is
        what makes incremental decoding line up with the cached keys.
        """
        seq_len = x.size(1)

        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # Cast to float to match the reference implementation.
        # [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        return x_out.flatten(3).type_as(x)


__all__ = ["RotaryPositionalEmbeddings"]
