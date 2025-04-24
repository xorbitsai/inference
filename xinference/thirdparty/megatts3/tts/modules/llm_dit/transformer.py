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


from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class AdaLNZero(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLNZero_Out(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class Attention(nn.Module):
    def __init__(self, encoder_dim, encoder_n_heads, max_seq_len):
        super().__init__()
        self.encoder_n_kv_heads = encoder_n_heads
        model_parallel_size = 1
        self.n_local_heads = encoder_n_heads // model_parallel_size
        self.n_local_kv_heads = self.encoder_n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = encoder_dim // encoder_n_heads

        self.wq = nn.Linear(
            encoder_dim,
            encoder_n_heads * self.head_dim,
        )
        self.wk = nn.Linear(
            encoder_dim,
            self.encoder_n_kv_heads * self.head_dim,
        )
        self.wv = nn.Linear(
            encoder_dim,
            self.encoder_n_kv_heads * self.head_dim,
        )
        self.wo = nn.Linear(
            encoder_n_heads * self.head_dim,
            encoder_dim,
        )

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = xk.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = xv.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)

        output = F.scaled_dot_product_attention(xq, keys, values, mask[:, None, None, :], is_causal=False)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
            ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim
        )
        self.w2 = nn.Linear(
            hidden_dim, dim
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, encoder_dim, encoder_n_heads, max_seq_len):
        super().__init__()
        self.encoder_n_heads = encoder_n_heads
        self.encoder_dim = encoder_dim
        self.head_dim = encoder_dim // encoder_n_heads
        self.attention = Attention(encoder_dim, encoder_n_heads, max_seq_len)
        self.feed_forward = FeedForward(
            dim=encoder_dim,
            hidden_dim=2 * encoder_dim,
            multiple_of=256,
            ffn_dim_multiplier=None,
        )
        self.attention_norm = AdaLNZero(encoder_dim)
        self.ffn_norm = nn.LayerNorm(encoder_dim, elementwise_affine=False, eps=1e-6)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attention_norm(x, emb=t)

        # attention
        attn_output = self.attention(norm, start_pos, freqs_cis, mask=mask)

        # process attention output for input x
        h = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ffn_norm(h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.feed_forward(norm)
        out = h + gate_mlp.unsqueeze(1) * ff_output

        return out


class Transformer(nn.Module):
    def __init__(self, encoder_n_layers, encoder_dim, encoder_n_heads, max_seq_len):
        super().__init__()
        # Decoder
        self.layers = torch.nn.ModuleList()
        for _ in range(encoder_n_layers):
            self.layers.append(TransformerBlock(encoder_dim, encoder_n_heads, max_seq_len))

        self.norm = AdaLNZero_Out(encoder_dim)
        self.out_proj = nn.Linear(encoder_dim, encoder_dim)

        # Rope embedding
        freqs_cis = precompute_freqs_cis(
            encoder_dim // encoder_n_heads, max_seq_len
        )
        self.register_buffer("freqs_cis", torch.view_as_real(freqs_cis), persistent=False)
    
    def forward(self, x, t, attn_mask, start_pos=0):
        freqs_cis = torch.view_as_complex(self.freqs_cis.float())[start_pos: start_pos + x.size(1)]
        for i, layer in enumerate(self.layers):
            x = layer(x, t, start_pos, freqs_cis, attn_mask)
        x = self.norm(x, t)
        x = self.out_proj(x)
        return x