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

import math
import torch
from typing import Optional, Tuple
from torch import nn
from torch.nn import Parameter, Linear
from tts.modules.ar_dur.commons.layers import LayerNorm, Embedding
from tts.modules.ar_dur.commons.transformer import TransformerFFNLayer, MultiheadAttention
from tts.modules.ar_dur.commons.seq_utils import get_incremental_state, set_incremental_state, softmax, make_positions
import torch.nn.functional as F

DEFAULT_MAX_SOURCE_POSITIONS = 3000
DEFAULT_MAX_TARGET_POSITIONS = 3000


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None, positions=None, **kwargs):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.shape[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = make_positions(input, self.padding_idx) if positions is None else positions
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class RotaryEmbeddings(nn.Module):
    cos: torch.Tensor
    sin: torch.Tensor
    theta: torch.Tensor

    def __init__(
            self,
            width: int,
            *,
            seq_len: int = 40000,
            base: int = 10000,
            device: Optional[torch.device] = None,
    ):
        """Rotary embeddings (Su et al., 2021) layer. The rotary embedding
        will be precomputed for up to 'seq _len' positions. The embedding
        will be recomputed when a longer sequence is found in the input.

        :param width:
            Rotary embedding dimensionality, must be even.
        :param seq_len:
            Number of positons to initially precompute.
        :param base:
            The base used for Θ_i, determines the cycle length of the
            embeddings.
        :param device: Device on which the module is to be initialized.
        """
        super().__init__()

        if width % 2:
            raise ValueError(f"Width of rotary embeddings must be even, was: {width}")

        # Ignore allocations on the meta device as we don't persist our buffer,
        # i.e., we don't expect the backing tensor to be replaced with pretrained weights.
        if device is not None and device.type == "meta":
            device = None
        # Θ_i = 10000^(-2(i-1)/d)
        theta = torch.pow(
            base, -torch.arange(0, width, 2, dtype=torch.float, device=device) / width
        )
        self.register_buffer("theta", theta, persistent=False)

        self._create_rotary_embed(width=width, length=seq_len)

    def _create_rotary_embed(self, *, width: int, length: int):
        # mΘ
        position = torch.arange(length, device=self.theta.device).unsqueeze(1)
        m_theta = position * self.theta.unsqueeze(0)

        # We apply both sin and cos twice (see Eq 15, 34), but the ordering
        # is changed for compatibility with most common implementations.
        m_theta = torch.cat([m_theta, m_theta], dim=-1)

        re_cos = m_theta.cos().view([length, width])
        re_sin = m_theta.sin().view([length, width])

        self.register_buffer("cos", re_cos, persistent=False)
        self.register_buffer("sin", re_sin, persistent=False)

    def _rotate(self, input: torch.Tensor):
        """Rotate the input tensor by half of its innermost width.

        input (Tensor): array to rotate.
        RETURNS (Tensor): rotated array.

        Shapes:
            input - (..., width)
            output - (..., width)
        """
        half_idx = input.shape[-1] // 2
        input_1 = -input[..., half_idx:]
        input_2 = input[..., :half_idx]
        return torch.cat([input_1, input_2], dim=-1)

    def forward(self, input: torch.Tensor, *, positions: Optional[torch.Tensor] = None):
        """
        Apply rotary embeddings to an array.

        :param input: Array to apply the rotary embeddings to.
        :param positions: positions of the inputs. If no positions are
            provided, they are assumed to be [0, seq_len).
        :return: Array with the rotary embeddings applied.

        Shapes:
            input - (batch_size, num_heads, seq_len, width_per_head)
            positions - (batch_size, seq_len)
            output - (batch_size, num_heads, seq_len, width_per_head)
        """
        batch_size, _, seq_len, width = input.shape

        if positions is None:
            # Fastpath: positions from [0..seq_len), avoid indexing.
            if self.cos.size(-2) < seq_len:
                self._create_rotary_embed(width=width, length=seq_len)
            rot_cos = self.cos[:seq_len, :].view(1, 1, seq_len, width)
            rot_sin = self.sin[:seq_len, :].view(1, 1, seq_len, width)
        else:
            max_len = int(positions.max()) + 1
            if self.cos.size(-2) < max_len:
                self._create_rotary_embed(width=width, length=max_len)

            # Flatten positions to index cos/sin arrays, then unflatten.
            #
            # Example shapes:
            #
            #   positions_flat - (batch_size * seq_len)
            #   self.cos - (max_len, width)
            #   rot_cos - (batch_size, seq_len, width)
            positions_flat = positions.view(-1)
            rot_cos = self.cos[positions_flat].view(batch_size, 1, seq_len, width)
            rot_sin = self.sin[positions_flat].view(batch_size, 1, seq_len, width)

        # Eq 34 with ordering changed for compatibility.
        return rot_cos * input + rot_sin * self._rotate(input)


class RotMultiheadAttention(MultiheadAttention):
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False):
        super().__init__(embed_dim, num_heads, kdim=kdim, vdim=vdim, dropout=dropout, bias=bias,
                         add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn, self_attention=self_attention,
                         encoder_decoder_attention=encoder_decoder_attention)
        self.rotary_embeds = RotaryEmbeddings(width=embed_dim // num_heads)

    def forward(
            self,
            query, key, value,
            spk_pos_ids_flat=None,
            key_padding_mask=None,
            incremental_state=None,
            need_weights=True,
            static_kv=False,
            attn_mask=None,
            before_softmax=False,
            need_head_weights=False,
            enc_dec_attn_constraint_mask=None,
            reset_attn_weight=None
    ):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.in_proj_k(key)
                v = self.in_proj_v(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # Apply rot embedding and store incremental_state
        q = self.rotary_embeds(q[None, :], positions=spk_pos_ids_flat)[0]
        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'], saved_state['prev_value'] = k.view(bsz, self.num_heads, -1, self.head_dim), v.view(
                bsz, self.num_heads, -1, self.head_dim)
            self._set_input_buffer(incremental_state, saved_state)
        if incremental_state is not None:
            key_pos = torch.arange(k.shape[-2], device=q.device).unsqueeze(0)
        else:
            key_pos = spk_pos_ids_flat
        k = self.rotary_embeds(k[None, :], positions=key_pos)[0]

        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            if len(attn_mask.shape) == 2:
                attn_mask = attn_mask.unsqueeze(0)
            elif len(attn_mask.shape) == 3:
                attn_mask = attn_mask[:, None].repeat([1, self.num_heads, 1, 1]).reshape(
                    bsz * self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights + attn_mask

        if enc_dec_attn_constraint_mask is not None:  # bs x head x L_kv
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                enc_dec_attn_constraint_mask.unsqueeze(2).bool(),
                -1e8,
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                -1e8,
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_logits = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)

        if reset_attn_weight is not None:
            if reset_attn_weight:
                self.last_attn_probs = attn_probs.detach()
            else:
                assert self.last_attn_probs is not None
                attn_probs = self.last_attn_probs
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        else:
            attn_weights = None

        return attn, (attn_weights, attn_logits)


class RotMultiheadAttention2(MultiheadAttention):
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False):
        super().__init__(embed_dim, num_heads, kdim=kdim, vdim=vdim, dropout=dropout, bias=bias,
                         add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn, self_attention=self_attention,
                         encoder_decoder_attention=encoder_decoder_attention)
        self.rotary_embeds = RotaryEmbeddings(width=embed_dim // num_heads)

    def forward(
            self,
            query, key, value,
            spk_pos_ids_flat=None,
            key_padding_mask=None,
            incremental_state=None,
            need_weights=True,
            static_kv=False,
            attn_mask=None,
            before_softmax=False,
            need_head_weights=False,
            enc_dec_attn_constraint_mask=None,
            reset_attn_weight=None
    ):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.in_proj_k(key)
                v = self.in_proj_v(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # Apply rot embedding and store incremental_state
        q = self.rotary_embeds(q[None, :], positions=spk_pos_ids_flat)[0]
        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'], saved_state['prev_value'] = k.view(bsz, self.num_heads, -1, self.head_dim), v.view(
                bsz, self.num_heads, -1, self.head_dim)
            self._set_input_buffer(incremental_state, saved_state)
        key_pos = torch.arange(k.shape[-2], device=q.device).unsqueeze(0)
        k = self.rotary_embeds(k[None, :], positions=key_pos)[0]

        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if attn_mask is not None:
            if len(attn_mask.shape) == 2:
                attn_mask = attn_mask.unsqueeze(0)
            elif len(attn_mask.shape) == 3:
                attn_mask = attn_mask[:, None].repeat([1, self.num_heads, 1, 1]).reshape(
                    bsz * self.num_heads, tgt_len, src_len)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0, is_causal=False)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_logits = None
        attn_weights = None
        return attn, (attn_weights, attn_logits)


class RotDecSALayer(nn.Module):
    def __init__(self, c, num_heads, dropout, attention_dropout=0.1, relu_dropout=0.1,
                 kernel_size=9, ffn_hidden_size=1024, act='gelu', post_ln=False, bias=True):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.layer_norm1 = LayerNorm(c)
        self.self_attn = RotMultiheadAttention(
            c, num_heads, self_attention=True, dropout=attention_dropout, bias=False
        )
        self.layer_norm2 = LayerNorm(c)
        self.ffn = TransformerFFNLayer(
            c, ffn_hidden_size, padding='LEFT', kernel_size=kernel_size,
            dropout=relu_dropout, act=act, bias=bias)
        self.post_ln = post_ln

    def forward(
            self,
            x,
            encoder_out=None,
            encoder_padding_mask=None,
            incremental_state=None,
            self_attn_mask=None,
            self_attn_padding_mask=None,
            attn_out=None,
            reset_attn_weight=None,
            spk_pos_ids_flat=None,
            **kwargs,
    ):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        residual = x
        if not self.post_ln:
            x = self.layer_norm1(x)

        x, (attn_weights, _) = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            attn_mask=self_attn_mask,
            spk_pos_ids_flat=spk_pos_ids_flat
        )
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.layer_norm1(x)

        residual = x
        if not self.post_ln:
            x = self.layer_norm2(x)
        x = self.ffn(x, incremental_state=incremental_state)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.layer_norm2(x)
        return x, attn_weights

    def clear_buffer(self, input, encoder_out=None, encoder_padding_mask=None, incremental_state=None):
        self.encoder_attn.clear_buffer(incremental_state)
        self.ffn.clear_buffer(incremental_state)

    def set_buffer(self, name, tensor, incremental_state):
        return set_incremental_state(self, incremental_state, name, tensor)


class RotDecSALayer2(RotDecSALayer):
    def __init__(self, c, num_heads, dropout, attention_dropout=0.1, relu_dropout=0.1, kernel_size=9,
                 ffn_hidden_size=1024, act='gelu', post_ln=False):
        super().__init__(c, num_heads, dropout, attention_dropout, relu_dropout, kernel_size, ffn_hidden_size, act,
                         post_ln)
        self.self_attn = RotMultiheadAttention2(
            c, num_heads, self_attention=True, dropout=attention_dropout, bias=False
        )


class RotTransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout, kernel_size=9, num_heads=8, ffn_hidden_size=1024, post_ln=False,
                 op_version=1, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        if op_version == 1:
            self.op = RotDecSALayer(
                hidden_size, num_heads, dropout=dropout,
                attention_dropout=0.0, relu_dropout=dropout,
                kernel_size=kernel_size, ffn_hidden_size=ffn_hidden_size,
                post_ln=post_ln, bias=bias)
        else:
            self.op = RotDecSALayer2(
                hidden_size, num_heads, dropout=dropout,
                attention_dropout=0.0, relu_dropout=dropout,
                kernel_size=kernel_size, ffn_hidden_size=ffn_hidden_size,
                post_ln=post_ln)

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)

    def clear_buffer(self, *args):
        return self.op.clear_buffer(*args)

    def set_buffer(self, *args):
        return self.op.set_buffer(*args)
