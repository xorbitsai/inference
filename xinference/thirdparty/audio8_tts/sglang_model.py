# SPDX-License-Identifier: Apache-2.0
"""SGLang-native slow AR plus Audio8's fixed-length fast codebook head."""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from torch import Tensor, nn

from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.layers import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RadixAttention,
    RMSNorm,
    RowParallelLinear,
    VocabParallelEmbedding,
    get_rope,
)
from xinference.thirdparty.audio8_tts.attention_backend import resolve_attention_backend
from sglang_omni.vendor.sglang.models import apply_qk_norm
from sglang_omni.vendor.sglang.utils import make_layers

logger = logging.getLogger(__name__)


@torch.library.custom_op(
    "audio8::fast_attention_with_cache",
    mutates_args=("key_cache", "value_cache"),
)
def _audio8_fast_attention_with_cache(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    key: Tensor,
    value: Tensor,
    cache_positions: Tensor,
) -> Tensor:
    attention_backend = resolve_attention_backend()
    if attention_backend == "fa3":
        return flash_attn_with_kvcache(
            q=query,
            k_cache=key_cache,
            v_cache=value_cache,
            k=key,
            v=value,
            cache_seqlens=cache_positions.contiguous(),
            causal=True,
            num_splits=0,
        )

    batch = query.shape[0]
    positions = cache_positions.to(torch.long)
    batch_indices = torch.arange(batch, device=query.device)
    key_cache[batch_indices, positions] = key[:, 0]
    value_cache[batch_indices, positions] = value[:, 0]

    sequence_length = key_cache.shape[1]
    valid_prefix = torch.arange(
        sequence_length,
        device=query.device,
    )[None] <= positions[:, None]
    output = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key_cache.transpose(1, 2),
        value_cache.transpose(1, 2),
        attn_mask=valid_prefix[:, None, None],
        is_causal=False,
        enable_gqa=True,
    )
    return output.transpose(1, 2).contiguous()


@_audio8_fast_attention_with_cache.register_fake
def _audio8_fast_attention_with_cache_fake(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    key: Tensor,
    value: Tensor,
    cache_positions: Tensor,
) -> Tensor:
    del key_cache, value_cache, key, value, cache_positions
    return torch.empty_like(query)


class Audio8FastKVCache(nn.Module):
    def __init__(
        self,
        max_batch_size: int,
        max_sequence_length: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        shape = (
            max_batch_size,
            max_sequence_length,
            num_kv_heads,
            head_dim,
        )
        self.register_buffer(
            "key",
            torch.zeros(shape, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "value",
            torch.zeros(shape, device=device, dtype=dtype),
            persistent=False,
        )

    def for_batch(self, batch_size: int) -> tuple[Tensor, Tensor]:
        return self.key[:batch_size], self.value[:batch_size]

    def clear(self) -> None:
        self.key.zero_()
        self.value.zero_()


class Audio8Attention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int,
        rope_base: float,
        max_position_embeddings: int,
        rms_norm_eps: float,
        qkv_bias: bool,
        output_bias: bool,
        qk_norm: bool,
    ) -> None:
        super().__init__()
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.head_dim = head_dim
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            num_heads * head_dim,
            hidden_size,
            bias=output_bias,
        )
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_base,
            is_neox_style=False,
        )
        self.attn = RadixAttention(
            num_heads,
            head_dim,
            head_dim**-0.5,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
        )
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        forward_batch: ForwardBatch,
    ) -> Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.qk_norm:
            q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(output)
        return output


class Audio8DecoderLayer(nn.Module):
    def __init__(self, config: Any, layer_id: int) -> None:
        super().__init__()
        self.self_attn = Audio8Attention(
            hidden_size=config.dim,
            num_heads=config.n_head,
            num_kv_heads=config.n_local_heads,
            head_dim=config.head_dim,
            layer_id=layer_id,
            rope_base=config.rope_base,
            max_position_embeddings=config.max_seq_len,
            rms_norm_eps=config.norm_eps,
            qkv_bias=config.attention_qkv_bias,
            output_bias=config.attention_o_bias,
            qk_norm=config.attention_qk_norm,
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            config.dim,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.dim,
            bias=False,
        )
        self.input_layernorm = RMSNorm(config.dim, eps=config.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = F.silu(gate) * up
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states, residual


class FastRMSNorm(RMSNorm):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__(dim, eps=eps)

    def forward(self, value: Tensor) -> Tensor:
        shape = value.shape
        normalized = super().forward(value.reshape(-1, shape[-1]))
        return normalized.view(shape)


def _rope(length: int, head_dim: int, base: float, device: torch.device) -> Tensor:
    frequencies = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    phases = torch.outer(torch.arange(length, device=device), frequencies)
    values = torch.polar(torch.ones_like(phases), phases)
    return torch.stack((values.real, values.imag), dim=-1).to(torch.bfloat16)


def _apply_rope(value: Tensor, rope_values: Tensor) -> Tensor:
    shaped = value.float().reshape(*value.shape[:-1], -1, 2)
    rope_values = rope_values[None, :, None]
    output = torch.stack(
        (
            shaped[..., 0] * rope_values[..., 0]
            - shaped[..., 1] * rope_values[..., 1],
            shaped[..., 1] * rope_values[..., 0]
            + shaped[..., 0] * rope_values[..., 1],
        ),
        dim=-1,
    )
    return output.flatten(3).to(value.dtype)


class FastAttention(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        total = (config.fast_n_head + 2 * config.fast_n_local_heads) * config.fast_head_dim
        self.wqkv = nn.Linear(
            config.fast_dim,
            total,
            bias=config.fast_attention_qkv_bias,
        )
        self.wo = nn.Linear(
            config.fast_n_head * config.fast_head_dim,
            config.fast_dim,
            bias=config.fast_attention_o_bias,
        )
        self.n_head = config.fast_n_head
        self.n_local_heads = config.fast_n_local_heads
        self.head_dim = config.fast_head_dim
        self.qk_norm = config.fast_attention_qk_norm
        self.audio8_cache: Audio8FastKVCache | None = None
        if self.qk_norm:
            self.q_norm = FastRMSNorm(self.head_dim, config.norm_eps)
            self.k_norm = FastRMSNorm(self.head_dim, config.norm_eps)

    def setup_audio8_cache(
        self,
        max_batch_size: int,
        max_sequence_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.audio8_cache = Audio8FastKVCache(
            max_batch_size,
            max_sequence_length,
            self.n_local_heads,
            self.head_dim,
            device=device,
            dtype=dtype,
        )

    def clear_audio8_cache(self) -> None:
        if self.audio8_cache is None:
            raise RuntimeError("Audio8 fast KV cache has not been initialized")
        self.audio8_cache.clear()

    def forward(self, value: Tensor, rope_values: Tensor) -> Tensor:
        batch, length, _ = value.shape
        q_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(value).split((q_size, kv_size, kv_size), dim=-1)
        q = q.view(batch, length, self.n_head, self.head_dim)
        k = k.view(batch, length, self.n_local_heads, self.head_dim)
        v = v.view(batch, length, self.n_local_heads, self.head_dim)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = _apply_rope(q, rope_values).transpose(1, 2)
        k = _apply_rope(k, rope_values).transpose(1, 2)
        v = v.transpose(1, 2)
        repeat = self.n_head // self.n_local_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        scores = q @ k.transpose(-2, -1) / (self.head_dim**0.5)
        causal = torch.ones(
            length,
            length,
            dtype=torch.bool,
            device=value.device,
        ).tril()
        scores = scores.masked_fill(~causal[None, None], -float("inf"))
        output = torch.softmax(scores, dim=-1) @ v
        return self.wo(output.transpose(1, 2).contiguous().view(batch, length, q_size))

    def forward_audio8_cached(
        self,
        value: Tensor,
        rope_values: Tensor,
        cache_positions: Tensor,
    ) -> Tensor:
        batch, length, _ = value.shape
        if length != 1:
            raise ValueError("Audio8 fast KV-cache decode expects one token per step")
        q_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(value).split((q_size, kv_size, kv_size), dim=-1)
        q = q.view(batch, length, self.n_head, self.head_dim)
        k = k.view(batch, length, self.n_local_heads, self.head_dim)
        v = v.view(batch, length, self.n_local_heads, self.head_dim)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = _apply_rope(q, rope_values)
        k = _apply_rope(k, rope_values)
        if self.audio8_cache is None:
            raise RuntimeError("Audio8 fast KV cache has not been initialized")
        key_cache, value_cache = self.audio8_cache.for_batch(batch)
        output = _audio8_fast_attention_with_cache(
            q,
            key_cache,
            value_cache,
            k,
            v,
            cache_positions,
        )
        return self.wo(output.contiguous().view(batch, length, q_size))


class FastFeedForward(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.fast_dim, config.fast_intermediate_size, bias=False)
        self.w2 = nn.Linear(config.fast_intermediate_size, config.fast_dim, bias=False)
        self.w3 = nn.Linear(config.fast_dim, config.fast_intermediate_size, bias=False)
        self.gate_up: nn.Linear | None = None

    def fuse_gate_up(self) -> None:
        if self.gate_up is not None:
            return
        gate_up = nn.Linear(
            self.w1.in_features,
            self.w1.out_features * 2,
            bias=False,
            device=self.w1.weight.device,
            dtype=self.w1.weight.dtype,
        )
        with torch.no_grad():
            gate_up.weight.copy_(torch.cat((self.w1.weight, self.w3.weight), dim=0))
        self.gate_up = gate_up
        del self.w1
        del self.w3

    def forward(self, value: Tensor) -> Tensor:
        if self.gate_up is None:
            gate = self.w1(value)
            up = self.w3(value)
        else:
            gate, up = self.gate_up(value).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class FastDecoderLayer(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.attention = FastAttention(config)
        self.feed_forward = FastFeedForward(config)
        self.ffn_norm = FastRMSNorm(config.fast_dim, config.norm_eps)
        self.attention_norm = FastRMSNorm(config.fast_dim, config.norm_eps)

    def forward(self, value: Tensor, rope_values: Tensor) -> Tensor:
        hidden = value + self.attention(self.attention_norm(value), rope_values)
        return hidden + self.feed_forward(self.ffn_norm(hidden))

    def forward_audio8_cached(
        self,
        value: Tensor,
        rope_values: Tensor,
        cache_positions: Tensor,
    ) -> Tensor:
        hidden = value + self.attention.forward_audio8_cached(
            self.attention_norm(value),
            rope_values,
            cache_positions,
        )
        return hidden + self.feed_forward(self.ffn_norm(hidden))


class Audio8SGLangModel(nn.Module):
    def __init__(self, config: Any, quant_config: Any = None) -> None:
        super().__init__()
        del quant_config
        self.config = config
        self.vocab_size = int(config.vocab_size)
        self.hidden_size = int(config.dim)
        self.num_layers = int(config.n_layer)
        self.tie_word_embeddings = bool(config.tie_word_embeddings)
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.dim)
        self.codebook_embeddings = nn.Embedding(
            config.codebook_size * config.num_codebooks,
            config.dim,
        )
        self.start_layer = 0
        self.end_layer = config.n_layer
        self.layers = make_layers(
            config.n_layer,
            lambda idx, prefix: Audio8DecoderLayer(config, idx),
            prefix="layers",
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        self.fast_project_in = (
            nn.Linear(config.dim, config.fast_dim)
            if config.fast_dim != config.dim
            else nn.Identity()
        )
        self.fast_embeddings = nn.Embedding(config.codebook_size, config.fast_dim)
        self.fast_layers = nn.ModuleList(
            [FastDecoderLayer(config) for _ in range(config.n_fast_layer)]
        )
        self.fast_norm = FastRMSNorm(config.fast_dim, config.norm_eps)
        self.fast_output = nn.Linear(config.fast_dim, config.codebook_size, bias=False)
        self._decode_ready = False

    def setup_audio8_decode(self, max_batch_size: int) -> None:
        device = self.embed_tokens.weight.device
        config = self.config
        self.register_buffer(
            "_codebook_offsets",
            torch.arange(config.num_codebooks, device=device, dtype=torch.long)
            * config.codebook_size,
            persistent=False,
        )
        self.register_buffer(
            "_vq_codes",
            torch.zeros(max_batch_size, config.num_codebooks, device=device, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_vq_mask",
            torch.zeros(max_batch_size, device=device, dtype=torch.bool),
            persistent=False,
        )
        semantic_bias = torch.full(
            (config.vocab_size,),
            -float("inf"),
            device=device,
            dtype=torch.float32,
        )
        semantic_bias[config.semantic_begin_id : config.semantic_end_id + 1] = 0.0
        semantic_bias[config.eos_token_id] = 0.0
        self.register_buffer("_semantic_bias", semantic_bias, persistent=False)
        self.register_buffer(
            "_output_codes",
            torch.zeros(
                max_batch_size,
                config.num_codebooks + 1,
                device=device,
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_output_semantic_ids",
            torch.zeros(max_batch_size, device=device, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_temperature",
            torch.full((max_batch_size,), 0.8, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_top_p",
            torch.full((max_batch_size,), 0.95, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_top_k",
            torch.full((max_batch_size,), 50, device=device, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_do_sample",
            torch.ones(max_batch_size, device=device, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "_previous_semantic",
            torch.zeros(
                max_batch_size,
                config.ras_window_size,
                device=device,
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_previous_valid",
            torch.zeros(
                max_batch_size,
                config.ras_window_size,
                device=device,
                dtype=torch.bool,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_fast_rope",
            _rope(
                config.num_codebooks,
                config.fast_head_dim,
                config.rope_base,
                device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_audio8_fast_position",
            torch.zeros(1, device=device, dtype=torch.long),
            persistent=False,
        )
        for layer in self.fast_layers:
            layer.feed_forward.fuse_gate_up()
            layer.attention.setup_audio8_cache(
                max_batch_size,
                config.num_codebooks + 1,
                device=device,
                dtype=self.embed_tokens.weight.dtype,
            )
        self._decode_ready = True

    def _embed_decode(self, input_ids: Tensor) -> Tensor:
        hidden = self.embed_tokens(input_ids)
        batch = hidden.shape[0]
        codes = self._vq_codes[:batch] + self._codebook_offsets[None]
        codebook_sum = self.codebook_embeddings(codes).sum(dim=1).to(hidden.dtype)
        return torch.where(
            self._vq_mask[:batch, None],
            hidden + codebook_sum,
            hidden,
        )

    @staticmethod
    def _sample(
        scores: Tensor,
        temperature: Tensor,
        top_p: Tensor,
        top_k: Tensor,
        do_sample: Tensor,
    ) -> Tensor:
        if os.getenv("AUDIO8_TTS_GREEDY_FASTPATH", "0") == "1":
            return scores.argmax(dim=-1)
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
        positions = torch.arange(scores.shape[-1], device=scores.device)[None]
        remove_sorted = (cumulative > top_p[:, None]) | (positions >= top_k[:, None])
        remove_sorted[:, 0] = False
        remove = torch.zeros_like(remove_sorted).scatter(1, sorted_indices, remove_sorted)
        filtered = scores.masked_fill(remove, -float("inf"))
        filtered = filtered / temperature[:, None].clamp_min(1e-5)
        greedy = filtered.argmax(dim=-1)
        probabilities = torch.softmax(filtered, dim=-1)
        random = torch.rand_like(probabilities).clamp_min(torch.finfo(probabilities.dtype).tiny)
        sampled = torch.argmax(probabilities / (-torch.log(random)), dim=-1)
        return torch.where(do_sample, sampled, greedy)

    def _sample_semantic(self, logits: Tensor) -> Tensor:
        batch = logits.shape[0]
        scores = logits.float()
        normal_index = self._sample(
            scores,
            self._temperature[:batch],
            self._top_p[:batch],
            self._top_k[:batch],
            self._do_sample[:batch],
        )
        ras_temperature = torch.full_like(
            self._temperature[:batch], self.config.ras_temperature
        )
        ras_top_p = torch.full_like(self._top_p[:batch], self.config.ras_top_p)
        high_index = self._sample(
            scores,
            ras_temperature,
            ras_top_p,
            self._top_k[:batch],
            self._do_sample[:batch],
        )
        semantic_begin = self.config.semantic_begin_id
        normal = torch.where(
            normal_index == 0,
            self.config.eos_token_id,
            semantic_begin + normal_index - 1,
        )
        high = torch.where(
            high_index == 0,
            self.config.eos_token_id,
            semantic_begin + high_index - 1,
        )
        repeated = (
            (self._previous_semantic[:batch] == normal[:, None])
            & self._previous_valid[:batch]
        ).any(dim=1)
        is_semantic = (normal >= self.config.semantic_begin_id) & (
            normal <= self.config.semantic_end_id
        )
        return torch.where(repeated & is_semantic, high, normal)

    def _semantic_logits(self, hidden_states: Tensor) -> Tensor:
        weight = self.embed_tokens.weight
        semantic_weight = weight[
            self.config.semantic_begin_id : self.config.semantic_end_id + 1
        ]
        eos_weight = weight[
            self.config.eos_token_id : self.config.eos_token_id + 1
        ]
        return torch.cat(
            (
                F.linear(hidden_states, eos_weight),
                F.linear(hidden_states, semantic_weight),
            ),
            dim=-1,
        )

    def _clear_audio8_fast_caches(self) -> None:
        for layer in self.fast_layers:
            layer.attention.clear_audio8_cache()

    def _run_audio8_fast_position(self, hidden: Tensor, position: int) -> Tensor:
        batch = hidden.shape[0]
        self._audio8_fast_position.fill_(position)
        rope_values = self._fast_rope[self._audio8_fast_position]
        cache_positions = self._audio8_fast_position.expand(batch).to(torch.int32)
        for layer in self.fast_layers:
            hidden = layer.forward_audio8_cached(
                hidden,
                rope_values,
                cache_positions,
            )
        return self.fast_output(self.fast_norm(hidden))[:, -1]

    @torch.no_grad()
    def _decode_codebooks(self, logits: Tensor, slow_hidden: Tensor) -> None:
        batch = logits.shape[0]
        semantic = self._sample_semantic(logits)
        current = (semantic - self.config.semantic_begin_id).clamp(
            0, self.config.codebook_size - 1
        )
        self._output_codes[:batch, 0] = semantic
        self._output_codes[:batch, 1] = current
        self._clear_audio8_fast_caches()
        fast_hidden = self.fast_project_in(slow_hidden).unsqueeze(1)
        self._run_audio8_fast_position(fast_hidden, 0)
        for position in range(1, self.config.num_codebooks):
            fast_hidden = self.fast_embeddings(current).unsqueeze(1)
            scores = self._run_audio8_fast_position(fast_hidden, position).float()
            current = self._sample(
                scores,
                self._temperature[:batch],
                self._top_p[:batch],
                self._top_k[:batch],
                self._do_sample[:batch],
            )
            self._output_codes[:batch, position + 1] = current
        self._output_semantic_ids[:batch] = semantic

    @torch.inference_mode()
    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[Tensor] = None,
    ) -> LogitsProcessorOutput:
        if input_embeds is None and forward_batch.input_embeds is not None:
            input_embeds = forward_batch.input_embeds
        if input_embeds is not None:
            hidden_states = input_embeds
        elif self._decode_ready:
            hidden_states = self._embed_decode(input_ids)
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None
        for layer_index in range(self.start_layer, self.end_layer):
            hidden_states, residual = self.layers[layer_index](
                positions, hidden_states, forward_batch, residual
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if forward_batch.forward_mode.is_extend():
            last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            hidden_states = hidden_states[last_index]
        if self._decode_ready:
            logits = self._semantic_logits(hidden_states)
            self._decode_codebooks(logits, hidden_states)
        else:
            logits = F.linear(hidden_states, self.embed_tokens.weight)
        return LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=hidden_states,
        )

    def get_embed_tokens(self) -> nn.Module:
        return self.embed_tokens

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> None:
        params = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if name in {"freqs_cis", "fast_freqs_cis"}:
                continue
            if self._load_slow_weight(name, loaded_weight, params):
                continue
            target_name = "embed_tokens.weight" if name == "embeddings.weight" else name
            target = params.get(target_name)
            if target is None:
                logger.debug("Skipping Audio8 weight: %s", name)
                continue
            loader = getattr(target, "weight_loader", _default_weight_loader)
            loader(target, loaded_weight)

    def _load_slow_weight(
        self,
        name: str,
        loaded_weight: Tensor,
        params: dict[str, nn.Parameter],
    ) -> bool:
        if not name.startswith("layers."):
            return False
        remap: dict[str, Any] = {
            "attention.wqkv.weight": None,
            "attention.wqkv.bias": None,
            "attention.wo.weight": "self_attn.o_proj.weight",
            "attention.wo.bias": "self_attn.o_proj.bias",
            "attention.q_norm.weight": "self_attn.q_norm.weight",
            "attention.k_norm.weight": "self_attn.k_norm.weight",
            "attention_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "feed_forward.w1.weight": ("gate_up_proj.weight", 0),
            "feed_forward.w3.weight": ("gate_up_proj.weight", 1),
            "feed_forward.w2.weight": "down_proj.weight",
        }
        for source_suffix, target in remap.items():
            if not name.endswith(source_suffix):
                continue
            prefix = name[: -len(source_suffix)]
            if target is None:
                self._load_fused_qkv(
                    prefix,
                    loaded_weight,
                    params,
                    is_bias=source_suffix.endswith("bias"),
                )
                return True
            if isinstance(target, tuple):
                target_suffix, shard_id = target
            else:
                target_suffix, shard_id = target, None
            parameter = params[prefix + target_suffix]
            if shard_id is None:
                loader = getattr(parameter, "weight_loader", _default_weight_loader)
                loader(parameter, loaded_weight)
            else:
                parameter.weight_loader(parameter, loaded_weight, shard_id)
            return True
        return False

    def _load_fused_qkv(
        self,
        prefix: str,
        loaded_weight: Tensor,
        params: dict[str, nn.Parameter],
        *,
        is_bias: bool,
    ) -> None:
        target_name = prefix + "self_attn.qkv_proj." + ("bias" if is_bias else "weight")
        parameter = params[target_name]
        layer = self.layers[int(prefix.split(".")[1])]
        q, k, v = loaded_weight.split(
            [layer.self_attn.q_size, layer.self_attn.kv_size, layer.self_attn.kv_size],
            dim=0,
        )
        for shard_id, weight in (("q", q), ("k", k), ("v", v)):
            parameter.weight_loader(parameter, weight, shard_id)


def _default_weight_loader(parameter: nn.Parameter, loaded_weight: Tensor) -> None:
    parameter.data.copy_(loaded_weight)


EntryClass = Audio8SGLangModel
