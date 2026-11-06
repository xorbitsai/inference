# SPDX-License-Identifier: Apache-2.0
"""Audio8 0.1B (Falcon-H1 hybrid) slow AR on the SGLang-Omni engine.

The 0.1B preview swaps the pure-attention slow backbone for a Falcon-H1
``Mamba + attention`` hybrid. SGLang's paged attention cannot cache Mamba
state, so this model runs the slow backbone eagerly with a per-request
``FalconHybridMambaAttentionDynamicCache`` (attention KV + Mamba conv/SSM
states). The fast codebook head, semantic sampling and the vocoder path are
shared with the 0.6B adapter.

The AR runtime keeps one hybrid cache on each request
(``Audio8SGLangRequestData._slow_cache``) and hands the current batch's caches
to the model through ``_batch_slow_caches`` before every forward, aligned with
the scheduler request order. Prefill runs the full prompt through the eager
Falcon-H1 backbone and seeds the cache; each decode step runs a single token
through the cache. No SGLang KV pages are consumed, so CUDA graphs are disabled
for this path (see ``factory.create_audio8_engine``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from torch import Tensor, nn

from sglang_omni.vendor.sglang.core import ForwardBatch

from xinference.thirdparty.audio8_tts.sglang_model import (
    FastDecoderLayer,
    FastRMSNorm,
    _default_weight_loader,
    _rope,
)

logger = logging.getLogger(__name__)


def _build_falcon_config(config: Any) -> Any:
    """Translate an ArkttsConfig into the Falcon-H1 config used by the model.

    Mirrors ``ArkttsModel._build_falcon_config`` in the 0.1B checkpoint's
    ``modeling_arktts.py``.
    """
    from transformers.models.falcon_h1.configuration_falcon_h1 import FalconH1Config

    return FalconH1Config(
        vocab_size=config.vocab_size,
        hidden_size=config.dim,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.n_layer,
        num_attention_heads=config.n_head,
        num_key_value_heads=config.n_local_heads,
        head_dim=config.head_dim,
        hidden_act=config.hidden_act,
        rms_norm_eps=config.norm_eps,
        rope_theta=config.rope_base,
        max_position_embeddings=config.max_seq_len,
        attention_bias=config.attention_bias,
        attention_dropout=config.attention_dropout,
        attention_in_multiplier=config.attention_in_multiplier,
        attention_out_multiplier=config.attention_out_multiplier,
        key_multiplier=config.key_multiplier,
        embedding_multiplier=config.embedding_multiplier,
        lm_head_multiplier=config.lm_head_multiplier,
        expansion_factor=config.expansion_factor,
        mlp_bias=config.mlp_bias,
        mlp_multipliers=config.mlp_multipliers,
        mamba_chunk_size=config.mamba_chunk_size,
        mamba_conv_bias=config.mamba_conv_bias,
        mamba_d_conv=config.mamba_d_conv,
        mamba_d_head=config.mamba_d_head,
        mamba_d_ssm=config.mamba_d_ssm,
        mamba_d_state=config.mamba_d_state,
        mamba_expand=config.mamba_expand,
        mamba_n_groups=config.mamba_n_groups,
        mamba_n_heads=config.mamba_n_heads,
        mamba_norm_before_gate=config.mamba_norm_before_gate,
        mamba_proj_bias=config.mamba_proj_bias,
        mamba_rms_norm=config.mamba_rms_norm,
        mamba_use_mlp=config.mamba_use_mlp,
        projectors_bias=config.projectors_bias,
        ssm_in_multiplier=config.ssm_in_multiplier,
        ssm_multipliers=config.ssm_multipliers,
        ssm_out_multiplier=config.ssm_out_multiplier,
        time_step_floor=config.time_step_floor,
        time_step_max=config.time_step_max,
        time_step_min=config.time_step_min,
        time_step_rank=config.time_step_rank,
        initializer_range=config.initializer_range,
        use_cache=config.use_cache,
        tie_word_embeddings=config.tie_word_embeddings,
        pad_token_id=config.pad_token_id,
        eos_token_id=config.eos_token_id,
        bos_token_id=config.bos_token_id,
    )


class Audio8HybridSGLangModel(nn.Module):
    """Eager Falcon-H1 slow backbone plus Audio8's fixed-length fast head.

    The slow backbone is the checkpoint's ``FalconH1Model`` (Mamba + attention)
    run eagerly with one ``FalconHybridMambaAttentionDynamicCache`` per request.
    The fast AR branch, semantic sampling and codebook embeddings mirror the
    existing ``Audio8SGLangModel``.
    """

    def __init__(self, config: Any, quant_config: Any = None) -> None:
        super().__init__()
        del quant_config
        self.config = config
        self.vocab_size = int(config.vocab_size)
        self.hidden_size = int(config.dim)
        self.num_layers = int(config.n_layer)
        self.tie_word_embeddings = bool(config.tie_word_embeddings)

        from transformers.models.falcon_h1.modeling_falcon_h1 import FalconH1Model

        self.slow = FalconH1Model(_build_falcon_config(config))
        self.slow.eval()

        self.codebook_embeddings = nn.Embedding(
            config.codebook_size * config.num_codebooks,
            config.dim,
        )
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
        # Compact slow-AR head: 4096 semantic tokens + 1 EOS, matching the
        # checkpoint's ``semantic_output`` layout (index codebook_size = EOS).
        self.semantic_output = nn.Linear(
            config.dim, config.codebook_size + 1, bias=False
        )
        self._decode_ready = False
        # Per-request Falcon-H1 hybrid caches; set by the AR runtime before
        # every forward (aligned with the scheduler request order).
        self._batch_slow_caches: list[Any] = []

    @property
    def embed_tokens(self) -> nn.Module:
        return self.slow.embed_tokens

    def get_embed_tokens(self) -> nn.Module:
        return self.slow.embed_tokens

    # ------------------------------------------------------------------
    # Decode buffers (shared layout with Audio8SGLangModel.setup_audio8_decode)
    # ------------------------------------------------------------------

    def setup_audio8_decode(self, max_batch_size: int) -> None:
        device = self.slow.embed_tokens.weight.device
        config = self.config
        self.register_buffer(
            "_codebook_offsets",
            torch.arange(config.num_codebooks, device=device, dtype=torch.long)
            * config.codebook_size,
            persistent=False,
        )
        self.register_buffer(
            "_vq_codes",
            torch.zeros(
                max_batch_size, config.num_codebooks, device=device, dtype=torch.long
            ),
            persistent=False,
        )
        self.register_buffer(
            "_vq_mask",
            torch.zeros(max_batch_size, device=device, dtype=torch.bool),
            persistent=False,
        )
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
                dtype=self.slow.embed_tokens.weight.dtype,
            )
        self._decode_ready = True

    # ------------------------------------------------------------------
    # Fast AR (shared with Audio8SGLangModel)
    # ------------------------------------------------------------------

    def _embed_decode(self, input_ids: Tensor) -> Tensor:
        hidden = self.slow.embed_tokens(input_ids)
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
        """Sample compact semantic logits in the checkpoint's layout.

        ``semantic_output`` emits ``[codebook_size + 1]`` logits where indices
        ``0..codebook_size-1`` are semantic tokens (offset by
        ``semantic_begin_id``) and index ``codebook_size`` is the EOS token.
        """
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
        eos_idx = self.config.codebook_size
        begin = self.config.semantic_begin_id
        normal = torch.where(
            normal_index == eos_idx,
            self.config.eos_token_id,
            begin + normal_index,
        )
        high = torch.where(
            high_index == eos_idx,
            self.config.eos_token_id,
            begin + high_index,
        )
        repeated = (
            (self._previous_semantic[:batch] == normal[:, None])
            & self._previous_valid[:batch]
        ).any(dim=1)
        is_semantic = (normal >= begin) & (normal <= self.config.semantic_end_id)
        return torch.where(repeated & is_semantic, high, normal)

    def _semantic_logits(self, hidden_states: Tensor) -> Tensor:
        return self.semantic_output(hidden_states)

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

    # ------------------------------------------------------------------
    # Slow AR (eager Falcon-H1 hybrid)
    # ------------------------------------------------------------------

    def _new_slow_cache(self, device: torch.device, dtype: torch.dtype) -> Any:
        from transformers.models.falcon_h1.modeling_falcon_h1 import (
            FalconHybridMambaAttentionDynamicCache,
        )

        falcon_config = _build_falcon_config(self.config)
        return FalconHybridMambaAttentionDynamicCache(
            falcon_config,
            1,
            dtype,
            devices=[
                self.slow.layers[i].mamba.conv1d.weight.device
                for i in range(falcon_config.num_hidden_layers)
            ],
        )

    def _slow_prefill(
        self,
        input_ids: Tensor,
        input_embeds: Optional[Tensor],
        forward_batch: ForwardBatch,
    ) -> Tensor:
        """Run the full prompt through Falcon-H1 and seed per-request caches."""
        seq_lens = forward_batch.extend_seq_lens
        batch = len(self._batch_slow_caches)
        starts = torch.cumsum(
            torch.cat(
                (
                    torch.zeros(1, dtype=torch.long, device=seq_lens.device),
                    seq_lens[:-1],
                )
            ),
            dim=0,
        )
        multiplier = self.slow.embedding_multiplier
        hidden_list: list[Tensor] = []
        for index in range(batch):
            length = int(seq_lens[index])
            start = int(starts[index])
            cache = self._batch_slow_caches[index]
            if cache is None:
                cache = self._new_slow_cache(
                    self.slow.embed_tokens.weight.device,
                    self.slow.embed_tokens.weight.dtype,
                )
            positions = torch.arange(length, device=input_ids.device)
            attention_mask = torch.ones(
                1, length, device=input_ids.device, dtype=torch.long
            )
            if input_embeds is not None:
                # ``_inject_reference_embeds`` provides embeddings without the
                # Falcon-H1 embedding multiplier, so apply it here.
                embeds = input_embeds[start : start + length].unsqueeze(0) * multiplier
                outputs = self.slow(
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    position_ids=positions.unsqueeze(0),
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=positions,
                )
            else:
                tokens = input_ids[start : start + length].unsqueeze(0)
                outputs = self.slow(
                    input_ids=tokens,
                    attention_mask=attention_mask,
                    position_ids=positions.unsqueeze(0),
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=positions,
                )
            self._batch_slow_caches[index] = outputs.past_key_values
            hidden_list.append(outputs.last_hidden_state[:, -1])
        return torch.cat(hidden_list, dim=0)

    def _slow_decode(self, input_ids: Tensor, forward_batch: ForwardBatch) -> Tensor:
        """Run one cached token per request through Falcon-H1."""
        del forward_batch
        embeds = self._embed_decode(input_ids)
        embeds = embeds * self.slow.embedding_multiplier
        hidden_list: list[Tensor] = []
        batch = len(self._batch_slow_caches)
        for index in range(batch):
            cache = self._batch_slow_caches[index]
            if cache is None:
                raise RuntimeError(
                    "Audio8 hybrid slow cache is missing for a decode request"
                )
            position = cache.get_seq_length()
            position_t = torch.tensor(
                [position], device=input_ids.device, dtype=torch.long
            )
            attention_mask = torch.ones(
                1, position + 1, device=input_ids.device, dtype=torch.long
            )
            outputs = self.slow(
                inputs_embeds=embeds[index : index + 1].unsqueeze(1),
                attention_mask=attention_mask,
                position_ids=position_t.unsqueeze(0),
                past_key_values=cache,
                use_cache=True,
                cache_position=position_t,
            )
            self._batch_slow_caches[index] = outputs.past_key_values
            hidden_list.append(outputs.last_hidden_state[:, -1])
        return torch.cat(hidden_list, dim=0)

    # ------------------------------------------------------------------
    # SGLang model interface
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[Tensor] = None,
    ) -> LogitsProcessorOutput:
        del positions
        if input_embeds is None and forward_batch.input_embeds is not None:
            input_embeds = forward_batch.input_embeds
        if forward_batch.forward_mode.is_extend():
            hidden_states = self._slow_prefill(input_ids, input_embeds, forward_batch)
        else:
            hidden_states = self._slow_decode(input_ids, forward_batch)
        logits = self._semantic_logits(hidden_states)
        if self._decode_ready:
            self._decode_codebooks(logits, hidden_states)
        return LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=hidden_states,
        )

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> None:
        params = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if name in {"freqs_cis", "fast_freqs_cis"}:
                continue
            target = params.get(name)
            if target is None:
                logger.debug("Skipping Audio8 hybrid weight: %s", name)
                continue
            loader = getattr(target, "weight_loader", _default_weight_loader)
            loader(target, loaded_weight)


EntryClass = Audio8HybridSGLangModel
