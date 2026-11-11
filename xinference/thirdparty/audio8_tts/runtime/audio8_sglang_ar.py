# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from sglang.srt.mem_cache.common import release_kv_cache

from sglang_omni.engines.omni.runtime.sglang_ar import (
    SGLangARRequestData,
    SGLangBatchPlanner,
    SGLangResourceManager,
)
from sglang_omni.engines.omni.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)

if TYPE_CHECKING:
    from sglang_omni.engines.ar.sglang_backend.model_worker import ModelWorker

logger = logging.getLogger(__name__)


@dataclass
class Audio8StepOutput:
    codes: torch.Tensor


@dataclass
class Audio8SGLangRequestData(SGLangARRequestData):
    vq_mask_tokens: torch.Tensor | None = None
    vq_parts: list[torch.Tensor] | None = None
    num_codebooks: int = 10
    codebook_size: int = 4096
    output_codes: list[torch.Tensor] = field(default_factory=list)
    max_new_tokens: int | None = None
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    do_sample: bool = True
    _previous_semantic_tokens: list[int] = field(default_factory=list)
    _last_codebook_values: torch.Tensor | None = None
    # Falcon-H1 hybrid slow cache (Mamba states + attention KV) used by the
    # 0.1B path. Created at prefill, carried through decode, freed at the end.
    _slow_cache: Any | None = None


class Audio8IterationController:
    def __init__(self, tree_cache: Any, eos_token_id: int, max_new_tokens: int) -> None:
        self.tree_cache = tree_cache
        self.eos_token_id = int(eos_token_id)
        self.max_new_tokens = int(max_new_tokens)

    def update_request(self, request: SchedulerRequest, output: RequestOutput) -> None:
        data: Audio8SGLangRequestData = request.data
        req = data.req
        if req.is_chunked > 0:
            output.data = None
            req.is_chunked -= 1
            return
        codes = output.data.codes.clone()
        semantic = int(codes[0, -1].item())
        if semantic != self.eos_token_id:
            # HF generation does not send the EOS step to the waveform codec.
            if data.output_codes:
                data._previous_semantic_tokens.append(semantic)
            data.output_codes.append(codes)
            data._last_codebook_values = codes[1:, 0].clone()
        req.output_ids.append(semantic)
        if not req.finished() and req.decode_batch_idx == 0:
            self.tree_cache.cache_unfinished_req(req)

    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool:
        data: Audio8SGLangRequestData = request.data
        if data.req.is_chunked > 0:
            return False
        if output.data is None:
            return False
        semantic = int(output.data.codes[0, -1].item())
        if semantic == self.eos_token_id:
            return True
        return len(data.output_codes) >= (data.max_new_tokens or self.max_new_tokens)


class Audio8ModelRunner:
    def __init__(self, model_worker: "ModelWorker", batch_planner: SGLangBatchPlanner):
        self.model_worker = model_worker
        self.batch_planner = batch_planner

    def _inject_reference_embeds(
        self,
        model_worker_batch: Any,
        scheduler_output: SchedulerOutput,
    ) -> None:
        device = model_worker_batch.input_ids.device
        model = self.model_worker.model_runner.model
        text_embeds = model.get_embed_tokens()(model_worker_batch.input_ids)
        offset = 0
        for scheduled in scheduler_output.requests:
            data: Audio8SGLangRequestData = scheduled.data
            request_length = int(data.req.extend_input_len)
            prefix_length = len(data.req.prefix_indices)
            if data.vq_mask_tokens is not None and data.vq_parts:
                mask = data.vq_mask_tokens.to(device=device, dtype=torch.bool).flatten()
                expected_length = prefix_length + request_length
                if expected_length > mask.numel():
                    raise ValueError(
                        f"Audio8 reference mask length {mask.numel()} < request span {expected_length}"
                    )
                mask_slice = mask[prefix_length : prefix_length + request_length]
                parts = [part.to(device=device, dtype=torch.long).T for part in data.vq_parts]
                all_codes = torch.cat(parts, dim=0)
                if all_codes.shape[0] != int(mask.sum().item()):
                    raise ValueError("Audio8 reference mask/code length mismatch")
                before = int(mask[:prefix_length].sum().item())
                count = int(mask_slice.sum().item())
                if count:
                    codes = all_codes[before : before + count]
                    if codes.shape[1] != model.config.num_codebooks:
                        raise ValueError("Audio8 reference has the wrong number of codebooks")
                    if int(codes.min().item()) < 0 or int(codes.max().item()) >= model.config.codebook_size:
                        raise ValueError("Audio8 reference code is outside the embedding range")
                    embedded_codes = codes + model._codebook_offsets[None]
                    codebook_sum = model.codebook_embeddings(embedded_codes).sum(dim=1)
                    indices = mask_slice.nonzero(as_tuple=True)[0] + offset
                    text_embeds[indices] += codebook_sum.to(text_embeds.dtype)
            offset += request_length
        model_worker_batch.input_embeds = text_embeds

    def _update_runtime_buffers(
        self,
        model_worker_batch: Any,
        scheduler_output: SchedulerOutput,
        *,
        is_prefill: bool,
    ) -> None:
        model = self.model_worker.model_runner.model
        batch = len(scheduler_output.requests)
        model._previous_valid[:batch].zero_()
        if not is_prefill:
            input_ids = model_worker_batch.input_ids
            semantic_mask = (input_ids >= model.config.semantic_begin_id) & (
                input_ids <= model.config.semantic_end_id
            )
            model._vq_mask[:batch].copy_(semantic_mask)
        else:
            model._vq_mask[:batch].zero_()

        for index, scheduled in enumerate(scheduler_output.requests):
            data: Audio8SGLangRequestData = scheduled.data
            model._temperature[index] = max(float(data.temperature), 1e-5)
            model._top_p[index] = min(max(float(data.top_p), 1e-5), 1.0)
            model._top_k[index] = min(
                max(int(data.top_k), 1),
                int(model.config.codebook_size),
            )
            model._do_sample[index] = bool(data.do_sample)
            history = data._previous_semantic_tokens[-model.config.ras_window_size :]
            if history:
                length = len(history)
                model._previous_semantic[index, -length:] = torch.tensor(
                    history,
                    device=model._previous_semantic.device,
                    dtype=torch.long,
                )
                model._previous_valid[index, -length:] = True
            if (
                not is_prefill
                and data._last_codebook_values is not None
                and bool(model._vq_mask[index].item())
            ):
                model._vq_codes[index].copy_(data._last_codebook_values)
            elif (
                not is_prefill
                and data._last_codebook_values is None
                and bool(model._vq_mask[index].item())
            ):
                logger.warning(
                    "AUDIO8-STALE-VQ index=%d rid=%s mask=True last_codebook=None",
                    index,
                    data.req.rid,
                )

        # Hand the per-request Falcon-H1 hybrid caches to the model (0.1B path
        # only; Audio8SGLangModel has no ``_batch_slow_caches`` attribute).
        if hasattr(model, "_batch_slow_caches"):
            model._batch_slow_caches = [
                scheduled.data._slow_cache for scheduled in scheduler_output.requests
            ]

    def _build_outputs(self, scheduler_output: SchedulerOutput) -> dict[str, RequestOutput]:
        model = self.model_worker.model_runner.model
        outputs: dict[str, RequestOutput] = {}
        for index, scheduled in enumerate(scheduler_output.requests):
            data: Audio8SGLangRequestData = scheduled.data
            outputs[scheduled.request_id] = RequestOutput(
                request_id=scheduled.request_id,
                data=(
                    None
                    if data.req.is_chunked > 0
                    else Audio8StepOutput(model._output_codes[index].unsqueeze(-1))
                ),
                finished=False,
            )
        return outputs

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        schedule_batch = scheduler_output.batch_data
        worker_batch = schedule_batch.get_model_worker_batch()
        is_prefill = schedule_batch.forward_mode.is_extend()
        model = self.model_worker.model_runner.model
        if len(scheduler_output.requests) != len(schedule_batch.reqs):
            logger.warning(
                "AUDIO8-ALIGN requests=%d batch_reqs=%d batch_rows=%s mode=%s",
                len(scheduler_output.requests),
                len(schedule_batch.reqs),
                tuple(worker_batch.input_ids.shape),
                schedule_batch.forward_mode,
            )
        self._update_runtime_buffers(
            worker_batch,
            scheduler_output,
            is_prefill=is_prefill,
        )
        if is_prefill:
            self._inject_reference_embeds(worker_batch, scheduler_output)
        forward_batch = ForwardBatch.init_new(worker_batch, self.model_worker.model_runner)
        batch_result = self.model_worker.forward_batch_generation(forward_batch)
        if hasattr(model, "_batch_slow_caches"):
            # Prefill seeds / decode advances each request's Falcon-H1 hybrid
            # cache; persist it back onto the request data for the next step.
            for index, scheduled in enumerate(scheduler_output.requests):
                scheduled.data._slow_cache = model._batch_slow_caches[index]
        if schedule_batch.is_prefill_only:
            batch_result.next_token_ids = torch.zeros(
                len(worker_batch.seq_lens),
                dtype=torch.long,
                device=worker_batch.input_ids.device,
            )
        self.batch_planner.record_last_batch(schedule_batch)
        outputs = self._build_outputs(scheduler_output)
        batch = len(scheduler_output.requests)
        schedule_batch.output_ids = model._output_semantic_ids[:batch].clone()
        request_ids = [request.request_id for request in scheduler_output.requests]
        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=request_ids,
            req_id_to_index={request_id: index for index, request_id in enumerate(request_ids)},
        )


class Audio8ResourceManager(SGLangResourceManager):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._freed_since_compact = 0

    def free(self, request: SchedulerRequest) -> None:
        data: Audio8SGLangRequestData = request.data
        self._freed_since_compact += 1
        release_kv_cache(data.req, self.tree_cache)
        data._previous_semantic_tokens.clear()
        data._last_codebook_values = None
        data._slow_cache = None
        if self._freed_since_compact >= 64:
            self._freed_since_compact = 0
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                logger.info(
                    "Audio8 memory: allocated=%.2f GB reserved=%.2f GB",
                    allocated / 1024**3,
                    reserved / 1024**3,
                )
                torch.cuda.empty_cache()
