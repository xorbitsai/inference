# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch

from xinference.thirdparty.audio8_tts.io import Audio8TTSState
from xinference.thirdparty.audio8_tts.runtime.audio8_sglang_ar import (
    Audio8SGLangRequestData,
)


def build_tts_request(
    state: Audio8TTSState,
    tokenizer: Any,
    request_id: str,
) -> Audio8SGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    input_ids = list(state.input_ids)
    sampling_params = SamplingParams(
        max_new_tokens=state.max_new_tokens,
        temperature=state.temperature,
    )
    req = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=input_ids,
        sampling_params=sampling_params,
        vocab_size=tokenizer.vocab_size,
    )
    return Audio8SGLangRequestData(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        req=req,
        vq_mask_tokens=(
            torch.as_tensor(state.vq_mask_tokens, dtype=torch.bool)
            if state.vq_mask_tokens is not None
            else None
        ),
        vq_parts=(
            [torch.as_tensor(part, dtype=torch.long) for part in state.vq_parts]
            if state.vq_parts is not None
            else None
        ),
        num_codebooks=state.num_codebooks,
        codebook_size=state.codebook_size,
        max_new_tokens=state.max_new_tokens,
        temperature=state.temperature,
        top_p=state.top_p,
        top_k=state.top_k,
        do_sample=state.do_sample,
    )


def apply_tts_result(state: Audio8TTSState, result: Audio8SGLangRequestData) -> None:
    if result.output_codes:
        all_codes = torch.cat(result.output_codes, dim=1)
        state.output_codes = all_codes[1:]
        state.completion_tokens = int(all_codes.shape[1])
        result.output_codes.clear()
    else:
        state.output_codes = None
    state.prompt_tokens = len(result.input_ids) if result.input_ids is not None else 0
