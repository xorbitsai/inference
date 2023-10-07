# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
import uuid
from typing import Iterator, Tuple

import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from ....types import CompletionChoice, CompletionChunk, CompletionUsage

logger = logging.getLogger(__name__)


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op, so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def get_context_length(config):
    """Get the context length of a model from a huggingface model config."""
    if (
        hasattr(config, "max_sequence_length")
        and config.max_sequence_length is not None
    ):
        return config.max_sequence_length
    elif hasattr(config, "seq_length") and config.seq_length is not None:
        return config.seq_length
    elif (
        hasattr(config, "max_position_embeddings")
        and config.max_position_embeddings is not None
    ):
        return config.max_position_embeddings
    else:
        return 2048


def normalize_logits(logits_processor: LogitsProcessorList, output_ids, logits):
    if logits_processor:
        last_token_logits = logits_processor(
            torch.as_tensor([output_ids], device=logits.device).long(), logits[:, -1, :]
        )[0]
    else:
        last_token_logits = logits[0, -1, :]

    return last_token_logits


def sample(last_token_logits, temperature, top_p):
    if temperature < 1e-5 or top_p < 1e-8:  # greedy
        _, indices = torch.topk(last_token_logits, 2)
        tokens = [int(index) for index in indices.tolist()]
    else:
        probs = torch.softmax(last_token_logits, dim=-1)
        indices = torch.multinomial(probs, num_samples=2)
        tokens = [int(token) for token in indices.tolist()]
    return tokens[0]


def rollback_kv_cache(
    kv_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], n: int
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    ret = []
    for k_cache, v_cache in kv_cache:
        k_cache = k_cache[:, :, :-n, :]  # [1, n_head, n_seq - n, n_dim]
        v_cache = v_cache[:, :, :-n, :]

        assert isinstance(k_cache, torch.Tensor)
        assert isinstance(v_cache, torch.Tensor)
        ret.append((k_cache, v_cache))

    return tuple(ret)


@torch.inference_mode()
def speculative_generate_stream(
    draft_model,
    model,
    tokenizer,
    prompt,
    device,
    generate_config,
) -> Iterator[Tuple[CompletionChunk, CompletionUsage]]:
    context_len = get_context_length(model.config)
    # stream = generate_config.get("stream", False)
    gamma = generate_config.get("gamma", 4)

    temperature = float(generate_config.get("temperature", 1.0))
    repetition_penalty = float(generate_config.get("repetition_penalty", 1.0))
    top_p = float(generate_config.get("top_p", 1.0))
    top_k = int(generate_config.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(generate_config.get("max_tokens", 256))
    # stop_str = generate_config.get("stop", None)
    stop_token_ids = generate_config.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    if "qwen" in str(type(model)).lower():
        # TODO: hacky
        input_ids = tokenizer(prompt, allowed_special="all").input_ids
    else:
        input_ids = tokenizer(prompt).input_ids
    num_prompt_tokens = len(input_ids)
    output_ids = list(input_ids)

    max_tokens = max_new_tokens + len(input_ids)
    max_src_len = context_len - max_new_tokens - 1
    input_ids = input_ids[-max_src_len:]

    draft_model_kv_cache = None
    kv_cache = None

    num_total_draft_tokens = 0
    num_total_accepted_tokens = 0

    while len(output_ids) < max_tokens:
        num_draft_tokens = 0
        draft_logits = None
        draft_token = None
        while num_draft_tokens < gamma:
            # allow the draft model to generate more than max_tokens since some of the generated
            # tokens could be rejected.
            if draft_model_kv_cache is None:
                # prefill.
                draft_model_out = draft_model(
                    torch.as_tensor([input_ids], device=draft_model.device),
                    use_cache=True,
                )
                draft_logits = draft_model_out.logits
            else:
                draft_model_out = draft_model(
                    torch.as_tensor([[draft_token]], device=draft_model.device),
                    use_cache=True,
                    past_key_values=draft_model_kv_cache,
                )
                draft_logits = torch.cat((draft_logits, draft_model_out.logits), dim=1)
            draft_model_kv_cache = draft_model_out.past_key_values
            draft_token = sample(
                normalize_logits(
                    logits_processor=logits_processor,
                    output_ids=output_ids,
                    logits=draft_model_out.logits,
                ),
                temperature,
                top_p,
            )
            output_ids.append(draft_token)
            num_draft_tokens += 1
            num_total_draft_tokens += 1

        if kv_cache is None:
            # prefill.
            out = model(
                torch.as_tensor([output_ids], device=model.device), use_cache=True
            )
        else:
            out = model(
                torch.as_tensor([output_ids[-num_draft_tokens:]], device=model.device),
                use_cache=True,
                past_key_values=kv_cache,
            )
        kv_cache = out.past_key_values
        logits = out.logits

        assert draft_logits is not None
        # TODO: remove
        logger.info(f"draft logits shape: {draft_logits.shape}")
        logger.info(f"logits shape: {logits.shape}")
        if logits.device != draft_logits.device:
            draft_logits = draft_logits.to(logits.device)

        accepted = 0
        for draft_token_idx in range(-num_draft_tokens, 0):
            r = torch.rand(1, device=device)
            draft_token = output_ids[draft_token_idx]

            if r < torch.min(
                torch.tensor([1], device=device),
                draft_logits[0, draft_token_idx, draft_token]
                / logits[0, draft_token_idx, draft_token],
            ):
                accepted += 1
                num_total_accepted_tokens += 1
            else:
                # rollback.
                output_ids = output_ids[:draft_token_idx]
                assert draft_model_kv_cache is not None
                draft_model_kv_cache = rollback_kv_cache(
                    draft_model_kv_cache, num_draft_tokens - accepted
                )
                kv_cache = rollback_kv_cache(kv_cache, num_draft_tokens - accepted)

                next_token = sample(
                    normalize_logits(
                        logits_processor=logits_processor,
                        output_ids=output_ids,
                        logits=logits[:, :draft_token_idx, :],
                    ),
                    temperature,
                    top_p,
                )
                output_ids.append(next_token)
                break

        logger.info(f"{accepted}/{num_draft_tokens} draft tokens are accepted")
        if accepted == num_draft_tokens:
            next_token = sample(
                normalize_logits(
                    logits_processor=logits_processor,
                    output_ids=output_ids,
                    logits=logits,
                ),
                temperature,
                top_p,
            )
            output_ids.append(next_token)

    logger.info(
        f"In total, {num_total_accepted_tokens}/{num_total_draft_tokens} draft tokens are "
        f"accepted, acceptance rate: {num_total_accepted_tokens / num_total_draft_tokens:.2f}"
    )

    output_ids = output_ids[: min(max_tokens, len(output_ids))]
    output = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        spaces_between_special_tokens=False,
        clean_up_tokenization_spaces=True,
    )
    completion_choice = CompletionChoice(
        text=output, index=0, logprobs=None, finish_reason=None
    )
    completion_chunk = CompletionChunk(
        id=str(uuid.uuid1()),
        object="text_completion",
        created=int(time.time()),
        model=generate_config["model"],
        choices=[completion_choice],
    )
    completion_usage = CompletionUsage(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=len(output_ids) - num_prompt_tokens,
        total_tokens=len(output_ids),
    )

    yield completion_chunk, completion_usage
