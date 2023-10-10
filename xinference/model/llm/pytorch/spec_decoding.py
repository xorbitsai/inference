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
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    import torch
    from torch.nn import functional as F
except ImportError:
    raise ImportError(
        f"Failed to import module 'torch'. Please make sure 'torch' is installed.\n\n"
    )

try:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )
except ImportError:
    error_message = "Failed to import module 'transformers'"
    installation_guide = [
        "Please make sure 'transformers' is installed. ",
        "You can install it by `pip install transformers`\n",
    ]

    raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")


from ....types import CompletionChoice, CompletionChunk, CompletionUsage

logger = logging.getLogger(__name__)


def prepare_logits_processor(
    temperature: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op, so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
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


def normalize_logits(
    logits_processor: LogitsProcessorList,
    input_ids: List[int],
    logits: torch.FloatTensor,  # [1, n_seq, n_vocab]
) -> torch.Tensor:
    """
    Parameters
    ----------
    logits : torch.Tensor
        Logits of shape `(n_batch, n_seq, n_vocab)`.

    Returns
    -------
    torch.Tensor
        Normalized logits of shape `(n_batch, n_seq, n_vocab)`.
    """

    def _helper(
        _input_ids: torch.LongTensor, _logits: torch.FloatTensor  # [1, n_vocab]
    ) -> torch.Tensor:
        if logits_processor:
            last_token_logits = logits_processor(
                _input_ids,
                _logits,
            )[0]
        else:
            return _logits[0]

        return last_token_logits  # [n_vocab,]

    input_ids = torch.as_tensor([input_ids], device=logits.device).long()
    for i in range(logits.shape[1]):
        normalized = _helper(
            input_ids[
                : -logits.shape[1] + i
            ],  # input_ids may not equal logits.shape[1]
            logits[:, i, :],
        )
        logits[:, i, :] = normalized.clone()
    return F.softmax(logits, dim=-1)


def sample(
    last_token_logits: torch.FloatTensor, temperature: float, top_p: float
) -> int:
    """
    Parameters
    ----------
    last_token_logits : torch.FloatTensor
        Last token logits of shape [n_vocab,]

    Returns
    -------
    int
        Token ID.
    """
    if temperature < 1e-5 or top_p < 1e-8:  # greedy
        _, indices = torch.topk(last_token_logits, 2)
        tokens = [int(index) for index in indices.tolist()]
    else:
        indices = torch.multinomial(last_token_logits, num_samples=2)
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


def rollback_logits(logits: torch.Tensor, n: int):
    return logits[:, :-n, :]  # [1, n_seq, n_vocab]


def is_partial_stop(output: str, stop_str: str):
    """Check whether the output contains a partial stop str."""
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False


def draft(
    input_ids: List[int],
    kv_cache: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
    logits: Optional[torch.FloatTensor],
    draft_model: PreTrainedModel,
    gamma: int,
    logits_processor: LogitsProcessorList,
    temperature: float,
    top_p: float,
):
    """
    Parameters
    ----------
    input_ids : List[int]
        On the prefill stage, `input_ids` are the prompt tokens.

        On the decode stage. It includes the prompt tokens, the token generated by the original model
        at the end of each full iteration, or the token generated by the draft model draft
        iteration.

    Returns
    -------
    int
        The number of generated draft tokens.
    List[int]
        Outputs, including the draft tokens.
    Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
        KV cache.
    torch.FloatTensor
        Logits.
    """
    draft_output_ids = input_ids.copy()

    if kv_cache is not None:
        input_ids = draft_output_ids[-2:]

    num_draft_tokens = 0
    while num_draft_tokens < gamma:
        if kv_cache is None:
            # prefill.
            draft_model_out = draft_model(
                torch.as_tensor([input_ids], device=draft_model.device),
                use_cache=True,
            )
            logits = normalize_logits(
                logits_processor, input_ids, draft_model_out.logits
            )
        else:
            draft_model_out = draft_model(
                torch.as_tensor([input_ids], device=draft_model.device),
                use_cache=True,
                past_key_values=kv_cache,
            )
            normalized = normalize_logits(
                logits_processor, draft_output_ids, draft_model_out.logits
            )
            assert logits is not None
            logits = torch.cat((logits, normalized), dim=1)
        kv_cache = draft_model_out.past_key_values
        draft_token = sample(
            logits[0, -1, :],
            temperature,
            top_p,
        )
        draft_output_ids.append(draft_token)
        input_ids = [draft_token]
        num_draft_tokens += 1

    assert kv_cache is not None
    return num_draft_tokens, draft_output_ids, kv_cache, logits


@torch.inference_mode()
def speculative_generate_stream(
    draft_model: PreTrainedModel,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    device: str,
    generate_config: Dict[str, Any],
) -> Iterator[Tuple[CompletionChunk, CompletionUsage]]:
    # TODO: currently, repetition penalty leads to garbled outputs.
    if float(generate_config.get("repetition_penalty", 1.0)) != 1.0:
        raise ValueError(
            "repetition penalty is not supported by speculative decoding yet"
        )

    stream_interval = generate_config.get("stream_interval", 2)
    stream = generate_config.get("stream", False)
    gamma = generate_config.get("gamma", 4)
    temperature = float(generate_config.get("temperature", 1.0))
    top_p = float(generate_config.get("top_p", 1.0))
    top_k = int(generate_config.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(generate_config.get("max_tokens", 256))
    stop_str = generate_config.get("stop", None)
    stop_token_ids = generate_config.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(temperature, top_p, top_k)

    if "qwen" in str(type(model)).lower():
        # TODO: hacky.
        input_ids = tokenizer(prompt, allowed_special="all").input_ids
    else:
        input_ids = tokenizer(prompt).input_ids

    num_prompt_tokens = len(input_ids)
    output_ids = list(input_ids)

    # internal states.
    draft_kv_cache = None
    draft_logits = None
    kv_cache = None
    logits = None
    next_token = (
        None  # the token generated by the original model at each full iteration.
    )

    # performance stats.
    total_seconds_on_drafting = 0.0
    total_seconds_on_eval = 0.0
    total_seconds_on_accepting = 0.0
    total_num_draft_tokens = 0
    total_num_accepted_tokens = 0

    while len(output_ids) < max_new_tokens:
        # allow the draft model to generate more than max_tokens since some of the generated
        # tokens could be rejected.
        start = time.time()
        num_draft_tokens, output_ids, draft_kv_cache, draft_logits = draft(
            input_ids=output_ids,
            kv_cache=draft_kv_cache,
            logits=draft_logits,
            draft_model=draft_model,
            gamma=gamma,
            logits_processor=logits_processor,
            temperature=temperature,
            top_p=top_p,
        )
        total_seconds_on_drafting += time.time() - start
        total_num_draft_tokens += num_draft_tokens

        # eval stage.
        start = time.time()
        if kv_cache is None:
            # prefill.
            out = model(
                torch.as_tensor([output_ids], device=model.device), use_cache=True
            )
            logits = normalize_logits(logits_processor, output_ids, out.logits)
        else:
            out = model(
                torch.as_tensor(
                    [[next_token] + output_ids[-num_draft_tokens:]], device=model.device
                ),
                use_cache=True,
                past_key_values=kv_cache,
            )
            normalized = normalize_logits(logits_processor, output_ids, out.logits)
            logits = torch.cat((logits, normalized), dim=1)
        kv_cache = out.past_key_values
        total_seconds_on_eval += time.time() - start

        # accepting stage.
        start = time.time()
        assert draft_logits is not None
        assert draft_kv_cache is not None
        accepted = 0
        for draft_token_idx in range(-num_draft_tokens, 0):
            r = torch.rand(1, device=device)
            draft_token = output_ids[draft_token_idx]
            tmp_draft_logits = draft_logits.to(logits.device)
            if r < torch.min(
                torch.tensor([1], device=device),
                tmp_draft_logits[0, draft_token_idx, draft_token]
                / logits[0, draft_token_idx - 1, draft_token],
            ):
                accepted += 1
                total_num_accepted_tokens += 1
            else:
                # rollback.
                output_ids = output_ids[:draft_token_idx]
                draft_kv_cache = rollback_kv_cache(
                    draft_kv_cache, num_draft_tokens - accepted
                )
                kv_cache = rollback_kv_cache(kv_cache, num_draft_tokens - accepted)
                draft_logits = rollback_logits(
                    draft_logits, num_draft_tokens - accepted
                )
                logits = rollback_logits(logits, num_draft_tokens - accepted)

                # sample the next token according to the modified distribution.
                modified_dist = (
                    logits[:, draft_token_idx, :]
                    - tmp_draft_logits[:, draft_token_idx, :]
                )
                modified_dist = torch.where(
                    modified_dist > 0, modified_dist, torch.zeros_like(modified_dist)
                )
                normalized = normalize_logits(
                    logits_processor,
                    output_ids,
                    modified_dist.unsqueeze(1),  # [1, 1, n_vocab]
                )
                next_token = sample(
                    normalized[0, -1, :],
                    temperature,
                    top_p,
                )
                output_ids.append(next_token)
                break

        if accepted == num_draft_tokens:
            next_token = sample(
                logits[0, -1, :],
                temperature,
                top_p,
            )
            output_ids.append(next_token)
        total_seconds_on_accepting += time.time() - start

    logger.info(
        f"In total, {total_num_accepted_tokens}/{total_num_draft_tokens} draft tokens are "
        f"accepted, acceptance rate: {total_num_accepted_tokens / total_num_draft_tokens:.2f}"
    )
    logger.info(f"In total, {total_seconds_on_drafting:.2f}s, {total_seconds_on_eval:.2f}s and "
                f"{total_seconds_on_accepting:.2f}s are spent on drafting, eval, and accepting "
                f"respectively.")

    output_ids = output_ids[num_prompt_tokens : min(max_new_tokens, len(output_ids))]
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
