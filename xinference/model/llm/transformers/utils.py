# Copyright 2022-2026 XProbe Inc.
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
import os
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from ....device_utils import empty_cache
from ....types import (
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionUsage,
    max_tokens_field,
)
from ...scheduler.request import InferenceRequest

if TYPE_CHECKING:
    from ...llm.transformers.core import PytorchModel

logger = logging.getLogger(__name__)


def get_context_length(config) -> int:
    """Get the context length of a model from a huggingface model config."""
    if (
        hasattr(config, "max_sequence_length")
        and config.max_sequence_length is not None
    ):
        max_sequence_length = config.max_sequence_length
    else:
        max_sequence_length = 2048
    if hasattr(config, "seq_length") and config.seq_length is not None:
        seq_length = config.seq_length
    else:
        seq_length = 2048
    if (
        hasattr(config, "max_position_embeddings")
        and config.max_position_embeddings is not None
    ):
        max_position_embeddings = config.max_position_embeddings
    else:
        max_position_embeddings = 2048
    return max(max_sequence_length, seq_length, max_position_embeddings)


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def _get_token_from_logits(
    req: InferenceRequest, i: int, logits, temperature, repetition_penalty, top_p, top_k
):
    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    if logits_processor:
        if repetition_penalty > 1.0:
            tmp_output_ids = torch.as_tensor(
                [req.prompt_tokens + req.new_tokens], device=logits.device
            )
        else:
            tmp_output_ids = None
        last_token_logits = logits_processor(tmp_output_ids, logits[i : i + 1, -1, :])[
            0
        ]
    else:
        last_token_logits = logits[i : i + 1, -1, :]

    if temperature < 1e-5 or top_p < 1e-8:  # greedy
        _, indices = torch.topk(last_token_logits, 2)
    else:
        probs = torch.softmax(last_token_logits, dim=-1)
        indices = torch.multinomial(probs, num_samples=2)
    token = indices[0].int().item()
    return token


def _pad_to_max_length(x: List[int], max_len: int, pad: int) -> List[int]:
    assert len(x) <= max_len
    return [pad] * (max_len - len(x)) + x


def _pad_seqs_inplace(seqs: List[List[int]], reqs: List[InferenceRequest], pad: int):
    max_len = max(len(seq) for seq in seqs)
    n = len(seqs)
    i = 0
    while i < n:
        prev_seq_len = len(seqs[i])
        seqs[i] = _pad_to_max_length(seqs[i], max_len, pad)
        padding_len = len(seqs[i]) - prev_seq_len
        reqs[i].padding_len = padding_len
        i += 1


def get_max_src_len(context_len: int, r: InferenceRequest) -> int:
    # if max_tokens not set, we just treat max_src_len = context_len - 8
    max_new_tokens = int(
        r.sanitized_generate_config.get("max_tokens") or max_tokens_field.default or 0
    )
    return context_len - max_new_tokens - 8


def pad_prefill_tokens(
    input_ids: List[List[int]], context_len: int, req_list: List[InferenceRequest]
):
    prompt_tokens = []
    for i, input_id in enumerate(input_ids):
        req = req_list[i]
        max_src_len = get_max_src_len(context_len, req)
        req.prompt_tokens = input_id[-max_src_len:]
        prompt_tokens.append(req.prompt_tokens)
    _pad_seqs_inplace(prompt_tokens, req_list, 0)
    return prompt_tokens


def _get_completion(
    output: str,
    chunk_id: str,
    finish_reason: Optional[str],
    model_uid: str,
    r: InferenceRequest,
    completion_tokens: int,
):
    completion_choice = CompletionChoice(
        text=output, index=0, logprobs=None, finish_reason=finish_reason
    )

    completion_chunk = CompletionChunk(
        id=chunk_id,
        object="text_completion",
        created=int(time.time()),
        model=model_uid,
        choices=[completion_choice],
    )
    completion_usage = CompletionUsage(
        prompt_tokens=len(r.prompt_tokens),
        completion_tokens=completion_tokens,
        total_tokens=len(r.prompt_tokens) + completion_tokens,
    )
    completion = Completion(
        id=completion_chunk["id"],
        object=completion_chunk["object"],
        created=completion_chunk["created"],
        model=completion_chunk["model"],
        choices=completion_chunk["choices"],
        usage=completion_usage,
    )
    return completion


def _get_pad_param(seq_len_idx: int, pad_len: int) -> Tuple:
    dimensions = [0] * 8
    dimensions[-2 * (seq_len_idx + 1)] = pad_len
    return tuple(dimensions)


def get_batch_size_and_seq_len_from_kv_cache(kv, xinf_model_obj: "PytorchModel"):
    from transformers import HybridCache

    bs_idx, seq_len_idx = xinf_model_obj.get_batch_size_and_seq_len_indexes_from_kv()

    if isinstance(kv, HybridCache):
        return kv.key_cache[0].shape[bs_idx], kv.get_seq_length()
    return kv[0][0].shape[bs_idx], kv[0][0].shape[seq_len_idx] + 1


def convert_to_cache_cls(cache) -> DynamicCache:
    """
    Compatible with some old models
    """
    if isinstance(cache, tuple):
        return DynamicCache.from_legacy_cache(cache)
    return cache


@torch.inference_mode()
def _batch_inference_one_step_internal(
    xinf_model_obj: "PytorchModel",
    req_list: List[InferenceRequest],
    model_uid,
    model,
    tokenizer,
    decode_round: int = 16,
    bos_flag: str = "<bos_stream>",
    eos_flag: str = "<eos_stream>",
):
    from ..utils import generate_completion_chunk

    # need to judge stopped here,
    # since some requests state may change to stopped due to invalid parameters, e.g. max_src_len
    valid_req_list = [r for r in req_list if not r.stopped]
    if not valid_req_list:
        return
    generate_config_mapping: Dict[InferenceRequest, Tuple] = {
        r: r.get_generate_configs(
            tokenizer.eos_token_id, xinf_model_obj.get_builtin_stop_token_ids()
        )
        for r in valid_req_list
    }
    s_time = time.time()

    prefill_reqs = []
    prompts = []
    decode_reqs = []
    for r in valid_req_list:
        if r.is_prefill:
            prompts.append(r.full_prompt if r.full_prompt is not None else r.prompt)
            prefill_reqs.append(r)
        else:
            decode_reqs.append(r)

    if prompts:  # prefill first
        prefill_kws = xinf_model_obj.build_prefill_kwargs(prompts, prefill_reqs)
        out = model(**prefill_kws, use_cache=True)

        logits = out.logits
        past_key_values = convert_to_cache_cls(out.past_key_values)

        for i, r in enumerate(prefill_reqs):
            (
                max_new_tokens,
                stream_interval,
                include_usage,
                stop_str,
                stop_token_ids,
                temperature,
                repetition_penalty,
                top_p,
                top_k,
            ) = generate_config_mapping[r]

            if max_new_tokens == 0:
                # max_tokens not set, we change it to the possible maximum
                max_new_tokens = xinf_model_obj.get_context_len() - len(r.prompt_tokens)
                new_gen_conf = list(generate_config_mapping[r])
                new_gen_conf[0] = max_new_tokens
                generate_config_mapping[r] = tuple(new_gen_conf)
                logger.debug("No max_tokens set, setting to: %s", max_new_tokens)

            token = _get_token_from_logits(
                r, i, logits, temperature, repetition_penalty, top_p, top_k
            )
            r.is_prefill = False
            r.append_new_token(token)

        if decode_reqs:
            # Ensure all decode requests have the same kv_cache reference
            # This prevents batch size mismatches during merging
            decode_kv = decode_reqs[0].kv_cache

            # prefill and decode kv cache need to be merged at `batch_size` and `seq_len` dimensions.
            merged_kv_cache = xinf_model_obj.merge_kv_cache(decode_kv, past_key_values)
            for r in valid_req_list:
                r.kv_cache = merged_kv_cache
            empty_cache()
        else:
            for r in valid_req_list:
                r.kv_cache = past_key_values

    past_key_values = valid_req_list[0].kv_cache
    stop_token_mapping: Dict[InferenceRequest, int] = {}
    output_mapping: Dict[InferenceRequest, str] = {}
    # here, only decode phase, just run some rounds
    for _i in range(decode_round):
        batch_size, seq_len = get_batch_size_and_seq_len_from_kv_cache(
            past_key_values, xinf_model_obj
        )
        decode_tokens: List[List[int]] = [[r.new_tokens[-1]] for r in valid_req_list]
        inf_kws = xinf_model_obj.build_decode_kwargs(
            decode_tokens, valid_req_list, batch_size, seq_len
        )
        out = model(**inf_kws, use_cache=True, past_key_values=past_key_values)
        logits = out.logits
        past_key_values = convert_to_cache_cls(out.past_key_values)

        for i, r in enumerate(valid_req_list):
            (
                max_new_tokens,
                stream_interval,
                include_usage,
                stop_str,
                stop_token_ids,
                temperature,
                repetition_penalty,
                top_p,
                top_k,
            ) = generate_config_mapping[r]

            token = _get_token_from_logits(
                r, i, logits, temperature, repetition_penalty, top_p, top_k
            )
            r.kv_cache = past_key_values
            r.append_new_token(token)

            output = None
            if not r.stopped:
                stopped = token in stop_token_ids

                if stopped:
                    finish_reason = "stop"
                elif len(r.new_tokens) == max_new_tokens:
                    finish_reason = "length"
                    stopped = True
                else:
                    finish_reason = None

                # handle stop str
                if stop_str and r not in output_mapping:
                    output = tokenizer.decode(
                        r.new_tokens,
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                        clean_up_tokenization_spaces=True,
                    )
                    if isinstance(stop_str, str):
                        stop_str = [stop_str]
                    for stop in stop_str:
                        pos = output.rfind(stop)
                        if pos != -1:
                            output = output[:pos]
                            output_mapping[r] = output
                            stopped = True
                            finish_reason = "stop"
                            break

                r.stopped = stopped
                r.finish_reason = finish_reason

            if r.stopped and r not in stop_token_mapping:
                stop_token_mapping[r] = _i + 1

            if r.stream:
                """
                Note that you can't just decode based on the newest r.new_tokens here,
                which may destroy the integrity of the parsed characters,
                and at the same time is not good at handling some special characters.
                So the implementation here is to decode all the tokens that have been generated each time,
                and then take the slice.
                """
                if r.stopped or len(r.new_tokens) % stream_interval == 0:
                    if output is None:
                        output = tokenizer.decode(
                            r.new_tokens,
                            skip_special_tokens=True,
                            spaces_between_special_tokens=False,
                            clean_up_tokenization_spaces=True,
                        )

                    if r.last_output_length == 0:
                        r.completion.append(bos_flag)

                    # this special character is mainly for qwen
                    output = output.strip("�")
                    output = output[r.last_output_length :]
                    r.last_output_length += len(output)
                    r.outputs.append(output)

                    completion_chunk = generate_completion_chunk(
                        chunk_text=output,
                        finish_reason=None,
                        chunk_id=r.chunk_id,
                        model_uid=model_uid,
                        prompt_tokens=len(r.prompt_tokens),
                        completion_tokens=len(r.new_tokens),
                        total_tokens=len(r.prompt_tokens) + len(r.new_tokens),
                    )
                    r.completion.append(completion_chunk)
                    if r.stopped:
                        # OpenAI compatible chunk
                        completion_chunk = generate_completion_chunk(
                            chunk_text="",
                            finish_reason=r.finish_reason,
                            chunk_id=r.chunk_id,
                            model_uid=model_uid,
                            prompt_tokens=len(r.prompt_tokens),
                            completion_tokens=len(r.new_tokens),
                            total_tokens=len(r.prompt_tokens) + len(r.new_tokens),
                        )
                        r.completion.append(completion_chunk)
                        r.completion.append(eos_flag)
                        r.outputs.append(eos_flag)

                    # last round, handle stream result
                    # append usage information when enable `include_usage` for OPENAI API compatibility
                    # The reason for counting the usage in the last round of the iteration is that,
                    # these tokens are real generated and should be counted.
                    if r.stopped and _i == decode_round - 1 and include_usage:
                        r.completion.append(
                            generate_completion_chunk(
                                chunk_text=None,
                                finish_reason=None,
                                chunk_id=r.chunk_id,
                                model_uid=model_uid,
                                prompt_tokens=len(r.prompt_tokens),
                                completion_tokens=len(r.new_tokens),
                                total_tokens=len(r.prompt_tokens) + len(r.new_tokens),
                                has_choice=False,
                                has_content=False,
                            )
                        )
            else:
                # last round, handle non-stream result
                if r.stopped and _i == decode_round - 1:
                    invalid_token_num = (
                        (decode_round - stop_token_mapping[r] + 1)
                        if r.finish_reason == "stop"
                        else (decode_round - stop_token_mapping[r])
                    )
                    outputs = (
                        tokenizer.decode(
                            r.new_tokens[:-invalid_token_num],
                            skip_special_tokens=True,
                            spaces_between_special_tokens=False,
                            clean_up_tokenization_spaces=True,
                        )
                        if r not in output_mapping
                        else output_mapping[r]
                    )
                    completion = _get_completion(
                        outputs,
                        r.chunk_id,
                        r.finish_reason,
                        model_uid,
                        r,
                        len(r.new_tokens) - invalid_token_num,
                    )
                    r.completion = [completion]

    e_time = time.time()
    logger.debug(
        f"Average throughput for a step: {(len(valid_req_list) * decode_round + len(prompts)) / (e_time - s_time)} token/s."
    )


def batch_inference_one_step(
    xinf_model_obj: "PytorchModel",
    req_list: List[InferenceRequest],
    model_uid,
    model,
    tokenizer,
):
    from ....core.model import OutOfMemoryError

    try:
        _batch_inference_one_step_internal(
            xinf_model_obj, req_list, model_uid, model, tokenizer
        )
    except OutOfMemoryError:
        logger.exception(
            f"Batch inference out of memory. "
            f"Xinference will restart the model: {model_uid}. "
            f"Please be patient for a few moments."
        )
        # Just kill the process and let xinference auto-recover the model
        os._exit(1)
    except Exception as e:
        logger.exception(f"Internal error for batch inference: {e}.")
        # If internal error happens, just skip all the requests in this batch.
        # If not handle here, the client will hang.
        for r in req_list:
            r.stopped = True
            r.error_msg = str(e)
