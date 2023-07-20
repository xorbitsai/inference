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

import gc
import time
import uuid
from typing import Iterable, Iterator, Tuple

import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from ....types import CompletionChoice, CompletionChunk, CompletionUsage


def is_sentence_complete(output: str):
    """Check whether the output is a complete sentence."""
    end_symbols = (".", "?", "!", "...", "。", "？", "！", "…", '"', "'", "”")
    return output.endswith(end_symbols)


def is_partial_stop(output: str, stop_str: str):
    """Check whether the output contains a partial stop str."""
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False


def get_context_length(config):
    """Get the context length of a model from a huggingface model config."""
    if hasattr(config, "max_sequence_length"):
        return config.max_sequence_length
    elif hasattr(config, "seq_length"):
        return config.seq_length
    elif hasattr(config, "max_position_embeddings"):
        return config.max_position_embeddings
    else:
        return 2048


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


@torch.inference_mode()
def generate_stream(
    model,
    tokenizer,
    prompt,
    device,
    generate_config,
    judge_sent_end=False,
) -> Iterator[Tuple[CompletionChunk, CompletionUsage]]:
    context_len = get_context_length(model.config)
    stream_interval = generate_config.get("stream_interval", 2)
    stream = generate_config.get("stream", False)

    len_prompt = len(prompt)

    temperature = float(generate_config.get("temperature", 1.0))
    repetition_penalty = float(generate_config.get("repetition_penalty", 1.0))
    top_p = float(generate_config.get("top_p", 1.0))
    top_k = int(generate_config.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(generate_config.get("max_new_tokens", 256))
    echo = bool(generate_config.get("echo", False))
    stop_str = generate_config.get("stop", None)
    stop_token_ids = generate_config.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:
        max_src_len = context_len - max_new_tokens - 8

    input_ids = input_ids[-max_src_len:]
    input_echo_len = len(input_ids)

    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )

    past_key_values = out = None
    sent_interrupt = False
    token = None
    last_output_length = 0
    for i in range(max_new_tokens):
        if i == 0:
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
            past_key_values = out.past_key_values
        else:
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids], device=device
                    ),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids], device=device
                    ),
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False
                logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]
        token = tokens[0]
        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

            # TODO: For the issue of incomplete sentences interrupting output, apply a patch and others can also modify it to a more elegant way
            if judge_sent_end and stopped and not is_sentence_complete(output):
                if len(tokens) > 1:
                    token = tokens[1]
                    output_ids[-1] = token
                else:
                    output_ids.pop()
                stopped = False
                sent_interrupt = True

            partially_stopped = False
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            if stream:
                tmp_output_length = len(output)
                output = output[last_output_length:]
                last_output_length = tmp_output_length

            # prevent yielding partial stop sequence
            if not partially_stopped:
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
                    prompt_tokens=input_echo_len,
                    completion_tokens=i,
                    total_tokens=(input_echo_len + i),
                )

                yield completion_chunk, completion_usage

        if stopped:
            break

    # finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    completion_choice = CompletionChoice(
        text=output, index=0, logprobs=None, finish_reason=finish_reason
    )
    completion_chunk = CompletionChunk(
        id=str(uuid.uuid1()),
        object="text_completion",
        created=int(time.time()),
        model=generate_config["model"],
        choices=[completion_choice],
    )
    completion_usage = CompletionUsage(
        prompt_tokens=input_echo_len,
        completion_tokens=i,
        total_tokens=(input_echo_len + i),
    )

    yield completion_chunk, completion_usage

    # clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()
