# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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

import json
import logging
import math
import random
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerFast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

RangeRatio = Union[float, Dict[str, float]]

# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    **kwargs,
) -> "PreTrainedTokenizerBase":
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    if (
        "llama" in tokenizer_name.lower()
        and kwargs.get("use_fast", True)
        and tokenizer_name != _FAST_LLAMA_TOKENIZER
    ):
        logger.info(
            "For some LLaMA-based models, initializing the fast tokenizer may "
            "take a long time. To eliminate the initialization time, consider "
            f"using '{_FAST_LLAMA_TOKENIZER}' instead of the original "
            "tokenizer."
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, *args, trust_remote_code=trust_remote_code, **kwargs
        )
    except TypeError as e:
        # The LLaMA tokenizer causes a protobuf error in some environments.
        err_msg = (
            "Failed to load the tokenizer. If you are using a LLaMA-based "
            f"model, use '{_FAST_LLAMA_TOKENIZER}' instead of the original "
            "tokenizer."
        )
        raise RuntimeError(err_msg) from e
    except ValueError as e:
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if not trust_remote_code and (
            "does not exist or is not currently imported." in str(e)
            or "requires you to execute the tokenizer file" in str(e)
        ):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        logger.warning(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )
    return tokenizer


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: "PreTrainedTokenizerBase",
    prompt_len_limit: int = 1024,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if (
            prompt_len > prompt_len_limit
            or prompt_len + output_len > prompt_len_limit * 2
        ):
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


def _resolve_range_ratios(range_ratio: RangeRatio) -> Tuple[float, float]:
    if isinstance(range_ratio, dict):
        try:
            return float(range_ratio["input"]), float(range_ratio["output"])
        except KeyError as exc:
            raise ValueError(
                "When range_ratio is a dict it must contain 'input' and "
                f"'output' keys, got: {sorted(range_ratio)}"
            ) from exc
    ratio = float(range_ratio)
    return ratio, ratio


def _get_vocab_size(tokenizer: "PreTrainedTokenizerBase") -> int:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        vocab_size = len(tokenizer)
    if vocab_size <= 0:
        raise ValueError("Tokenizer vocab size must be positive.")
    return vocab_size


def _num_special_tokens_to_add(tokenizer: "PreTrainedTokenizerBase") -> int:
    num_special_tokens_to_add = getattr(tokenizer, "num_special_tokens_to_add", None)
    if callable(num_special_tokens_to_add):
        return int(num_special_tokens_to_add())
    if isinstance(num_special_tokens_to_add, int):
        return num_special_tokens_to_add
    return 0


def _encode_prompt(
    tokenizer: "PreTrainedTokenizerBase",
    prompt: str,
    add_special_tokens: bool = False,
) -> List[int]:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        return list(encode(prompt, add_special_tokens=add_special_tokens))
    return list(tokenizer(prompt, add_special_tokens=add_special_tokens).input_ids)


def _get_sampling_params(
    rng: np.random.Generator,
    num_requests: int,
    range_ratio: RangeRatio,
    input_len: int,
    output_len: int,
    tokenizer: "PreTrainedTokenizerBase",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_range_ratio, output_range_ratio = _resolve_range_ratios(range_ratio)
    if not (0.0 <= input_range_ratio < 1.0):
        raise ValueError("input_range_ratio must be in [0, 1).")
    if not (0.0 <= output_range_ratio < 1.0):
        raise ValueError("output_range_ratio must be in [0, 1).")

    num_special_tokens = _num_special_tokens_to_add(tokenizer)
    real_input_len = max(0, int(input_len) - num_special_tokens)
    input_low = math.floor(real_input_len * (1 - input_range_ratio))
    input_high = math.ceil(real_input_len * (1 + input_range_ratio))
    output_low = math.floor(output_len * (1 - output_range_ratio))
    output_high = math.ceil(output_len * (1 + output_range_ratio))
    output_low = max(output_low, 1)
    output_high = max(output_high, 1)

    if input_low > input_high:
        raise ValueError(
            f"Invalid input sampling interval: low={input_low} > high={input_high}"
        )
    if output_low > output_high:
        raise ValueError(
            f"Invalid output sampling interval: low={output_low} > high={output_high}"
        )

    logger.info(
        "Sampling input_len from [%s, %s] and output_len from [%s, %s]",
        input_low,
        input_high,
        output_low,
        output_high,
    )
    input_lens = rng.integers(input_low, input_high + 1, size=num_requests)
    output_lens = rng.integers(output_low, output_high + 1, size=num_requests)
    offsets = rng.integers(0, _get_vocab_size(tokenizer), size=num_requests)
    return input_lens, output_lens, offsets


def _gen_prompt_decode_to_target_len(
    tokenizer: "PreTrainedTokenizerBase",
    token_sequence: List[int],
    target_token_len: int,
    allowed_tokens: np.ndarray,
    rng: np.random.Generator,
    max_retry: int = 10,
) -> Tuple[str, List[int], int]:
    remain_num_try = max_retry
    token_mismatch = 0
    while True:
        prompt = tokenizer.decode(token_sequence)
        token_sequence = _encode_prompt(tokenizer, prompt, add_special_tokens=False)

        if remain_num_try <= 0:
            if len(token_sequence) != target_token_len:
                token_mismatch = len(token_sequence) - target_token_len
            break
        if len(token_sequence) == target_token_len:
            break
        if len(token_sequence) < target_token_len:
            extra_tokens = allowed_tokens[
                rng.integers(
                    0,
                    len(allowed_tokens),
                    size=target_token_len - len(token_sequence),
                )
            ].tolist()
            token_sequence.extend(extra_tokens)
        else:
            token_sequence = token_sequence[:target_token_len]

        remain_num_try -= 1
    return prompt, token_sequence, token_mismatch


def _get_random_prefix(
    tokenizer: "PreTrainedTokenizerBase",
    allowed_tokens: np.ndarray,
    prefix_len: int,
    rng: np.random.Generator,
) -> List[int]:
    if prefix_len <= 0:
        return []

    prefix_tokens = allowed_tokens[
        rng.integers(0, len(allowed_tokens), size=prefix_len)
    ].tolist()
    _, adjusted_tokens, token_mismatch = _gen_prompt_decode_to_target_len(
        tokenizer=tokenizer,
        token_sequence=prefix_tokens,
        target_token_len=prefix_len,
        allowed_tokens=allowed_tokens,
        rng=rng,
    )
    if token_mismatch != 0:
        sign = "more" if token_mismatch > 0 else "fewer"
        logger.warning(
            "Prefix tokenization produced %d %s tokens than expected after "
            "decoding and re-encoding.",
            abs(token_mismatch),
            sign,
        )
    return adjusted_tokens


def _generate_prompt_with_target_len(
    tokenizer: "PreTrainedTokenizerBase",
    prefix_token_ids: List[int],
    prefix_len: int,
    input_len: int,
    offset: int,
    index: int,
    allowed_tokens: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[str, int, int]:
    inner_sequence = allowed_tokens[
        (offset + index + np.arange(input_len)) % len(allowed_tokens)
    ].tolist()
    token_sequence = prefix_token_ids + inner_sequence
    total_input_len = prefix_len + int(input_len)
    prompt, adjusted_token_sequence, token_mismatch = _gen_prompt_decode_to_target_len(
        tokenizer=tokenizer,
        token_sequence=token_sequence,
        target_token_len=total_input_len,
        allowed_tokens=allowed_tokens,
        rng=rng,
    )
    return prompt, len(adjusted_token_sequence), token_mismatch


def sample_random_requests(
    num_requests: int,
    tokenizer: "PreTrainedTokenizerBase",
    input_len: int = 1024,
    output_len: int = 128,
    range_ratio: RangeRatio = 0.0,
    prefix_len: int = 0,
    seed: int = 0,
) -> List[Tuple[str, int, int]]:
    """Generate vLLM-style synthetic requests with reproducible token lengths."""
    if num_requests <= 0:
        raise ValueError("num_requests must be positive.")
    if input_len <= 0:
        raise ValueError("input_len must be positive.")
    if output_len <= 0:
        raise ValueError("output_len must be positive.")
    if prefix_len < 0:
        raise ValueError("prefix_len must be non-negative.")

    rng = np.random.default_rng(seed)

    input_range_ratio, _ = _resolve_range_ratios(range_ratio)
    num_special_tokens = _num_special_tokens_to_add(tokenizer)
    real_input_len = max(0, int(input_len) - num_special_tokens)
    min_sampled_input = math.floor(real_input_len * (1.0 - input_range_ratio))
    min_total_input = int(prefix_len) + min_sampled_input
    if min_total_input < 1:
        raise ValueError(
            "--random-input-len is too small: with tokenizer special tokens "
            f"{num_special_tokens} and input range ratio {input_range_ratio}, "
            "the minimum possible total input tokens is "
            f"{min_total_input}."
        )

    input_lens, output_lens, offsets = _get_sampling_params(
        rng,
        num_requests,
        range_ratio,
        input_len,
        output_len,
        tokenizer,
    )
    vocab_size = _get_vocab_size(tokenizer)
    prohibited_tokens = getattr(tokenizer, "all_special_ids", []) or []
    all_tokens = np.arange(vocab_size)
    allowed_tokens = np.setdiff1d(all_tokens, prohibited_tokens)
    if len(allowed_tokens) == 0:
        raise ValueError("Tokenizer has no non-special tokens for random sampling.")

    prefix_token_ids = _get_random_prefix(tokenizer, allowed_tokens, prefix_len, rng)

    sampled_requests = []
    token_mismatch_total = 0
    for i in range(num_requests):
        prompt, prompt_len, token_mismatch = _generate_prompt_with_target_len(
            tokenizer=tokenizer,
            prefix_token_ids=prefix_token_ids,
            prefix_len=prefix_len,
            input_len=int(input_lens[i]),
            offset=int(offsets[i]),
            index=i,
            allowed_tokens=allowed_tokens,
            rng=rng,
        )
        token_mismatch_total += token_mismatch
        sampled_requests.append((prompt, prompt_len, int(output_lens[i])))

    if token_mismatch_total != 0:
        sign = "more" if token_mismatch_total > 0 else "fewer"
        logger.warning(
            "Across all generated prompts, there were %d %s tokens than "
            "expected after decoding and re-encoding.",
            abs(token_mismatch_total),
            sign,
        )
    return sampled_requests


def generate_sorting_prompts(
    num_prompts: int,
    context_length: int,
    prompt_len_limit: int,
    tokenizer: "PreTrainedTokenizerBase",
) -> List[Tuple[str, int, int]]:
    prompts = []
    for i in range(0, num_prompts):
        random_nums = []
        _prompt_len = 0
        while True:
            r_str = "%s" % random.randint(0, 99)
            r_len = len(r_str) + 1
            if r_len + _prompt_len > prompt_len_limit:
                break
            random_nums.append(r_str)
            _prompt_len += r_len
        prompt = "Sort the numbers:" + ",".join(random_nums)
        prompts.append(prompt)
    prompt_token_ids = tokenizer(prompts).input_ids
    dataset = []
    for i in range(0, len(prompts)):
        prompt_len = len(prompt_token_ids[i])
        dataset.append((prompts[i], prompt_len, context_length - prompt_len))
    return dataset
