# Logits Processor for vLLM V1 Engine.

from typing import Any

import torch
from vllm.config import VllmConfig

try:
    from vllm.v1.sample.logits_processor.interface import (
        BatchUpdate,
        LogitsProcessor,
        MoveDirectionality,
    )
except ImportError as e:
    class BatchUpdate:
        pass

    class LogitsProcessor:
        pass

    class MoveDirectionality:
        pass


def _get_int_value(extra_args: dict[str, Any] | None, key: str) -> int | None:
    if isinstance(extra_args, dict):
        arg_value = extra_args.get(key)
        if arg_value is not None:
            try:
                return int(arg_value)
            except Exception:
                pass
    return None


class VllmV1NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """
    Prevents repeating the same n-gram of specified size in the output.
    Inspired by Hugging Face's NoRepeatNGramLogitsProcessor.
    Handled Extra Args:
        no_repeat_ngram_size (int): Size of the n-gram to avoid repeating.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
        # mapping: index -> (no_repeat_ngram_size, output_tok_ids, cached_ngrams)
        self.req_info: dict[int, tuple[int, list[int], dict[tuple, list[int]]]] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if not batch_update:
            return

        for index in batch_update.removed:
            self.req_info.pop(index, None)

        for index, params, _, output_tok_ids in batch_update.added:
            val = _get_int_value(params.extra_args, "no_repeat_ngram_size")
            no_repeat_ngram_size = 0 if (val is None or val < 0) else val
            if isinstance(params.extra_args, dict) and params.extra_args.get("debug"):
                print(f"Request {index}: no_repeat_ngram_size = {no_repeat_ngram_size}")
            self.req_info[index] = (no_repeat_ngram_size, output_tok_ids, {})

        for a_index, b_index, direct in batch_update.moved:
            a_info = self.req_info.pop(a_index, None)
            b_info = self.req_info.pop(b_index, None)
            if a_info is not None:
                self.req_info[b_index] = a_info
            if direct == MoveDirectionality.SWAP and b_info is not None:
                self.req_info[a_index] = b_info

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for index in range(len(logits)):
            req_info = self.req_info.get(index)
            if req_info is None:
                continue
            no_repeat_ngram_size, output_tok_ids, cached_ngrams = req_info
            if no_repeat_ngram_size <= 0:
                continue
            # Skip if there are not enough tokens to form an n-gram
            if len(output_tok_ids) < no_repeat_ngram_size:
                continue

            # Get the n-gram prefix (all but the last token)
            prev_ngram = tuple(output_tok_ids[-no_repeat_ngram_size:-1])
            last_token = output_tok_ids[-1]

            # Store this n-gram occurrence
            cached_ngrams.setdefault(prev_ngram, []).append(last_token)

            # Get the next-token candidates to ban based on current prefix
            current_prefix = tuple(output_tok_ids[-no_repeat_ngram_size + 1 :])
            banned_tokens = cached_ngrams.get(current_prefix, [])

            # Set the logits of banned tokens to negative infinity
            for token in banned_tokens:
                logits[index][token] = -float("inf")

        return logits
