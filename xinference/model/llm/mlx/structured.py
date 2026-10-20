# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Request-local JSON constraints for MLX generation."""

from typing import Any, Dict, List, Optional


class ReasoningBudgetLogitsProcessor:
    """Close thinking at its token budget while leaving answer tokens available."""

    def __init__(
        self,
        tokenizer: Any,
        end: str,
        budget: int,
        eos_token_ids: Optional[List[int]] = None,
    ):
        if type(budget) is not int or budget < 1:
            raise ValueError("Thinking budget must be a positive integer")
        self._end = tokenizer.encode(end, add_special_tokens=False) if end else []
        unknown = getattr(tokenizer, "unk_token_id", None)
        if not self._end or unknown is not None and unknown in self._end:
            raise NotImplementedError("MLX thinking budgets require known end markers")
        eos = (
            eos_token_ids
            if eos_token_ids is not None
            else getattr(tokenizer, "eos_token_ids", None)
        )
        if eos is None:
            eos = [tokenizer.eos_token_id]
        self._eos = [eos] if isinstance(eos, int) else list(eos)
        self._budget = budget
        self._seen: Optional[int] = None
        self._generated: list = []
        self._forced = 0
        self._done = False

    def __call__(self, tokens: Any, logits: Any) -> Any:
        import mlx.core as mx

        current = tokens.reshape(-1)
        if self._seen is None:
            self._seen = len(current)
        self._generated.extend(current[self._seen :].tolist())
        self._seen = len(current)
        if self._generated[-len(self._end) :] == self._end:
            self._done = True
        if self._done:
            return logits
        if len(self._generated) >= self._budget:
            if not self._forced:
                for width in range(len(self._end) - 1, 0, -1):
                    if self._generated[-width:] == self._end[:width]:
                        self._forced = width
                        break
            token = self._end[self._forced]
            self._forced += 1
            return mx.where(
                mx.arange(logits.shape[-1]) == token, 0.0, -float("inf")
            ).reshape(logits.shape)
        for token in self._eos:
            logits = mx.where(
                mx.arange(logits.shape[-1]) == token, -float("inf"), logits
            )
        return logits


class JSONLogitsProcessor:
    def __init__(
        self,
        tokenizer: Any,
        schema: Dict,
        reasoning_end: Optional[str] = None,
        eos_token_ids: Optional[List[int]] = None,
    ):
        try:
            from llguidance import LLMatcher
            from llguidance.hf import from_tokenizer
        except ImportError as exc:
            raise NotImplementedError(
                "MLX structured output requires llguidance; install xinference[mlx]"
            ) from exc

        from transformers import PreTrainedTokenizerFast

        eos = (
            eos_token_ids
            if eos_token_ids is not None
            else getattr(tokenizer, "eos_token_ids", None)
        )
        if eos is None:
            eos = [tokenizer.eos_token_id]
        self._eos = [eos] if isinstance(eos, int) else list(eos)
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
        self.matcher = LLMatcher(
            from_tokenizer(tokenizer, eos_token=self._eos),
            LLMatcher.grammar_from_json_schema(schema),
        )
        if self.matcher.is_error():
            raise ValueError(self.matcher.get_error())
        self._reasoning_end = (
            tokenizer.encode(reasoning_end, add_special_tokens=False)
            if reasoning_end
            else []
        )
        self._reasoning = bool(self._reasoning_end)
        self._seen: Optional[int] = None
        self._tail: list = []

    def __call__(self, tokens: Any, logits: Any) -> Any:
        import numpy as np
        from llguidance.mlx import apply_token_bitmask

        current = tokens.reshape(-1)
        if self._seen is None:
            # mlx-lm passes prompt+output, mlx-vlm passes output only. Neither
            # has generated a token on the first call.
            self._seen = len(current)
        for token in current[self._seen :].tolist():
            if self._reasoning:
                self._tail = (self._tail + [token])[-len(self._reasoning_end) :]
                if self._tail == self._reasoning_end:
                    self._reasoning = False
            elif not self.matcher.consume_token(token):
                raise RuntimeError(self.matcher.get_error() or "Invalid JSON token")
        self._seen = len(current)
        if self._reasoning:
            import mlx.core as mx

            for token in self._eos:
                logits = mx.where(
                    mx.arange(logits.shape[-1]) == token, -float("inf"), logits
                )
            return logits
        mask = np.frombuffer(self.matcher.compute_bitmask(), dtype=np.int32)
        result = apply_token_bitmask(logits, mask)
        return result.reshape(logits.shape)


def json_schema(response_format: Dict) -> Dict:
    kind = response_format.get("type")
    if kind == "json_object":
        return {"type": "object"}
    if kind != "json_schema":
        raise ValueError("MLX structured output requires json_object or json_schema")
    block = response_format.get("json_schema") or {}
    schema = block.get("schema_", block.get("schema"))
    if not isinstance(schema, dict):
        raise ValueError("Structured output requires a JSON schema object")
    return schema
