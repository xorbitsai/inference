# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Request-local JSON constraints for MLX generation."""

from typing import Any, Dict, Optional


class JSONLogitsProcessor:
    def __init__(
        self, tokenizer: Any, schema: Dict, reasoning_end: Optional[str] = None
    ):
        try:
            from llguidance import LLMatcher
            from llguidance.hf import from_tokenizer
        except ImportError as exc:
            raise NotImplementedError(
                "MLX structured output requires llguidance; install xinference[mlx]"
            ) from exc

        from transformers import PreTrainedTokenizerFast

        eos = getattr(tokenizer, "eos_token_ids", None)
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

        current = tokens.reshape(-1).tolist()
        if self._seen is None:
            # mlx-lm passes prompt+output, mlx-vlm passes output only. Neither
            # has generated a token on the first call.
            self._seen = len(current)
        for token in current[self._seen :]:
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
