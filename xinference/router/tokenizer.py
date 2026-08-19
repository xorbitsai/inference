from __future__ import annotations

import importlib.util
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from tokenizers import Tokenizer


class TokenizationError(ValueError):
    """Raised when a request cannot be rendered or tokenized safely."""


@dataclass(frozen=True)
class TokenBudget:
    prompt_tokens: int
    output_tokens: int
    reserve_tokens: int
    total_tokens: int
    enable_thinking: bool


def _load_encoding_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"deepseek_v4_router_encoding_{abs(hash(path))}", path
    )
    if spec is None or spec.loader is None:
        raise TokenizationError(f"Unable to load DeepSeek-V4 encoding module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "encode_messages", None)):
        raise TokenizationError(f"Encoding module has no encode_messages(): {path}")
    return module


def _normalize_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for original in messages:
        if not isinstance(original, dict):
            raise TokenizationError("Each chat message must be an object")
        message = dict(original)
        content = message.get("content")
        if content is None:
            message["content"] = ""
        elif isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text":
                    raise TokenizationError(
                        "The phase-1 Router only supports text chat content"
                    )
                text_parts.append(str(part.get("text", "")))
            message["content"] = "\n".join(text_parts)
        elif not isinstance(content, str):
            raise TokenizationError("Message content must be a string or text list")
        normalized.append(message)
    return normalized


def _attach_tools(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    prepared = [dict(message) for message in messages]
    for message in prepared:
        if message.get("role") in {"system", "developer"}:
            existing_tools = message.get("tools") or []
            message["tools"] = [*existing_tools, *tools]
            return prepared
    return [{"role": "system", "content": "", "tools": tools}, *prepared]


def _chat_template_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("chat_template_kwargs") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TokenizationError("chat_template_kwargs must be valid JSON") from exc
    if not isinstance(raw, dict):
        raise TokenizationError("chat_template_kwargs must be an object")
    return dict(raw)


class DeepSeekV4TokenEstimator:
    """Mirror Xinference's DeepSeek-V4 chat rendering before routing."""

    def __init__(
        self,
        tokenizer_path: Path,
        *,
        reserve_tokens: int,
        default_output_tokens: int,
    ) -> None:
        tokenizer_json = tokenizer_path / "tokenizer.json"
        encoding_path = tokenizer_path / "encoding" / "encoding_dsv4.py"
        if not tokenizer_json.is_file():
            raise TokenizationError(f"Missing tokenizer.json: {tokenizer_json}")
        if not encoding_path.is_file():
            raise TokenizationError(f"Missing encoding module: {encoding_path}")
        self._tokenizer = Tokenizer.from_file(str(tokenizer_json))
        self._encoding = _load_encoding_module(encoding_path)
        self._reserve_tokens = reserve_tokens
        self._default_output_tokens = default_output_tokens

    def estimate(self, payload: dict[str, Any]) -> TokenBudget:
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise TokenizationError("messages must be a non-empty list")
        messages = _normalize_content(raw_messages)
        tools = payload.get("tools") or []
        if tools:
            if not isinstance(tools, list):
                raise TokenizationError("tools must be a list")
            messages = _attach_tools(messages, tools)

        template_kwargs = _chat_template_kwargs(payload)
        enable_thinking = bool(template_kwargs.get("enable_thinking", False))
        encode_messages = self._encoding.encode_messages
        kwargs: dict[str, Any] = {
            "thinking_mode": "thinking" if enable_thinking else "chat"
        }
        signature = inspect.signature(encode_messages)
        for name in signature.parameters:
            if name in template_kwargs:
                kwargs[name] = template_kwargs[name]
        try:
            prompt = encode_messages(messages, **kwargs)
        except Exception as exc:
            raise TokenizationError(
                f"DeepSeek-V4 chat rendering failed: {exc}"
            ) from exc
        if not isinstance(prompt, str):
            raise TokenizationError("DeepSeek-V4 renderer did not return text")

        prompt_tokens = len(self._tokenizer.encode(prompt, add_special_tokens=True).ids)
        output_value = payload.get("max_tokens")
        if output_value is None:
            output_value = payload.get("max_completion_tokens")
        if output_value is None:
            output_value = self._default_output_tokens
        if isinstance(output_value, bool) or not isinstance(output_value, int):
            raise TokenizationError("max_tokens must be an integer")
        if output_value < 1:
            raise TokenizationError("max_tokens must be at least 1")

        total_tokens = prompt_tokens + output_value + self._reserve_tokens
        return TokenBudget(
            prompt_tokens=prompt_tokens,
            output_tokens=output_value,
            reserve_tokens=self._reserve_tokens,
            total_tokens=total_tokens,
            enable_thinking=enable_thinking,
        )
