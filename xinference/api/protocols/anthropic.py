# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Anthropic Messages protocol adapter.

The public Anthropic protocol terminates in the Supervisor.  Both physical
models and Token Router virtual models consume the same OpenAI-compatible
canonical request and produce the same Anthropic response representation.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional


class AnthropicProtocolError(ValueError):
    def __init__(
        self,
        status_code: int,
        error_type: str,
        message: str,
        *,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.message = message
        self.headers = headers or {}


@dataclass(frozen=True)
class CanonicalChatRequest:
    requested_model: str
    messages: List[Dict[str, Any]]
    max_output_tokens: int
    stream: bool
    stop: Optional[List[str]]
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    tools: Optional[List[Dict[str, Any]]]
    tool_choice: Any
    metadata: Dict[str, Any]
    thinking_enabled: Optional[bool]
    thinking_budget_tokens: Optional[int]

    def to_openai_body(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.requested_model,
            "messages": self.messages,
            "max_tokens": self.max_output_tokens,
            "stream": self.stream,
        }
        if self.stop:
            body["stop"] = self.stop
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if self.top_k is not None:
            body["top_k"] = self.top_k
        if self.tools:
            body["tools"] = self.tools
        if self.tool_choice is not None:
            body["tool_choice"] = self.tool_choice
        if self.thinking_enabled is not None:
            chat_template_kwargs: Dict[str, Any] = {
                "enable_thinking": self.thinking_enabled,
                "thinking": self.thinking_enabled,
            }
            if self.thinking_budget_tokens is not None:
                # Preserve Anthropic's requested reasoning budget for backends
                # and custom chat templates that support a token budget.
                chat_template_kwargs["thinking_budget"] = self.thinking_budget_tokens
            body["chat_template_kwargs"] = chat_template_kwargs
        if self.stream:
            # Accurate Anthropic message_delta usage requires a final OpenAI
            # usage chunk. Backends that support this option will emit one.
            body["stream_options"] = {"include_usage": True}
        return body


def _invalid(message: str) -> AnthropicProtocolError:
    return AnthropicProtocolError(400, "invalid_request_error", message)


def _text_from_blocks(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise _invalid("Message content must be a string or an array of blocks")
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            raise _invalid("Anthropic content blocks must be JSON objects")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise _invalid("Anthropic text blocks require a string 'text' field")
            if text.startswith("x-anthropic-billing-header"):
                continue
            parts.append(text)
        else:
            raise _invalid(f"Unsupported Anthropic content block type: {block_type}")
    return "\n".join(parts)


def _normalize_messages(raw_system: Any, raw_messages: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_messages, list) or not raw_messages:
        raise _invalid("Please specify at least one message")

    system_parts: List[str] = []
    if raw_system is not None:
        system_text = _text_from_blocks(raw_system)
        if system_text:
            system_parts.append(system_text)

    normalized: List[Dict[str, Any]] = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            raise _invalid("Every Anthropic message must be a JSON object")
        role = raw_message.get("role")
        content = raw_message.get("content")
        if role == "system":
            text = _text_from_blocks(content)
            if text:
                system_parts.append(text)
            continue
        if role not in {"user", "assistant"}:
            raise _invalid(f"Unsupported Anthropic message role: {role}")

        if isinstance(content, str):
            normalized.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise _invalid("Message content must be a string or an array of blocks")

        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        tool_results: List[Dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                raise _invalid("Anthropic content blocks must be JSON objects")
            block_type = block.get("type")
            if block_type == "text":
                block_text = block.get("text")
                if not isinstance(block_text, str):
                    raise _invalid(
                        "Anthropic text blocks require a string 'text' field"
                    )
                if not block_text.startswith("x-anthropic-billing-header"):
                    text_parts.append(block_text)
            elif block_type == "tool_use" and role == "assistant":
                tool_id = block.get("id")
                name = block.get("name")
                input_data = block.get("input", {})
                if not isinstance(tool_id, str) or not isinstance(name, str):
                    raise _invalid("tool_use blocks require string 'id' and 'name'")
                tool_calls.append(
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(input_data, ensure_ascii=False),
                        },
                    }
                )
            elif block_type == "tool_result" and role == "user":
                tool_id = block.get("tool_use_id")
                if not isinstance(tool_id, str):
                    raise _invalid("tool_result blocks require a string 'tool_use_id'")
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    result_content = _text_from_blocks(result_content)
                elif not isinstance(result_content, str):
                    result_content = json.dumps(result_content, ensure_ascii=False)
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result_content,
                    }
                )
            else:
                raise _invalid(
                    f"Unsupported Anthropic content block type: {block_type}"
                )

        if role == "assistant":
            message: Dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_parts) or None,
            }
            if tool_calls:
                message["tool_calls"] = tool_calls
            normalized.append(message)
        else:
            if text_parts:
                normalized.append({"role": "user", "content": "\n".join(text_parts)})
            normalized.extend(tool_results)

    if system_parts:
        normalized.insert(0, {"role": "system", "content": "\n".join(system_parts)})
    if not normalized or normalized[-1].get("role") not in {
        "user",
        "assistant",
        "tool",
    }:
        raise _invalid("Please specify the prompt")
    return normalized


def _normalize_tools(raw_tools: Any) -> Optional[List[Dict[str, Any]]]:
    if raw_tools is None:
        return None
    if not isinstance(raw_tools, list):
        raise _invalid("The 'tools' field must be an array")
    tools: List[Dict[str, Any]] = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            raise _invalid("Every tool must be a JSON object")
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            tools.append(tool)
            continue
        name = tool.get("name")
        schema = tool.get("input_schema")
        if not isinstance(name, str) or not isinstance(schema, dict):
            raise _invalid("Anthropic tools require 'name' and object 'input_schema'")
        function: Dict[str, Any] = {"name": name, "parameters": schema}
        if isinstance(tool.get("description"), str):
            function["description"] = tool["description"]
        tools.append({"type": "function", "function": function})
    return tools or None


def _normalize_tool_choice(raw_choice: Any) -> Any:
    if raw_choice is None:
        return None
    if isinstance(raw_choice, str):
        return {"any": "required"}.get(raw_choice, raw_choice)
    if not isinstance(raw_choice, dict):
        raise _invalid("The 'tool_choice' field must be a string or object")
    choice_type = raw_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "none":
        return "none"
    if choice_type in {"tool", "function"}:
        name = raw_choice.get("name")
        if name is None and isinstance(raw_choice.get("function"), dict):
            name = raw_choice["function"].get("name")
        if not isinstance(name, str) or not name:
            raise _invalid("A named tool_choice requires a tool name")
        return {"type": "function", "function": {"name": name}}
    raise _invalid(f"Unsupported Anthropic tool_choice type: {choice_type}")


def _optional_number(
    raw_body: Dict[str, Any],
    name: str,
    *,
    minimum: float,
    maximum: Optional[float] = None,
) -> Optional[float]:
    value = raw_body.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid(f"The '{name}' field must be a number")
    number = float(value)
    if number < minimum or (maximum is not None and number > maximum):
        if maximum is None:
            raise _invalid(f"The '{name}' field must be at least {minimum}")
        raise _invalid(f"The '{name}' field must be between {minimum} and {maximum}")
    return number


def parse_anthropic_request(raw_body: Any) -> CanonicalChatRequest:
    if not isinstance(raw_body, dict):
        raise _invalid("Request body must be a JSON object")
    model = raw_body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise _invalid("The 'model' field is required")
    max_tokens = raw_body.get("max_tokens")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise _invalid("The 'max_tokens' field must be a positive integer")
    stream = raw_body.get("stream", False)
    if not isinstance(stream, bool):
        raise _invalid("The 'stream' field must be a boolean")
    temperature = _optional_number(raw_body, "temperature", minimum=0.0, maximum=1.0)
    top_p = _optional_number(raw_body, "top_p", minimum=0.0, maximum=1.0)
    top_k = raw_body.get("top_k")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0
    ):
        raise _invalid("The 'top_k' field must be a positive integer")
    stop_sequences = raw_body.get("stop_sequences")
    if stop_sequences is not None:
        if not isinstance(stop_sequences, list) or not all(
            isinstance(item, str) for item in stop_sequences
        ):
            raise _invalid("The 'stop_sequences' field must be an array of strings")
    metadata = raw_body.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise _invalid("The 'metadata' field must be an object")

    thinking = raw_body.get("thinking")
    thinking_enabled: Optional[bool] = None
    thinking_budget: Optional[int] = None
    if thinking is not None:
        if not isinstance(thinking, dict):
            raise _invalid("The 'thinking' field must be an object")
        thinking_type = thinking.get("type")
        if thinking_type not in {"enabled", "disabled"}:
            raise _invalid("Unsupported thinking type")
        thinking_enabled = thinking_type == "enabled"
        if thinking_enabled:
            thinking_budget = thinking.get("budget_tokens")
            if (
                isinstance(thinking_budget, bool)
                or not isinstance(thinking_budget, int)
                or thinking_budget <= 0
            ):
                raise _invalid("Enabled thinking requires a positive budget_tokens")
            if thinking_budget >= max_tokens:
                raise _invalid("thinking.budget_tokens must be less than max_tokens")

    return CanonicalChatRequest(
        requested_model=model.strip(),
        messages=_normalize_messages(raw_body.get("system"), raw_body.get("messages")),
        max_output_tokens=max_tokens,
        stream=stream,
        stop=stop_sequences,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        tools=_normalize_tools(raw_body.get("tools")),
        tool_choice=_normalize_tool_choice(raw_body.get("tool_choice")),
        metadata=metadata,
        thinking_enabled=thinking_enabled,
        thinking_budget_tokens=thinking_budget,
    )


def _stop_reason(finish_reason: Any, stop_sequence: Any = None) -> str:
    if stop_sequence:
        return "stop_sequence"
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }.get(finish_reason, "end_turn")


def openai_to_anthropic(openai_response: Dict[str, Any], model: str) -> Dict[str, Any]:
    content_blocks: List[Dict[str, Any]] = []
    finish_reason: Any = "stop"
    stop_sequence = openai_response.get("stop_sequence")
    choices = openai_response.get("choices") or []
    if choices:
        choice = choices[0]
        finish_reason = choice.get("finish_reason") or "stop"
        message = choice.get("message") or {}
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            content_blocks.append({"type": "thinking", "thinking": reasoning})
        content = message.get("content")
        if isinstance(content, str) and content:
            content_blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    content_blocks.append({"type": "text", "text": block["text"]})
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments", "{}")
            try:
                input_data = (
                    json.loads(arguments) if isinstance(arguments, str) else arguments
                )
            except json.JSONDecodeError:
                input_data = {}
            if not isinstance(input_data, dict):
                input_data = {}
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id") or f"toolu_{uuid.uuid4().hex}",
                    "name": function.get("name", ""),
                    "input": input_data,
                }
            )
    usage = openai_response.get("usage") or {}
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": _stop_reason(finish_reason, stop_sequence),
        "stop_sequence": stop_sequence,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


def anthropic_error_response(
    error_type: str, message: str, request_id: str
) -> Dict[str, Any]:
    return {
        "type": "error",
        "error": {"type": error_type, "message": message},
        "request_id": request_id,
    }


async def anthropic_stream_events(
    chunks: AsyncIterator[Dict[str, Any]], model: str, request_id: str
) -> AsyncIterator[Dict[str, str]]:
    """Convert OpenAI chat-completion chunks into Anthropic SSE events.

    Anthropic content blocks are strictly sequential: a block must be stopped
    before the next block starts. OpenAI tool calls can be fragmented and even
    interleaved by index, so tool fragments are buffered and emitted as complete
    sequential Anthropic blocks after text/reasoning streaming finishes.
    """

    message_id = f"msg_{uuid.uuid4().hex}"
    yield {
        "event": "message_start",
        "data": json.dumps(
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }
        ),
    }

    next_index = 0
    active_index: Optional[int] = None
    active_kind: Optional[str] = None
    finish_reason: Any = "stop"
    stop_sequence: Any = None
    output_tokens = 0
    tool_buffers: Dict[int, Dict[str, Any]] = {}

    def block_stop(index: int) -> Dict[str, str]:
        return {
            "event": "content_block_stop",
            "data": json.dumps({"type": "content_block_stop", "index": index}),
        }

    try:
        async for chunk in chunks:
            if "error" in chunk:
                error = chunk.get("error") or {}
                yield {
                    "event": "error",
                    "data": json.dumps(
                        anthropic_error_response(
                            error.get("type", "api_error"),
                            error.get("message", "Upstream stream failed"),
                            request_id,
                        )
                    ),
                }
                return

            usage = chunk.get("usage") or {}
            output_tokens = int(usage.get("completion_tokens") or output_tokens)
            choices = chunk.get("choices") or []
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta") or {}
            for kind, value, delta_type, field in (
                (
                    "thinking",
                    delta.get("reasoning_content"),
                    "thinking_delta",
                    "thinking",
                ),
                ("text", delta.get("content"), "text_delta", "text"),
            ):
                if not isinstance(value, str) or not value:
                    continue
                if active_kind != kind:
                    if active_index is not None:
                        yield block_stop(active_index)
                    active_index = next_index
                    next_index += 1
                    active_kind = kind
                    content_block = {"type": kind, field: ""}
                    yield {
                        "event": "content_block_start",
                        "data": json.dumps(
                            {
                                "type": "content_block_start",
                                "index": active_index,
                                "content_block": content_block,
                            }
                        ),
                    }
                yield {
                    "event": "content_block_delta",
                    "data": json.dumps(
                        {
                            "type": "content_block_delta",
                            "index": active_index,
                            "delta": {"type": delta_type, field: value},
                        }
                    ),
                }

            for tool_call in delta.get("tool_calls") or []:
                try:
                    source_index = int(tool_call.get("index") or 0)
                except (TypeError, ValueError):
                    source_index = 0
                state = tool_buffers.setdefault(
                    source_index,
                    {"id": None, "name": "", "argument_fragments": []},
                )
                if tool_call.get("id"):
                    state["id"] = tool_call["id"]
                function = tool_call.get("function") or {}
                if function.get("name"):
                    state["name"] = function["name"]
                arguments = function.get("arguments")
                if isinstance(arguments, str) and arguments:
                    state["argument_fragments"].append(arguments)

            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
                stop_sequence = choice.get("stop_sequence") or stop_sequence

    finally:
        close = getattr(chunks, "aclose", None)
        if close is not None:
            await close()

    if active_index is not None:
        yield block_stop(active_index)

    for source_index in sorted(tool_buffers):
        state = tool_buffers[source_index]
        block_index = next_index
        next_index += 1
        yield {
            "event": "content_block_start",
            "data": json.dumps(
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": state["id"] or f"toolu_{uuid.uuid4().hex}",
                        "name": state["name"],
                        "input": {},
                    },
                }
            ),
        }
        for arguments in state["argument_fragments"]:
            yield {
                "event": "content_block_delta",
                "data": json.dumps(
                    {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": arguments,
                        },
                    }
                ),
            }
        yield block_stop(block_index)

    yield {
        "event": "message_delta",
        "data": json.dumps(
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": _stop_reason(finish_reason, stop_sequence),
                    "stop_sequence": stop_sequence,
                },
                "usage": {"output_tokens": output_tokens},
            }
        ),
    }
    yield {"event": "message_stop", "data": json.dumps({"type": "message_stop"})}
