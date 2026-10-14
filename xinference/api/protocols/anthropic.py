# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Anthropic Messages protocol adapter.

The public Anthropic protocol terminates in the Supervisor.  Both physical
models and Token Router virtual models consume the same OpenAI-compatible
canonical request and produce the same Anthropic response representation.
"""

from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from .anthropic_content import ContentBudget, normalize_content
from .anthropic_thinking import sign_thinking, verify_thinking


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
    response_format: Optional[Dict[str, Any]] = None
    reasoning_effort: Optional[str] = None
    parallel_tool_calls: Optional[bool] = None
    cache_control: Optional[Dict[str, Any]] = None

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
        if self.response_format is not None:
            body["response_format"] = self.response_format
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        if self.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = self.parallel_tool_calls
        if self.cache_control is not None:
            body["_anthropic_cache_control"] = self.cache_control
        if self.thinking_enabled is not None:
            chat_template_kwargs: Dict[str, Any] = {
                "enable_thinking": self.thinking_enabled,
                "thinking": self.thinking_enabled,
            }
            if self.thinking_enabled:
                chat_template_kwargs["reasoning_content"] = True
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


def _normalize_messages(
    raw_system: Any,
    raw_messages: Any,
    tool_names: Optional[set] = None,
    *,
    budget: Optional[ContentBudget] = None,
    model: str = "",
) -> List[Dict[str, Any]]:
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
        content_parts: List[Dict[str, Any]] = []
        reasoning_parts: List[str] = []
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
                    content_parts.append({"type": "text", "text": block_text})
            elif block_type in {"image", "document", "tool_reference"}:
                try:
                    converted = normalize_content([block], tool_names, budget)
                except ValueError as exc:
                    raise _invalid(str(exc)) from exc
                if isinstance(converted, str):
                    text_parts.append(converted)
                    content_parts.append({"type": "text", "text": converted})
                else:
                    content_parts.extend(converted)
            elif block_type == "thinking" and role == "assistant":
                reasoning = block.get("thinking")
                if not isinstance(reasoning, str):
                    raise _invalid("thinking blocks require a string 'thinking' field")
                signature = block.get("signature")
                # Stale, foreign and unsigned reasoning is opaque history. Keep
                # the answer/tool calls usable across model or signing-key changes.
                if (
                    isinstance(signature, str)
                    and signature.startswith("xinference_v1:")
                    and verify_thinking(reasoning, model, signature)
                ):
                    reasoning_parts.append(reasoning)
            elif block_type == "redacted_thinking" and role == "assistant":
                if not isinstance(block.get("data"), str):
                    raise _invalid("redacted_thinking blocks require string data")
            elif block_type == "tool_use" and role == "assistant":
                tool_id = block.get("id")
                name = block.get("name")
                input_data = block.get("input", {})
                if not isinstance(input_data, dict):
                    raise _invalid("tool_use input must be an object")
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
                    try:
                        result_content = normalize_content(
                            result_content, tool_names, budget
                        )
                    except ValueError as exc:
                        raise _invalid(str(exc)) from exc
                elif not isinstance(result_content, str):
                    result_content = json.dumps(result_content, ensure_ascii=False)
                if block.get("is_error"):
                    if isinstance(result_content, str):
                        result_content = "Tool error: " + result_content
                    else:
                        result_content.insert(
                            0, {"type": "text", "text": "Tool error:"}
                        )
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
                "content": (
                    content_parts
                    if any(p["type"] != "text" for p in content_parts)
                    else "\n".join(text_parts) or None
                ),
            }
            if reasoning_parts:
                message["reasoning_content"] = "\n".join(reasoning_parts)
            if tool_calls:
                message["tool_calls"] = tool_calls
            normalized.append(message)
        else:
            normalized.extend(tool_results)
            if content_parts:
                normalized.append(
                    {
                        "role": "user",
                        "content": (
                            content_parts
                            if any(p["type"] != "text" for p in content_parts)
                            else "\n".join(text_parts)
                        ),
                    }
                )

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
        if raw_choice not in {"auto", "any", "required", "none"}:
            raise _invalid(f"Unsupported Anthropic tool_choice type: {raw_choice}")
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


def _parse_cache_control(
    raw_body: Dict[str, Any],
    tool_names: set,
    messages: List[Dict[str, Any]],
    budget: ContentBudget,
) -> Optional[Dict[str, Any]]:
    """Retain the furthest explicit cache boundary, including its lifetime."""
    controls: List[Dict[str, Any]] = []
    system = raw_body.get("system")
    prefix_system = system
    prefix_tools = None
    for index, tool in enumerate(raw_body.get("tools") or []):
        if tool.get("cache_control") is not None:
            controls.append(tool["cache_control"])
            prefix_tools = _normalize_tools(raw_body["tools"][: index + 1])
    system_blocks = system if isinstance(system, list) else []
    for index, block in enumerate(system_blocks):
        if isinstance(block, dict) and block.get("cache_control") is not None:
            controls.append(block["cache_control"])
            prefix_system = system_blocks[: index + 1]
            # Chat templates may place tools after the system text. Do not
            # extend such a template's prefix past the marked system block.
            prefix_tools = []
    prefix = None
    for index, message in enumerate(raw_body.get("messages") or []):
        content = message.get("content")
        for block_index, block in enumerate(
            content if isinstance(content, list) else []
        ):
            if isinstance(block, dict) and block.get("cache_control") is not None:
                controls.append(block["cache_control"])
                prefix = [
                    *raw_body["messages"][:index],
                    {**message, "content": content[: block_index + 1]},
                ]
                prefix_system = system
                prefix_tools = None
    if raw_body.get("cache_control") is not None:
        controls.append(raw_body["cache_control"])
        prefix = raw_body.get("messages")
        prefix_system = system
        prefix_tools = None
    if not controls:
        return None
    if len(controls) > 4:
        raise _invalid("At most four cache_control breakpoints are supported")
    for control in controls:
        if not isinstance(control, dict) or control.get("type") != "ephemeral":
            raise _invalid("cache_control requires type 'ephemeral'")
        if control.get("ttl", "5m") not in {"5m", "1h"}:
            raise _invalid("cache_control.ttl must be '5m' or '1h'")
    ttl = 3600 if controls[-1].get("ttl") == "1h" else 300
    has_images = any(
        isinstance(message.get("content"), list)
        and any(part.get("type") == "image_url" for part in message["content"])
        for message in messages
    )
    if has_images:
        # Current image paths use automatic engine caching. Do not carry a
        # second copy of rendered pages through the preparation/generation RPCs.
        prefix_messages = []
    elif prefix is None:
        prefix_messages = (
            [{"role": "system", "content": _text_from_blocks(prefix_system)}]
            if prefix_system
            else []
        )
    else:
        prefix_messages = _normalize_messages(
            system, prefix, tool_names, budget=budget, model=raw_body["model"].strip()
        )
    result: Dict[str, Any] = {"messages": prefix_messages, "ttl": ttl}
    if prefix_tools is not None:
        result["tools"] = prefix_tools
    return result


def anthropic_request_model(raw_body: Any) -> str:
    """Validate the model name without decoding or fetching request content."""
    if not isinstance(raw_body, dict):
        raise _invalid("Request body must be a JSON object")
    model = raw_body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise _invalid("The 'model' field is required")
    return model.strip()


def _parse_context_management(raw_value: Any) -> Optional[Dict[str, Any]]:
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise _invalid("context_management must be an object")
    edits = raw_value.get("edits")
    if not isinstance(edits, list):
        raise _invalid("context_management.edits must be an array")

    clear_thinking = None
    seen = set()
    for edit in edits:
        if not isinstance(edit, dict):
            raise _invalid("Every context management edit must be an object")
        edit_type = edit.get("type")
        if not isinstance(edit_type, str) or not edit_type:
            raise _invalid("Context management edits require a string 'type'")
        if edit_type in seen:
            raise _invalid(f"Duplicate Anthropic context management edit: {edit_type}")
        seen.add(edit_type)
        if edit_type != "clear_thinking_20251015":
            raise _invalid(
                f"Unsupported Anthropic context management edit type: {edit_type}"
            )

        keep = edit.get("keep")
        if keep == "all":
            clear_thinking = {"keep_all": True}
        elif (
            isinstance(keep, dict)
            and keep.get("type") == "thinking_turns"
            and isinstance(keep.get("value"), int)
            and not isinstance(keep.get("value"), bool)
            and keep["value"] > 0
        ):
            clear_thinking = {"keep_turns": keep["value"]}
        else:
            raise _invalid(
                "clear_thinking_20251015.keep must be 'all' or a positive "
                "thinking_turns value"
            )
    return clear_thinking


def _apply_context_management(
    messages: List[Dict[str, Any]], clear_thinking: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if clear_thinking is None or clear_thinking.get("keep_all"):
        return messages

    prepared = copy.deepcopy(messages)
    resolved_tool_ids = {
        message.get("tool_call_id")
        for message in prepared
        if message.get("role") == "tool"
    }
    thinking_indexes = [
        index
        for index, message in enumerate(prepared)
        if message.get("role") == "assistant" and message.get("reasoning_content")
    ]
    keep_indexes = set(thinking_indexes[-clear_thinking["keep_turns"] :])
    for index in thinking_indexes:
        tool_calls = prepared[index].get("tool_calls") or []
        if any(call.get("id") not in resolved_tool_ids for call in tool_calls):
            keep_indexes.add(index)
    for index in thinking_indexes:
        if index not in keep_indexes:
            prepared[index].pop("reasoning_content", None)
    return prepared


def parse_anthropic_request(raw_body: Any) -> CanonicalChatRequest:
    model = anthropic_request_model(raw_body)
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
        if thinking_type not in {"enabled", "disabled", "adaptive"}:
            raise _invalid("Unsupported thinking type")
        thinking_enabled = thinking_type != "disabled"
        if thinking_type == "enabled":
            thinking_budget = thinking.get("budget_tokens")
            if (
                isinstance(thinking_budget, bool)
                or not isinstance(thinking_budget, int)
                or thinking_budget <= 0
            ):
                raise _invalid("Enabled thinking requires a positive budget_tokens")
            if thinking_budget < 1024:
                raise _invalid(
                    "thinking.budget_tokens must be greater than or equal to 1024"
                )
            if thinking_budget >= max_tokens:
                raise _invalid("thinking.budget_tokens must be less than max_tokens")

    clear_thinking = _parse_context_management(raw_body.get("context_management"))
    tools = _normalize_tools(raw_body.get("tools"))
    tool_names = {tool["function"]["name"] for tool in tools or []}
    budget = ContentBudget()
    messages = _normalize_messages(
        raw_body.get("system"),
        raw_body.get("messages"),
        tool_names,
        budget=budget,
        model=model,
    )
    messages = _apply_context_management(messages, clear_thinking)
    output_config = raw_body.get("output_config") or {}
    if not isinstance(output_config, dict):
        raise _invalid("output_config must be an object")
    effort = output_config.get("effort")
    if effort is not None and effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise _invalid("Unsupported output_config.effort")
    response_format = None
    format_config = output_config.get("format") or raw_body.get("output_format")
    if format_config is not None:
        if (
            not isinstance(format_config, dict)
            or format_config.get("type") != "json_schema"
            or not isinstance(format_config.get("schema"), dict)
        ):
            raise _invalid("Structured output requires a JSON schema")
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "anthropic_output",
                "schema": format_config["schema"],
                "strict": True,
            },
        }
    parallel = None
    choice = raw_body.get("tool_choice")
    if isinstance(choice, dict) and "disable_parallel_tool_use" in choice:
        disabled = choice["disable_parallel_tool_use"]
        if not isinstance(disabled, bool):
            raise _invalid("disable_parallel_tool_use must be a boolean")
        parallel = not disabled
    cache_control = _parse_cache_control(raw_body, tool_names, messages, budget)
    if cache_control is not None:
        cache_control["messages"] = _apply_context_management(
            cache_control["messages"], clear_thinking
        )
    return CanonicalChatRequest(
        requested_model=model,
        messages=messages,
        max_output_tokens=max_tokens,
        stream=stream,
        stop=stop_sequences,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        tools=tools,
        tool_choice=_normalize_tool_choice(raw_body.get("tool_choice")),
        metadata=metadata,
        thinking_enabled=thinking_enabled,
        thinking_budget_tokens=thinking_budget,
        response_format=response_format,
        reasoning_effort=effort,
        parallel_tool_calls=parallel,
        cache_control=cache_control,
    )


def _stop_reason(finish_reason: Any, stop_sequence: Any = None) -> str:
    if stop_sequence:
        return "stop_sequence"
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }.get(finish_reason, "end_turn")


def _anthropic_usage(usage: Dict[str, Any]) -> Dict[str, int]:
    details = usage.get("prompt_tokens_details") or {}
    cached = int(
        details.get("cached_tokens") or usage.get("cache_read_input_tokens") or 0
    )
    created = int(usage.get("cache_creation_input_tokens") or 0)
    result = {
        "input_tokens": max(0, int(usage.get("prompt_tokens") or 0) - cached - created),
        "output_tokens": int(usage.get("completion_tokens") or 0),
    }
    if (
        details
        or "cache_read_input_tokens" in usage
        or "cache_creation_input_tokens" in usage
    ):
        result.update(
            cache_read_input_tokens=cached, cache_creation_input_tokens=created
        )
    return result


def openai_to_anthropic(
    openai_response: Dict[str, Any], model: str, include_thinking: bool = False
) -> Dict[str, Any]:
    content_blocks: List[Dict[str, Any]] = []
    finish_reason: Any = "stop"
    stop_sequence = openai_response.get("stop_sequence")
    choices = openai_response.get("choices") or []
    if choices:
        choice = choices[0]
        finish_reason = choice.get("finish_reason") or "stop"
        message = choice.get("message") or {}
        reasoning = message.get("reasoning_content")
        if include_thinking and isinstance(reasoning, str) and reasoning:
            content_blocks.append(
                {
                    "type": "thinking",
                    "thinking": reasoning,
                    "signature": sign_thinking(reasoning, model),
                }
            )
        # Reasoning is exposed only when requested, with this endpoint's opaque
        # signature. Legacy callers continue to omit unsigned reasoning.
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
        "usage": _anthropic_usage(usage),
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
    chunks: AsyncIterator[Dict[str, Any]],
    model: str,
    request_id: str,
    input_tokens: int = 0,
    include_thinking: bool = False,
) -> AsyncIterator[Dict[str, str]]:
    """Convert OpenAI chat-completion chunks into Anthropic SSE events.

    Anthropic content blocks are strictly sequential: a block must be stopped
    before the next block starts. OpenAI tool calls can be fragmented and even
    interleaved by index, so tool fragments are buffered and emitted as complete
    sequential Anthropic blocks after text streaming finishes. Explicit thinking
    requests receive reasoning with an opaque Xinference signature.
    """

    message_id = f"msg_{uuid.uuid4().hex}"

    next_index = 0
    active_index: Optional[int] = None
    active_kind: Optional[str] = None
    finish_reason: Any = "stop"
    stop_sequence: Any = None
    output_tokens = 0
    total_usage: Dict[str, Any] = {"prompt_tokens": input_tokens}
    thinking_text = ""
    tool_buffers: Dict[int, Dict[str, Any]] = {}

    def block_stop(index: int) -> Dict[str, str]:
        return {
            "event": "content_block_stop",
            "data": json.dumps({"type": "content_block_stop", "index": index}),
        }

    def finish_block(index: int) -> List[Dict[str, str]]:
        events = []
        if active_kind == "thinking":
            events.append(
                {
                    "event": "content_block_delta",
                    "data": json.dumps(
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "signature_delta",
                                "signature": sign_thinking(thinking_text, model),
                            },
                        }
                    ),
                }
            )
        events.append(block_stop(index))
        return events

    try:
        iterator = chunks.__aiter__()
        try:
            first = await anext(iterator)
        except StopAsyncIteration:
            first = None
        # Native engines expose prompt/cache counters with their first output.
        # If an upstream reports usage only at the end, start with unknown (0)
        # rather than mislabeling cached tokens as uncached input.
        initial_usage = _anthropic_usage((first or {}).get("usage") or {})
        initial_usage["output_tokens"] = 0
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
                        "usage": initial_usage,
                    },
                }
            ),
        }

        async def remaining_chunks():
            if first is not None:
                yield first
            async for item in iterator:
                yield item

        async for chunk in remaining_chunks():
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
            total_usage.update(usage)
            output_tokens = int(usage.get("completion_tokens") or output_tokens)
            choices = chunk.get("choices") or []
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta") or {}
            reasoning = delta.get("reasoning_content")
            if include_thinking and isinstance(reasoning, str) and reasoning:
                if active_kind != "thinking":
                    if active_index is not None:
                        for event in finish_block(active_index):
                            yield event
                    active_index = next_index
                    next_index += 1
                    active_kind = "thinking"
                    thinking_text = ""
                    yield {
                        "event": "content_block_start",
                        "data": json.dumps(
                            {
                                "type": "content_block_start",
                                "index": active_index,
                                "content_block": {"type": "thinking", "thinking": ""},
                            }
                        ),
                    }
                thinking_text += reasoning
                yield {
                    "event": "content_block_delta",
                    "data": json.dumps(
                        {
                            "type": "content_block_delta",
                            "index": active_index,
                            "delta": {"type": "thinking_delta", "thinking": reasoning},
                        }
                    ),
                }
            value = delta.get("content")
            if isinstance(value, str) and value:
                if active_kind != "text":
                    if active_index is not None:
                        for event in finish_block(active_index):
                            yield event
                    active_index = next_index
                    next_index += 1
                    active_kind = "text"
                    yield {
                        "event": "content_block_start",
                        "data": json.dumps(
                            {
                                "type": "content_block_start",
                                "index": active_index,
                                "content_block": {"type": "text", "text": ""},
                            }
                        ),
                    }
                yield {
                    "event": "content_block_delta",
                    "data": json.dumps(
                        {
                            "type": "content_block_delta",
                            "index": active_index,
                            "delta": {"type": "text_delta", "text": value},
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
        for event in finish_block(active_index):
            yield event

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
                "usage": _anthropic_usage(
                    {**total_usage, "completion_tokens": output_tokens}
                ),
            }
        ),
    }
    yield {"event": "message_stop", "data": json.dumps({"type": "message_stop"})}
