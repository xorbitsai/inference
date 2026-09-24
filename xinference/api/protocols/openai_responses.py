# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Stateless OpenAI Responses API on top of chat completions (Codex CLI, openai SDK).

Mapping follows xeonvs/responses-proxy; events match Codex's codex-api SSE parser."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Set

_UNSUPPORTED_FIELDS = ("previous_response_id", "conversation", "prompt")
_TEXT_PART_TYPES = {"input_text", "output_text", "text", "summary_text", "refusal"}
_ECHO_FIELDS = (
    "instructions",
    "max_output_tokens",
    "metadata",
    "parallel_tool_calls",
    "reasoning",
    "temperature",
    "text",
    "tool_choice",
    "tools",
    "top_p",
)


class ResponsesProtocolError(ValueError):
    def __init__(
        self,
        message: str,
        status_code: int = 400,
        param: Optional[str] = None,
        code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.param = param
        self.code = code


@dataclass
class ResponsesRequest:
    model: str
    stream: bool
    chat_body: Dict[str, Any]
    custom_tools: Set[str]
    echo: Dict[str, Any]
    namespaced: Dict[str, tuple[str, str]] = field(default_factory=dict)
    response_id: str = field(default_factory=lambda: f"resp_{uuid.uuid4().hex}")
    created_at: int = field(default_factory=lambda: int(time.time()))


def responses_error(
    message: str,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "error": {"message": message, "type": error_type, "param": param, "code": code}
    }


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ResponsesProtocolError("Message content must be a string or an array")
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") in _TEXT_PART_TYPES:
            parts.append(part.get("text") or part.get("refusal") or "")
    return "\n".join(parts)


def _user_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ResponsesProtocolError("Message content must be a string or an array")
    parts: List[Dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            raise ResponsesProtocolError("Content parts must be JSON objects")
        part_type = part.get("type")
        if part_type in _TEXT_PART_TYPES:
            parts.append({"type": "text", "text": part.get("text") or ""})
        elif part_type == "input_image":
            url = part.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            if not url:
                raise ResponsesProtocolError(
                    "input_image requires image_url; file_id is not supported"
                )
            image: Dict[str, Any] = {"url": url}
            if part.get("detail"):
                image["detail"] = part["detail"]
            parts.append({"type": "image_url", "image_url": image})
        else:
            raise ResponsesProtocolError(f"Unsupported content part type: {part_type}")
    if all(part["type"] == "text" for part in parts):
        return "\n".join(part["text"] for part in parts)
    return parts


def _output_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return _text(output)
    return json.dumps(output, ensure_ascii=False)


def _reasoning_text(item: Dict[str, Any]) -> str:
    parts = [
        part.get("text") or ""
        for key in ("summary", "content")
        for part in (item.get(key) or [])
        if isinstance(part, dict)
    ]
    return "\n".join(part for part in parts if part)


def _normalize_system(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    leading = 0
    while leading < len(messages) and messages[leading]["role"] == "system":
        leading += 1
    head = [m["content"] for m in messages[:leading] if m["content"]]
    # Many chat templates reject a system message anywhere but first, e.g.
    # Codex's mid-session <model_switch> developer message.
    rest = [
        {"role": "user", "content": m["content"]} if m["role"] == "system" else m
        for m in messages[leading:]
    ]
    return ([{"role": "system", "content": "\n\n".join(head)}] if head else []) + rest


def _flat_name(namespace: Optional[str], name: str) -> str:
    if not namespace:
        return name
    return namespace + name if namespace.endswith("__") else f"{namespace}__{name}"


def _input_to_messages(instructions: Any, raw_input: Any) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if instructions:
        if not isinstance(instructions, str):
            raise ResponsesProtocolError("instructions must be a string")
        messages.append({"role": "system", "content": instructions})
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
        return messages
    if not isinstance(raw_input, list):
        raise ResponsesProtocolError("input must be a string or an array of items")

    pending_calls: List[Dict[str, Any]] = []
    tool_outputs: List[Dict[str, Any]] = []
    pending_reasoning: Optional[str] = None
    # Text followed by calls in one turn must stay one assistant message, or
    # the calls lose the reasoning DeepSeek requires to be sent back with them.
    open_assistant: Optional[Dict[str, Any]] = None

    def flush() -> None:
        nonlocal pending_reasoning, open_assistant
        if pending_calls:
            message = open_assistant
            if message is None:
                message = {"role": "assistant", "content": None}
                messages.append(message)
            message["tool_calls"] = list(pending_calls)
            if pending_reasoning and "reasoning_content" not in message:
                message["reasoning_content"] = pending_reasoning
            pending_reasoning = None
            pending_calls.clear()
            open_assistant = None
        if tool_outputs:
            messages.extend(tool_outputs)
            tool_outputs.clear()
            open_assistant = None

    for item in raw_input:
        if not isinstance(item, dict):
            raise ResponsesProtocolError("input items must be JSON objects")
        item_type = item.get("type") or ("message" if "role" in item else None)
        if item_type == "message":
            flush()
            role = item.get("role")
            content = item.get("content")
            open_assistant = None
            if role in ("system", "developer"):
                messages.append({"role": "system", "content": _text(content)})
            elif role == "user":
                messages.append({"role": "user", "content": _user_content(content)})
            elif role == "assistant":
                open_assistant = {"role": "assistant", "content": _text(content)}
                if pending_reasoning:
                    open_assistant["reasoning_content"] = pending_reasoning
                messages.append(open_assistant)
            else:
                raise ResponsesProtocolError(f"Unsupported message role: {role}")
            pending_reasoning = None
        elif item_type == "reasoning":
            flush()
            open_assistant = None
            text = _reasoning_text(item)
            if text:
                pending_reasoning = (
                    f"{pending_reasoning}\n{text}" if pending_reasoning else text
                )
        elif item_type in ("function_call", "custom_tool_call"):
            # Outputs already seen mean this call was made after them, not in
            # parallel with the pending ones.
            if tool_outputs:
                flush()
            arguments = (
                item.get("arguments") or "{}"
                if item_type == "function_call"
                else json.dumps({"input": item.get("input") or ""}, ensure_ascii=False)
            )
            name = _flat_name(item.get("namespace"), item.get("name") or "")
            pending_calls.append(
                {
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
        elif item_type in ("function_call_output", "custom_tool_call_output"):
            tool_outputs.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": _output_text(item.get("output")),
                }
            )
        elif item_type == "item_reference":
            raise ResponsesProtocolError(
                "item_reference needs stored responses, which are not supported",
                code="unsupported_parameter",
            )
    flush()
    return _normalize_system(messages)


def _chat_tool(tool: Dict[str, Any], name: str) -> Dict[str, Any]:
    if tool.get("type") == "function":
        function: Dict[str, Any] = {
            "name": name,
            "parameters": tool.get("parameters")
            or {"type": "object", "properties": {}},
        }
        if tool.get("description"):
            function["description"] = tool["description"]
        if tool.get("strict") is not None:
            function["strict"] = tool["strict"]
        return {"type": "function", "function": function}
    description = tool.get("description") or ""
    fmt = tool.get("format") or {}
    if fmt.get("type") == "grammar" and fmt.get("definition"):
        description += (
            f"\n\nThe input must follow this {fmt.get('syntax', '')} "
            f"grammar:\n{fmt['definition']}"
        )
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description.strip(),
            "parameters": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            },
        },
    }


def _convert_tools(
    raw_tools: Any,
) -> tuple[List[Dict[str, Any]], Set[str], Dict[str, tuple[str, str]]]:
    tools: List[Dict[str, Any]] = []
    custom: Set[str] = set()
    namespaced: Dict[str, tuple[str, str]] = {}

    def add(tool: Any, namespace: Optional[str] = None) -> None:
        if not isinstance(tool, dict):
            raise ResponsesProtocolError("tools must be JSON objects")
        # Hosted tools (web_search, file_search, mcp, ...) have no chat
        # equivalent here; the model simply does not see them.
        if tool.get("type") not in ("function", "custom"):
            return
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise ResponsesProtocolError(f"{tool['type']} tools require a name")
        flat = _flat_name(namespace, name)
        if namespace:
            namespaced[flat] = (namespace, name)
        if tool["type"] == "custom":
            custom.add(flat)
        tools.append(_chat_tool(tool, flat))

    for tool in raw_tools or []:
        if isinstance(tool, dict) and tool.get("type") == "namespace":
            for inner in tool.get("tools") or []:
                add(inner, tool.get("name"))
        else:
            add(tool)
    return tools, custom, namespaced


def _convert_tool_choice(choice: Any) -> Any:
    if choice is None or isinstance(choice, str):
        return choice
    if isinstance(choice, dict) and choice.get("type") in ("function", "custom"):
        return {"type": "function", "function": {"name": choice.get("name")}}
    return "auto"


def _response_format(text: Any) -> Optional[Dict[str, Any]]:
    fmt = (text or {}).get("format") if isinstance(text, dict) else None
    if not isinstance(fmt, dict) or fmt.get("type") in (None, "text"):
        return None
    if fmt["type"] == "json_object":
        return {"type": "json_object"}
    if fmt["type"] == "json_schema":
        schema = {"name": fmt.get("name") or "response", "schema": fmt.get("schema")}
        for key in ("description", "strict"):
            if fmt.get(key) is not None:
                schema[key] = fmt[key]
        return {"type": "json_schema", "json_schema": schema}
    raise ResponsesProtocolError(f"Unsupported text.format type: {fmt['type']}")


def parse_responses_request(raw: Any) -> ResponsesRequest:
    if not isinstance(raw, dict):
        raise ResponsesProtocolError("Request body must be a JSON object")
    model = raw.get("model")
    if not isinstance(model, str) or not model:
        raise ResponsesProtocolError("model is required", param="model")
    for name in _UNSUPPORTED_FIELDS:
        if raw.get(name):
            raise ResponsesProtocolError(
                f"{name} is not supported; send the full conversation in input",
                param=name,
                code="unsupported_parameter",
            )
    if raw.get("background"):
        raise ResponsesProtocolError(
            "background responses are not supported",
            param="background",
            code="unsupported_parameter",
        )

    stream = raw.get("stream") is True
    tools, custom_tools, namespaced = _convert_tools(raw.get("tools"))
    body: Dict[str, Any] = {
        "model": model,
        "messages": _input_to_messages(raw.get("instructions"), raw.get("input")),
        "stream": stream,
    }
    if raw.get("max_output_tokens") is not None:
        body["max_tokens"] = raw["max_output_tokens"]
    for key in ("temperature", "top_p"):
        if raw.get(key) is not None:
            body[key] = raw[key]
    if tools:
        body["tools"] = tools
        choice = _convert_tool_choice(raw.get("tool_choice"))
        if choice is not None:
            body["tool_choice"] = choice
    response_format = _response_format(raw.get("text"))
    if response_format:
        body["response_format"] = response_format
    if stream:
        body["stream_options"] = {"include_usage": True}

    echo = {key: raw[key] for key in _ECHO_FIELDS if key in raw}
    return ResponsesRequest(
        model=model,
        stream=stream,
        chat_body=body,
        custom_tools=custom_tools,
        echo=echo,
        namespaced=namespaced,
    )


def _usage(usage: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens") or 0
    completion = usage.get("completion_tokens") or 0
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "input_tokens": prompt,
        "input_tokens_details": {
            "cached_tokens": prompt_details.get("cached_tokens") or 0
        },
        "output_tokens": completion,
        "output_tokens_details": {
            "reasoning_tokens": completion_details.get("reasoning_tokens") or 0
        },
        "total_tokens": usage.get("total_tokens") or prompt + completion,
    }


def _response_object(
    req: ResponsesRequest,
    status: str,
    output: List[Dict[str, Any]],
    usage: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
    incomplete_reason: Optional[str] = None,
) -> Dict[str, Any]:
    echo = req.echo
    return {
        "id": req.response_id,
        "object": "response",
        "created_at": req.created_at,
        "status": status,
        "error": error,
        "incomplete_details": (
            {"reason": incomplete_reason} if incomplete_reason else None
        ),
        "instructions": echo.get("instructions"),
        "max_output_tokens": echo.get("max_output_tokens"),
        "model": req.model,
        "output": output,
        "parallel_tool_calls": echo.get("parallel_tool_calls", True),
        "previous_response_id": None,
        "reasoning": echo.get("reasoning") or {"effort": None, "summary": None},
        "store": False,
        "temperature": echo.get("temperature"),
        "text": echo.get("text") or {"format": {"type": "text"}},
        "tool_choice": echo.get("tool_choice", "auto"),
        "tools": echo.get("tools", []),
        "top_p": echo.get("top_p"),
        "truncation": "disabled",
        "usage": usage,
        "metadata": echo.get("metadata") or {},
    }


def _reasoning_item(item_id: str, text: str, status: str) -> Dict[str, Any]:
    return {
        "id": item_id,
        "type": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}] if text else [],
        "encrypted_content": None,
        "status": status,
    }


def _text_part(text: str) -> Dict[str, Any]:
    return {"type": "output_text", "text": text, "annotations": [], "logprobs": []}


def _message_item(item_id: str, text: Optional[str], status: str) -> Dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": status,
        "content": [] if text is None else [_text_part(text)],
    }


def _call_item(
    req: ResponsesRequest,
    item_id: str,
    call_id: str,
    name: str,
    arguments: str,
    status: str,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {"id": item_id, "call_id": call_id, "status": status}
    namespace, item["name"] = req.namespaced.get(name, (None, name))
    if namespace:
        item["namespace"] = namespace
    if name not in req.custom_tools:
        return {"type": "function_call", "arguments": arguments, **item}
    try:
        parsed = json.loads(arguments)
    except ValueError:
        parsed = arguments
    tool_input = parsed.get("input", "") if isinstance(parsed, dict) else parsed
    if not isinstance(tool_input, str):
        tool_input = json.dumps(tool_input, ensure_ascii=False)
    return {"type": "custom_tool_call", "input": tool_input, **item}


def _reasoning_of(message: Dict[str, Any]) -> str:
    # vLLM's OpenAI server names the field ``reasoning``.
    value = message.get("reasoning_content") or message.get("reasoning")
    return value if isinstance(value, str) else ""


def responses_error_code(message: str) -> str:
    lowered = message.lower()
    if (
        "context length" in lowered
        or "context window" in lowered
        or ("maximum" in lowered and "token" in lowered)
    ):
        return "context_length_exceeded"
    if "rate limit" in lowered:
        return "rate_limit_exceeded"
    return "server_error"


def chat_to_response(chat: Dict[str, Any], req: ResponsesRequest) -> Dict[str, Any]:
    choices = chat.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    output: List[Dict[str, Any]] = []
    reasoning = _reasoning_of(message)
    if reasoning:
        output.append(_reasoning_item(f"rs_{uuid.uuid4().hex}", reasoning, "completed"))
    if isinstance(message.get("content"), str) and message["content"]:
        output.append(
            _message_item(f"msg_{uuid.uuid4().hex}", message["content"], "completed")
        )
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {}, ensure_ascii=False)
        output.append(
            _call_item(
                req,
                f"fc_{uuid.uuid4().hex}",
                call.get("id") or f"call_{uuid.uuid4().hex}",
                function.get("name") or "",
                arguments,
                "completed",
            )
        )
    truncated = choice.get("finish_reason") == "length"
    return _response_object(
        req,
        "incomplete" if truncated else "completed",
        output,
        usage=_usage(chat.get("usage")),
        incomplete_reason="max_output_tokens" if truncated else None,
    )


async def responses_stream_events(
    chunks: AsyncIterator[Dict[str, Any]], req: ResponsesRequest
) -> AsyncIterator[Dict[str, str]]:
    """Chat-completion chunks to Responses SSE events."""
    sequence = 0
    output: Dict[int, Dict[str, Any]] = {}
    active: Optional[Dict[str, Any]] = None
    calls: Dict[Any, Dict[str, Any]] = {}
    latest_key: Dict[Any, Any] = {}
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    started = False

    def event(event_type: str, **payload: Any) -> Dict[str, str]:
        nonlocal sequence
        data = {"type": event_type, "sequence_number": sequence, **payload}
        sequence += 1
        return {"event": event_type, "data": json.dumps(data, ensure_ascii=False)}

    def start() -> List[Dict[str, str]]:
        snapshot = _response_object(req, "in_progress", [])
        return [
            event("response.created", response=snapshot),
            event("response.in_progress", response=snapshot),
        ]

    def open_item(kind: str) -> List[Dict[str, str]]:
        nonlocal active
        index = len(output) + len(calls)
        if kind == "reasoning":
            item = _reasoning_item(f"rs_{uuid.uuid4().hex}", "", "in_progress")
            part: Dict[str, Any] = {"type": "reasoning_text", "text": ""}
        else:
            item = _message_item(f"msg_{uuid.uuid4().hex}", None, "in_progress")
            part = _text_part("")
        active = {"kind": kind, "item": item, "index": index, "text": ""}
        return [
            event("response.output_item.added", output_index=index, item=item),
            event(
                "response.content_part.added",
                item_id=item["id"],
                output_index=index,
                content_index=0,
                part=part,
            ),
        ]

    def close_active() -> List[Dict[str, str]]:
        nonlocal active
        if active is None:
            return []
        kind, index, text = active["kind"], active["index"], active["text"]
        item_id = active["item"]["id"]
        if kind == "reasoning":
            item = _reasoning_item(item_id, text, "completed")
            part: Dict[str, Any] = {"type": "reasoning_text", "text": text}
            done = event(
                "response.reasoning_text.done",
                item_id=item_id,
                output_index=index,
                content_index=0,
                text=text,
            )
        else:
            item = _message_item(item_id, text, "completed")
            part = _text_part(text)
            done = event(
                "response.output_text.done",
                item_id=item_id,
                output_index=index,
                content_index=0,
                text=text,
                logprobs=[],
            )
        output[index] = item
        active = None
        return [
            done,
            event(
                "response.content_part.done",
                item_id=item_id,
                output_index=index,
                content_index=0,
                part=part,
            ),
            event("response.output_item.done", output_index=index, item=item),
        ]

    def append(kind: str, delta: str) -> List[Dict[str, str]]:
        events: List[Dict[str, str]] = []
        if active is None or active["kind"] != kind:
            events += close_active()
            events += open_item(kind)
        assert active is not None
        active["text"] += delta
        name = (
            "response.reasoning_text.delta"
            if kind == "reasoning"
            else "response.output_text.delta"
        )
        extra: Dict[str, Any] = {} if kind == "reasoning" else {"logprobs": []}
        events.append(
            event(
                name,
                item_id=active["item"]["id"],
                output_index=active["index"],
                content_index=0,
                delta=delta,
                **extra,
            )
        )
        return events

    def call_delta(fragment: Dict[str, Any]) -> List[Dict[str, str]]:
        events = close_active()
        position = fragment.get("index", 0)
        function = fragment.get("function") or {}
        upstream_id = fragment.get("id") or None
        key = latest_key.get(position, position)
        state = calls.get(key)
        # Some backends put every parallel call at index 0, told apart by id only;
        # others repeat the call with id="" on every later fragment.
        if (
            state is not None
            and upstream_id
            and state["upstream_id"]
            and upstream_id != state["upstream_id"]
        ):
            key = (position, upstream_id)
            state = calls.get(key)
        latest_key[position] = key
        if state is None:
            state = {
                "index": len(output) + len(calls),
                "item_id": f"fc_{uuid.uuid4().hex}",
                "call_id": upstream_id or f"call_{uuid.uuid4().hex}",
                "upstream_id": upstream_id,
                "name": function.get("name") or "",
                "arguments": "",
            }
            calls[key] = state
            item = _call_item(
                req,
                state["item_id"],
                state["call_id"],
                state["name"],
                "",
                "in_progress",
            )
            events.append(
                event(
                    "response.output_item.added", output_index=state["index"], item=item
                )
            )
        elif function.get("name") and not state["name"]:
            state["name"] = function["name"]
        delta = function.get("arguments")
        if delta is not None and not isinstance(delta, str):
            delta = json.dumps(delta, ensure_ascii=False)
        if delta:
            state["arguments"] += delta
            if state["name"] not in req.custom_tools:
                events.append(
                    event(
                        "response.function_call_arguments.delta",
                        item_id=state["item_id"],
                        output_index=state["index"],
                        delta=delta,
                    )
                )
        return events

    def close_calls() -> List[Dict[str, str]]:
        events: List[Dict[str, str]] = []
        for state in sorted(calls.values(), key=lambda s: s["index"]):
            item = _call_item(
                req,
                state["item_id"],
                state["call_id"],
                state["name"],
                state["arguments"] or "{}",
                "completed",
            )
            if item["type"] == "function_call":
                events.append(
                    event(
                        "response.function_call_arguments.done",
                        item_id=state["item_id"],
                        output_index=state["index"],
                        name=item["name"],
                        arguments=item["arguments"],
                    )
                )
            events.append(
                event(
                    "response.output_item.done", output_index=state["index"], item=item
                )
            )
            output[state["index"]] = item
        return events

    try:
        async for chunk in chunks:
            if not started:
                started = True
                for item in start():
                    yield item
            error = chunk.get("error")
            if error:
                message = (
                    error.get("message") if isinstance(error, dict) else str(error)
                ) or "Model stream failed"
                failed = _response_object(
                    req,
                    "failed",
                    [output[i] for i in sorted(output)],
                    error={"code": responses_error_code(message), "message": message},
                )
                yield event("response.failed", response=failed)
                return
            if chunk.get("usage"):
                usage = _usage(chunk["usage"])
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                reasoning = _reasoning_of(delta)
                if reasoning:
                    for item in append("reasoning", reasoning):
                        yield item
                content = delta.get("content")
                if isinstance(content, str) and content:
                    for item in append("message", content):
                        yield item
                for fragment in delta.get("tool_calls") or []:
                    for item in call_delta(fragment):
                        yield item
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    finally:
        # Releases the model's serve count now rather than when GC collects it.
        close = getattr(chunks, "aclose", None)
        if close is not None:
            await close()

    if not started:
        for item in start():
            yield item
    for item in close_active() + close_calls():
        yield item
    truncated = finish_reason == "length"
    final = _response_object(
        req,
        "incomplete" if truncated else "completed",
        [output[i] for i in sorted(output)],
        usage=usage,
        incomplete_reason="max_output_tokens" if truncated else None,
    )
    yield event(
        "response.incomplete" if truncated else "response.completed", response=final
    )
