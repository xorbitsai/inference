# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Tool controls implemented through the Xinference vLLM engine."""

import copy
import json
import uuid
from typing import Any, AsyncGenerator, Dict, List, Tuple


def is_forced_tool_choice(config: Dict[str, Any]) -> bool:
    choice = config.get("tool_choice")
    return choice == "required" or isinstance(choice, dict)


def prepare_forced_tools(
    messages: List[Dict[str, Any]], config: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    config = copy.deepcopy(config)
    tools = config.pop("tools", [])
    choice = config.pop("tool_choice")
    parallel = config.pop("parallel_tool_calls", True)
    selected = tools
    if isinstance(choice, dict):
        name = choice.get("function", {}).get("name")
        selected = [t for t in tools if t["function"]["name"] == name]
    if not selected:
        raise ValueError("Forced tool_choice requires a declared tool")

    definitions = {}
    variants = []
    for index, tool in enumerate(selected):
        function = tool["function"]
        schema = copy.deepcopy(function["parameters"])
        if schema.get("type", "object") != "object":
            raise ValueError("Tool parameter schemas must describe objects")
        schema["type"] = "object"
        key = f"tool_{index}"

        def relocate_refs(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    if k == "$ref" and isinstance(v, str) and v.startswith("#"):
                        value[k] = f"#/$defs/{key}" + v[1:]
                    else:
                        relocate_refs(v)
            elif isinstance(value, list):
                for v in value:
                    relocate_refs(v)

        relocate_refs(schema)
        definitions[key] = schema
        variants.append(
            {
                "type": "object",
                "properties": {
                    "name": {"const": function["name"]},
                    "arguments": {"$ref": f"#/$defs/{key}"},
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
        )
    constraint: Dict[str, Any] = {
        "type": "array",
        "items": {"oneOf": variants},
        "minItems": 1,
        "$defs": definitions,
    }
    if parallel is False:
        constraint["maxItems"] = 1
    config["response_format"] = {
        "type": "json_schema",
        "json_schema": {"name": "tool_calls", "schema": constraint, "strict": True},
    }
    config["_require_structured_output"] = True
    instruction = (
        "For this response, call the required tool(s). Return only a JSON array "
        "of objects with name and arguments. Use the following tool definitions:\n"
        + json.dumps(selected, ensure_ascii=False)
    )
    messages = copy.deepcopy(messages)
    if messages and messages[0].get("role") == "system":
        if isinstance(messages[0]["content"], list):
            messages[0]["content"].append({"type": "text", "text": instruction})
        else:
            messages[0]["content"] += "\n" + instruction
    else:
        messages.insert(0, {"role": "system", "content": instruction})
    return messages, config


def convert_forced_completion(completion: Dict[str, Any]) -> Dict[str, Any]:
    for choice in completion["choices"]:
        if choice.get("finish_reason") == "length":
            # Truncated arguments cannot be executed as a tool call.
            continue
        try:
            calls = json.loads(choice["message"]["content"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Constrained tool generation returned invalid JSON"
            ) from exc
        if not isinstance(calls, list) or not calls:
            raise RuntimeError("Constrained tool generation returned no tool calls")
        converted = []
        for call in calls:
            if (
                not isinstance(call, dict)
                or not isinstance(call.get("name"), str)
                or not isinstance(call.get("arguments"), dict)
            ):
                raise RuntimeError(
                    "Constrained tool generation returned invalid arguments"
                )
            converted.append(
                {
                    "id": f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                    },
                }
            )
        choice["message"]["content"] = None
        choice["message"]["tool_calls"] = converted
        choice["finish_reason"] = "tool_calls"
    return completion


async def stream_forced_completion(
    completion: Dict[str, Any], include_usage: bool
) -> AsyncGenerator[Dict[str, Any], None]:
    # Emit only complete, constrained arguments. Partial/truncated JSON must
    # never become an executable tool call.
    metadata = {k: completion[k] for k in ("id", "created", "model")}
    choices = []
    for choice in completion["choices"]:
        message = choice["message"]
        delta = {"role": "assistant", "content": message.get("content")}
        if message.get("reasoning_content"):
            delta["reasoning_content"] = message["reasoning_content"]
        if message.get("tool_calls"):
            delta["tool_calls"] = [
                {**call, "index": i} for i, call in enumerate(message["tool_calls"])
            ]
        choices.append(
            {
                "index": choice["index"],
                "delta": delta,
                "finish_reason": choice["finish_reason"],
            }
        )
    yield {
        **metadata,
        "object": "chat.completion.chunk",
        "choices": choices,
        "usage": completion.get("usage"),
    }
    if include_usage:
        yield {
            **metadata,
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": completion.get("usage"),
        }


def single_tool_completion(completion: Dict[str, Any]) -> Dict[str, Any]:
    for choice in completion.get("choices", []):
        message = choice["message"]
        if message.get("tool_calls"):
            message["tool_calls"] = message["tool_calls"][:1]
    return completion


async def single_tool_stream(chunks) -> AsyncGenerator[Dict[str, Any], None]:
    first_indexes: Dict[int, int] = {}
    try:
        async for chunk in chunks:
            chunk = copy.deepcopy(chunk)
            for choice in chunk.get("choices", []):
                calls = choice["delta"].get("tool_calls")
                if calls:
                    index = first_indexes.setdefault(choice["index"], calls[0]["index"])
                    choice["delta"]["tool_calls"] = [
                        c for c in calls if c["index"] == index
                    ]
            yield chunk
    finally:
        await chunks.aclose()
