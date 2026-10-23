# CI: linux

import json

import pytest

from ...utils import ChatModelMixin
from ..abstract_tool_parser import ToolParser, request_tools_var
from ..glm5_tool_parser import Glm5ToolParser
from ..minicpm5_tool_parser import MiniCPM5ToolParser
from ..minimax_m3_tool_parser import MiniMaxM3ToolParser
from ..minimax_tool_parser import MiniMaxToolParser
from ..qwen_tool_parser import QwenToolParser

# The raw text each argument has in the model output. These formats write
# string values unquoted, so the parser needs the tool schema to tell them
# apart from JSON values.
ARGS = [
    ("content", '{"name": "demo"}'),
    ("status", "404"),
    ("enabled", "true"),
    ("note", "null"),
    ("count", "3"),
    ("options", '{"overwrite": true}'),
]


def _tools(function_name):
    return [
        {
            "type": "function",
            "function": {
                "name": function_name,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {"type": "string"},
                        "enabled": {"type": "string"},
                        "note": {"type": "string"},
                        "count": {"type": "integer"},
                        "options": {"type": "object"},
                    },
                },
            },
        }
    ]


TOOLS = _tools("save")

TYPED_ARGS = {
    "content": '{"name": "demo"}',
    "status": "404",
    "enabled": "true",
    "note": "null",
    "count": 3,
    "options": {"overwrite": True},
}

DECODED_ARGS = {
    "content": {"name": "demo"},
    "status": 404,
    "enabled": True,
    "note": None,
    "count": 3,
    "options": {"overwrite": True},
}


def _qwen(args):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args)
    return f"<tool_call>\n<function=save>\n{body}</function>\n</tool_call>"


def _glm5(args):
    body = "".join(f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>" for k, v in args)
    return f"<tool_call>save{body}</tool_call>"


def _minimax(args):
    body = "".join(f'<parameter name="{k}">{v}</parameter>' for k, v in args)
    return f'<minimax:tool_call><invoke name="save">{body}</invoke></minimax:tool_call>'


def _minicpm5(args):
    body = "".join(f'<param name="{k}">{v}</param>' for k, v in args)
    return f'<function name="save">{body}</function>'


PARSERS = {
    "qwen": (QwenToolParser, _qwen),
    "glm5": (Glm5ToolParser, _glm5),
    "minimax": (MiniMaxToolParser, _minimax),
    "minimax-m3": (MiniMaxM3ToolParser, _minimax),
    "minicpm5": (MiniCPM5ToolParser, _minicpm5),
}


def _model(parser_cls):
    model = ChatModelMixin()
    model.tool_parser = parser_cls()
    model.reasoning_parser = None
    model.model_family = None
    model.model_uid = "test-model"
    return model


def _completion(text):
    return {
        "id": "test",
        "object": "text_completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _chunks(text, size=7):
    for start in range(0, len(text), size):
        yield {
            "id": "test",
            "object": "text_completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {"index": 0, "text": text[start : start + size], "finish_reason": None}
            ],
        }


def _streamed_arguments(chunks):
    arguments = {}
    for chunk in chunks:
        for choice in chunk.get("choices") or []:
            for tool_call in (choice.get("delta") or {}).get("tool_calls") or []:
                index = tool_call.get("index", 0)
                delta = tool_call.get("function", {}).get("arguments") or ""
                arguments[index] = arguments.get(index, "") + delta
    return [json.loads(value) for value in arguments.values()]


@pytest.mark.parametrize("parser_name", PARSERS)
def test_string_parameters_stay_text(parser_name):
    parser_cls, render = PARSERS[parser_name]
    model = _model(parser_cls)

    result = model._post_process_completion(
        None, "test-model", _completion(render(ARGS)), TOOLS
    )

    tool_calls = result["choices"][0]["message"]["tool_calls"]
    assert [json.loads(call["function"]["arguments"]) for call in tool_calls] == [
        TYPED_ARGS
    ]
    assert request_tools_var.get() is None


@pytest.mark.parametrize("parser_name", PARSERS)
def test_streamed_string_parameters_stay_text(parser_name):
    parser_cls, render = PARSERS[parser_name]
    model = _model(parser_cls)

    chunks = model._to_tool_completion_chunks(_chunks(render(ARGS)), tools=TOOLS)

    assert _streamed_arguments(chunks) == [TYPED_ARGS]
    assert request_tools_var.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("parser_name", PARSERS)
async def test_async_streamed_string_parameters_stay_text(parser_name):
    parser_cls, render = PARSERS[parser_name]
    model = _model(parser_cls)

    async def chunks():
        for chunk in _chunks(render(ARGS)):
            yield chunk

    results = [
        chunk
        async for chunk in model._async_to_tool_completion_chunks(chunks(), tools=TOOLS)
    ]

    assert _streamed_arguments(results) == [TYPED_ARGS]
    assert request_tools_var.get() is None


@pytest.mark.parametrize("tools", [None, _tools("other")], ids=["no-tools", "other"])
@pytest.mark.parametrize("parser_name", PARSERS)
def test_values_are_decoded_without_a_matching_schema(parser_name, tools):
    parser_cls, render = PARSERS[parser_name]
    model = _model(parser_cls)

    result = model._post_process_completion(
        None, "test-model", _completion(render(ARGS)), tools
    )

    tool_calls = result["choices"][0]["message"]["tool_calls"]
    assert [json.loads(call["function"]["arguments"]) for call in tool_calls] == [
        DECODED_ARGS
    ]


@pytest.mark.parametrize(
    "schema, expected",
    [
        ({"type": "string"}, True),
        ({"type": "integer"}, False),
        ({"type": ["string", "null"]}, False),
        ({"anyOf": [{"type": "string"}, {"type": "null"}]}, False),
        ({}, False),
    ],
)
def test_is_string_parameter(schema, expected):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "save",
                "parameters": {"type": "object", "properties": {"value": schema}},
            },
        }
    ]
    token = request_tools_var.set(tools)
    try:
        assert ToolParser._is_string_parameter("save", "value") is expected
        assert ToolParser._is_string_parameter("save", "missing") is False
        assert ToolParser._is_string_parameter("other", "value") is False
    finally:
        request_tools_var.reset(token)
    assert ToolParser._is_string_parameter("save", "value") is False


def test_qwen_string_parameters_keep_whitespace():
    # The Qwen3-Coder / Qwen3.5 template writes "<parameter=k>\n{value}\n</parameter>",
    # so only those two newlines are markup.
    content = "    def foo():\n        return 1\n"
    model = _model(QwenToolParser)
    tools = _tools("save")
    text = _qwen([("content", content), ("count", "3")])

    result = model._post_process_completion(
        None, "test-model", _completion(text), tools
    )
    tool_calls = result["choices"][0]["message"]["tool_calls"]
    expected = {"content": content, "count": 3}
    assert json.loads(tool_calls[0]["function"]["arguments"]) == expected

    chunks = model._to_tool_completion_chunks(_chunks(text), tools=tools)
    assert _streamed_arguments(chunks) == [expected]
