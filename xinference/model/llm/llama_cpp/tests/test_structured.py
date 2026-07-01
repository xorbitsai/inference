import importlib
import importlib.util
import json
import sys
from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict

import openai
import pytest
from pydantic import BaseModel

from xinference.client import Client

from ...reasoning_parser import ReasoningParser
from ...tool_parsers.qwen_tool_parser import QwenToolParser
from ..core import XllamaCppModel, _apply_response_format


class _InlineExecutor:
    def submit(self, fn):
        fn()


class _FakeLlamaCppServer:
    def __init__(self, responses):
        self.responses = responses
        self.requests = []

    def handle_chat_completions(self, data, callback):
        self.requests.append(data)
        for response in self.responses:
            callback(response)

    def handle_completions(self, data, callback):
        self.requests.append(data)
        for response in self.responses:
            callback(response)


def _new_fake_llamacpp_model(responses):
    model = XllamaCppModel.__new__(XllamaCppModel)
    model.model_uid = "test-model"
    model.reasoning_parser = None
    model.tool_parser = None
    model._executor = _InlineExecutor()
    model._llm = _FakeLlamaCppServer(responses)
    return model


def test_llamacpp_stream_tool_parse_error_finishes_partial_stream():
    first_chunk = {
        "choices": [
            {
                "finish_reason": None,
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "select_execution_pattern",
                                "arguments": '{"answer":"unterminated',
                            },
                        }
                    ]
                },
            }
        ],
        "created": 123,
        "id": "chatcmpl-test",
        "model": "test-model",
        "object": "chat.completion.chunk",
    }
    model = _new_fake_llamacpp_model(
        [
            first_chunk,
            {
                "code": 500,
                "message": (
                    "Failed to parse tool call arguments as JSON: "
                    "[json.exception.parse_error.101] missing closing quote"
                ),
            },
        ]
    )

    chunks = list(
        model.chat(
            [{"role": "user", "content": "hi"}],
            {
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "select_execution_pattern",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            },
        )
    )

    assert chunks[0] == first_chunk
    assert chunks[-1]["id"] == first_chunk["id"]
    assert chunks[-1]["model"] == first_chunk["model"]
    assert chunks[-1]["created"] == first_chunk["created"]
    assert chunks[-1]["choices"] == [
        {"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}
    ]


def test_llamacpp_stream_non_tool_parse_error_still_raises():
    model = _new_fake_llamacpp_model(
        [
            {
                "code": 500,
                "message": (
                    "Failed to parse tool call arguments as JSON: "
                    "[json.exception.parse_error.101] missing closing quote"
                ),
            }
        ]
    )

    with pytest.raises(Exception, match="Failed to parse tool call arguments"):
        list(model.chat([{"role": "user", "content": "hi"}], {"stream": True}))


@pytest.mark.parametrize("method", ["generate", "chat"])
@pytest.mark.parametrize("stream", [False, True])
def test_llamacpp_raises_nested_error_response(method, stream):
    message = (
        "Field 'max_tokens': [json.exception.type_error.302] "
        "type must be number, but is null"
    )
    model = _new_fake_llamacpp_model(
        [
            {
                "error": {
                    "code": 400,
                    "message": message,
                    "type": "invalid_request_error",
                }
            }
        ]
    )

    with pytest.raises(Exception, match="type must be number, but is null") as exc_info:
        if method == "generate":
            result = model.generate("hi", {"stream": stream})
        else:
            result = model.chat([{"role": "user", "content": "hi"}], {"stream": stream})
        if stream:
            list(result)

    assert str(exc_info.value) == message


@pytest.mark.parametrize("method", ["generate", "chat"])
def test_llamacpp_does_not_forward_null_max_tokens(method):
    response = {
        "choices": [],
        "created": 123,
        "id": "completion-test",
        "model": "test-model",
        "object": "text_completion" if method == "generate" else "chat.completion",
        "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
    }
    model = _new_fake_llamacpp_model([response])

    if method == "generate":
        model.generate("hi", {"max_tokens": None})
    else:
        model.chat([{"role": "user", "content": "hi"}], {"max_tokens": None})

    assert "max_tokens" not in model._llm.requests[0]


def test_llamacpp_chat_reparses_tool_call_from_reasoning_content():
    model = _new_fake_llamacpp_model(
        [
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": (
                                "I should call the tool.\n\n"
                                "<tool_call>\n"
                                "<function=execute_python_code>\n"
                                "<parameter=code>\n"
                                "import random\n"
                                "random.randint(1, 100)\n"
                                "</parameter>\n"
                                "</function>\n"
                                "</tool_call>"
                            ),
                        },
                    }
                ],
                "created": 123,
                "id": "chatcmpl-test",
                "model": "test-model",
                "object": "chat.completion",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }
        ]
    )
    model.reasoning_parser = ReasoningParser(
        reasoning_content=True,
        reasoning_start_tag="<think>",
        reasoning_end_tag="</think>",
        enable_thinking=False,
    )
    model.tool_parser = QwenToolParser()

    response = model.chat(
        [{"role": "user", "content": "run python"}],
        {
            "stream": False,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "execute_python_code",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    message = response["choices"][0]["message"]
    assert response["choices"][0]["finish_reason"] == "tool_calls"
    assert message["content"] == "I should call the tool.\n\n"
    assert message["tool_calls"][0]["function"]["name"] == "execute_python_code"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "code": "import random\nrandom.randint(1, 100)"
    }
    assert "reasoning_content" not in message


def test_llamacpp_chat_moves_reasoning_content_to_content_when_thinking_disabled():
    model = _new_fake_llamacpp_model(
        [
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "visible answer",
                        },
                    }
                ],
                "created": 123,
                "id": "chatcmpl-test",
                "model": "test-model",
                "object": "chat.completion",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }
        ]
    )
    model.reasoning_parser = ReasoningParser(
        reasoning_content=True,
        reasoning_start_tag="<think>",
        reasoning_end_tag="</think>",
        enable_thinking=False,
    )

    response = model.chat(
        [{"role": "user", "content": "hi"}],
        {"stream": False},
    )

    message = response["choices"][0]["message"]
    assert message["content"] == "visible answer"
    assert "reasoning_content" not in message


class CarType(str, Enum):
    sedan = "sedan"
    suv = "SuV"
    truck = "Truck"
    coupe = "Coupe"


class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType


def _load_json_from_message(message: Any) -> Dict[str, Any]:
    def _strip_think(text: str) -> str:
        stripped = text.lstrip()
        if stripped.startswith("<think>"):
            if "</think>" in stripped:
                stripped = stripped.split("</think>", 1)[1]
            else:
                stripped = stripped.split("<think>", 1)[1]
        return stripped.lstrip()

    raw_content = message.content
    if isinstance(raw_content, str):
        return json.loads(_strip_think(raw_content))

    if isinstance(raw_content, list):
        text_blocks = []
        for block in raw_content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    text_blocks.append(_strip_think(block["text"]))
                continue

            block_type = getattr(block, "type", None)
            block_text = getattr(block, "text", None)
            if block_type == "text" and block_text:
                text_blocks.append(_strip_think(block_text))

        if text_blocks:
            return json.loads("".join(text_blocks))

    pytest.fail(f"Unexpected message content format: {raw_content!r}")
    raise AssertionError("Unreachable")


def test_apply_response_format_sets_grammar(monkeypatch):
    fake_xllamacpp = SimpleNamespace(json_schema_to_grammar=lambda schema: "GRAMMAR")
    monkeypatch.setitem(sys.modules, "xllamacpp", fake_xllamacpp)

    cfg = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": ["a"],
                }
            },
        }
    }

    _apply_response_format(cfg)

    assert "response_format" not in cfg
    assert "json_schema" not in cfg
    assert cfg["grammar"] == "GRAMMAR"


def test_apply_response_format_handles_conversion_failure(monkeypatch):
    def _raise(_):
        raise ValueError("bad schema")

    fake_xllamacpp = SimpleNamespace(json_schema_to_grammar=_raise)
    monkeypatch.setitem(sys.modules, "xllamacpp", fake_xllamacpp)

    cfg = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {"b": {"type": "string"}},
                    "required": ["b"],
                }
            },
        }
    }

    _apply_response_format(cfg)

    assert "response_format" not in cfg
    assert cfg["json_schema"]["required"] == ["b"]
    assert "grammar" not in cfg


def test_apply_response_format_ignores_non_schema(monkeypatch):
    cfg = {"response_format": {"type": "json_object"}}
    _apply_response_format(cfg)
    assert "grammar" not in cfg
    assert "json_schema" not in cfg


def test_apply_response_format_uses_real_xllamacpp_if_available():
    if importlib.util.find_spec("xllamacpp") is None:
        pytest.skip("xllamacpp not installed")
    xllamacpp = importlib.import_module("xllamacpp")
    if not hasattr(xllamacpp, "json_schema_to_grammar"):
        pytest.skip("xllamacpp does not expose json_schema_to_grammar")

    cfg = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {"c": {"type": "integer"}},
                    "required": ["c"],
                }
            },
        }
    }

    _apply_response_format(cfg)

    assert "response_format" not in cfg
    # Real xllamacpp should prefer grammar to avoid passing both
    assert "json_schema" not in cfg
    assert "grammar" in cfg and cfg["grammar"]


def test_llamacpp_qwen3_json_schema(setup):
    endpoint, _ = setup
    client = Client(endpoint)
    model_uid = client.launch_model(
        model_name="qwen3",
        model_engine="llama.cpp",
        model_size_in_billions="0_6",
        model_format="ggufv2",
        quantization="Q4_K_M",
        n_gpu=None,
    )

    try:
        api_client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
        completion = api_client.chat.completions.create(
            model=model_uid,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Generate a JSON containing the brand, model, and car_type of"
                        " an iconic 90s car."
                    ),
                }
            ],
            temperature=0,
            max_tokens=128,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "car-description",
                    "schema": CarDescription.model_json_schema(),
                },
            },
        )

        parsed = _load_json_from_message(completion.choices[0].message)
        car_description = CarDescription.model_validate(parsed)
        assert car_description.brand
        assert car_description.model
    finally:
        if model_uid is not None:
            client.terminate_model(model_uid)
