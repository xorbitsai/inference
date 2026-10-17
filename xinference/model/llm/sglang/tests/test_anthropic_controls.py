# Copyright 2022-2026 Xinference Holdings Pte. Ltd
import copy
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ...core import chat_context_var
from ...reasoning_parser import ReasoningParser
from ..core import SGLANGChatModel, SGLANGModel, SGLANGVisionModel


def model(cls=SGLANGChatModel):
    instance = object.__new__(cls)
    instance.model_uid = "test"
    instance.model_family = SimpleNamespace(
        model_name="qwen3",
        model_family="qwen3",
        stop=None,
        chat_template="template",
        model_ability=["chat", "vision"],
    )
    instance.reasoning_parser = ReasoningParser(
        reasoning_content=True,
        enable_thinking=False,
        allow_disable_thinking=True,
        reasoning_start_tag="<think>",
        reasoning_end_tag="</think>",
    )
    instance._model_config = {"reasoning_parser": "xinference_test"}
    instance._active_request_ids = set()
    return instance


def tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def completion(text):
    return {
        "id": "test",
        "model": "test",
        "created": 1,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 10, "total_tokens": 22},
    }


@pytest.mark.parametrize("schema", [{}, {"type": "object"}])
def test_schema_is_preserved_and_input_not_mutated(schema):
    response_format = {"type": "json_schema", "json_schema": {"schema": schema}}
    cfg = {"response_format": response_format, "_require_structured_output": True}
    expected = copy.deepcopy(response_format)
    sanitized = SGLANGModel._sanitize_generate_config(cfg)
    import json

    assert json.loads(sanitized["json_schema"]) == schema
    assert response_format == expected
    assert "_require_structured_output" not in sanitized


def test_required_constraint_never_silently_dropped():
    with pytest.raises(NotImplementedError):
        SGLANGModel._sanitize_generate_config({"_require_structured_output": True})
    with pytest.raises(ValueError):
        SGLANGModel._sanitize_generate_config(
            {"response_format": {"type": "json_schema"}}
        )


@pytest.mark.parametrize("thinking", [False, True])
def test_reasoning_flag_is_lifted_out_of_sampling_params(thinking):
    sampling = {"require_reasoning": thinking, "json_schema": "{}", "temperature": 0}
    top = SGLANGModel._lift_logprob_request_params(sampling)
    assert top == {"require_reasoning": thinking}
    assert "require_reasoning" not in sampling
    assert sampling["json_schema"] == "{}"


def test_reasoning_capability_and_manual_thinking(monkeypatch):
    instance = model()
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.io_struct",
        SimpleNamespace(
            GenerateReqInput=SimpleNamespace(
                __dataclass_fields__={"require_reasoning": None}
            )
        ),
    )
    instance._validate_structured_reasoning(True)
    instance._model_config = {}
    with pytest.raises(NotImplementedError):
        instance._validate_structured_reasoning(True)
    with pytest.raises(ValueError, match="manual thinking"):
        instance._validate_forced_tools(
            {
                "tool_choice": "required",
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "thinking_budget": 128,
                },
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", [SGLANGChatModel, SGLANGVisionModel])
async def test_forced_tool_uses_schema_and_buffers_stream(cls):
    instance = model(cls)
    instance._prepare_chat_inputs = AsyncMock(
        return_value=("prompt", None, None, {}, {})
    )
    instance.async_generate = AsyncMock(
        return_value=completion('[{"name":"Read","arguments":{"file_path":"a"}}]')
    )
    config = {
        "tools": tools(),
        "tool_choice": "required",
        "_anthropic_request": True,
        "parallel_tool_calls": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    chunks = await instance.async_chat(
        [{"role": "user", "content": "read a"}], config, request_id="rid"
    )
    chunks = [chunk async for chunk in chunks]
    args = instance._prepare_chat_inputs.call_args.args[1]
    assert args["response_format"]["json_schema"]["schema"]["maxItems"] == 1
    assert args["_require_structured_output"] is True
    assert args["stream"] is False
    assert instance.async_generate.call_args.kwargs["request_id"] == "rid"
    calls = [c["choices"][0]["delta"].get("tool_calls") for c in chunks if c["choices"]]
    assert any(
        call and call[0]["function"].get("arguments") == '{"file_path": "a"}'
        for call in calls
    )
    assert chunks[-1]["usage"]["prompt_tokens"] == 12
    assert config["tools"] == tools()


@pytest.mark.asyncio
async def test_text_count_uses_same_tools_and_template():
    instance = model()
    instance.get_full_context = lambda messages, template, **kwargs: str(
        (messages, kwargs)
    )
    instance._engine = SimpleNamespace(
        get_tokenizer=lambda: SimpleNamespace(encode=lambda text: list(text))
    )
    messages = [{"role": "user", "content": "read a"}]
    config = {"tools": tools(), "chat_template_kwargs": {"enable_thinking": False}}
    count = await instance.async_count_tokens(messages, config)
    prompt, _, render_tools, _, _ = await instance._prepare_chat_inputs(
        messages, copy.deepcopy(config)
    )
    assert render_tools == tools()
    assert count == len(prompt)
    assert config["tools"] == tools()


@pytest.mark.asyncio
async def test_image_count_uses_native_zero_output_prefill():
    instance = model(SGLANGVisionModel)
    instance._engine = SimpleNamespace(get_tokenizer=lambda: object())
    instance._prepare_chat_inputs = AsyncMock(
        return_value=("image prompt", ["data:image"], None, {}, {})
    )
    instance._non_stream_generate = AsyncMock(
        return_value={"meta_info": {"prompt_tokens": 237, "completion_tokens": 0}}
    )
    assert await instance.async_count_tokens([{"role": "user", "content": []}]) == 237
    args = instance._non_stream_generate.call_args
    assert args.args == ("image prompt",)
    assert args.kwargs["image_data"] == ["data:image"]
    assert args.kwargs["max_new_tokens"] == 0


@pytest.mark.parametrize(
    "supported,strict", [(True, True), (False, True), (True, False)]
)
def test_manual_budget_requires_native_support(monkeypatch, supported, strict):
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.io_struct",
        SimpleNamespace(
            GenerateReqInput=SimpleNamespace(
                __dataclass_fields__={"max_thinking_tokens": None} if supported else {}
            )
        ),
    )
    instance = model()
    instance._model_config["enable_strict_thinking"] = strict
    config = {
        "chat_template_kwargs": {"enable_thinking": True, "thinking_budget": 1024}
    }
    if supported and strict:
        instance._apply_thinking_budget(config)
        assert config["max_thinking_tokens"] == 1024
        assert instance._lift_logprob_request_params(config) == {
            "max_thinking_tokens": 1024,
            "require_reasoning": True,
        }
    else:
        with pytest.raises(NotImplementedError, match="thinking budgets"):
            instance._apply_thinking_budget(config)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["QwQ-32B", "deepseek-r1-0528"])
@pytest.mark.parametrize("content", [None, "omitted"])
async def test_text_preparation_renders_tool_call_turn_without_content(
    model_name, content
):
    from ... import BUILTIN_LLM_FAMILIES

    instance = model()
    instance.model_family = next(
        f for f in BUILTIN_LLM_FAMILIES if f.model_name == model_name
    )
    call = {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": "call_read",
                "type": "function",
                "function": {"name": "Read", "arguments": '{"file_path":"a"}'},
            }
        ],
    }
    if content == "omitted":
        del call["content"]

    def history(assistant):
        return [
            {"role": "user", "content": "read a"},
            assistant,
            {"role": "tool", "tool_call_id": "call_read", "content": "TOOL_RESULT_42"},
        ]

    messages = history(call)
    original = copy.deepcopy(messages)
    prompt, *_ = await instance._prepare_chat_inputs(messages, {"tools": tools()})
    expected, *_ = await instance._prepare_chat_inputs(
        history({**call, "content": ""}), {"tools": tools()}
    )
    assert prompt == expected
    assert "TOOL_RESULT_42" in prompt
    assert messages == original


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", [False, True])
async def test_generate_passes_per_request_reasoning_and_preserves_markers(thinking):
    instance = model()
    instance._validate_structured_reasoning = lambda _: None
    instance._non_stream_generate = AsyncMock(
        return_value={
            "text": "{}",
            "meta_info": {
                "prompt_tokens": 12,
                "completion_tokens": 1,
                "finish_reason": {"type": "stop"},
            },
        }
    )
    token = chat_context_var.set({"enable_thinking": thinking})
    try:
        await instance.async_generate(
            "prompt", {"json_schema": "{}", "_anthropic_request": True}
        )
    finally:
        chat_context_var.reset(token)
    params = instance._non_stream_generate.call_args.kwargs
    assert params["require_reasoning"] is thinking
    if thinking:
        assert params["skip_special_tokens"] is False


def test_non_reasoning_model_does_not_require_reasoning_plugin():
    instance = model()
    instance.reasoning_parser = ReasoningParser()
    instance._model_config = {}
    instance._validate_forced_tools({"tool_choice": "required"})


@pytest.mark.asyncio
@pytest.mark.parametrize("image_count", [0, 1, 2])
async def test_vision_preparation_includes_tools_and_consumes_template_kwargs(
    monkeypatch,
    image_count,
):
    import base64
    from io import BytesIO

    from PIL import Image

    instance = model(SGLANGVisionModel)
    colors = [(255, 0, 0), (0, 0, 255)][:image_count]
    source_images = [Image.new("RGB", (16, 16), color) for color in colors]
    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        SimpleNamespace(process_vision_info=lambda messages: (source_images, None)),
    )
    instance._transform_messages = lambda messages: messages
    captured = {}

    def render(messages, template, **kwargs):
        captured.update(kwargs)
        return "prompt"

    instance.get_full_context = render
    prompt, images, render_tools, context, config = await instance._prepare_chat_inputs(
        [{"role": "user", "content": "read a"}],
        {"tools": tools(), "chat_template_kwargs": {"enable_thinking": False}},
    )
    assert prompt == "prompt"
    if image_count:
        assert len(images) == image_count
        for image, color in zip(images, colors):
            header, encoded = image.split(",", 1)
            assert header == "data:image/jpeg;base64"
            decoded = Image.open(BytesIO(base64.b64decode(encoded)))
            assert decoded.size == (16, 16)
            assert all(abs(a - b) <= 1 for a, b in zip(decoded.getpixel((0, 0)), color))
    else:
        assert images is None
    assert captured["tools"] == render_tools == tools()
    assert context["enable_thinking"] is False
    assert "chat_template_kwargs" not in config


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", [SGLANGChatModel, SGLANGVisionModel])
@pytest.mark.parametrize(
    "choice", ["required", {"type": "function", "function": {"name": "Read"}}]
)
async def test_openai_choice_does_not_enter_forced_decoding(cls, choice):
    instance = model(cls)
    instance.reasoning_parser = None
    instance._prepare_chat_inputs = AsyncMock(
        return_value=("prompt", None, None, {}, {})
    )
    instance.async_generate = AsyncMock(return_value=completion("answer"))
    instance._forced_tool_chat = AsyncMock(
        side_effect=AssertionError("not an Anthropic request")
    )
    result = await instance.async_chat([], {"tools": tools(), "tool_choice": choice})
    assert result["choices"][0]["message"]["content"] == "answer"
    instance._forced_tool_chat.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel,expected", [(None, 2), (True, 2), (False, 1)])
async def test_only_explicit_false_disables_parallel_tools(parallel, expected):
    instance = model()
    instance._prepare_chat_inputs = AsyncMock(
        return_value=("prompt", None, tools(), {}, {})
    )
    instance.async_generate = AsyncMock(return_value=completion(""))
    instance._post_process_completion = lambda *args: {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": str(i),
                            "type": "function",
                            "function": {"name": "Read", "arguments": "{}"},
                        }
                        for i in range(2)
                    ]
                }
            }
        ]
    }
    result = await instance.async_chat([], {"parallel_tool_calls": parallel})
    assert len(result["choices"][0]["message"]["tool_calls"]) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [False, True])
async def test_openai_json_schema_works_without_request_reasoning_capability(
    monkeypatch, configured
):
    import json

    instance = model()
    instance._model_config = {"reasoning_parser": "native"} if configured else {}
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.io_struct",
        SimpleNamespace(GenerateReqInput=SimpleNamespace(__dataclass_fields__={})),
    )
    instance._non_stream_generate = AsyncMock(
        return_value={
            "text": '{"answer":42}',
            "meta_info": {
                "prompt_tokens": 12,
                "completion_tokens": 2,
                "finish_reason": {"type": "stop"},
            },
        }
    )
    result = await instance.async_generate(
        "prompt",
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}},
            }
        },
    )
    assert json.loads(result["choices"][0]["text"]) == {"answer": 42}
    assert "require_reasoning" not in instance._non_stream_generate.call_args.kwargs
    assert "_anthropic_request" not in instance._non_stream_generate.call_args.kwargs
