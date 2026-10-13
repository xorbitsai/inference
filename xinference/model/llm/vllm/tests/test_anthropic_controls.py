# Copyright 2022-2026 Xinference Holdings Pte. Ltd
import copy
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import jsonschema
import pytest

from ...tool_controls import (
    convert_forced_completion,
    prepare_forced_tools,
    single_tool_stream,
)
from .. import core


def tools():
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "$defs": {"path": {"type": "string"}},
                    "properties": {"file_path": {"$ref": "#/$defs/path"}},
                    "required": ["file_path"],
                    "additionalProperties": False,
                },
            },
        }
        for name in ("Read", "Delete")
    ]


def completion(text, finish_reason="stop"):
    return {
        "id": "chat-test",
        "model": "qwen",
        "created": 1,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 42, "completion_tokens": 10, "total_tokens": 52},
    }


def test_forced_named_tool_schema_restricts_name_arguments_and_count():
    config = {
        "tools": tools(),
        "tool_choice": {"type": "function", "function": {"name": "Read"}},
        "parallel_tool_calls": False,
    }
    original = copy.deepcopy(config)
    messages = [{"role": "user", "content": "Delete the file"}]
    prepared, generated = prepare_forced_tools(messages, config)
    schema = generated["response_format"]["json_schema"]["schema"]
    valid = [{"name": "Read", "arguments": {"file_path": "/tmp/a"}}]
    jsonschema.validate(valid, schema)
    for bad in (
        [{"name": "Delete", "arguments": {"file_path": "/tmp/a"}}],
        [{"name": "Read", "arguments": {"file_path": 7}}],
        valid * 2,
        [],
    ):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)
    assert generated["_require_structured_output"] is True
    assert "tool definitions" in prepared[0]["content"]
    assert config == original
    assert messages == [{"role": "user", "content": "Delete the file"}]


def test_required_choice_preserves_alternatives_and_rejects_unknown_names():
    _, generated = prepare_forced_tools(
        [], {"tools": tools(), "tool_choice": "required"}
    )
    schema = generated["response_format"]["json_schema"]["schema"]
    jsonschema.validate(
        [
            {"name": "Delete", "arguments": {"file_path": "/tmp/a"}},
            {"name": "Read", "arguments": {"file_path": "/tmp/b"}},
        ],
        schema,
    )
    with pytest.raises(ValueError, match="declared tool"):
        prepare_forced_tools(
            [], {"tools": tools(), "tool_choice": {"function": {"name": "missing"}}}
        )


def test_truncated_arguments_never_become_executable_calls():
    result = convert_forced_completion(completion('[{"name":', "length"))
    assert result["choices"][0]["finish_reason"] == "length"
    assert not result["choices"][0]["message"].get("tool_calls")
    assert result["choices"][0]["message"]["content"] is None
    with pytest.raises(RuntimeError, match="invalid JSON"):
        convert_forced_completion(completion("not JSON"))


@pytest.mark.asyncio
async def test_forced_stream_uses_constrained_generation_and_reports_usage(monkeypatch):
    model = object.__new__(core.VLLMChatModel)
    model.reasoning_parser = None
    monkeypatch.setattr(core, "StructuredOutputsParams", object)
    model._engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            structured_outputs_config=SimpleNamespace(
                reasoning_parser="qwen3", enable_in_reasoning=False
            )
        )
    )
    generated_completion = completion(
        '[{"name":"Read","arguments":{"file_path":"/tmp/a"}}]'
    )
    generated_completion["choices"][0]["message"]["reasoning_content"] = "plan"
    model.async_chat = AsyncMock(return_value=generated_completion)
    config = {
        "tools": tools(),
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": True},
    }
    stream = await model._forced_tool_chat([], config, "request")
    chunks = [c async for c in stream]
    generated = model.async_chat.call_args.args[1]
    assert generated["_require_structured_output"] is True
    assert generated["stream"] is False
    assert "tool_choice" not in generated
    assert len(chunks[0]["choices"][0]["delta"]["tool_calls"]) == 1
    assert chunks[0]["choices"][0]["delta"]["reasoning_content"] == "plan"
    assert chunks[0]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["prompt_tokens"] == 42


@pytest.mark.asyncio
async def test_serial_stream_keeps_one_call_and_closes_upstream():
    closed = []

    async def source():
        try:
            for calls in (
                [{"index": 2, "id": "a"}, {"index": 3, "id": "b"}],
                [
                    {"index": 3, "function": {"arguments": "ignored"}},
                    {"index": 2, "function": {"arguments": "kept"}},
                ],
            ):
                yield {"choices": [{"index": 0, "delta": {"tool_calls": calls}}]}
            yield {"choices": [], "usage": {"prompt_tokens": 42}}
        finally:
            closed.append(True)

    chunks = [c async for c in single_tool_stream(source())]
    assert [
        [c["index"] for c in chunk["choices"][0]["delta"]["tool_calls"]]
        for chunk in chunks[:-1]
    ] == [[2], [2]]
    assert chunks[-1]["usage"]["prompt_tokens"] == 42
    assert closed == [True]


@pytest.mark.asyncio
async def test_vllm_counting_uses_generation_preparation_and_loaded_tokenizer():
    model = object.__new__(core.VLLMChatModel)
    model.lora_requests = []
    tokenizer = object()
    model._prepare_chat_inputs = AsyncMock(
        return_value=("rendered-with-tools", tools(), {}, {})
    )
    model._get_tokenizer = AsyncMock(return_value=tokenizer)
    model._gen_tokens_prompt = AsyncMock(
        return_value={"prompt_token_ids": [1, 2, 3, 4]}
    )
    messages = [{"role": "user", "content": "hello"}]
    config = {"tools": tools()}
    assert await model.async_count_tokens(messages, config) == 4
    model._gen_tokens_prompt.assert_awaited_once_with(
        tokenizer, "rendered-with-tools", {}
    )
    assert "tools" in config


@pytest.mark.asyncio
async def test_multimodal_count_uses_engine_preprocessor_without_generation(
    monkeypatch,
):
    model = object.__new__(core.VLLMMultiModel)
    model.lora_requests = []
    model._prepare_chat_inputs = AsyncMock(
        return_value=(
            {"prompt": "vision", "multi_modal_data": {"image": [object()]}},
            [],
            {},
            {},
        )
    )
    model._get_tokenizer = AsyncMock(return_value=object())
    engine_prompt = {
        "prompt_token_ids": [1, 2],
        "multi_modal_data": {"image": [object()]},
    }
    model._gen_tokens_prompt = AsyncMock(return_value=engine_prompt)
    seen = []

    def process_inputs(request_id, prompt, params, **kwargs):
        assert prompt is engine_prompt
        seen.append(request_id)
        return SimpleNamespace(prompt_token_ids=list(range(87)))

    original_cache = object()
    mm_processor = SimpleNamespace(cache=original_cache)
    renderer = SimpleNamespace(mm_processor=mm_processor)

    class Processor:
        def __init__(self):
            self.renderer = renderer

        def process_inputs(self, *args, **kwargs):
            assert self.renderer is not renderer
            assert self.renderer.mm_processor.cache is None
            return process_inputs(*args, **kwargs)

    model._engine = SimpleNamespace(
        input_processor=Processor(),
        get_supported_tasks=AsyncMock(return_value=("generate",)),
    )
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: kwargs)
    )
    assert await model.async_count_tokens([], {}) == 87
    assert seen[0].startswith("count-")
    assert renderer.mm_processor.cache is original_cache


def test_forced_choice_requires_engine_constraints_and_disallows_thinking(monkeypatch):
    from ...reasoning_parser import ReasoningParser

    model = object.__new__(core.VLLMChatModel)
    model.reasoning_parser = ReasoningParser(
        reasoning_start_tag="<think>", reasoning_end_tag="</think>"
    )
    model._engine = None
    monkeypatch.setattr(core, "GuidedDecodingParams", None)
    monkeypatch.setattr(core, "StructuredOutputsParams", None)
    with pytest.raises(NotImplementedError, match="structured decoding"):
        model._validate_forced_tools({"tool_choice": "required"})
    monkeypatch.setattr(core, "StructuredOutputsParams", object)
    with pytest.raises(NotImplementedError, match="with thinking"):
        model._validate_forced_tools(
            {
                "tool_choice": "required",
                "chat_template_kwargs": {"enable_thinking": True},
            }
        )
    model._engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            structured_outputs_config=SimpleNamespace(
                reasoning_parser="qwen3", enable_in_reasoning=False
            )
        )
    )
    model._validate_forced_tools(
        {"tool_choice": "required", "chat_template_kwargs": {"enable_thinking": True}}
    )
    with pytest.raises(ValueError, match="manual thinking"):
        model._validate_forced_tools(
            {
                "tool_choice": "required",
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "thinking_budget": 1024,
                },
            }
        )


def test_non_reasoning_model_can_force_tools_without_reasoning_plugin(monkeypatch):
    from ...reasoning_parser import ReasoningParser

    model = object.__new__(core.VLLMChatModel)
    model.reasoning_parser = ReasoningParser(enable_thinking=True)
    model._engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            structured_outputs_config=SimpleNamespace(reasoning_parser="")
        )
    )
    monkeypatch.setattr(core, "StructuredOutputsParams", object)
    model._validate_forced_tools({"tool_choice": "required"})


@pytest.mark.asyncio
async def test_required_constraint_cannot_fall_back_to_unconstrained_generation(
    monkeypatch,
):
    from packaging import version

    from .. import utils

    # A host without vLLM uses RuntimeError as its engine-death fallback;
    # this test intentionally raises a normal NotImplementedError.
    monkeypatch.setattr(utils, "VLLM_ENGINE_DEAD_ERRORS", ())

    def broken_constraint(**kwargs):
        raise ValueError("unsupported schema")

    package = SimpleNamespace()
    sampling = SimpleNamespace(SamplingParams=lambda **kwargs: None)
    package.sampling_params = sampling
    monkeypatch.setitem(sys.modules, "vllm", package)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling)
    monkeypatch.setattr(core, "VLLM_INSTALLED", True)
    monkeypatch.setattr(core, "VLLM_VERSION", version.parse("0.19.0"))
    monkeypatch.setattr(core, "GuidedDecodingParams", None)
    monkeypatch.setattr(core, "StructuredOutputsParams", broken_constraint)
    model = object.__new__(core.VLLMChatModel)
    model.reasoning_parser = None
    model.lora_requests = []
    with pytest.raises(
        NotImplementedError, match="required structured output constraint"
    ):
        await model.async_generate(
            "prompt",
            {"_require_structured_output": True, "guided_json": {"type": "object"}},
        )


@pytest.mark.asyncio
async def test_backend_preparation_preserves_controls_and_counts_prompt(monkeypatch):
    model = object.__new__(core.VLLMChatModel)
    model.reasoning_parser = None
    model.model_family = SimpleNamespace(model_ability=["chat", "tools"])
    model.async_count_tokens = AsyncMock(return_value=42)
    monkeypatch.setattr(core, "StructuredOutputsParams", object)
    prepared = await model.async_prepare_anthropic_request(
        [{"role": "user", "content": "read"}],
        {"tools": tools(), "tool_choice": "required", "parallel_tool_calls": False},
    )
    assert prepared["generate_config"]["tool_choice"] == "required"
    assert prepared["generate_config"]["parallel_tool_calls"] is False
    assert prepared["input_tokens"] == 42


@pytest.mark.asyncio
@pytest.mark.parametrize("format_type", ["json_schema", "json_object"])
async def test_json_thinking_requires_structured_reasoning_at_preparation(
    monkeypatch, format_type
):
    from ...reasoning_parser import ReasoningParser

    model = object.__new__(core.VLLMChatModel)
    model.reasoning_parser = ReasoningParser(
        reasoning_start_tag="<think>", reasoning_end_tag="</think>"
    )
    model._engine = None
    monkeypatch.setattr(core, "StructuredOutputsParams", object)
    with pytest.raises(NotImplementedError, match="with thinking"):
        await model.async_prepare_anthropic_request(
            [],
            {
                "response_format": {"type": format_type},
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", [core.VLLMChatModel, core.VLLMMultiModel])
@pytest.mark.parametrize(
    "choice", ["required", {"type": "function", "function": {"name": "Read"}}]
)
async def test_openai_tool_choice_keeps_the_legacy_path(cls, choice):
    instance = object.__new__(cls)
    instance._engine = None
    instance.reasoning_parser = None
    prompt = {"prompt": "prompt"} if cls is core.VLLMMultiModel else "prompt"
    instance._prepare_chat_inputs = AsyncMock(return_value=(prompt, [], {}, {}))
    instance._forced_tool_chat = AsyncMock(
        side_effect=AssertionError("OpenAI must not opt into new forced decoding")
    )
    generated = {
        "id": "test",
        "model": "test",
        "created": 1,
        "choices": [{"text": "answer", "finish_reason": "stop"}],
        "usage": {},
    }
    instance.async_generate = AsyncMock(return_value=generated)
    result = await cls.async_chat.__wrapped__(
        instance, [], {"tools": tools(), "tool_choice": choice}
    )
    assert result["choices"][0]["message"]["content"] == "answer"
    assert "tool_choice" not in instance._prepare_chat_inputs.call_args.args[1]
    instance._forced_tool_chat.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", [core.VLLMChatModel, core.VLLMMultiModel])
@pytest.mark.parametrize("format_type", ["json_schema", "json_object", "text"])
@pytest.mark.parametrize(
    "prefix", ["qwen3_hybrid", "assistant<think>\n", "assistant<think>\n</think>\n"]
)
async def test_openai_json_schema_starts_after_closed_prompt_reasoning(
    cls, format_type, prefix
):
    import json

    from ...reasoning_parser import ReasoningParser

    instance = object.__new__(cls)
    instance.reasoning_parser = ReasoningParser(
        reasoning_start_tag="<think>",
        reasoning_end_tag="</think>",
        reasoning_content=False,
        enable_thinking=True,
    )
    instance._engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            structured_outputs_config=SimpleNamespace(
                reasoning_parser="xinference_auto"
            )
        )
    )
    if prefix == "qwen3_hybrid":
        from pathlib import Path

        family = json.loads(
            (Path(__file__).parents[2] / "models/qwen3.json").read_text()
        )[0]
        prefix = instance._build_from_raw_template(
            [{"role": "user", "content": "hello"}],
            family["chat_template"],
            enable_thinking=True,
        )
        assert prefix.endswith("<|im_start|>assistant\n")
        assert "<think>" not in prefix
    prompt = {"prompt": prefix} if cls is core.VLLMMultiModel else prefix
    instance._prepare_chat_inputs = AsyncMock(return_value=(prompt, [], {}, {}))
    generated = {
        "id": "test",
        "model": "test",
        "created": 1,
        "choices": [{"text": '{"answer":42}', "finish_reason": "stop"}],
        "usage": {},
    }
    instance.async_generate = AsyncMock(return_value=generated)
    result = await cls.async_chat.__wrapped__(
        instance, [], {"response_format": {"type": format_type}}
    )
    rendered = instance.async_generate.call_args.args[0]
    if isinstance(rendered, dict):
        rendered = rendered["prompt"]
    if format_type == "text":
        assert rendered == prefix
    else:
        assert rendered.endswith("</think>\n")
        assert rendered.count("<think>") == 1
    content = result["choices"][0]["message"]["content"]
    if format_type == "text":
        assert content.startswith("<think>\n")
    else:
        assert json.loads(content) == {"answer": 42}
    assert "reasoning_content" not in result["choices"][0]["message"]


@pytest.mark.asyncio
async def test_forced_json_extraction_does_not_require_reasoning_content_display(
    monkeypatch,
):
    from ...core import chat_context_var
    from ...reasoning_parser import ReasoningParser

    instance = object.__new__(core.VLLMChatModel)
    instance.reasoning_parser = ReasoningParser(
        reasoning_start_tag="<think>",
        reasoning_end_tag="</think>",
        reasoning_content=False,
    )
    instance._validate_forced_tools = lambda _: None

    async def chat(messages, config, **kwargs):
        token = chat_context_var.set(config["chat_template_kwargs"])
        try:
            generated = {
                "id": "test",
                "model": "test",
                "created": 1,
                "choices": [
                    {
                        "text": '<think>plan</think>[{"name":"Read","arguments":{"file_path":"a"}}]',
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }
            return instance._to_chat_completion(generated, instance.reasoning_parser)
        finally:
            chat_context_var.reset(token)

    instance.async_chat = chat
    result = await instance._forced_tool_chat(
        [], {"tools": tools(), "tool_choice": "required"}, "rid"
    )
    assert (
        result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "Read"
    )
    assert "reasoning_content" not in result["choices"][0]["message"]
