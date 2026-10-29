# Copyright 2022-2026 Xinference Holdings Pte. Ltd
import copy
import json
import sys
from pathlib import Path
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


@pytest.mark.parametrize("unsupported", [None, "sampling", "markers", "disabled"])
def test_manual_thinking_budget_native_mapping(monkeypatch, unsupported):
    from ...reasoning_parser import ReasoningParser

    def sampling(*, thinking_token_budget=None):
        pass

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            SamplingParams=(lambda: None) if unsupported == "sampling" else sampling
        ),
    )
    instance = object.__new__(core.VLLMChatModel)
    instance.reasoning_parser = ReasoningParser(
        reasoning_start_tag="<think>", reasoning_end_tag="</think>"
    )
    reasoning = SimpleNamespace(
        reasoning_start_str="<think>",
        reasoning_end_str="custom" if unsupported == "markers" else "</think>",
    )
    instance._engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            reasoning_config=None if unsupported == "disabled" else reasoning
        )
    )
    config = {
        "chat_template_kwargs": {"enable_thinking": True, "thinking_budget": 1024}
    }
    if unsupported:
        with pytest.raises(NotImplementedError, match="thinking budgets require"):
            instance._apply_thinking_budget(config)
    else:
        instance._apply_thinking_budget(config)
        assert config["thinking_token_budget"] == 1024


@pytest.mark.parametrize("anthropic", [False, True])
def test_native_budget_forwarding_is_anthropic_only(monkeypatch, anthropic):
    from packaging.version import Version

    monkeypatch.setattr(core, "VLLM_VERSION", Version("0.28.0"))
    config = {"thinking_token_budget": 16, "_anthropic_request": anthropic}
    sanitized = core.VLLMModel._sanitize_generate_config(config)
    assert sanitized.get("thinking_token_budget") == (16 if anthropic else None)


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", [core.VLLMChatModel, core.VLLMMultiModel])
@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize("preopened", [False, True])
async def test_adaptive_structured_output_after_replayed_tool_reasoning(
    cls, forced, preopened
):
    from ...reasoning_parser import ReasoningParser

    instance = object.__new__(cls)
    instance.reasoning_parser = ReasoningParser(
        reasoning_start_tag="<think>",
        reasoning_end_tag="</think>",
        reasoning_content=True,
    )
    tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text.encode()))
    instance.reasoning_parser.bind_tokenizer(tokenizer)
    family = json.loads((Path(__file__).parents[2] / "models/qwen3.json").read_text())[
        0
    ]
    messages = [
        {"role": "user", "content": "Read the file"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Inspect it first",
            "tool_calls": [
                {
                    "id": "call1",
                    "type": "function",
                    "function": {"name": "Read", "arguments": {"file_path": "a"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call1", "content": "file data"},
    ]
    prompt = instance._build_from_raw_template(
        messages, family["chat_template"], enable_thinking=True
    )
    assert prompt.endswith("<|im_start|>assistant\n") and "</think>" in prompt
    if preopened:
        prompt += "<think>\n"
    context = {"enable_thinking": True, "reasoning_content": True}
    instance._validate_forced_tools = lambda _: None

    async def prepare(_messages, config):
        core.chat_context_var.set(context)
        rendered = {"prompt": prompt} if cls is core.VLLMMultiModel else prompt
        return rendered, [], context, config

    instance._prepare_chat_inputs = prepare
    payload = (
        '[{"name":"Read","arguments":{"file_path":"a"}}]' if forced else '{"answer":42}'
    )
    generated_prompts = []

    async def generate(rendered, *args, **kwargs):
        assert args[0]["_anthropic_request"] is True
        text = rendered["prompt"] if isinstance(rendered, dict) else rendered
        generated_prompts.append(text)
        # Match vLLM's initial grammar gate on the complete prompt token IDs.
        ended = instance.reasoning_parser.is_reasoning_end(tokenizer.encode(text))
        output = payload if ended else "plan</think>" + payload
        return {
            "id": "test",
            "model": "test",
            "created": 1,
            "choices": [{"index": 0, "text": output, "finish_reason": "stop"}],
            "usage": {},
        }

    instance.async_generate = generate
    instance.async_chat = lambda *args, **kwargs: cls.async_chat.__wrapped__(
        instance, *args, **kwargs
    )
    config = {
        "_anthropic_request": True,
        "chat_template_kwargs": context,
        "response_format": {"type": "json_object"},
    }
    if forced:
        config.update(tools=tools(), tool_choice="required")
    instance.lora_requests = []
    instance._get_tokenizer = AsyncMock(return_value=tokenizer)
    instance._gen_tokens_prompt = AsyncMock(
        side_effect=lambda _tokenizer, rendered, _config: {
            "prompt_token_ids": tokenizer.encode(
                rendered["prompt"] if isinstance(rendered, dict) else rendered
            )
        }
    )
    token = core.chat_context_var.set(context)
    try:
        count = await instance.async_count_tokens(messages, config)
        result = await instance.async_chat(messages, config)
    finally:
        core.chat_context_var.reset(token)
    message = result["choices"][0]["message"]
    assert generated_prompts[0].endswith("assistant\n<think>\n")
    assert generated_prompts[0].count("<think>") == 2
    assert count == len(tokenizer.encode(generated_prompts[0]))
    assert message["reasoning_content"].strip() == "plan"
    if forced:
        assert message["tool_calls"][0]["function"]["name"] == "Read"
    else:
        assert json.loads(message["content"]) == {"answer": 42}


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


@pytest.mark.parametrize(
    "function", [{"name": "now"}, {"name": "now", "parameters": None}]
)
def test_forced_tool_without_parameters_takes_empty_arguments(function):
    # OpenAI allows omitting ``parameters`` for a function with no arguments.
    _, generated = prepare_forced_tools(
        [],
        {
            "tools": [{"type": "function", "function": function}],
            "tool_choice": "required",
        },
    )
    schema = generated["response_format"]["json_schema"]["schema"]
    jsonschema.validate([{"name": "now", "arguments": {}}], schema)


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
    assert "_anthropic_request" not in instance.async_generate.call_args.args[1]
    instance._forced_tool_chat.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", [core.VLLMChatModel, core.VLLMMultiModel])
@pytest.mark.parametrize(
    "config,output",
    [
        ({"response_format": {"type": "json_schema"}}, '{"answer":42}'),
        ({"response_format": {"type": "json_object"}}, '{"answer":42}'),
        ({"guided_json": {"type": "object"}}, '{"answer":42}'),
        ({"guided_json": {}}, '{"answer":42}'),
        ({"guided_json_object": True}, '{"answer":42}'),
        ({"guided_choice": ["yes", "no"]}, "yes"),
        ({"guided_regex": "yes|no"}, "yes"),
        ({"guided_grammar": 'root ::= "yes"'}, "yes"),
        ({"response_format": {"type": "text"}}, '{"answer":42}'),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "prefix", ["qwen3_hybrid", "assistant<think>\n", "assistant<think>\n</think>\n"]
)
async def test_openai_json_schema_starts_after_closed_prompt_reasoning(
    cls, config, output, stream, prefix
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
    instance._prepare_chat_inputs = AsyncMock(
        return_value=(prompt, [], {}, {"stream": stream})
    )
    generated = {
        "id": "test",
        "model": "test",
        "created": 1,
        "choices": [{"index": 0, "text": output, "finish_reason": "stop"}],
        "usage": {},
    }

    async def chunks():
        yield generated

    instance.async_generate = AsyncMock(return_value=chunks() if stream else generated)
    result = await cls.async_chat.__wrapped__(
        instance, [], {**config, "stream": stream}
    )
    rendered = instance.async_generate.call_args.args[0]
    if isinstance(rendered, dict):
        rendered = rendered["prompt"]
    unconstrained = config == {"response_format": {"type": "text"}}
    if unconstrained:
        assert rendered == prefix
    else:
        assert rendered.endswith("</think>\n")
        assert rendered.count("<think>") == 1
    if stream:
        content = "".join(
            choice["delta"].get("content") or ""
            for chunk in [c async for c in result]
            for choice in chunk["choices"]
        )
    else:
        content = result["choices"][0]["message"]["content"]
        assert "reasoning_content" not in result["choices"][0]["message"]
    if unconstrained:
        assert content.startswith("<think>\n")
    else:
        assert content == output


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


@pytest.mark.asyncio
@pytest.mark.parametrize("anthropic", [False, True])
@pytest.mark.parametrize("multimodal", [False, True])
async def test_completion_constraints_start_before_thinking(
    monkeypatch, anthropic, multimodal
):
    from ...reasoning_parser import ReasoningParser

    class StopBeforeGeneration(Exception):
        pass

    monkeypatch.setitem(
        sys.modules,
        "vllm.sampling_params",
        SimpleNamespace(
            SamplingParams=lambda **kwargs: SimpleNamespace(max_tokens=None)
        ),
    )
    monkeypatch.setattr(core, "VLLM_INSTALLED", False)
    monkeypatch.setattr(core, "VLLM_VERSION", core.version.parse("0.19.0"))
    model = object.__new__(core.VLLMChatModel)
    model.reasoning_parser = ReasoningParser(
        reasoning_start_tag="<think>", reasoning_end_tag="</think>"
    )
    model._engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            structured_outputs_config=SimpleNamespace(
                reasoning_parser="xinference_auto"
            )
        )
    )
    model.lora_requests = []
    model._get_tokenizer = AsyncMock(return_value=object())
    model._gen_tokens_prompt = AsyncMock(side_effect=StopBeforeGeneration)
    prompt = (
        {"prompt": "assistant\n", "multi_modal_data": {"image": object()}}
        if multimodal
        else "assistant\n"
    )
    with pytest.raises(StopBeforeGeneration):
        await core.VLLMModel.async_generate.__wrapped__(
            model, prompt, {"guided_choice": ["yes"], "_anthropic_request": anthropic}
        )
    rendered = model._gen_tokens_prompt.call_args.args[1]
    text = rendered["prompt"] if multimodal else rendered
    assert text == ("assistant\n" if anthropic else "assistant\n<think></think>\n")
    if multimodal:
        assert rendered["multi_modal_data"] is prompt["multi_modal_data"]
        assert prompt["prompt"] == "assistant\n"


@pytest.mark.parametrize(
    "key,value",
    [
        ("guided_json", {"type": "object"}),
        ("guided_json_object", True),
        ("guided_choice", ["yes"]),
        ("guided_regex", "yes|no"),
        ("guided_grammar", 'root ::= "yes"'),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_api_guided_params_reach_legacy_constraints(key, value, nested):
    from xinference.api.restful_api import RESTfulAPI

    raw = {"extra_body": {key: value}} if nested else {key: value}
    config = RESTfulAPI.extract_guided_params(raw)
    model = object.__new__(core.VLLMChatModel)
    model._engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            structured_outputs_config=SimpleNamespace(
                reasoning_parser="xinference_auto"
            )
        )
    )
    assert config[key] == value
    assert model._use_legacy_constraints(config, False)
    assert not model._use_legacy_constraints(config, True)
