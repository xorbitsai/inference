# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
"""Request-level multimodal limits and expanded-token budgeting for vLLM."""

import copy
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from packaging.version import Version

from ...context_length import (
    ContextLengthExceeded,
    classify_vllm_context_error,
    normalize_runtime_positive_int,
    parse_vllm_prompt_length_error,
)
from ...llm_family import match_llm
from .. import core


class RuntimeContextLength(int):
    pass


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(core, "VLLM_VERSION", Version("0.30.0"))
    family = match_llm("qwen2.5-vl-instruct", "pytorch", "3", "none")
    instance = core.VLLMMultiModel("adaptive-test", family, "/models/test", {})
    instance._device_count = 1
    return instance


def test_unset_limit_is_not_forwarded_to_engine(model):
    config = model._sanitize_model_config({})

    assert "limit_mm_per_prompt" not in config
    assert model._explicit_limit_mm_per_prompt is None
    assert model.get_limit_mm_per_prompt() == {"image": 2, "video": 2}


@pytest.mark.parametrize(
    "message, expected",
    [
        (
            "The decoder prompt (length 280175) is longer than the maximum "
            "model length of 262144.",
            (280175, 262144),
        ),
        (
            "The encoder prompt (length 5000) is longer than the maximum "
            "model length of 4096.",
            (5000, 4096),
        ),
        (
            "[address=worker:1234, pid=5678] The decoder prompt (length 4096) "
            "plus the number of requested output tokens (at least 1) is longer "
            "than the maximum model length of 4096.",
            (4096, 4096),
        ),
        (
            "This model's maximum context length is 4096 tokens. However, "
            "your prompt contains 5000 input tokens, which exceeds this limit "
            "even without any output tokens. Please reduce the length of the "
            "input prompt.",
            (5000, 4096),
        ),
        (
            "[address=worker:1234, pid=5678] This model's maximum context "
            "length is 1048576 tokens. However, your prompt contains at least "
            "1048577 input tokens, which exceeds this limit even without any "
            "output tokens. Please reduce the length of the input prompt. "
            "(parameter=input_tokens, value=1048577)",
            (1048577, 1048576),
        ),
    ],
)
def test_parse_vllm_prompt_length_error(message, expected):
    assert parse_vllm_prompt_length_error(message) == expected


@pytest.mark.parametrize(
    "message",
    [
        "No module named 'vllm'",
        "CUDA out of memory while allocating context cache",
        "The decoder prompt is invalid",
        "The decoder prompt (length 5000) is longer than 4096.",
        "This model's maximum context length is 4096 tokens.",
        "max_tokens=8192 cannot be greater than max_model_len=4096. "
        "Please request fewer output tokens.",
    ],
)
def test_parse_vllm_prompt_length_error_does_not_guess(message):
    assert parse_vllm_prompt_length_error(message) is None


def test_runtime_context_length_normalizes_integer_subclass():
    value = normalize_runtime_positive_int(RuntimeContextLength(262144))

    assert value == 262144
    assert type(value) is int


@pytest.mark.parametrize("value", [None, True, False, "262144", 262144.0, 0, -1])
def test_runtime_context_length_rejects_non_positive_or_non_integral(value):
    assert normalize_runtime_positive_int(value) is None


def test_classify_vllm_context_error_uses_structured_attributes():
    error_type = type(
        "VLLMValidationError",
        (ValueError,),
        {"__module__": "vllm.exceptions"},
    )
    error = error_type("engine-specific validation failed")
    error.prompt_len = RuntimeContextLength(263175)
    error.max_model_len = RuntimeContextLength(262144)

    parsed = classify_vllm_context_error(error)

    assert parsed == (263175, 262144)
    assert parsed is not None
    assert all(type(value) is int for value in parsed)


def test_classify_vllm_context_error_does_not_guess_unknown_validation():
    error_type = type(
        "VLLMValidationError",
        (ValueError,),
        {"__module__": "vllm.exceptions"},
    )
    assert (
        classify_vllm_context_error(
            error_type("invalid image processor parameters"), 262144
        )
        is None
    )


def test_classify_vllm_context_error_checks_worker_local_exception_chain():
    error_type = type(
        "VLLMValidationError",
        (ValueError,),
        {"__module__": "vllm.exceptions"},
    )
    engine_error = error_type(
        "The decoder prompt (length 263175) is longer than the maximum "
        "model length of 262144."
    )
    wrapper = RuntimeError("worker wrapper")
    wrapper.__cause__ = engine_error

    assert classify_vllm_context_error(wrapper) == (263175, 262144)


def test_classify_vllm_context_error_recognizes_renderer_rejection():
    error_type = type(
        "VLLMValidationError",
        (ValueError,),
        {"__module__": "vllm.exceptions"},
    )
    engine_error = error_type(
        "This model's maximum context length is 1048576 tokens. However, your "
        "prompt contains at least 1048577 input tokens, which exceeds this "
        "limit even without any output tokens. Please reduce the length of "
        "the input prompt."
    )
    engine_error.parameter = "input_tokens"
    engine_error.value = 1048577
    wrapper = RuntimeError("worker wrapper")
    wrapper.__cause__ = engine_error

    assert classify_vllm_context_error(wrapper) == (1048577, 1048576)


@pytest.mark.parametrize(
    "limits, engine_limits, preflight_limits",
    [
        ({"image": 4, "video": 2}, {"image": 4, "video": 2}, {"image": 4, "video": 2}),
        (
            '{"image":{"count":5,"width":512,"height":512},"video":{"count":0}}',
            {
                "image": {"count": 5, "width": 512, "height": 512},
                "video": {"count": 0},
            },
            {"image": 5, "video": 0},
        ),
        (
            ['{"image": 4', '"video": {"count": 2, "num_frames": 8}}'],
            {"image": 4, "video": {"count": 2, "num_frames": 8}},
            {"image": 4, "video": 2},
        ),
        (
            {"image": {"count": 0, "width": 256}},
            {"image": {"count": 0, "width": 256}},
            {"image": 0},
        ),
        ({}, {}, {}),
    ],
)
def test_explicit_limits_are_preserved(model, limits, engine_limits, preflight_limits):
    config = model._sanitize_model_config({"limit_mm_per_prompt": limits})

    assert config["limit_mm_per_prompt"] == engine_limits
    assert model.get_limit_mm_per_prompt() == preflight_limits


@pytest.mark.parametrize(
    "limits",
    [
        "not-json",
        {"image": -1},
        {"image": True},
        {"image": {"count": -1}},
        {"image": {"count": False}},
        {"image": {"width": 512}},
        ["not-json"],
    ],
)
def test_invalid_explicit_limit_uses_compatibility_fallback(model, limits):
    config = model._sanitize_model_config({"limit_mm_per_prompt": limits})

    assert "limit_mm_per_prompt" not in config
    assert model._explicit_limit_mm_per_prompt is None
    assert model.get_limit_mm_per_prompt() == {"image": 2, "video": 2}


@pytest.mark.asyncio
async def test_count_tokens_omits_tools_disabled_by_tool_choice(model):
    observed = {}

    async def prepare_inputs(messages, config):
        observed["config"] = copy.deepcopy(config)
        prompt = {
            "prompt": "without disabled tools",
            "multi_modal_data": {"image": [object()]},
        }
        return prompt, [], {}, config

    model._prepare_chat_inputs = prepare_inputs
    model._get_tokenizer = AsyncMock(return_value=object())
    model._gen_tokens_prompt = AsyncMock(
        return_value={
            "prompt_token_ids": list(range(100)),
            "multi_modal_data": {"image": [object()]},
        }
    )
    model._count_prompt_tokens = AsyncMock(return_value=100)
    config = {
        "tool_choice": "none",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "large_unused_tool",
                    "description": "unused " * 1000,
                    "parameters": {"type": "object"},
                },
            }
        ],
    }

    assert await model.async_count_tokens([], config) == 100
    assert "tool_choice" not in observed["config"]
    assert "tools" not in observed["config"]
    assert "tool_choice" in config
    assert "tools" in config


@pytest.mark.asyncio
@pytest.mark.parametrize("anthropic", [False, True])
async def test_forced_tool_preflight_counts_the_generation_prompt(
    model, monkeypatch, anthropic
):
    from .....types import CreateChatCompletion

    tool = {
        "type": "function",
        "function": {
            "name": "describe_image",
            "parameters": {
                "type": "object",
                "properties": {"caption": {"type": "string"}},
            },
        },
    }
    # Build kwargs exactly like the REST handler: pydantic dumps the OpenAI
    # ``Iterable[...]`` tools field as a one-shot list_iterator, not a list.
    body = CreateChatCompletion.parse_obj(
        {
            "model": "test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe it"},
                        {"type": "image_url", "image_url": {"url": "unused"}},
                    ],
                }
            ],
            "tools": [tool],
            "tool_choice": "required",
            "max_tokens": 64,
        }
    )
    messages = list(body.messages)
    config = body.dict(exclude_unset=True, exclude={"model", "messages"})
    if anthropic:
        config["_anthropic_request"] = True
    forced = Mock(side_effect=core.prepare_forced_tools)
    monkeypatch.setattr(core, "prepare_forced_tools", forced)
    observed = {}

    async def prepare_inputs(messages, config):
        observed["messages"] = copy.deepcopy(messages)
        # Mirrors VLLMMultiModel._prepare_chat_inputs.
        observed["tools"] = list(config.pop("tools", []))
        observed["config"] = copy.deepcopy(config)
        prompt = {"prompt": "rendered", "multi_modal_data": {"image": [object()]}}
        return prompt, observed["tools"], {}, config

    structured = Mock(side_effect=lambda prompt, config, context: prompt)
    model._prepare_chat_inputs = prepare_inputs
    model._prepare_structured_reasoning_prompt = structured
    model._get_tokenizer = AsyncMock(return_value=object())
    model._gen_tokens_prompt = AsyncMock(
        return_value={
            "prompt_token_ids": list(range(100)),
            "multi_modal_data": {"image": [object()]},
        }
    )
    model._count_prompt_tokens = AsyncMock(return_value=100)
    model._context_length = 512

    result = await model.async_preflight_multimodal_request(messages, config)

    assert result["input_tokens"] == 100
    assert result["effective_max_tokens"] == 64
    assert "tool_choice" not in observed["config"]
    # The private marker is restored for prompt post-processing only, as in
    # async_chat, and never reaches template rendering.
    assert "_anthropic_request" not in observed["config"]
    assert structured.call_args.args[1].get("_anthropic_request", False) is anthropic
    if anthropic:
        # Anthropic generation goes through _forced_tool_chat: constrained JSON
        # output with tool definitions injected into the system prompt.
        forced.assert_called_once()
        assert observed["tools"] == []
        assert observed["messages"][0]["role"] == "system"
        assert "describe_image" in observed["messages"][0]["content"]
        assert observed["config"]["response_format"]["type"] == "json_schema"
    else:
        # OpenAI generation renders tools natively, even for "required".
        forced.assert_not_called()
        assert observed["tools"] == [tool]
        assert observed["messages"] == messages
        assert "response_format" not in observed["config"]


@pytest.mark.asyncio
async def test_processor_options_survive_tokenization(model, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(TokensPrompt=dict))
    model._tokenize = Mock(return_value=[1, 2])
    prompt = {
        "prompt": "images",
        "multi_modal_data": {"image": [object(), object(), object()]},
        "mm_processor_kwargs": {"max_pixels": 65536},
    }

    tokens = await model._gen_tokens_prompt(object(), prompt, {})

    assert tokens["mm_processor_kwargs"] == {"max_pixels": 65536}
    assert tokens["multi_modal_data"] is prompt["multi_modal_data"]


@pytest.mark.asyncio
async def test_preflight_adapts_output_budget_without_mutating_request(model):
    model._context_length = 32
    model.async_count_tokens = AsyncMock(return_value=12)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "unused"}},
                {"type": "image_url", "image_url": {"url": "unused"}},
                {"type": "image_url", "image_url": {"url": "unused"}},
            ],
        }
    ]
    config = {"max_tokens": 24, "mm_processor_kwargs": {"max_pixels": 4096}}
    original = copy.deepcopy((messages, config))

    result = await model.async_preflight_multimodal_request(messages, config)

    assert result == {
        "mode": "adaptive",
        "limits": None,
        "input_tokens": 12,
        "max_model_len": 32,
        "effective_max_tokens": 20,
    }
    assert (messages, config) == original


@pytest.mark.asyncio
async def test_explicit_limit_wins_before_token_count(model):
    model._model_config = model._sanitize_model_config(
        {"limit_mm_per_prompt": {"image": 2}}
    )
    model._context_length = 32
    model.async_count_tokens = AsyncMock(return_value=4)
    image = {"type": "image_url", "image_url": {"url": "unused"}}

    with pytest.raises(ValueError, match=r"At most 2 image\(s\).+received 3"):
        await model.async_preflight_multimodal_request(
            [{"role": "user", "content": [image, image, image]}],
            {"max_tokens": 8},
        )

    model.async_count_tokens.assert_not_awaited()


@pytest.mark.asyncio
async def test_structured_zero_count_disables_modality_before_token_count(model):
    model._model_config = model._sanitize_model_config(
        {"limit_mm_per_prompt": {"image": {"count": 0, "width": 512, "height": 512}}}
    )
    model._context_length = 32
    model.async_count_tokens = AsyncMock(return_value=4)

    with pytest.raises(ValueError, match=r"At most 0 image\(s\).+received 1"):
        await model.async_preflight_multimodal_request(
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": "unused"}],
                }
            ],
            {"max_tokens": 8},
        )

    assert model._model_config["limit_mm_per_prompt"] == {
        "image": {"count": 0, "width": 512, "height": 512}
    }
    model.async_count_tokens.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_runtime_counting_uses_compatibility_limit(model):
    model._context_length = 32
    model.async_count_tokens = AsyncMock(
        side_effect=NotImplementedError("multimodal token counting")
    )
    image = {"type": "image_url", "image_url": {"url": "unused"}}

    result = await model.async_preflight_multimodal_request(
        [{"role": "user", "content": [image, image]}], {"max_tokens": 8}
    )
    assert result["mode"] == "compat_fallback"
    assert result["limits"] == {"image": 2, "video": 2}

    with pytest.raises(ValueError, match=r"At most 2 image\(s\).+received 3"):
        await model.async_preflight_multimodal_request(
            [{"role": "user", "content": [image, image, image]}],
            {"max_tokens": 8},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("input_tokens", [32, 33])
async def test_preflight_rejects_exhausted_context(model, input_tokens):
    model._context_length = 32
    model.async_count_tokens = AsyncMock(return_value=input_tokens)

    with pytest.raises(ContextLengthExceeded, match="max_model_len=32"):
        await model.async_preflight_multimodal_request(
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": "unused"}],
                }
            ],
            {"max_tokens": 8},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "The decoder prompt (length 280175) is longer than the maximum "
        "model length of 262144.",
        "The encoder prompt (length 262144) plus the number of requested "
        "output tokens (at least 1) is longer than the maximum model length "
        "of 262144.",
    ],
)
async def test_preflight_normalizes_vllm_prompt_length_error(model, message):
    model._context_length = 262144
    model.async_count_tokens = AsyncMock(side_effect=RuntimeError(message))

    with pytest.raises(ContextLengthExceeded) as exc_info:
        await model.async_preflight_multimodal_request(
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": "unused"}],
                }
            ],
            {"max_tokens": 8},
        )

    assert "max_model_len=262144" in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.asyncio
async def test_preflight_preserves_unknown_counting_error(model):
    model._context_length = 32
    error = RuntimeError("processor crashed")
    model.async_count_tokens = AsyncMock(side_effect=error)

    with pytest.raises(RuntimeError) as exc_info:
        await model.async_preflight_multimodal_request(
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": "unused"}],
                }
            ],
            {"max_tokens": 8},
        )

    assert exc_info.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("base_input_tokens", [4, 5])
async def test_counting_rejects_base_prompt_before_multimodal_processor(
    model, base_input_tokens
):
    model._context_length = RuntimeContextLength(4)
    processor = Mock()
    model._engine = SimpleNamespace(input_processor=processor)

    with pytest.raises(ContextLengthExceeded) as exc_info:
        await model._count_prompt_tokens(
            {
                "prompt_token_ids": list(range(base_input_tokens)),
                "multi_modal_data": {"image": [object()]},
            }
        )

    error = exc_info.value
    assert error.input_tokens == base_input_tokens
    assert error.max_model_len == 4
    assert type(error.max_model_len) is int
    assert error.stage == "base_prompt_budget"
    assert error.diagnostic_code == "BASE_PROMPT_EXHAUSTS_CONTEXT"
    processor.process_inputs.assert_not_called()


@pytest.mark.asyncio
async def test_counting_uses_cache_isolated_processor_copy(model, monkeypatch):
    class SamplingParams:
        def __init__(self, max_tokens):
            self.max_tokens = max_tokens

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(SamplingParams=SamplingParams)
    )
    model._context_length = RuntimeContextLength(4)
    original_cache = object()
    original_stats = object()
    mm_processor = SimpleNamespace(cache=original_cache)
    original_sender_cache = object()
    original_processor_only_cache = object()
    renderer = SimpleNamespace(
        mm_processor=mm_processor,
        _mm_cache_stats=original_stats,
        _mm_processor_cache=original_sender_cache,
        _mm_processor_only_cache=original_processor_only_cache,
    )
    preprocessor = SimpleNamespace(renderer=renderer)
    observed = {}

    def process_inputs(*args, **kwargs):
        observed["processor"] = processor_copy_target["value"]
        observed["renderer"] = observed["processor"].renderer
        return SimpleNamespace(prompt_token_ids=[1, 2, 3, 4])

    processor = SimpleNamespace(
        renderer=renderer,
        input_preprocessor=preprocessor,
        process_inputs=process_inputs,
    )
    processor_copy_target = {"value": None}
    real_copy = copy.copy

    def tracking_copy(value):
        result = real_copy(value)
        if value is processor:
            processor_copy_target["value"] = result
        return result

    monkeypatch.setattr(core.copy, "copy", tracking_copy)
    model._engine = SimpleNamespace(
        input_processor=processor,
        get_supported_tasks=AsyncMock(return_value={"generate"}),
    )

    count = await model._count_prompt_tokens(
        {"prompt_token_ids": [1], "multi_modal_data": {"image": [object()]}}
    )

    assert count == 4
    assert observed["processor"] is not processor
    assert observed["renderer"] is not renderer
    assert observed["renderer"].mm_processor is not mm_processor
    assert observed["renderer"].mm_processor.cache is None
    assert observed["renderer"]._mm_cache_stats is None
    assert observed["processor"].input_preprocessor.renderer is observed["renderer"]
    assert mm_processor.cache is original_cache
    assert renderer._mm_cache_stats is original_stats
    assert observed["renderer"]._mm_processor_cache is None
    assert observed["renderer"]._mm_processor_only_cache is None
    assert renderer._mm_processor_cache is original_sender_cache
    assert renderer._mm_processor_only_cache is original_processor_only_cache
