# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from packaging.version import Version

from xinference.api.protocols.openai_responses import (
    chat_to_response,
    parse_responses_request,
    responses_stream_events,
)

from ..utils import ChatModelMixin, generate_completion_chunk

MISSING = object()


def _usage(cached_tokens):
    usage = {"prompt_tokens": 100, "completion_tokens": 2, "total_tokens": 102}
    if cached_tokens is not MISSING and cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return usage


@pytest.fixture(params=["vllm", "sglang", "mlx"])
def engine(request):
    return request.param


@pytest.fixture
def model_factory(engine, monkeypatch):
    def create(cached_tokens):
        if engine == "vllm":
            from ..vllm import core

            class SamplingParams:
                def __init__(self, n=1, max_tokens=2):
                    self.max_tokens = max_tokens

            sampling_params = ModuleType("vllm.sampling_params")
            sampling_params.SamplingParams = SamplingParams
            monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)
            monkeypatch.setattr(core, "VLLM_INSTALLED", False)
            monkeypatch.setattr(core, "VLLM_VERSION", Version("0.5.0"))
            model = object.__new__(core.VLLMModel)
            model._nixl_config = model._xavier_config = None
            model.reasoning_parser = None

            async def generate(*args, **kwargs):
                for text, token_ids, finished in [
                    ("hello", [1], False),
                    ("hello world", [1, 2], True),
                ]:
                    output = SimpleNamespace(
                        prompt_token_ids=list(range(100)),
                        outputs=[
                            SimpleNamespace(
                                text=text,
                                token_ids=token_ids,
                                index=0,
                                logprobs=None,
                                finish_reason="stop" if finished else None,
                            )
                        ],
                        finished=finished,
                    )
                    if cached_tokens is not MISSING:
                        output.num_cached_tokens = cached_tokens
                    yield output

            model._engine = SimpleNamespace(generate=generate)
            # Exercise generation without the worker's process-exit guard.
            model.async_generate = core.VLLMModel.async_generate.__wrapped__.__get__(
                model
            )
        elif engine == "sglang":
            from ..sglang.core import SGLANGModel

            model = object.__new__(SGLANGModel)

            def meta_info(completion_tokens, finished):
                meta = dict(
                    prompt_tokens=100,
                    completion_tokens=completion_tokens,
                    finish_reason={"type": "stop"} if finished else None,
                )
                if cached_tokens is not MISSING:
                    meta["cached_tokens"] = cached_tokens
                # The total already includes these sources; do not add them again.
                meta["cached_tokens_details"] = dict(device=16, host=16, storage=32)
                return meta

            async def generate(*args, **kwargs):
                return dict(text="hello world", meta_info=meta_info(2, True))

            async def generate_stream(*args, **kwargs):
                yield meta_info(1, False), "hello"
                yield meta_info(2, True), " world"

            model._non_stream_generate = generate
            model._stream_generate = generate_stream
        else:
            from ..mlx.core import MLXModel

            model = object.__new__(MLXModel)
            model._model = model._tokenizer = object()

            async def generate(*args, **kwargs):
                return "hello world", _usage(cached_tokens)

            async def generate_stream(*args, **kwargs):
                for text, finished in [("hello world", False), ("", True)]:
                    chunk = generate_completion_chunk(
                        text,
                        finish_reason="stop" if finished else None,
                        chunk_id="request-1",
                        model_uid="test-model",
                        prompt_tokens=100,
                        completion_tokens=2,
                        total_tokens=102,
                    )
                    chunk["usage"] = _usage(cached_tokens)
                    yield chunk

            model._batch_model = SimpleNamespace(
                generate=generate, generate_stream=generate_stream
            )

        model.model_uid = "test-model"
        model._active_request_ids = set()
        model._sanitize_generate_config = lambda config: dict(
            {"stream": False, "stream_options": None, "n": 1, "lora_name": None},
            **(config or {}),
        )
        return model

    return create


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_tokens", [0, 64])
async def test_non_stream_cached_tokens_reach_chat_and_responses(
    model_factory, cached_tokens
):
    completion = await model_factory(cached_tokens).async_generate(
        "prompt", {"max_tokens": 2}
    )
    assert completion["usage"] == _usage(cached_tokens)
    chat = ChatModelMixin._to_chat_completion(completion)
    assert chat["usage"] == _usage(cached_tokens)
    response = chat_to_response(
        chat, parse_responses_request({"model": "test-model", "input": "prompt"})
    )
    assert response["usage"]["input_tokens_details"] == {"cached_tokens": cached_tokens}


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_tokens", [0, 64])
@pytest.mark.parametrize("include_usage", [False, True])
async def test_stream_cached_tokens_reach_final_and_usage_chunks(
    model_factory, cached_tokens, include_usage
):
    stream = await model_factory(cached_tokens).async_generate(
        "prompt",
        {
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_usage": include_usage},
        },
    )
    chunks = [chunk async for chunk in stream]
    final = next(
        chunk
        for chunk in reversed(chunks)
        if chunk["choices"] and chunk["choices"][0]["finish_reason"]
    )
    assert final["usage"] == _usage(cached_tokens)
    usage_chunks = [chunk for chunk in chunks if not chunk["choices"]]
    assert len(usage_chunks) == int(include_usage)
    if include_usage:
        assert chunks[-1]["choices"] == []
        assert usage_chunks[0]["usage"] == _usage(cached_tokens)

    async def source():
        for chunk in chunks:
            yield chunk

    chat_stream = ChatModelMixin._async_to_chat_completion_chunks(source())
    request = parse_responses_request(
        {"model": "test-model", "input": "prompt", "stream": True}
    )
    events = [
        json.loads(event["data"])
        async for event in responses_stream_events(chat_stream, request)
    ]
    assert events[-1]["response"]["usage"]["input_tokens_details"] == {
        "cached_tokens": cached_tokens
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("cached_tokens", [MISSING, None])
@pytest.mark.parametrize("stream", [False, True])
async def test_unreported_cached_tokens_remain_optional(
    model_factory, cached_tokens, stream
):
    result = await model_factory(cached_tokens).async_generate(
        "prompt",
        {
            "max_tokens": 2,
            "stream": stream,
            "stream_options": {"include_usage": True},
        },
    )
    if stream:
        chunks = [chunk async for chunk in result]
        assert chunks[-1]["choices"] == []
        assert chunks[-1]["usage"] == _usage(cached_tokens)

        async def source():
            for chunk in chunks:
                yield chunk

        events = [
            json.loads(event["data"])
            async for event in responses_stream_events(
                ChatModelMixin._async_to_chat_completion_chunks(source()),
                parse_responses_request(
                    {"model": "test-model", "input": "prompt", "stream": True}
                ),
            )
        ]
        response = events[-1]["response"]
    else:
        assert result["usage"] == _usage(cached_tokens)
        response = chat_to_response(
            ChatModelMixin._to_chat_completion(result),
            parse_responses_request({"model": "test-model", "input": "prompt"}),
        )
    assert response["usage"]["input_tokens_details"] == {"cached_tokens": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_tokens", [0, 64])
async def test_tool_stream_preserves_cached_tokens_in_usage_chunk(
    model_factory, cached_tokens
):
    from ..tool_parsers.qwen_tool_parser import QwenToolParser

    completions = await model_factory(cached_tokens).async_generate(
        "prompt",
        {"max_tokens": 2, "stream": True, "stream_options": {"include_usage": True}},
    )

    async def tool_completions():
        sent = False
        async for chunk in completions:
            for choice in chunk["choices"]:
                choice["text"] = (
                    '<tool_call>{"name":"get_weather","arguments":{"city":"Beijing"}}'
                    "</tool_call>"
                    if not sent
                    else ""
                )
                sent = True
            yield chunk

    mixin = ChatModelMixin()
    mixin.model_family = "qwen2-instruct"
    mixin.model_uid = "test-model"
    mixin.reasoning_parser = None
    mixin.tool_parser = QwenToolParser()
    chunks = [
        chunk
        async for chunk in mixin._async_to_tool_completion_chunks(tool_completions())
    ]
    final = next(
        chunk
        for chunk in reversed(chunks)
        if chunk["choices"] and chunk["choices"][0]["finish_reason"]
    )
    assert final["choices"][0]["finish_reason"] == "tool_calls"
    assert final["usage"] is None
    assert any(
        call["function"].get("name") == "get_weather"
        for chunk in chunks
        for choice in chunk["choices"]
        for call in choice["delta"].get("tool_calls", [])
    )
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == _usage(cached_tokens)

    async def source():
        for chunk in chunks:
            yield chunk

    events = [
        json.loads(event["data"])
        async for event in responses_stream_events(
            source(),
            parse_responses_request(
                {"model": "test-model", "input": "prompt", "stream": True}
            ),
        )
    ]
    assert events[-1]["response"]["usage"]["input_tokens_details"] == {
        "cached_tokens": cached_tokens
    }


def test_vllm_multiple_outputs_do_not_multiply_cached_tokens():
    from ..vllm.core import VLLMModel

    output = SimpleNamespace(
        prompt_token_ids=list(range(100)),
        num_cached_tokens=64,
        outputs=[SimpleNamespace(token_ids=[1, 2]), SimpleNamespace(token_ids=[3])],
    )
    assert VLLMModel._get_completion_usage(output) == dict(
        prompt_tokens=100,
        completion_tokens=3,
        total_tokens=103,
        prompt_tokens_details={"cached_tokens": 64},
    )
