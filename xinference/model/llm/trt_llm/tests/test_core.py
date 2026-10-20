# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ..core import TRTLLMModel


class Request:
    def __init__(self):
        self.prompt_token_ids = [10, 20]
        self.outputs = []
        self.finished = False
        self.aborted = False
        self.position = 0

    def abort(self):
        self.aborted = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.position == 2:
            raise StopAsyncIteration
        await asyncio.sleep(0)
        self.position += 1
        self.finished = self.position == 2
        self.outputs = [
            SimpleNamespace(
                index=0,
                text="ab" if self.position == 1 else "abc",
                text_diff="ab" if self.position == 1 else "c",
                token_ids=[1, 2] if self.position == 1 else [1, 2, 3],
                finish_reason="length" if self.finished else None,
            )
        ]
        return self

    def __await__(self):
        async def collect():
            async for _ in self:
                pass
            return self

        return collect().__await__()


class Engine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.requests = []
        self.tokenizer = SimpleNamespace(
            encode=lambda prompt, **kwargs: list(range(len(prompt))),
            chat_template="template",
            apply_chat_template=lambda messages, **kwargs: "hi",
        )

    def generate_async(self, tokens, params, streaming):
        self.tokens, self.params = tokens, params
        request = Request()
        self.requests.append(request)
        return request

    def shutdown(self):
        self.stopped = True


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(
        "xinference.model.llm.trt_llm.compat.prepare_runtime",
        lambda config: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm",
        SimpleNamespace(LLM=Engine, SamplingParams=SimpleNamespace),
    )
    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.2.1")
    family = SimpleNamespace(
        model_specs=[SimpleNamespace(quantization="none")],
        model_name="qwen3",
        model_family="qwen3",
        context_length=32,
        stop=["DEFAULT"],
        stop_token_ids=[999],
        model_ability=["chat", "hybrid"],
        reasoning_start_tag="<think>",
        reasoning_end_tag="</think>",
        chat_template=None,
    )
    result = TRTLLMModel(
        "tiny", family, "/model", {"max_seq_len": 32, "enable_thinking": False}
    )
    result.load()
    yield result
    result.stop()


@pytest.mark.asyncio
async def test_stream_is_incremental_and_usage_keeps_last_token(model):
    stream = await model.async_generate(
        "hi",
        {
            "max_tokens": 3,
            "stream": True,
            "stop": ["CUSTOM"],
            "stream_options": {"include_usage": True},
        },
        "request",
    )
    assert not model._engine.requests  # An unstarted/closed stream allocates nothing.
    first = await anext(stream)
    request = model._engine.requests[0]
    assert not request.finished
    assert first["choices"][0]["text"] == "ab"
    rest = [chunk async for chunk in stream]
    assert rest[0]["choices"][0]["text"] == "c"
    assert rest[0]["choices"][0]["finish_reason"] == "length"
    assert rest[-1]["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }
    assert model._engine.params.max_tokens == 3
    assert model._engine.params.stop == ["CUSTOM"]
    assert not model._requests and not request.aborted


@pytest.mark.asyncio
async def test_close_stream_aborts_and_releases_request(model):
    stream = await model.async_generate("hi", {"stream": True}, "close")
    await anext(stream)
    request = model._engine.requests[-1]
    await stream.aclose()
    assert request.aborted and not model._requests
    assert await model.abort_request("close") == "NO_OP"


@pytest.mark.asyncio
async def test_explicit_abort_and_shutdown(model):
    stream = await model.async_generate("hi", {"stream": True}, "cancel")
    await anext(stream)
    assert await model.abort_request("cancel") == "DONE"
    assert model._engine.requests[-1].aborted
    engine = model._engine
    model.stop()
    assert engine.stopped
    await stream.aclose()


@pytest.mark.asyncio
async def test_cancel_nonstream_aborts_engine_request(model, monkeypatch):
    class WaitingRequest(Request):
        def __await__(self):
            return asyncio.Event().wait().__await__()

    request = WaitingRequest()
    monkeypatch.setattr(
        model._engine, "generate_async", lambda *args, **kwargs: request
    )
    task = asyncio.create_task(model.async_generate("hi", request_id="waiting"))
    await asyncio.sleep(0)
    assert model._requests["waiting"] is request
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert request.aborted and not model._requests


@pytest.mark.asyncio
async def test_duplicate_id_does_not_remove_original_request(model):
    first = await model.async_generate("hi", {"stream": True}, "same")
    await anext(first)
    request = model._requests["same"]
    duplicate = await model.async_generate("hi", {"stream": True}, "same")
    with pytest.raises(ValueError, match="already active"):
        await anext(duplicate)
    assert model._requests["same"] is request
    await first.aclose()


@pytest.mark.asyncio
async def test_late_stream_close_does_not_abort_a_shutdown_executor(model, monkeypatch):
    stream = await model.async_generate("hi", {"stream": True})
    await anext(stream)
    request = model._engine.requests[-1]
    model.stop()
    calls = []

    def unavailable():
        calls.append("abort")
        raise RuntimeError("executor was already shut down")

    monkeypatch.setattr(request, "abort", unavailable)
    await stream.aclose()
    assert not calls and not model._requests


@pytest.mark.asyncio
async def test_nonstream_clamps_to_context_without_mutating_input(model):
    config = {"max_tokens": 100, "temperature": 0.5}
    result = await model.async_generate("hi", config)
    assert result["choices"][0]["text"] == "abc"
    assert result["usage"]["completion_tokens"] == 3
    assert model._engine.params.max_tokens == 30
    assert config["max_tokens"] == 100
    with pytest.raises(ValueError, match="no room"):
        await model.async_generate("x" * 32)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        {"max_tokens": 0},
        {"max_tokens": True},
        {"n": 2},
        {"logprobs": 2},
        {"cache_salt": "tenant"},
    ],
)
async def test_unsupported_request_fails_before_submission(model, config):
    with pytest.raises(ValueError):
        await model.async_generate("hi", config)
    assert not model._engine.requests


@pytest.mark.asyncio
async def test_chat_and_chat_stream_use_shared_conversion(model):
    messages = [{"role": "user", "content": "hello"}]
    result = await model.async_chat(messages)
    assert result["choices"][0]["message"]["content"] == "abc"
    stream = await model.async_chat(messages, {"stream": True})
    chunks = [chunk async for chunk in stream]
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "abc"
    with pytest.raises(ValueError, match="plain text"):
        await model.async_chat([{"role": "user", "content": [{"type": "image_url"}]}])


def test_platform_format_and_model_guards(monkeypatch, model):
    family = model.model_family
    spec = SimpleNamespace(model_format="pytorch")
    monkeypatch.setattr(sys, "platform", "linux")
    assert TRTLLMModel.match_json(family, spec, "none") is True
    assert TRTLLMModel.match_json(family, spec, "fp8")[0] is False
    family.model_family = "qwen3.5"
    assert TRTLLMModel.match_json(family, spec, "none")[0] is False


def test_engine_construction_failure_clears_state(monkeypatch, model):
    monkeypatch.setattr(
        "xinference.model.llm.trt_llm.compat.prepare_runtime", lambda _: None
    )
    monkeypatch.setattr(
        sys.modules["tensorrt_llm"],
        "LLM",
        Mock(side_effect=RuntimeError("load failed")),
    )
    failed = TRTLLMModel("failed", model.model_family, "/model")
    with pytest.raises(RuntimeError, match="load failed"):
        failed.load()
    assert failed._engine is None and not failed._requests
