# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import struct
import sys

import pytest

from .....request_priority import to_trt_request_priority
from ..core import TRTLLMModel
from . import test_core

model = test_core.model


class NativePriorityEngine(test_core.Engine):
    def generate_async(self, tokens, params, streaming, priority=0.5):
        self.priority = priority
        return super().generate_async(tokens, params, streaming)


def _new_model(legacy, monkeypatch, config=None, version="1.3.0rc24"):
    monkeypatch.setattr("importlib.metadata.version", lambda name: version)
    monkeypatch.setattr(sys.modules["tensorrt_llm"], "LLM", NativePriorityEngine)
    return TRTLLMModel(
        "native",
        legacy.model_family,
        "/model",
        {"max_seq_len": 32, "enable_thinking": False, **(config or {})},
    )


@pytest.fixture
def native_model(model, monkeypatch):
    result = _new_model(model, monkeypatch)
    result.load()
    yield result
    result.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "priority,expected",
    [(-10000, 1.0), (-10, 0.5005), (0, 0.5), (10, 0.4995), (10000, 0.0)],
)
async def test_native_priority_is_converted_and_forwarded(
    native_model, stream, priority, expected
):
    config = {"priority": priority, "stream": stream, "max_tokens": 2}
    result = await native_model.async_generate("hi", config)
    if stream:
        assert [chunk async for chunk in result]
    assert native_model._engine.priority == expected
    assert native_model.supports_native_request_priority
    assert (
        native_model._engine.kwargs["scheduler_config"]["waiting_queue_policy"]
        == "priority"
    )
    assert "priority" not in native_model._engine.params.__dict__
    assert config["priority"] == priority


def test_all_integer_priorities_remain_distinct_in_native_float32():
    values = [
        struct.unpack("f", struct.pack("f", to_trt_request_priority(p)))[0]
        for p in range(-10000, 10001)
    ]
    assert values[0] == 1.0
    assert values[-1] == 0.0
    assert all(left > right for left, right in zip(values, values[1:]))


@pytest.mark.asyncio
async def test_explicit_fifo_rejects_nonzero_priority(model, monkeypatch):
    result = _new_model(
        model, monkeypatch, {"scheduler_config": {"waiting_queue_policy": "fcfs"}}
    )
    result.load()
    try:
        with pytest.raises(ValueError, match="waiting_queue_policy=priority"):
            await result.async_generate("hi", {"priority": -1})
        assert not result._engine.requests
        await result.async_generate("hi", {"priority": 0})
        assert result._engine.priority == 0.5
    finally:
        result.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["async_generate", "async_chat"])
@pytest.mark.parametrize("priority", [-10000, 0, 10000])
@pytest.mark.parametrize("stream", [False, True])
async def test_legacy_engine_rejects_explicit_priority(model, method, priority, stream):
    inputs = "hi" if method == "async_generate" else [{"role": "user", "content": "hi"}]
    with pytest.raises(ValueError, match="native priority support"):
        await getattr(model, method)(inputs, {"priority": priority, "stream": stream})
    assert not model._engine.requests


def test_removed_draft_target_api_reports_a_configuration_error(model, monkeypatch):
    def unsupported(config):
        config.pop("decoding_type")
        raise ValueError("Unknown decoding type")

    monkeypatch.setattr(
        sys.modules["tensorrt_llm.llmapi.llm_args"].DecodingBaseConfig,
        "from_dict",
        unsupported,
    )
    result = _new_model(model, monkeypatch, {"draft_model_path": "/draft"}, "1.3.0rc29")
    with pytest.raises(ValueError, match="does not support this Draft/Target"):
        result.load()
    assert result._engine is None


def test_public_draft_target_api_is_not_rejected_by_version_alone(model, monkeypatch):
    result = _new_model(model, monkeypatch, {"draft_model_path": "/draft"})
    result.load()
    try:
        assert (
            result._engine.kwargs["speculative_config"].decoding_type == "DraftTarget"
        )
    finally:
        result.stop()


def test_ngram_keeps_using_the_public_native_api(model, monkeypatch):
    result = _new_model(
        model,
        monkeypatch,
        {"speculative_config": {"decoding_type": "NGram", "max_draft_len": 4}},
    )
    result.load()
    try:
        assert result._engine.kwargs["speculative_config"].decoding_type == "NGram"
    finally:
        result.stop()


def test_legacy_engine_does_not_use_the_common_queue(model):
    assert not model.supports_native_request_priority
    assert not model.supports_request_priority_fallback
    assert "scheduler_config" not in model._engine.kwargs
