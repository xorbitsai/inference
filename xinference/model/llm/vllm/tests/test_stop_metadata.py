# Copyright 2022-2026 Xinference Holdings Pte. Ltd
import json
import sys
from types import SimpleNamespace

import pytest
from packaging.version import Version

from xinference.api.protocols import anthropic_stream_events, openai_to_anthropic

from ...core import chat_context_var
from ...reasoning_parser import ReasoningParser
from ...tool_parsers.qwen_tool_parser import QwenToolParser
from .. import core


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode", ["plain", "reasoning", "tools", "openai"])
@pytest.mark.parametrize(
    "finish_reason,stop_reason,expected",
    [
        ("stop", "HALT", "stop_sequence"),
        ("stop", "\n停止", "stop_sequence"),
        ("stop", None, "end_turn"),
        ("stop", 2, "end_turn"),
        ("stop", "<im_end>", "end_turn"),
        ("length", None, "max_tokens"),
    ],
)
async def test_native_stop_metadata_reaches_anthropic(
    monkeypatch, stream, mode, finish_reason, stop_reason, expected
):
    class SamplingParams:
        def __init__(self, n=1, max_tokens=2):
            self.max_tokens = max_tokens

    monkeypatch.setitem(
        sys.modules,
        "vllm.sampling_params",
        SimpleNamespace(SamplingParams=SamplingParams),
    )
    monkeypatch.setattr(core, "VLLM_INSTALLED", False)
    monkeypatch.setattr(core, "VLLM_VERSION", Version("0.5.0"))
    model = object.__new__(core.VLLMChatModel)
    model.model_uid = "test"
    model.model_family = SimpleNamespace(model_name="qwen3", model_ability=["chat"])
    model._nixl_config = model._xavier_config = None
    model._active_request_ids = set()
    model.reasoning_parser = (
        ReasoningParser(reasoning_content=True) if mode == "reasoning" else None
    )
    model.tool_parser = QwenToolParser() if mode == "tools" else None

    async def generate(*args, **kwargs):
        yield SimpleNamespace(
            prompt_token_ids=[1, 2, 3],
            finished=True,
            outputs=[
                SimpleNamespace(
                    text="hello",
                    token_ids=[4, 5],
                    index=0,
                    finish_reason=finish_reason,
                    stop_reason=stop_reason,
                )
            ],
        )

    async def prepare(messages, config):
        return "prompt", ([{"type": "function"}] if mode == "tools" else []), {}, config

    model._engine = SimpleNamespace(generate=generate)
    model._prepare_chat_inputs = prepare
    model._sanitize_generate_config = lambda config: dict(
        n=1, lora_name=None, max_tokens=2, **config
    )
    model.async_generate = core.VLLMModel.async_generate.__wrapped__.__get__(model)
    token = chat_context_var.set({"enable_thinking": False})
    try:
        response = await core.VLLMChatModel.async_chat.__wrapped__(
            model,
            [],
            {
                "stream": stream,
                "stream_options": {"include_usage": True},
                "_anthropic_request": mode != "openai",
            },
        )
        if stream:
            events = [
                json.loads(event["data"])
                async for event in anthropic_stream_events(
                    response, "test", "req_test", stop_sequences=["HALT", "\n停止"]
                )
            ]
            result = next(e["delta"] for e in events if e["type"] == "message_delta")
            assert events[-1]["type"] == "message_stop"
        else:
            result = openai_to_anthropic(
                response, "test", stop_sequences=["HALT", "\n停止"]
            )
        if mode == "openai" and expected == "stop_sequence":
            expected = "end_turn"
        assert result["stop_reason"] == expected
        assert result["stop_sequence"] == (
            stop_reason if expected == "stop_sequence" else None
        )
        assert not model._active_request_ids
    finally:
        chat_context_var.reset(token)
