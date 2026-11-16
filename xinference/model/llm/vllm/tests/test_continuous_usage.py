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
# CI: linux
"""``stream_options.continuous_usage_stats`` on vLLM chat streams."""

import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ...tool_parsers.qwen_tool_parser import QwenToolParser


def _usage(prompt, completion, cached=None):
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return usage


def _chunk(text, completion, finish_reason=None):
    # The shape VLLMModel.async_generate streams: cumulative usage on every chunk.
    return {
        "id": "test",
        "object": "text_completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": _usage(5, completion, cached=2),
    }


RAW = [
    _chunk("Hey", 1),
    _chunk(" there", 2),
    _chunk("!", 3),
    _chunk("", 3, finish_reason="stop"),
    {
        "id": "test",
        "object": "text_completion",
        "created": 1,
        "model": "test-model",
        "choices": [],
        "usage": _usage(5, 3, cached=2),
    },
]


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({"include_usage": True, "continuous_usage_stats": True}, True),
        ({"continuous_usage_stats": True}, False),
        ({"include_usage": True}, False),
        ({"include_usage": True, "continuous_usage_stats": "true"}, False),
        (None, False),
    ],
)
def test_continuous_usage_needs_include_usage(options, expected):
    from .. import core

    config = {} if options is None else {"stream_options": options}
    assert core.continuous_usage_requested(config) is expected
    assert core.continuous_usage_requested(None) is False


def _model(monkeypatch, model_class, closed):
    from .. import core

    model = object.__new__(getattr(core, model_class))
    family = json.loads((Path(__file__).parents[2] / "models/qwen3.json").read_text())[
        0
    ]
    model.model_family = types.SimpleNamespace(
        **family, model_family=None, has_architecture=lambda _: False
    )
    model.model_uid = "test-model"
    model.reasoning_parser = None
    model.tool_parser = QwenToolParser()
    model._get_tokenizer = AsyncMock(return_value=None)
    monkeypatch.setattr(
        core.asyncio,
        "to_thread",
        AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
    )
    monkeypatch.setitem(
        sys.modules,
        "qwen_omni_utils",
        types.SimpleNamespace(
            process_audio_info=None, process_mm_info=None, process_vision_info=None
        ),
    )

    async def generate(prompt, generate_config, tools, request_id=None):
        async def chunks():
            try:
                for chunk in RAW:
                    yield json.loads(json.dumps(chunk))
            finally:
                closed.append(True)

        return chunks()

    model.async_generate = generate
    return model


TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
    }
]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_class", ["VLLMChatModel", "VLLMMultiModel"])
@pytest.mark.parametrize("variant", ["plain", "tools", "single_tool"])
async def test_every_chat_chunk_carries_cumulative_usage(
    monkeypatch, model_class, variant
):
    closed: list = []
    model = _model(monkeypatch, model_class, closed)
    config = {
        "stream": True,
        "stop_token_ids": [151645],
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    if variant != "plain":
        config["tools"] = TOOLS
    if variant == "single_tool":
        config["parallel_tool_calls"] = False
    result = await model.async_chat([{"role": "user", "content": "Hi"}], config)
    chunks = [chunk async for chunk in result]
    content = [chunk for chunk in chunks if chunk["choices"]]
    assert content and all(chunk.get("usage") for chunk in content)
    completions = [chunk["usage"]["completion_tokens"] for chunk in chunks]
    assert completions == sorted(completions)
    # The last content chunk agrees with the final usage-only chunk.
    assert chunks[-1]["choices"] == []
    assert content[-1]["usage"] == chunks[-1]["usage"] == _usage(5, 3, cached=2)
    assert content[0]["usage"]["prompt_tokens_details"] == {"cached_tokens": 2}
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_class", ["VLLMChatModel", "VLLMMultiModel"])
@pytest.mark.parametrize(
    "options", [{"include_usage": True}, {"continuous_usage_stats": True}, None]
)
async def test_without_the_option_content_chunks_are_unchanged(
    monkeypatch, model_class, options
):
    closed: list = []
    model = _model(monkeypatch, model_class, closed)
    config = {"stream": True, "stop_token_ids": [151645]}
    if options is not None:
        config["stream_options"] = options
    result = await model.async_chat([{"role": "user", "content": "Hi"}], config)
    chunks = [chunk async for chunk in result]
    # As before: only the finish chunk (and the usage-only chunk) carry usage.
    assert [
        chunk["usage"] is not None
        for chunk in chunks
        if chunk["choices"] and chunk["choices"][0]["finish_reason"] is None
    ] == [False, False, False]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_class", ["VLLMChatModel", "VLLMMultiModel"])
async def test_early_close_still_closes_the_engine_stream(monkeypatch, model_class):
    closed: list = []
    model = _model(monkeypatch, model_class, closed)
    config = {
        "stream": True,
        "stop_token_ids": [151645],
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    result = await model.async_chat([{"role": "user", "content": "Hi"}], config)
    first = await result.__anext__()
    assert first["usage"]["completion_tokens"] >= 1
    await result.aclose()
    assert closed == [True]
