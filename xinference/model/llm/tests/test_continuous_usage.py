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
"""``stream_options.continuous_usage_stats`` on chat streams, across engines.

Every engine reports cumulative counters on its raw chunks; the shared chat
conversion attaches the latest value to each chat chunk with choices.
"""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ..tool_parsers.qwen_tool_parser import QwenToolParser
from ..utils import continuous_usage_requested

CONTINUOUS = {"include_usage": True, "continuous_usage_stats": True}
TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
    }
]
PROMPT, CACHED = 5, 2
TEXTS = ["Hey", " there", "!"]


def _usage(completion):
    return {
        "prompt_tokens": PROMPT,
        "completion_tokens": completion,
        "total_tokens": PROMPT + completion,
        "prompt_tokens_details": {"cached_tokens": CACHED},
    }


def _raw_chunks():
    # The raw shape vLLM and MLX stream: cumulative usage on every chunk.
    def chunk(text, completion, finish_reason=None, choices=True):
        return {
            "id": "test",
            "object": "text_completion",
            "created": 1,
            "model": "test-model",
            "choices": (
                [
                    {
                        "index": 0,
                        "text": text,
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ]
                if choices
                else []
            ),
            "usage": _usage(completion),
        }

    n = len(TEXTS)
    return [chunk(t, i + 1) for i, t in enumerate(TEXTS)] + [
        chunk("", n, "stop"),
        chunk("", n, choices=False),
    ]


class _Upstream:
    """An engine stream that records whether it was closed."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def agen(self):
        try:
            for chunk in self.chunks:
                yield json.loads(json.dumps(chunk))
        finally:
            self.closed = True

    def gen(self):
        try:
            for chunk in self.chunks:
                yield json.loads(json.dumps(chunk))
        finally:
            self.closed = True


def _qwen3_family():
    family = json.loads((Path(__file__).parents[1] / "models/qwen3.json").read_text())
    return SimpleNamespace(
        **family[0], model_family=None, has_architecture=lambda _: False
    )


def _vllm(monkeypatch, model_class, upstream):
    from ..vllm import core

    model = object.__new__(getattr(core, model_class))
    model.model_family = _qwen3_family()
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
        return upstream.agen()

    model.async_generate = generate
    return model


def _sglang(monkeypatch, upstream):
    from ..sglang.core import SGLANGChatModel

    # SGLang's native /generate stream: cumulative meta_info per event.
    events = [
        {
            "text": "".join(TEXTS[: i + 1]),
            "meta_info": {
                "prompt_tokens": PROMPT,
                "completion_tokens": i + 1,
                "cached_tokens": CACHED,
                "finish_reason": ({"type": "stop"} if i == len(TEXTS) - 1 else None),
            },
        }
        for i in range(len(TEXTS))
    ]

    async def iter_chunks():
        for index, event in enumerate(events):
            data = "data: " + json.dumps(event) + "\n\n"
            if index == len(events) - 1:
                data += "data: [DONE]\n\n"
            yield data.encode(), True

    async def abort_request(request_id):
        # SGLang releases an unfinished request by aborting it.
        upstream.closed = True

    response = MagicMock()
    response.__aenter__.return_value = response
    response.content.iter_chunks = iter_chunks
    session = MagicMock()
    session.__aenter__.return_value = session
    session.post.return_value = response
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    model = object.__new__(SGLANGChatModel)
    model.model_uid = "test-model"
    model.model_family = _qwen3_family()
    model.reasoning_parser = None
    model.tool_parser = QwenToolParser()
    model._active_request_ids = set()
    model._engine = SimpleNamespace(generate_url="http://unused/generate")
    model.abort_request = abort_request
    return model


def _mlx(model_class, upstream):
    from ..mlx import core

    model = object.__new__(getattr(core, model_class))
    model.model_family = _qwen3_family()
    model.model_uid = "test-model"
    model.reasoning_parser = None
    model.tool_parser = QwenToolParser()
    model._tokenizer = None
    model._prompt_cache = None
    model.get_full_context = lambda *args, **kwargs: "prompt"
    model._set_explicit_cache_boundaries = lambda *args, **kwargs: None
    model._get_reusable_prompt_prefix_len = lambda *args, **kwargs: None
    model._sanitize_generate_config = lambda config: dict(config or {})

    async def async_generate(prompt, generate_config, *args, **kwargs):
        return upstream.agen()

    def generate(inputs, generate_config):
        return upstream.gen()

    def prepare_chat_inputs(messages, generate_config):
        tools = generate_config.pop("tools", None)
        context = dict(generate_config.get("chat_template_kwargs") or {})
        return "prompt", tools, context, generate_config

    model.async_generate = async_generate
    model.generate = generate
    model._prepare_chat_inputs = prepare_chat_inputs
    return model


ENGINES = [
    "vllm-chat",
    "vllm-multi",
    "sglang",
    "mlx-chat",
    "mlx-vision",
]


async def _chat(monkeypatch, engine, config, upstream):
    messages = [{"role": "user", "content": "Hi"}]
    config = {"stream": True, "stop_token_ids": [151645], **config}
    if engine.startswith("vllm"):
        model_class = "VLLMChatModel" if engine == "vllm-chat" else "VLLMMultiModel"
        model = _vllm(monkeypatch, model_class, upstream)
        return await model.async_chat(messages, config)
    if engine == "sglang":
        model = _sglang(monkeypatch, upstream)
        model._prepare_chat_inputs = AsyncMock(
            side_effect=lambda messages, config: (
                "prompt",
                None,
                config.pop("tools", None),
                dict(config.get("chat_template_kwargs") or {}),
                config,
            )
        )
        return await model.async_chat(messages, config)
    if engine == "mlx-chat":
        return await _mlx("MLXChatModel", upstream).async_chat(messages, config)
    stream = _mlx("MLXVisionModel", upstream).chat(messages, config)

    async def as_async():
        try:
            for chunk in stream:
                yield chunk
        finally:
            stream.close()

    return as_async()


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        (CONTINUOUS, True),
        ({"continuous_usage_stats": True}, False),
        ({"include_usage": True}, False),
        ({"include_usage": True, "continuous_usage_stats": "true"}, False),
        (None, False),
    ],
)
def test_continuous_usage_needs_include_usage(options, expected):
    config = {} if options is None else {"stream_options": options}
    assert continuous_usage_requested(config) is expected
    assert continuous_usage_requested(None) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("variant", ["plain", "tools", "single_tool"])
async def test_every_chat_chunk_carries_cumulative_usage(monkeypatch, engine, variant):
    upstream = _Upstream(_raw_chunks())
    config = {"stream_options": dict(CONTINUOUS)}
    if variant != "plain":
        config["tools"] = TOOLS
    if variant == "single_tool":
        config["parallel_tool_calls"] = False
    result = await _chat(monkeypatch, engine, config, upstream)
    chunks = [chunk async for chunk in result]
    content = [chunk for chunk in chunks if chunk["choices"]]
    assert content and all(chunk.get("usage") for chunk in content)
    completions = [chunk["usage"]["completion_tokens"] for chunk in chunks]
    assert completions == sorted(completions)
    # Cached input is reported from the first chunk on.
    assert content[0]["usage"]["prompt_tokens"] == PROMPT
    assert content[0]["usage"]["prompt_tokens_details"] == {"cached_tokens": CACHED}
    # The final usage-only chunk is unchanged and agrees with the last content.
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == content[-1]["usage"]
    assert chunks[-1]["usage"]["completion_tokens"] == len(TEXTS)


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "options", [{"include_usage": True}, {"continuous_usage_stats": True}, None]
)
async def test_without_the_option_content_chunks_are_unchanged(
    monkeypatch, engine, options
):
    upstream = _Upstream(_raw_chunks())
    config = {} if options is None else {"stream_options": dict(options)}
    result = await _chat(monkeypatch, engine, config, upstream)
    chunks = [chunk async for chunk in result]
    # As before: content chunks before the finish chunk carry no usage.
    intermediate = [
        chunk
        for chunk in chunks
        if chunk["choices"] and chunk["choices"][0]["finish_reason"] is None
    ]
    assert intermediate
    assert all(chunk.get("usage") is None for chunk in intermediate)


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ENGINES)
async def test_chat_template_kwargs_cannot_enable_it(monkeypatch, engine):
    # Only stream_options turns it on; a same-named client template kwarg is
    # overwritten, the same way ``_anthropic_request`` is.
    upstream = _Upstream(_raw_chunks())
    config = {
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"_continuous_usage": True},
    }
    result = await _chat(monkeypatch, engine, config, upstream)
    chunks = [chunk async for chunk in result]
    intermediate = [
        chunk
        for chunk in chunks
        if chunk["choices"] and chunk["choices"][0]["finish_reason"] is None
    ]
    assert intermediate
    assert all(chunk.get("usage") is None for chunk in intermediate)


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ENGINES)
async def test_early_close_still_closes_the_engine_stream(monkeypatch, engine):
    upstream = _Upstream(_raw_chunks())
    config = {"stream_options": dict(CONTINUOUS)}
    result = await _chat(monkeypatch, engine, config, upstream)
    first = await result.__anext__()
    assert first["usage"]["completion_tokens"] >= 1
    await result.aclose()
    assert upstream.closed


@pytest.mark.asyncio
async def test_harmony_close_reaches_the_source_when_the_parser_fails(monkeypatch):
    from .. import harmony

    upstream = _Upstream(_raw_chunks())

    class _FailingParser:
        def __init__(self, chunks):
            self.chunks = chunks

        def __aiter__(self):
            return self

        async def __anext__(self):
            return await self.chunks.__anext__()

        async def aclose(self):
            raise RuntimeError("parser close failed")

    monkeypatch.setattr(
        harmony,
        "async_stream_harmony_chat_completion",
        lambda chunks, **kwargs: _FailingParser(chunks),
    )
    result = harmony.async_stream_harmony_chat_completion_chunks(upstream.agen())
    await result.__anext__()
    with pytest.raises(RuntimeError, match="parser close failed"):
        await result.aclose()
    assert upstream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [CONTINUOUS, None])
async def test_harmony_early_close_closes_the_engine_stream(monkeypatch, options):
    from ..vllm import core

    upstream = _Upstream(_raw_chunks())
    model = _vllm(monkeypatch, "VLLMChatModel", upstream)
    monkeypatch.setattr(core, "_is_gpt_oss_model", lambda family: True)
    config = {"stream": True, "stop_token_ids": [151645]}
    if options is not None:
        config["stream_options"] = dict(options)
    result = await model.async_chat([{"role": "user", "content": "Hi"}], config)
    first = await result.__anext__()
    assert first["choices"]
    if options is not None:
        assert first["usage"]["completion_tokens"] >= 1
    await result.aclose()
    assert upstream.closed
