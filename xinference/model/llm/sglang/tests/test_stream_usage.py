# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ..core import SGLANGChatModel


@pytest.fixture
def stream_model(monkeypatch):
    events = []

    async def iter_chunks():
        for index, event in enumerate(events):
            data = "data: " + json.dumps(event) + "\n\n"
            if index == len(events) - 1:
                data += "data: [DONE]\n\n"
            yield data.encode(), True

    response = MagicMock()
    response.__aenter__.return_value = response
    response.content.iter_chunks = iter_chunks
    session = MagicMock()
    session.__aenter__.return_value = session
    session.post.return_value = response
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    model = object.__new__(SGLANGChatModel)
    model.model_uid = "test"
    model.reasoning_parser = None
    model._active_request_ids = set()
    model._engine = SimpleNamespace(generate_url="http://unused/generate")
    return model, events


def event(text, tokens, reason=None):
    return {
        "text": text,
        "meta_info": {
            "prompt_tokens": 19,
            "completion_tokens": tokens,
            "finish_reason": {"type": reason} if reason else None,
            "cached_tokens": 7 if reason else 0,
        },
    }


async def chunks(model, chat=False, tools=None):
    config = {"stream": True, "stream_options": {"include_usage": True}}
    if chat:
        model._prepare_chat_inputs = AsyncMock(
            return_value=("prompt", None, None, {}, config)
        )
        result = await model.async_chat([{"role": "user", "content": "prompt"}], config)
    else:
        result = await model.async_generate("prompt", config, tools=tools)
    return [chunk async for chunk in result]


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [False, True], ids=["completion", "chat"])
@pytest.mark.parametrize(
    "texts,reason",
    [
        (["Hello", "Hello"], "stop"),
        ([""], "stop"),
        ([""], "length"),
        (["Hello", "Hello world"], "length"),
    ],
    ids=["unchanged-stop", "first-stop", "empty-length", "visible-terminal"],
)
async def test_terminal_usage(stream_model, chat, texts, reason):
    model, events = stream_model
    events.append(event("", 0))  # Nonterminal no-growth events stay skipped.
    events.extend(event(text, i + 1) for i, text in enumerate(texts[:-1]))
    events.append(event(texts[-1], len(texts), reason))
    output = await chunks(model, chat)
    expected_usage = {
        "prompt_tokens": 19,
        "completion_tokens": len(texts),
        "total_tokens": 19 + len(texts),
        "prompt_tokens_details": {"cached_tokens": 7},
    }
    assert output[-1]["choices"] == []
    assert output[-2]["choices"][0]["finish_reason"] == reason
    assert output[-1]["usage"] == expected_usage
    if not chat:
        assert output[-2]["usage"] == expected_usage
    content = [
        c["choices"][0]["delta"].get("content", "") if chat else c["choices"][0]["text"]
        for c in output[:-2]
    ]
    expected_content = [texts[0]] if texts[0] else []
    if len(texts) > 1 and texts[1] != texts[0]:
        expected_content.append(texts[1][len(texts[0]) :])
    assert content == expected_content
    assert not model._active_request_ids


@pytest.mark.asyncio
async def test_terminal_logprobs_are_sliced_once(stream_model):
    model, events = stream_model
    logprobs = [[-0.5, 1, "Hello"], [-1.2, 2, " world"]]
    for text, count, reason in [
        ("Hello", 1, None),
        ("Hello world", 2, None),
        ("Hello world", 2, "stop"),
    ]:
        item = event(text, count, reason)
        item["meta_info"].update(
            output_token_logprobs=logprobs[:count],
            output_top_logprobs=[[lp] for lp in logprobs[:count]],
        )
        events.append(item)
    raw = [item async for item in model._stream_generate("prompt")]
    assert [text for _, text in raw] == ["Hello", " world", ""]
    for (meta, _), expected in zip(raw, [logprobs[:1], logprobs[1:], []]):
        assert meta["output_token_logprobs"] == expected
        assert meta["output_top_logprobs"] == [[lp] for lp in expected]
    output = await chunks(model)
    assert [c["choices"][0]["logprobs"]["tokens"] for c in output[:-2]] == [
        ["Hello"],
        [" world"],
    ]
    assert output[-1]["usage"]["prompt_tokens_details"] == {"cached_tokens": 7}


@pytest.mark.asyncio
async def test_terminal_metadata_preserves_qwen_tool_buffer(stream_model):
    model, events = stream_model
    text = '<tool_call>\n{"name":"Read","arguments":{}}\n</tool_call>'
    events.extend([event("<tool", 1), event(text, 2), event(text, 3, "stop")])
    output = await chunks(model, tools=[{"type": "function"}])
    assert [c["choices"][0]["text"] for c in output[:-2]] == [text]
    assert output[-1]["usage"]["completion_tokens"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [False, True], ids=["completion", "chat"])
@pytest.mark.parametrize("reason", [None, "None", {"type": None}, {"type": "None"}])
async def test_nonterminal_finish_sentinels_preserve_logprobs(
    stream_model, chat, reason
):
    model, events = stream_model
    logprobs = [[-0.5, 1, "A"], [-1.2, 2, "B"]]
    first = event("", 1)
    first["meta_info"]["finish_reason"] = reason
    terminal = event("AB", 2, "length")
    for item, count in [(first, 1), (terminal, 2)]:
        item["meta_info"].update(
            output_token_logprobs=logprobs[:count],
            output_top_logprobs=[[lp] for lp in logprobs[:count]],
        )
        events.append(item)
    output = await chunks(model, chat)
    lp = output[0]["choices"][0]["logprobs"]
    tokens = [entry["token"] for entry in lp["content"]] if chat else lp["tokens"]
    assert tokens == ["A", "B"]
    assert output[-2]["choices"][0]["finish_reason"] == "length"
    assert output[-1]["usage"]["completion_tokens"] == 2
    assert [text async for _, text in model._stream_generate("prompt")] == ["AB"]
