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

import asyncio
from contextlib import aclosing
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ...reasoning_parser import ReasoningParser
from .. import core


def make_model(monkeypatch, parts, engine, max_tokens=100, family_stop=None):
    family = SimpleNamespace(
        model_specs=[SimpleNamespace(model_revision="test", quantization="4bit")],
        model_name="stop-test",
        model_family="stop-test",
        model_ability=["chat"],
        chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}",
        stop=family_stop,
        stop_token_ids=None,
    )
    cls = core.MLXVisionModel if engine == "vision" else core.MLXChatModel
    model = cls("stop-test", family, "/unused", {})
    model._model = object()
    model._context_length = 1024
    tokenizer = SimpleNamespace(
        encode=lambda prompt: [1, 2, 3],
        decode=lambda tokens, **kwargs: "".join(parts[token - 10] for token in tokens),
        eos_token_ids={999},
    )
    model._tokenizer = tokenizer
    cache = SimpleNamespace(tokens=[1, 2, 3])
    model._prompt_cache = core.PromptCache(cache=[cache], tokens=[1, 2, 3])
    model._prepare_inputs = lambda *args: ([3], 3, 2)
    model._structured_processor = lambda config: []
    consumed = []
    closed = Mock()

    def source(**kwargs):
        try:
            for index, text in enumerate(parts[: kwargs["max_tokens"]]):
                token = 10 + index
                consumed.append(token)
                cache.tokens.append(token)
                yield SimpleNamespace(
                    token=token,
                    text=text,
                    generation_tokens=index + 1,
                    finish_reason=None,
                )
        finally:
            closed()

    model._generate_stream_inner = source
    calls = []
    generator = SimpleNamespace(remove=Mock())
    state = {
        "generator": generator,
        "queues": {99: object()},
        "pending": {99: [object()]},
        "active": {99},
        "cache_boundaries": {99: 17},
        "cache_salts": {99: "other-request"},
    }
    if engine == "batch":

        class RecordingBatchModel(core.MLXBatchModel):
            async def generate_stream(self, *args, **kwargs):
                calls.append(kwargs.copy())
                async with aclosing(super().generate_stream(*args, **kwargs)) as chunks:
                    async for chunk in chunks:
                        yield chunk

        batch = object.__new__(RecordingBatchModel)
        batch._prompt_cache = None
        batch._xavier = None
        batch._get_or_create_generator = lambda *args: state
        batch._ensure_background_worker = lambda gen_dict: None
        uids = iter(range(7, 100))

        def insert(*args, **kwargs):
            uid = next(uids)
            state["pending"][uid] = [
                SimpleNamespace(uid=uid, token=10 + index, finish_reason=None)
                for index, _ in enumerate(parts[:max_tokens])
            ] + [
                SimpleNamespace(
                    uid=uid,
                    token=None,
                    finish_reason="length" if len(parts) >= max_tokens else "stop",
                )
            ]
            return uid, 2, 2

        batch._insert_request = insert
        monkeypatch.setattr(core.MLXBatchModel, "_tokenizer_ref", tokenizer)
        monkeypatch.setattr(core.MLXBatchModel, "_lock", None)
        model._batch_model = batch
    return model, consumed, closed, calls, generator, state


@pytest.mark.parametrize("engine", ["serial", "vision", "batch"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "parts,stop,expected,count,maximum,matched",
    [
        (
            ["Hello ", "world ", "END after", "unread"],
            "END",
            "Hello world ",
            3,
            100,
            True,
        ),
        (["Hello ", "END after", "unread"], ["END"], "Hello ", 2, 100, True),
        (["Hello E", "ND after", "unread"], "END", "Hello ", 2, 100, True),
        (["Hello E", "N", "D after", "unread"], ["END"], "Hello ", 3, 100, True),
        (["Hello END LATER after", "unread"], ["LATER", "END"], "Hello ", 1, 100, True),
        (["Hello ", "EN"], "END", "Hello EN", 2, 100, False),
        (["Hello ", "EN", "D unread"], ["END"], "Hello EN", 2, 2, False),
        (["END after", "unread"], "END", "", 1, 1, True),
        (["Hello 停", "止 after", "unread"], ["停止"], "Hello ", 2, 100, True),
        (["Hello E", "NX", "END unread"], "END", "Hello ENX", 3, 100, True),
        (["Hello END"], None, "Hello END", 1, 100, False),
        (["Hello END"], [], "Hello END", 1, 100, False),
        (["Hello", "unread"], "", "", 1, 100, True),
        (["Hello", "unread"], [""], "", 1, 100, True),
        (["Hello E", "ND after"], ["END!", "END"], "Hello ", 2, 100, True),
        (["Hello a", "bab after"], ["bab", "abab"], "Hello ", 2, 100, True),
        (["END"], "END", "", 1, 100, True),
    ],
    ids=[
        "string",
        "list",
        "split",
        "split-three",
        "earliest",
        "partial-eof",
        "partial-limit",
        "first-token",
        "unicode",
        "false-prefix",
        "no-stop",
        "empty-list",
        "empty-string",
        "empty-string-list",
        "overlapping-prefix",
        "overlapping-position",
        "whole-output",
    ],
)
@pytest.mark.asyncio
async def test_stop_strings(
    monkeypatch, engine, stream, parts, stop, expected, count, maximum, matched
):
    model, consumed, closed, calls, generator, state = make_model(
        monkeypatch, parts, engine, maximum
    )
    config = {
        "stream": stream,
        "stream_options": {"include_usage": True},
        "stop": stop,
        "max_tokens": maximum,
    }
    try:
        if engine == "vision":
            result = model.generate({"prompt": "prompt"}, config)
            chunks = list(result) if stream else [result]
        else:
            result = await model.async_generate("prompt", config)
            chunks = [chunk async for chunk in result] if stream else [result]
        choices = [chunk["choices"][0] for chunk in chunks if chunk["choices"]]
        assert "".join(choice["text"] for choice in choices) == expected
        if matched:
            assert choices[-1]["finish_reason"] == "stop"
        elif stream:
            assert choices[-1]["finish_reason"] == (
                "length" if count >= maximum else "stop"
            )
        assert chunks[-1]["usage"]["completion_tokens"] == count
        assert chunks[-1]["usage"]["prompt_tokens"] == 3
        assert chunks[-1]["usage"]["total_tokens"] == 3 + count
        assert chunks[-1]["usage"]["prompt_tokens_details"] == {"cached_tokens": 2}
        if stream:
            assert chunks[-1]["choices"] == []
            assert all(
                choice["text"] == ""
                for choice in choices
                if choice["finish_reason"] is not None
            )
        if engine == "batch":
            if stop:
                assert calls[0]["stop"] == stop
            if matched:
                generator.remove.assert_called_once_with([7])
            assert set(state["queues"]) == set(state["pending"]) == {99}
            assert state["active"] == {99}
            assert state["cache_boundaries"] == {99: 17}
            assert state["cache_salts"] == {99: "other-request"}
        else:
            assert consumed == list(range(10, 10 + count))
            closed.assert_called_once()
            assert model._prompt_cache.tokens == [1, 2, 3] + consumed
            assert model._prompt_cache.cache[0].tokens == model._prompt_cache.tokens
    finally:
        model.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["serial", "vision", "batch"])
@pytest.mark.parametrize("stream", [False, True])
async def test_family_stop_strings(monkeypatch, engine, stream):
    model, _, _, calls, generator, _ = make_model(
        monkeypatch, ["Hello E", "ND after", "unread"], engine, family_stop=["END"]
    )
    try:
        config = {"stream": stream, "max_tokens": 100}
        if engine == "vision":
            result = model.generate({"prompt": "prompt"}, config)
            chunks = list(result) if stream else [result]
        else:
            result = await model.async_generate("prompt", config)
            chunks = [chunk async for chunk in result] if stream else [result]
        assert "".join(chunk["choices"][0]["text"] for chunk in chunks) == "Hello "
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        if engine == "batch":
            assert calls[0]["stop"] == ["END"]
            generator.remove.assert_called_once_with([7])
    finally:
        model.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("matched", [False, True])
async def test_batch_stream_close_ends_request(monkeypatch, matched):
    parts = ["Hello END after", "unread"] if matched else ["Hello ", "END after"]
    model, _, _, _, generator, state = make_model(monkeypatch, parts, "batch")
    chunks = await model.async_generate("prompt", {"stream": True, "stop": "END"})
    try:
        assert (await anext(chunks))["choices"][0]["text"] == "Hello "
        if matched:
            # End native generation as soon as a match is seen, even if the
            # client pauses before reading the terminal chunk.
            generator.remove.assert_called_once_with([7])
        await chunks.aclose()
        generator.remove.assert_called_once_with([7])
        assert state["active"] == {99}
        assert set(state["queues"]) == {99}
    finally:
        await chunks.aclose()
        model.stop()


@pytest.mark.asyncio
async def test_batch_stop_matching_is_request_local(monkeypatch):
    parts = ["Hello E", "ND after", "unread"]
    model, _, _, _, generator, state = make_model(monkeypatch, parts, "batch")

    async def collect(stop):
        result = await model.async_generate(
            "prompt", {"stream": True, "stop": stop, "max_tokens": 100}
        )
        return [chunk async for chunk in result]

    try:
        stopped, uninterrupted = await asyncio.gather(collect("END"), collect("OTHER"))
        assert "".join(c["choices"][0]["text"] for c in stopped) == "Hello "
        assert "".join(c["choices"][0]["text"] for c in uninterrupted) == "".join(parts)
        assert stopped[-1]["usage"]["completion_tokens"] == 2
        assert uninterrupted[-1]["usage"]["completion_tokens"] == 3
        generator.remove.assert_called_once_with([7])
        assert state["active"] == {99}
        assert set(state["queues"]) == {99}
    finally:
        model.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["serial", "vision", "batch"])
@pytest.mark.parametrize(
    "parts,stop,expected",
    [
        (["<think>reason E"], "END", "reason E"),
        (["<think>reason</think>"], "</think>END", "reason"),
    ],
    ids=["reasoning-suffix", "reasoning-end-tag"],
)
async def test_chat_terminal_flush_parses_reasoning(
    monkeypatch, engine, parts, stop, expected
):
    model, _, _, _, _, _ = make_model(monkeypatch, parts, engine)
    model.reasoning_parser = ReasoningParser(
        reasoning_content=True,
        reasoning_start_tag="<think>",
        reasoning_end_tag="</think>",
        enable_thinking=True,
    )
    if engine == "vision":
        model._prepare_chat_inputs = lambda messages, config: (
            {"prompt": "prompt"},
            [],
            {"enable_thinking": True},
            config,
        )
    messages = [{"role": "user", "content": "prompt"}]
    try:
        config = {"stop": stop, "max_tokens": 100}
        if engine == "vision":
            completion = model.chat(messages, dict(config))
            chunks = list(model.chat(messages, {**config, "stream": True}))
        else:
            completion = await model.async_chat(messages, dict(config))
            source = await model.async_chat(messages, {**config, "stream": True})
            chunks = [chunk async for chunk in source]
        message = completion["choices"][0]["message"]
        choices = [chunk["choices"][0] for chunk in chunks if chunk["choices"]]
        assert (
            "".join(
                choice["delta"].get("reasoning_content") or "" for choice in choices
            )
            == (message["reasoning_content"])
            == expected
        )
        assert (
            "".join(choice["delta"].get("content") or "" for choice in choices)
            == (message.get("content") or "")
            == ""
        )
        assert choices[-1]["finish_reason"] == "stop"
        assert chunks[-1]["usage"] == completion["usage"]
        assert completion["usage"]["completion_tokens"] == len(parts)
    finally:
        model.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "stop,expected,count",
    [(None, "Hello . after", 4), (".", "Hello ", 3)],
    ids=["ordinary-output", "period-stop"],
)
async def test_batch_slow_tokenizer_cleanup(monkeypatch, stream, stop, expected, count):
    from transformers import PreTrainedTokenizer

    parts = ["Hello", " ", ".", " after"]
    vocabulary = {
        character: index for index, character in enumerate(dict.fromkeys("prompt"))
    }
    vocabulary.update({part: 10 + index for index, part in enumerate(parts)})
    tokens = {index: token for token, index in vocabulary.items()}

    class SlowTokenizer(PreTrainedTokenizer):
        @property
        def vocab_size(self):
            return len(vocabulary)

        def get_vocab(self):
            return vocabulary.copy()

        def _tokenize(self, text):
            return list(text)

        def _convert_token_to_id(self, token):
            return vocabulary[token]

        def _convert_id_to_token(self, index):
            return tokens[index]

        def convert_tokens_to_string(self, tokens):
            return "".join(tokens)

    tokenizer = SlowTokenizer(clean_up_tokenization_spaces=True)
    assert not tokenizer.is_fast
    assert tokenizer.clean_up_tokenization_spaces
    assert tokenizer.decode([10, 11, 12, 13]) == "Hello. after"
    assert (
        tokenizer.decode([10, 11, 12, 13], clean_up_tokenization_spaces=False)
        == "Hello . after"
    )
    model, _, _, _, generator, state = make_model(monkeypatch, parts, "batch")
    monkeypatch.setattr(core.MLXBatchModel, "_tokenizer_ref", tokenizer)
    try:
        result = await model.async_generate(
            "prompt", {"stream": stream, "stop": stop, "max_tokens": 100}
        )
        chunks = [chunk async for chunk in result] if stream else [result]
        assert "".join(chunk["choices"][0]["text"] for chunk in chunks) == expected
        assert chunks[-1]["usage"]["completion_tokens"] == count
        assert chunks[-1]["usage"]["prompt_tokens"] == 6
        assert chunks[-1]["usage"]["total_tokens"] == 6 + count
        assert chunks[-1]["usage"]["prompt_tokens_details"] == {"cached_tokens": 2}
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        if stop:
            generator.remove.assert_called_once_with([7])
        else:
            generator.remove.assert_not_called()
        assert state["active"] == {99}
        assert set(state["queues"]) == set(state["pending"]) == {99}
    finally:
        model.stop()


def byte_fallback_tokenizer(parts):
    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast

    vocabulary = {
        character: index for index, character in enumerate(dict.fromkeys("prompt"))
    }
    vocabulary.update({f"unused-{index}": index for index in range(5, 10)})
    vocabulary.update({part: 10 + index for index, part in enumerate(parts)})
    tokenizer = Tokenizer(models.BPE(vocab=vocabulary, merges=[], byte_fallback=True))
    tokenizer.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, clean_up_tokenization_spaces=False
    )


@pytest.mark.parametrize("skip_special_tokens", [False, True])
def test_batch_raw_tokenizer_without_decode_stream(monkeypatch, skip_special_tokens):
    from tokenizers import decoders

    parts = ["Hello ", "<0xE5>", "<0x81>", "<0x9C>", " after"]
    tokenizer = byte_fallback_tokenizer(parts).backend_tokenizer
    monkeypatch.delattr(decoders, "DecodeStream", raising=False)
    detokenizer = core._BatchDetokenizer(tokenizer, skip_special_tokens)
    text = "".join(detokenizer.append(token) for token in range(10, 15))
    assert text + detokenizer.finalize() == "Hello 停 after"
    assert detokenizer.finalize() == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("stop", ["停", ["停"], "OTHER", None])
async def test_batch_byte_fallback_stop(monkeypatch, stream, fast, stop):
    parts = ["Hello ", "<0xE5>", "<0x81>", "<0x9C>", " after"]
    model, _, _, _, generator, state = make_model(monkeypatch, parts, "batch")
    tokenizer = byte_fallback_tokenizer(parts)
    monkeypatch.setattr(
        core.MLXBatchModel,
        "_tokenizer_ref",
        (
            tokenizer
            if fast
            else SimpleNamespace(encode=tokenizer.encode, decode=tokenizer.decode)
        ),
    )
    assert tokenizer.decode([10, 11, 12, 13, 14]) == "Hello 停 after"
    assert tokenizer.decode([11]) == "�"
    matched = stop in ("停", ["停"])
    count = 4 if matched else 5
    try:
        result = await model.async_generate(
            "prompt", {"stream": stream, "stop": stop, "max_tokens": 100}
        )
        chunks = [chunk async for chunk in result] if stream else [result]
        assert "".join(chunk["choices"][0]["text"] for chunk in chunks) == (
            "Hello " if matched else "Hello 停 after"
        )
        assert chunks[-1]["usage"]["completion_tokens"] == count
        assert chunks[-1]["usage"]["prompt_tokens"] == 6
        assert chunks[-1]["usage"]["total_tokens"] == 6 + count
        assert chunks[-1]["usage"]["prompt_tokens_details"] == {"cached_tokens": 2}
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        if matched:
            generator.remove.assert_called_once_with([7])
        else:
            generator.remove.assert_not_called()
        assert state["active"] == {99}
        assert set(state["queues"]) == set(state["pending"]) == {99}
    finally:
        model.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize(
    "parts,expected",
    [
        (["Hello ", "�", " after"], "Hello � after"),
        (["Hello ", "�"], "Hello �"),
        (["Hello ", "<0xE5>", "<0x81>"], "Hello ��"),
    ],
    ids=["literal-replacement", "literal-replacement-at-eof", "incomplete-at-eof"],
)
async def test_batch_byte_fallback_finalization(
    monkeypatch, stream, fast, parts, expected
):
    model, _, _, _, generator, _ = make_model(monkeypatch, parts, "batch")
    tokenizer = byte_fallback_tokenizer(parts)
    monkeypatch.setattr(
        core.MLXBatchModel,
        "_tokenizer_ref",
        (
            tokenizer
            if fast
            else SimpleNamespace(encode=tokenizer.encode, decode=tokenizer.decode)
        ),
    )
    try:
        result = await model.async_generate(
            "prompt", {"stream": stream, "max_tokens": 100, "stop": "END"}
        )
        chunks = [chunk async for chunk in result] if stream else [result]
        assert "".join(chunk["choices"][0]["text"] for chunk in chunks) == expected
        assert chunks[-1]["usage"]["completion_tokens"] == len(parts)
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        if stream:
            assert chunks[-1]["choices"][0]["text"] == ""
        generator.remove.assert_not_called()
    finally:
        model.stop()


@pytest.mark.asyncio
async def test_batch_byte_fallback_detokenizer_is_request_local(monkeypatch):
    parts = ["Hello ", "<0xE5>", "<0x81>", "<0x9C>", " after"]
    model, _, _, _, generator, state = make_model(monkeypatch, parts, "batch")
    tokenizer = byte_fallback_tokenizer(parts)
    monkeypatch.setattr(core.MLXBatchModel, "_tokenizer_ref", tokenizer)
    stopped = await model.async_generate("prompt", {"stream": True, "stop": "停"})
    uninterrupted = await model.async_generate("prompt", {"stream": True})
    try:
        first, second = [], []
        # Pause after an incomplete byte on the first request, then interleave
        # both streams to expose any shared incremental decoding state.
        for _ in range(2):
            first.append(await anext(stopped))
        for _ in range(3):
            second.append(await anext(uninterrupted))
        first.extend([chunk async for chunk in stopped])
        second.extend([chunk async for chunk in uninterrupted])
        assert "".join(chunk["choices"][0]["text"] for chunk in first) == "Hello "
        assert (
            "".join(chunk["choices"][0]["text"] for chunk in second) == "Hello 停 after"
        )
        assert first[-1]["usage"]["completion_tokens"] == 4
        assert second[-1]["usage"]["completion_tokens"] == 5
        generator.remove.assert_called_once_with([7])
        assert state["active"] == {99}
        assert set(state["queues"]) == {99}
    finally:
        await stopped.aclose()
        await uninterrupted.aclose()
        model.stop()
