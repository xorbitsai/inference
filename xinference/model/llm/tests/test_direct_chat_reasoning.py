from copy import deepcopy
from unittest.mock import Mock

import pytest

from ..core import chat_context_var
from ..reasoning_parser import ReasoningParser
from ..tool_parsers.qwen_tool_parser import QwenToolParser
from .test_minicpmv47 import adapter as adapter

MESSAGES = [{"role": "user", "content": "What is 17 * 23?"}]
CALL = '<tool_call>{"name": "answer", "arguments": {"value": 391}}</tool_call>'


@pytest.fixture
def model(adapter):
    adapter.model_uid = "mini47"
    adapter.model_family, adapter.tool_parser = Mock(), None
    adapter._tokenizer.encode = lambda text, **kw: list(text)
    return adapter


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("launch", [False, True])
@pytest.mark.parametrize("thinking", [None, False, True])
@pytest.mark.parametrize(
    "extract,override",
    [(True, None), (False, None), (True, False), (False, True), (None, None)],
)
def test_request_settings(model, stream, launch, thinking, extract, override):
    parser = ReasoningParser(bool(extract), "<think>", "</think>", launch, False)
    model.reasoning_parser = parser if extract is not None else None
    defaults = parser.get_config()
    config = {"stream": stream}
    kwargs = {}
    if thinking is not None:
        kwargs["enable_thinking"] = thinking
    if override is not None:
        kwargs["reasoning_content"] = override
    if kwargs:
        config["chat_template_kwargs"] = kwargs
    original = deepcopy(config)
    thinking = launch if thinking is None and extract is not None else bool(thinking)
    raw = "work</think>391" if thinking else "391"
    model._model.generate = lambda **kw: kw["streamer"].on_finalized_text(
        raw, stream_end=True
    )
    caller = {"sentinel": True}
    token = chat_context_var.set(caller)
    try:
        result = model.build_direct_chat_result(MESSAGES, config)
        assert chat_context_var.get() is caller
        if stream:
            chunks = []
            for chunk in result:
                assert chat_context_var.get() is caller
                chunks.append(chunk)
            deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
            assert deltas[0]["role"] == "assistant"
            assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        else:
            deltas = [result["choices"][0]["message"]]
            assert result["usage"]["completion_tokens"] == len(raw)
        enabled = extract if override is None else override
        assert "".join(d.get("content") or "" for d in deltas) == (
            "391" if enabled else raw
        )
        assert "".join(d.get("reasoning_content") or "" for d in deltas).strip() == (
            "work" if enabled and thinking else ""
        )
        assert (
            model._processor.apply_chat_template.call_args.kwargs["enable_thinking"]
            is thinking
        )
        assert config == original and parser.get_config() == defaults
    finally:
        chat_context_var.reset(token)


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("launch", [False, True])
@pytest.mark.parametrize("combined", [False, True])
def test_direct_tool_stream_parses_once(model, monkeypatch, thinking, launch, combined):
    model.tool_parser = QwenToolParser()
    model.reasoning_parser = ReasoningParser(True, "<think>", "</think>", launch, False)
    parser = model.reasoning_parser
    parse = Mock(wraps=parser.extract_reasoning_content_streaming)
    monkeypatch.setattr(parser, "extract_reasoning_content_streaming", parse)
    parts = ["work</think>", CALL] if thinking else [CALL]
    if combined:
        parts = ["".join(parts)]
    monkeypatch.setattr(model, "build_streaming_iter", lambda *args: (iter(parts), 3))
    chunks = list(
        model.build_direct_chat_result(
            MESSAGES,
            {
                "stream": True,
                "tools": [{"type": "function", "function": {"name": "answer"}}],
                "chat_template_kwargs": {"enable_thinking": thinking},
                "stream_options": {"include_usage": True},
            },
        )
    )
    deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
    calls = [call for d in deltas for call in d.get("tool_calls", [])]
    assert calls[0]["function"]["name"] == "answer"
    assert calls[0]["id"] and calls[0]["index"] == 0
    assert "391" in "".join(c["function"].get("arguments", "") for c in calls)
    assert "".join(d.get("reasoning_content") or "" for d in deltas) == (
        "work" if thinking else ""
    )
    assert not "".join(d.get("content") or "" for d in deltas)
    assert parse.call_count == (len(parts) if thinking else 0)
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["choices"] == []


@pytest.mark.parametrize("insert", [False, True])
@pytest.mark.parametrize("texts", [[], [""]])
def test_direct_empty_stream_keeps_finish_and_usage(model, monkeypatch, insert, texts):
    model.reasoning_parser = ReasoningParser(True, "<think>", "</think>", True, insert)
    monkeypatch.setattr(model, "build_streaming_iter", lambda *args: (iter(texts), 3))
    config = {"stream": True, "stream_options": {"include_usage": True}}
    chunks = list(model.build_direct_chat_result(MESSAGES, config))
    assert len(chunks) == len(texts) + 2
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["completion_tokens"] == 0


@pytest.mark.parametrize("stream", [False, True])
def test_direct_generation_error_restores_context(model, stream):
    model.reasoning_parser = ReasoningParser(True, "<think>", "</think>")
    model._model.generate = Mock(side_effect=ValueError("generation failed"))
    caller = chat_context_var.get()
    with pytest.raises(ValueError, match="generation failed"):
        result = model.build_direct_chat_result(MESSAGES, {"stream": stream})
        if stream:
            list(result)
    assert chat_context_var.get() is caller
