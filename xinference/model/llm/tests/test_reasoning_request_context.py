from ..core import chat_context_var
from ..reasoning_parser import ReasoningParser
from ..utils import ChatModelMixin


def test_explicit_request_extracts_reasoning_without_changing_model_defaults():
    parser = ReasoningParser(
        False, "<think>", "</think>", enable_thinking=False, auto_insert_start_tag=False
    )
    token = chat_context_var.set({"enable_thinking": True, "reasoning_content": True})
    try:
        assert parser.check_content_parser()
        assert parser.should_insert_start_tag()
    finally:
        chat_context_var.reset(token)
    assert not parser.check_content_parser()
    assert not parser.auto_insert_start_tag


def test_synchronous_stream_applies_context_without_leaking_it():
    parser = ReasoningParser(
        False, "<think>", "</think>", enable_thinking=False, auto_insert_start_tag=False
    )
    chunks = iter(
        [
            {
                "id": "test",
                "created": 1,
                "model": "qwen",
                "choices": [{"index": 0, "text": text, "finish_reason": None}],
            }
            for text in ("reasoning", "</think>answer")
        ]
    )
    stream = ChatModelMixin._to_chat_completion_chunks(
        chunks, parser, {"enable_thinking": True, "reasoning_content": True}
    )
    results = []
    for chunk in stream:
        assert not parser.check_content_parser()
        results.append(chunk)
    deltas = [chunk["choices"][0]["delta"] for chunk in results]
    assert (
        "".join(d.get("reasoning_content") or "" for d in deltas).strip() == "reasoning"
    )
    assert "".join(d.get("content") or "" for d in deltas) == "answer"
    assert not parser.auto_insert_start_tag


def test_raw_template_supports_model_json_filter_options():
    rendered = ChatModelMixin()._build_from_raw_template(
        [], "{{ tools | tojson(ensure_ascii=False) }}", tools=[{"name": "中文工具"}]
    )
    assert rendered == '[{"name": "中文工具"}]'
