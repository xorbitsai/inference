from ..reasoning_parser import KimiK3ReasoningParser


def test_extract_reasoning_and_response():
    parser = KimiK3ReasoningParser(reasoning_content=True)
    output = (
        "<|open|>think<|sep|>reason"
        "<|close|>think<|sep|>"
        "<|open|>response<|sep|>answer"
        "<|close|>response<|sep|><|close|>message<|sep|>"
    )
    assert parser.extract_reasoning_content(output) == ("reason", "answer")


def test_extract_reasoning_when_open_marker_is_generation_prefix():
    parser = KimiK3ReasoningParser(reasoning_content=True)
    output = (
        "reason<|close|>think<|sep|>"
        "<|open|>response<|sep|>answer<|close|>response<|sep|>"
    )
    assert parser.extract_reasoning_content(output) == ("reason", "answer")


def test_default_mode_normalizes_xtml_without_returning_reasoning():
    parser = KimiK3ReasoningParser(reasoning_content=False)
    output = (
        "reason<|close|>think<|sep|>"
        "<|open|>response<|sep|>answer<|close|>response<|sep|>"
    )
    assert parser.check_content_parser()
    assert parser.extract_reasoning_content(output) == (None, "answer")


def test_streaming_does_not_leak_split_xtml_markers():
    parser = KimiK3ReasoningParser(reasoning_content=True)
    first = parser.extract_reasoning_content_streaming("", "reason<|cl", "reason<|cl")
    assert first == {"reasoning_content": "reason", "content": None}

    current = (
        "reason<|close|>think<|sep|>"
        "<|open|>response<|sep|>answer<|close|>res"
    )
    second = parser.extract_reasoning_content_streaming(
        "reason<|cl", current, current
    )
    assert second == {"content": "answer"}
