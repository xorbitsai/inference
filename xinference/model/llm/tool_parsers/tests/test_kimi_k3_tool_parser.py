from ..kimi_k3_tool_parser import KimiK3ToolParser


def test_extract_response_and_typed_tool_call():
    parser = KimiK3ToolParser()
    output = (
        "<|open|>response<|sep|>Let me check."
        "<|close|>response<|sep|>"
        "<|open|>tools<|sep|>"
        '<|open|>call tool="weather" index="0"<|sep|>'
        '<|open|>argument key="city" type="string"<|sep|>杭州'
        "<|close|>argument<|sep|>"
        '<|open|>argument key="days" type="number"<|sep|>3'
        "<|close|>argument<|sep|>"
        "<|close|>call<|sep|>"
        "<|close|>tools<|sep|>"
        "<|close|>message<|sep|>"
    )

    assert parser.extract_tool_calls(output) == [
        ("Let me check.", None, None),
        (None, "weather", {"city": "杭州", "days": 3}),
    ]


def test_extract_response_without_tools():
    parser = KimiK3ToolParser()
    output = (
        "<|open|>response<|sep|>hello"
        "<|close|>response<|sep|><|close|>message<|sep|>"
    )
    assert parser.extract_tool_calls(output) == [("hello", None, None)]


def test_streaming_buffers_incomplete_tool_call():
    parser = KimiK3ToolParser()
    prefix = "answer<|open|>tools<|sep|>"
    partial = prefix + '<|open|>call tool="weather" index="0"<|sep|>'
    complete = (
        partial
        + '<|open|>argument key="city" type="string"<|sep|>Paris'
        + "<|close|>argument<|sep|><|close|>call<|sep|>"
    )

    assert parser.extract_tool_calls_streaming([prefix], partial, partial) is None
    assert parser.extract_tool_calls_streaming([partial], complete, complete) == (
        None,
        "weather",
        {"city": "Paris"},
        0,
    )


def test_streaming_does_not_leak_partial_tools_marker():
    parser = KimiK3ToolParser()
    assert parser.extract_tool_calls_streaming(
        ["answer"], "answer<|open|>too", "<|open|>too"
    ) is None
