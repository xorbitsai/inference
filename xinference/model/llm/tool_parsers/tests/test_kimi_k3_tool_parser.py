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
        "<|open|>response<|sep|>hello" "<|close|>response<|sep|><|close|>message<|sep|>"
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
    assert (
        parser.extract_tool_calls_streaming(
            ["answer"], "answer<|open|>too", "<|open|>too"
        )
        is None
    )


def test_streaming_long_call_in_small_deltas():
    parser = KimiK3ToolParser()
    code = "\n".join(["print(1)"] * 500)
    deltas = ["Writing.", "<|open|>tools<|sep|>"]
    deltas += ['<|open|>call tool="write_file" index="0"<|sep|>']
    deltas += ['<|open|>argument key="content" type="string"<|sep|>']
    deltas += [code[i : i + 3] for i in range(0, len(code), 3)]
    deltas += ["<|close|>argument<|sep|>", "<|close|>call<|sep|>"]
    deltas += ["<|close|>tools<|sep|>"]

    previous_texts = [""]
    events = []
    for delta in deltas:
        current_text = previous_texts[-1] + delta
        result = parser.extract_tool_calls_streaming(
            previous_texts, current_text, delta
        )
        previous_texts[-1] = current_text
        if result is not None:
            events.extend(result if isinstance(result, list) else [result])

    assert events == [
        ("Writing.", None, None),
        (None, "write_file", {"content": code}, 0),
    ]
