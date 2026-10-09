import pytest

from ..gemma_tool_parser import GemmaToolParser


@pytest.fixture
def parser():
    return GemmaToolParser()


def test_extract_tool_calls(parser):
    output = (
        "<|tool_call>call:get_weather"
        '{location:<|"|>上海<|"|>,unit:<|"|>celsius<|"|>}'
        "<tool_call|>"
    )
    result = parser.extract_tool_calls(output)
    assert result == [(None, "get_weather", {"location": "上海", "unit": "celsius"})]


def test_extract_tool_calls_with_surrounding_text(parser):
    output = (
        "Thought...\n"
        "<|tool_call>call:get_weather"
        '{location:<|"|>上海<|"|>}'
        "<tool_call|>\nThanks"
    )
    result = parser.extract_tool_calls(output)
    assert result == [
        ("Thought...\n", None, None),
        (None, "get_weather", {"location": "上海"}),
        ("\nThanks", None, None),
    ]


def test_extract_tool_calls_streaming(parser):
    previous = [""]
    block = "<|tool_call>call:get_weather" '{location:<|"|>上海<|"|>}' "<tool_call|>"
    result = parser.extract_tool_calls_streaming(previous, block, block)
    assert result == (None, "get_weather", {"location": "上海"})


def test_streaming_ignores_processed_block(parser):
    block = "<|tool_call>call:get_weather" '{location:<|"|>上海<|"|>}' "<tool_call|>"
    previous = [block]
    current = block + " more text"
    result = parser.extract_tool_calls_streaming(previous, current, " more text")
    assert result == (" more text", None, None)


def test_string_values_may_contain_quotes(parser):
    # Gemma delimits strings with <|"|> instead of quoting them, so a value is
    # free to contain double quotes. Replacing the delimiters textually and
    # handing the result to json.loads broke on exactly this, which is what
    # every routing prompt that quotes the user back produces.
    output = (
        "<|tool_call>call:select_execution_pattern"
        '{action:<|"|>final_answer<|"|>,'
        'reason:<|"|>The user said "你好" (Hello), a simple greeting.<|"|>,'
        "requires_current_or_external_facts:false}"
        "<tool_call|>"
    )

    result = parser.extract_tool_calls(output)

    assert result == [
        (
            None,
            "select_execution_pattern",
            {
                "action": "final_answer",
                "reason": 'The user said "你好" (Hello), a simple greeting.',
                "requires_current_or_external_facts": False,
            },
        )
    ]


def test_string_values_may_look_like_arguments(parser):
    # A key is only a key in the structural text between strings; `b:` inside a
    # value must survive verbatim rather than being quoted as one.
    output = (
        "<|tool_call>call:note"
        '{text:<|"|>a,b:c and a backslash \\ plus a\nnewline<|"|>}'
        "<tool_call|>"
    )

    result = parser.extract_tool_calls(output)

    assert result == [
        (None, "note", {"text": "a,b:c and a backslash \\ plus a\nnewline"})
    ]


def test_empty_and_repeated_arguments(parser):
    # Empty strings are two delimiters back to back, and Gemma sometimes emits a
    # key twice; the later value wins, as in JSON.
    output = (
        "<|tool_call>call:probe"
        '{missing_verification:<|"|><|"|>,flag:true,flag:false}'
        "<tool_call|>"
    )

    result = parser.extract_tool_calls(output)

    assert result == [(None, "probe", {"missing_verification": "", "flag": False})]


def test_unterminated_string_is_reported_as_text(parser):
    output = '<|tool_call>call:note{text:<|"|>never closed}<tool_call|>'

    result = parser.extract_tool_calls(output)

    # unparsable blocks come back as content rather than a bogus tool call
    assert result == [(output, None, None)]


def _stream(parser, deltas):
    previous = [""]
    content = ""
    calls = []
    for delta in deltas:
        current = previous[-1] + delta
        result = parser.extract_tool_calls_streaming(previous, current, delta)
        previous[-1] = current
        if result is None:
            continue
        events = result if isinstance(result, list) else [result]
        for event in events:
            text, name, args = event[:3]
            if text:
                content += text
            if name:
                calls.append((name, args))
    return content, calls


def test_streaming_second_tool_call_is_not_content(parser):
    deltas = [
        "<|tool_call>",
        'call:get_weather{location:<|"|>上海<|"|>}',
        "<tool_call|>",
        "<|tool_call>",
        'call:get_time{zone:<|"|>UTC<|"|>}',
        "<tool_call|>",
    ]
    content, calls = _stream(parser, deltas)
    assert content == ""
    assert calls == [
        ("get_weather", {"location": "上海"}),
        ("get_time", {"zone": "UTC"}),
    ]


def test_streaming_keeps_text_around_tool_calls(parser):
    deltas = [
        "Checking. <|tool_call>",
        'call:get_weather{location:<|"|>上海<|"|>}',
        "<tool_call|> Done.",
    ]
    content, calls = _stream(parser, deltas)
    assert content == "Checking.  Done."
    assert calls == [("get_weather", {"location": "上海"})]
