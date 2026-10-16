from ..minicpm5_tool_parser import MiniCPM5ToolParser


def test_extracts_minicpm5_xml_tool_calls():
    parser = MiniCPM5ToolParser()

    result = parser.extract_tool_calls(
        'Before <function name="weather"><param name="city">"Beijing"</param>'
        '<param name="days">3</param><param name="note"><![CDATA[a < b & c]]>'
        "</param></function> after"
    )

    assert result == [
        ("Before ", None, None),
        (None, "weather", {"city": "Beijing", "days": 3, "note": "a < b & c"}),
        (" after", None, None),
    ]


def test_waits_for_a_complete_minicpm5_tool_call_when_streaming():
    parser = MiniCPM5ToolParser()
    partial = 'before <function name="weather"><param name="city">Beijing'
    complete = partial + "</param></function>"

    assert parser.extract_tool_calls_streaming([], partial, partial) == (
        "before ",
        None,
        None,
    )
    assert parser.extract_tool_calls_streaming(
        [partial], complete, "</param></function>"
    ) == (
        None,
        "weather",
        {"city": "Beijing"},
        0,
    )


def test_preserves_cdata_with_literal_closing_tags():
    parser = MiniCPM5ToolParser()
    output = (
        '<function name="write_file"><param name="content"><![CDATA['
        "example </param> and </function> end]]></param></function>"
    )

    assert parser.extract_tool_calls(output) == [
        (None, "write_file", {"content": "example </param> and </function> end"})
    ]


def test_streaming_waits_for_cdata_with_literal_closing_tags():
    parser = MiniCPM5ToolParser()
    complete = (
        '<function name="write_file"><param name="content"><![CDATA['
        "example </param> and </function> end]]></param></function>"
    )
    split_param = complete.index("</param>") + len("</par")
    split_function = complete.index("</function>") + len("</func")
    partial_param = complete[:split_param]
    partial_function = complete[:split_function]

    assert parser.extract_tool_calls_streaming([], partial_param, partial_param) is None
    assert (
        parser.extract_tool_calls_streaming(
            [partial_param], partial_function, partial_function[len(partial_param) :]
        )
        is None
    )
    assert parser.extract_tool_calls_streaming(
        [partial_function], complete, complete[len(partial_function) :]
    ) == (
        None,
        "write_file",
        {"content": "example </param> and </function> end"},
        0,
    )


def test_streaming_buffers_every_function_opening_tag_split():
    parser = MiniCPM5ToolParser()
    prefix = "Before "
    opening_tag = '<function name="weather">'
    output = (
        prefix + opening_tag + '<param name="city">"Beijing"</param></function> After'
    )

    for split in range(1, len(opening_tag) + 1):
        partial = prefix + opening_tag[:split]
        assert parser.extract_tool_calls_streaming([], partial, partial) == (
            prefix,
            None,
            None,
        )
        assert parser.extract_tool_calls_streaming(
            [partial], output, output[len(partial) :]
        ) == [
            (None, "weather", {"city": "Beijing"}, 0),
            (" After", None, None),
        ]


def test_character_streaming_matches_complete_tool_calls_with_text():
    parser = MiniCPM5ToolParser()
    output = (
        'Before <function name="first"><param name="value">1</param></function>'
        ' between <function name="second"><param name="value">2</param></function>'
        " After"
    )
    events = []
    previous_text = []

    for index, _ in enumerate(output, start=1):
        current_text = output[:index]
        result = parser.extract_tool_calls_streaming(
            previous_text, current_text, output[index - 1 : index]
        )
        previous_text.append(current_text)
        if result is None:
            continue
        events.extend(result if isinstance(result, list) else [result])

    normalized_events = []
    call_index = 0
    for event in events:
        content, name, arguments = event[:3]
        if name is not None:
            assert event[3] == call_index
            call_index += 1
            normalized_events.append((content, name, arguments))
        elif normalized_events and normalized_events[-1][1] is None:
            normalized_events[-1] = (
                normalized_events[-1][0] + content,
                None,
                None,
            )
        else:
            normalized_events.append((content, name, arguments))

    assert normalized_events == [
        ("Before ", None, None),
        (None, "first", {"value": 1}),
        (" between ", None, None),
        (None, "second", {"value": 2}),
        (" After", None, None),
    ]
    assert normalized_events == parser.extract_tool_calls(output)
