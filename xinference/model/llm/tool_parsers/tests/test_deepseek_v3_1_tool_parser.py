from ..deepseek_v3_1_tool_parser import DeepseekV3_1ToolParser


def test_extract_tool_calls_single_call():
    """Test extracting a single tool call."""
    parser = DeepseekV3_1ToolParser()

    test_case = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "上海"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'

    expected_results = [(None, "get_current_weather", {"location": "上海"})]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results, f"Expected {expected_results}, but got {result}"


def test_extract_tool_calls_multiple_calls():
    """Test extracting multiple tool calls."""
    parser = DeepseekV3_1ToolParser()

    test_case = (
        '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "上海"}<｜tool▁call▁end｜>'
        '<｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{"timezone": "UTC+8"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
    )

    expected_results = [
        (None, "get_current_weather", {"location": "上海"}),
        (None, "get_time", {"timezone": "UTC+8"}),
    ]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results, f"Expected {expected_results}, but got {result}"


def test_extract_tool_calls_no_tool_call():
    """Test when no tool call is present."""
    parser = DeepseekV3_1ToolParser()

    test_case = "This is a normal response without any tool calls."

    expected_results = [(test_case, None, None)]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results, f"Expected {expected_results}, but got {result}"


STREAMING_TEST_CASES = [
    # (previous_texts, current_text, delta_text)
    ([""], "<｜tool▁calls▁begin｜>", "<｜tool▁calls▁begin｜>"),
    (["<｜tool▁calls▁begin｜>"], "<｜tool▁calls▁begin｜>get", "get"),
    (["<｜tool▁calls▁begin｜>get"], "<｜tool▁calls▁begin｜>get_current", "_current"),
    (
        ["<｜tool▁calls▁begin｜>get_current"],
        "<｜tool▁calls▁begin｜>get_current_weather",
        "_weather",
    ),
    (
        ["<｜tool▁calls▁begin｜>get_current_weather"],
        "<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>",
        "<｜tool▁sep｜>",
    ),
    (
        ["<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>"],
        "<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{",
        "{",
    ),
    (
        ["<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"],
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location',
        '"location',
    ),
    (
        ['<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location'],
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location":',
        '":',
    ),
    (
        ['<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location":'],
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "',
        ' "',
    ),
    (
        ['<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "'],
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "上海',
        "上海",
    ),
    (
        ['<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location":"上海'],
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "上海"}',
        '"}',
    ),
    (
        ['<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location":"上海"}'],
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "上海"}<｜tool▁call▁end｜>',
        "<｜tool▁call▁end｜>",
    ),
    (
        [
            '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location":"上海"}<｜tool▁call▁end｜>'
        ],
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "上海"}<｜tool▁call▁end｜><｜tool▁call▁end｜>',
        "<｜tool▁call▁end｜>",
    ),
    (
        [
            '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location":"上海"}<｜tool▁call▁end｜><｜tool▁call▁end｜>'
        ],
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "上海"}<｜tool▁call▁end｜><｜tool▁call▁end｜>',
        "",
    ),
]

STREAMING_TEST_MUTI_CASES = [
    # (previous_texts, current_text, delta_text)
    ([""], "<｜tool▁calls▁begin｜>", "<｜tool▁calls▁begin｜>"),
    (["<｜tool▁calls▁begin｜>"], "<｜tool▁calls▁begin｜>get", "get"),
    (["<｜tool▁calls▁begin｜>get"], "<｜tool▁calls▁begin｜>get_current", "_current"),
    (
        ["<｜tool▁calls▁begin｜>get_current"],
        "<｜tool▁calls▁begin｜>get_current_weather",
        "_weather",
    ),
    (
        ["<｜tool▁calls▁begin｜>get_current_weather"],
        "<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>",
        "<｜tool▁sep｜>",
    ),
    (
        ["<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>"],
        "<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{",
        "{",
    ),
    (
        ["<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"],
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location',
        '"location',
    ),
    (
        ['<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location'],
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location":',
        '":',
    ),
    (
        ['<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location":'],
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "',
        ' "',
    ),
    (
        ['<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "'],
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "上海',
        "上海",
    ),
    (
        ['<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location":"上海'],
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "上海"}',
        '"}',
    ),
    (
        ['<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location":"上海"}'],
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "上海"}<｜tool▁call▁end｜>',
        "<｜tool▁call▁end｜>",
    ),
    (
        [
            '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location":"上海"}<｜tool▁call▁end｜>'
        ],
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "上海"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>',
        "<｜tool▁call▁begin｜>",
    ),
    (
        [
            '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location":"上海"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>'
        ],
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "上海"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_time',
        "get_time",
    ),
    (
        [
            '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location":"上海"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_time'
        ],
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "上海"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>',
        "<｜tool▁sep｜>",
    ),
    (
        [
            '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location":"上海"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_time'
        ],
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "上海"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{"timezone": "UTC+8"}',
        '{"timezone": "UTC+8"}',
    ),
    (
        [
            '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location":"上海"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{"timezone": "UTC+8"}'
        ],
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "上海"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{"timezone": "UTC+8"}<｜tool▁call▁end｜>',
        "<｜tool▁call▁end｜>",
    ),
    (
        [
            '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location":"上海"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{"timezone": "UTC+8"}<｜tool▁call▁end｜>'
        ],
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "上海"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{"timezone": "UTC+8"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "<｜tool▁calls▁end｜>",
    ),
    (
        [
            '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location":"上海"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{"timezone": "UTC+8"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        ],
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "上海"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{"timezone": "UTC+8"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "",
    ),
]


def test_extract_tool_calls_streaming_full_sequence():
    """Test streaming extraction through a complete tool call sequence."""
    parser = DeepseekV3_1ToolParser()

    tool_call_detected = False
    detected_result = None

    for previous_texts, current_text, delta_text in STREAMING_TEST_CASES:
        result = parser.extract_tool_calls_streaming(
            previous_texts, current_text, delta_text
        )

        # When tool call is complete, we should get the parsed result
        if result is not None and result[1] is not None:
            tool_call_detected = True
            detected_result = result

    assert tool_call_detected, "Tool call should be detected during streaming"
    assert detected_result == (
        None,
        "get_current_weather",
        {"location": "上海"},
    ), f"Expected tool call result, but got {detected_result}"


def test_extract_tool_calls_streaming_multi_sequence():
    """Test streaming extraction through a complete multiple tool calls sequence."""
    parser = DeepseekV3_1ToolParser()
    detected_results = []

    for previous_texts, current_text, delta_text in STREAMING_TEST_MUTI_CASES:
        result = parser.extract_tool_calls_streaming(
            previous_texts, current_text, delta_text
        )

        if result is not None and result[1] is not None:
            detected_results.append(result)

    expected_results = [
        (None, "get_current_weather", {"location": "上海"}),
        (None, "get_time", {"timezone": "UTC+8"}),
    ]

    assert (
        len(detected_results) == 2
    ), f"Expected 2 tool calls, but got {len(detected_results)}"
    assert (
        detected_results == expected_results
    ), f"Expected {expected_results}, but got {detected_results}"


def test_extract_tool_calls_streaming_split_token():
    """Test streaming extraction when tokens are split across chunks."""
    parser = DeepseekV3_1ToolParser()

    # Case 1: Split end token <｜tool▁call▁end｜>
    # Split as "<｜tool▁call" and "▁end｜>"
    previous_texts = [
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "上海"}'
    ]
    current_text_base = (
        '<｜tool▁calls▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "上海"}'
    )

    # Step 1: Part 1 of end token
    delta1 = "<｜tool▁call"
    current_text1 = current_text_base + delta1
    result1 = parser.extract_tool_calls_streaming(previous_texts, current_text1, delta1)
    # Should be None (waiting)
    assert result1 is None, f"Expected None for partial token, got {result1}"

    # Step 2: Part 2 of end token
    delta2 = "▁end｜>"
    current_text2 = current_text1 + delta2
    previous_texts.append(current_text1)
    result2 = parser.extract_tool_calls_streaming(previous_texts, current_text2, delta2)

    # Should detect tool call now
    expected = (None, "get_current_weather", {"location": "上海"})
    assert (
        result2 == expected
    ), f"Expected {expected} after split token join, got {result2}"


def test_extract_tool_calls_invalid_json():
    """Test behavior with invalid JSON in tool calls."""
    parser = DeepseekV3_1ToolParser()

    # Non-streaming
    invalid_json_input = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>func<｜tool▁sep｜>{invalid_json<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    result = parser.extract_tool_calls(invalid_json_input)
    # Should return raw content or handle gracefully (current implementation returns raw content in tuple)
    assert result[0][0] is not None, "Should return raw content for invalid JSON"
    assert result[0][1] is None

    # Streaming
    # If JSON is invalid, streaming should eventually return None or log error, but not crash
    # Note: Current implementation swallows invalid JSON in streaming loop and returns None
    previous = [""]
    current = "<｜tool▁calls▁begin｜>func<｜tool▁sep｜>{invalid_json<｜tool▁call▁end｜>"
    delta = "<｜tool▁call▁end｜>"

    result_stream = parser.extract_tool_calls_streaming(previous, current, delta)
    # Since JSON is invalid, it won't extract. It returns None (swallowing content).
    assert result_stream is None
