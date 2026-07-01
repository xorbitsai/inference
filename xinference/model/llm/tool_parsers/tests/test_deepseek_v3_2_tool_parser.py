from ..deepseek_v3_2_tool_parser import DeepseekV3_2ToolParser


def test_extract_tool_calls_single_call():
    """Test extracting a single tool call."""
    parser = DeepseekV3_2ToolParser()

    test_case = (
        "<｜DSML｜function_calls>"
        '<｜DSML｜invoke name="get_current_weather">'
        '<｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter>'
        "</｜DSML｜invoke>"
        "</｜DSML｜function_calls>"
    )

    expected_results = [(None, "get_current_weather", {"location": "上海"})]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results, f"Expected {expected_results}, but got {result}"


def test_extract_tool_calls_multiple_calls():
    """Test extracting multiple tool calls."""
    parser = DeepseekV3_2ToolParser()

    test_case = (
        "<｜DSML｜function_calls>"
        '<｜DSML｜invoke name="get_current_weather">'
        '<｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter>'
        "</｜DSML｜invoke>"
        '<｜DSML｜invoke name="get_time">'
        '<｜DSML｜parameter name="timezone" string="true">UTC+8</｜DSML｜parameter>'
        "</｜DSML｜invoke>"
        "</｜DSML｜function_calls>"
    )

    expected_results = [
        (None, "get_current_weather", {"location": "上海"}),
        (None, "get_time", {"timezone": "UTC+8"}),
    ]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results, f"Expected {expected_results}, but got {result}"


def test_extract_tool_calls_multiple_params():
    """Test extracting a tool call with multiple parameters."""
    parser = DeepseekV3_2ToolParser()

    test_case = (
        "<｜DSML｜function_calls>"
        '<｜DSML｜invoke name="get_weather">'
        '<｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>'
        '<｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>'
        "</｜DSML｜invoke>"
        "</｜DSML｜function_calls>"
    )

    expected_results = [
        (None, "get_weather", {"location": "杭州", "date": "2024-01-16"}),
    ]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results, f"Expected {expected_results}, but got {result}"


def test_extract_tool_calls_no_tool_call():
    """Test when no tool call is present."""
    parser = DeepseekV3_2ToolParser()

    test_case = "This is a normal response without any tool calls."

    expected_results = [(test_case, None, None)]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results, f"Expected {expected_results}, but got {result}"


# Streaming test cases: simulate chunks arriving for a single tool call
STREAMING_TEST_CASES = [
    # (previous_texts, current_text, delta_text)
    ([""], "<｜DSML｜function_calls>", "<｜DSML｜function_calls>"),
    (
        ["<｜DSML｜function_calls>"],
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather">',
        '<｜DSML｜invoke name="get_current_weather">',
    ),
    (
        ['<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather">'],
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">',
        '<｜DSML｜parameter name="location" string="true">',
    ),
    (
        [
            '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">'
        ],
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海',
        "上海",
    ),
    (
        [
            '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海'
        ],
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter>',
        "</｜DSML｜parameter>",
    ),
    (
        [
            '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter>'
        ],
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter></｜DSML｜invoke>',
        "</｜DSML｜invoke>",
    ),
    (
        [
            '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter></｜DSML｜invoke>'
        ],
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜function_calls>',
        "</｜DSML｜function_calls>",
    ),
]

# Streaming test cases: simulate chunks arriving for multiple tool calls
STREAMING_TEST_MULTI_CASES = [
    # (previous_texts, current_text, delta_text)
    ([""], "<｜DSML｜function_calls>", "<｜DSML｜function_calls>"),
    (
        ["<｜DSML｜function_calls>"],
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather">',
        '<｜DSML｜invoke name="get_current_weather">',
    ),
    (
        ['<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather">'],
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">',
        '<｜DSML｜parameter name="location" string="true">',
    ),
    (
        [
            '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">'
        ],
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海',
        "上海",
    ),
    (
        [
            '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海'
        ],
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter>',
        "</｜DSML｜parameter>",
    ),
    (
        [
            '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter>'
        ],
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter></｜DSML｜invoke>',
        "</｜DSML｜invoke>",
    ),
    # First invoke complete, second invoke starts
    (
        [
            '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter></｜DSML｜invoke>'
        ],
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter></｜DSML｜invoke><｜DSML｜invoke name="get_time">',
        '<｜DSML｜invoke name="get_time">',
    ),
    (
        [
            '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter></｜DSML｜invoke><｜DSML｜invoke name="get_time">'
        ],
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter></｜DSML｜invoke><｜DSML｜invoke name="get_time"><｜DSML｜parameter name="timezone" string="true">',
        '<｜DSML｜parameter name="timezone" string="true">',
    ),
    (
        [
            '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter></｜DSML｜invoke><｜DSML｜invoke name="get_time"><｜DSML｜parameter name="timezone" string="true">'
        ],
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter></｜DSML｜invoke><｜DSML｜invoke name="get_time"><｜DSML｜parameter name="timezone" string="true">UTC+8',
        "UTC+8",
    ),
    (
        [
            '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter></｜DSML｜invoke><｜DSML｜invoke name="get_time"><｜DSML｜parameter name="timezone" string="true">UTC+8'
        ],
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter></｜DSML｜invoke><｜DSML｜invoke name="get_time"><｜DSML｜parameter name="timezone" string="true">UTC+8</｜DSML｜parameter>',
        "</｜DSML｜parameter>",
    ),
    (
        [
            '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter></｜DSML｜invoke><｜DSML｜invoke name="get_time"><｜DSML｜parameter name="timezone" string="true">UTC+8</｜DSML｜parameter>'
        ],
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter></｜DSML｜invoke><｜DSML｜invoke name="get_time"><｜DSML｜parameter name="timezone" string="true">UTC+8</｜DSML｜parameter></｜DSML｜invoke>',
        "</｜DSML｜invoke>",
    ),
    (
        [
            '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter></｜DSML｜invoke><｜DSML｜invoke name="get_time"><｜DSML｜parameter name="timezone" string="true">UTC+8</｜DSML｜parameter></｜DSML｜invoke>'
        ],
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_current_weather"><｜DSML｜parameter name="location" string="true">上海</｜DSML｜parameter></｜DSML｜invoke><｜DSML｜invoke name="get_time"><｜DSML｜parameter name="timezone" string="true">UTC+8</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜function_calls>',
        "</｜DSML｜function_calls>",
    ),
]


def test_extract_tool_calls_streaming_full_sequence():
    """Test streaming extraction through a complete single tool call sequence."""
    parser = DeepseekV3_2ToolParser()

    tool_call_detected = False
    detected_result = None

    for previous_texts, current_text, delta_text in STREAMING_TEST_CASES:
        result = parser.extract_tool_calls_streaming(
            previous_texts, current_text, delta_text
        )

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
    parser = DeepseekV3_2ToolParser()
    detected_results = []

    for previous_texts, current_text, delta_text in STREAMING_TEST_MULTI_CASES:
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
    """Test streaming extraction when the start token is split across chunks."""
    parser = DeepseekV3_2ToolParser()

    # Simulate start token being split: "<｜DSML｜function" + "_calls>"
    previous_texts = [""]
    delta1 = "<｜DSML｜function"
    current_text1 = delta1
    result1 = parser.extract_tool_calls_streaming(previous_texts, current_text1, delta1)
    # Should return None (waiting, potential start token forming)
    assert result1 is None, f"Expected None for partial token, got {result1}"

    # Complete the start token
    delta2 = "_calls>"
    current_text2 = current_text1 + delta2
    previous_texts.append(current_text1)
    result2 = parser.extract_tool_calls_streaming(previous_texts, current_text2, delta2)
    # Start token is now complete, but no invoke yet — should return None
    assert result2 is None, f"Expected None after start token completed, got {result2}"


def test_extract_tool_calls_plain_text_before_tool_call():
    """Test streaming with plain text before tool calls."""
    parser = DeepseekV3_2ToolParser()

    # Plain text before tool call
    previous_texts = [""]
    delta = "这是普通文本"
    current_text = delta
    result = parser.extract_tool_calls_streaming(previous_texts, current_text, delta)

    assert result == (
        "这是普通文本",
        None,
        None,
    ), f"Expected plain text passthrough, got {result}"


# ── Tests for plain (non-DSML) format ──────────────────────────────────────


def test_extract_tool_calls_plain_single_call():
    """Test extracting a single tool call in plain format."""
    parser = DeepseekV3_2ToolParser()

    test_case = (
        "<function_calls>"
        '<invoke name="get_current_weather">'
        '<parameter name="location" string="true">上海</parameter>'
        "</invoke>"
        "</function_calls>"
    )

    expected_results = [(None, "get_current_weather", {"location": "上海"})]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results, f"Expected {expected_results}, but got {result}"


def test_extract_tool_calls_plain_multiple_params():
    """Test extracting a tool call with multiple parameters in plain format."""
    parser = DeepseekV3_2ToolParser()

    test_case = (
        "<function_calls>"
        '<invoke name="get_weather">'
        '<parameter name="location" string="true">杭州</parameter>'
        '<parameter name="date" string="true">2024-01-16</parameter>'
        "</invoke>"
        "</function_calls>"
    )

    expected_results = [
        (None, "get_weather", {"location": "杭州", "date": "2024-01-16"}),
    ]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results, f"Expected {expected_results}, but got {result}"


def test_extract_tool_calls_plain_multiple_calls():
    """Test extracting multiple tool calls in plain format."""
    parser = DeepseekV3_2ToolParser()

    test_case = (
        "<function_calls>"
        '<invoke name="get_current_weather">'
        '<parameter name="location" string="true">上海</parameter>'
        "</invoke>"
        '<invoke name="get_time">'
        '<parameter name="timezone" string="true">UTC+8</parameter>'
        "</invoke>"
        "</function_calls>"
    )

    expected_results = [
        (None, "get_current_weather", {"location": "上海"}),
        (None, "get_time", {"timezone": "UTC+8"}),
    ]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results, f"Expected {expected_results}, but got {result}"


# Streaming test cases for plain format — single tool call
STREAMING_PLAIN_TEST_CASES = [
    ([""], "<function_calls>", "<function_calls>"),
    (
        ["<function_calls>"],
        '<function_calls><invoke name="get_current_weather">',
        '<invoke name="get_current_weather">',
    ),
    (
        ['<function_calls><invoke name="get_current_weather">'],
        '<function_calls><invoke name="get_current_weather"><parameter name="location" string="true">',
        '<parameter name="location" string="true">',
    ),
    (
        [
            '<function_calls><invoke name="get_current_weather"><parameter name="location" string="true">'
        ],
        '<function_calls><invoke name="get_current_weather"><parameter name="location" string="true">上海',
        "上海",
    ),
    (
        [
            '<function_calls><invoke name="get_current_weather"><parameter name="location" string="true">上海'
        ],
        '<function_calls><invoke name="get_current_weather"><parameter name="location" string="true">上海</parameter>',
        "</parameter>",
    ),
    (
        [
            '<function_calls><invoke name="get_current_weather"><parameter name="location" string="true">上海</parameter>'
        ],
        '<function_calls><invoke name="get_current_weather"><parameter name="location" string="true">上海</parameter></invoke>',
        "</invoke>",
    ),
    (
        [
            '<function_calls><invoke name="get_current_weather"><parameter name="location" string="true">上海</parameter></invoke>'
        ],
        '<function_calls><invoke name="get_current_weather"><parameter name="location" string="true">上海</parameter></invoke></function_calls>',
        "</function_calls>",
    ),
]


def test_extract_tool_calls_streaming_plain_full_sequence():
    """Test streaming extraction for plain format through a complete sequence."""
    parser = DeepseekV3_2ToolParser()

    tool_call_detected = False
    detected_result = None

    for previous_texts, current_text, delta_text in STREAMING_PLAIN_TEST_CASES:
        result = parser.extract_tool_calls_streaming(
            previous_texts, current_text, delta_text
        )

        if result is not None and result[1] is not None:
            tool_call_detected = True
            detected_result = result

    assert tool_call_detected, "Tool call should be detected during streaming"
    assert detected_result == (
        None,
        "get_current_weather",
        {"location": "上海"},
    ), f"Expected tool call result, but got {detected_result}"
