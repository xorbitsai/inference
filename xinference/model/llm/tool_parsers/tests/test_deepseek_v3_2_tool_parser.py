# CI: linux

import contextvars
import json
import re

import pytest

from ...llm_family import match_llm
from ...transformers.core import PytorchChatModel
from ..deepseek_v3_2_tool_parser import DeepseekV3_2ToolParser

_WEATHER_CALL = (
    '<function_calls><invoke name="weather">'
    '<parameter name="city" string="true">上海</parameter></invoke></function_calls>'
)


@pytest.mark.parametrize("dsml", [True, False])
@pytest.mark.parametrize(
    "before, after", [("Checking. ", ""), ("", " Done."), (" \n", "\t")]
)
def test_nonstreaming_preserves_surrounding_content(dsml, before, after):
    block = (
        _WEATHER_CALL.replace("<", "<｜DSML｜").replace("<｜DSML｜/", "</｜DSML｜")
        if dsml
        else _WEATHER_CALL
    )
    assert DeepseekV3_2ToolParser().extract_tool_calls(before + block + after) == [
        *([(before, None, None)] if before else []),
        (None, "weather", {"city": "上海"}),
        *([(after, None, None)] if after else []),
    ]


@pytest.mark.parametrize("between", [" Next. ", " \n"])
def test_nonstreaming_preserves_content_between_call_blocks(between):
    text = _WEATHER_CALL + between + _WEATHER_CALL.replace("上海", "北京")
    assert DeepseekV3_2ToolParser().extract_tool_calls(text) == [
        (None, "weather", {"city": "上海"}),
        (between, None, None),
        (None, "weather", {"city": "北京"}),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "plain text",
        _WEATHER_CALL[: -len("</function_calls>")],
        "<function_calls>broken</function_calls>",
    ],
)
def test_nonstreaming_keeps_unparseable_content(text):
    assert DeepseekV3_2ToolParser().extract_tool_calls(text) == [(text, None, None)]


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


def _typed_invoke(dsml=True):
    prefix = "<｜DSML｜" if dsml else "<"
    close_prefix = "｜DSML｜" if dsml else ""
    return (
        f"{prefix}function_calls>"
        f'{prefix}invoke name="typed_args">'
        f'{prefix}parameter name="text" string="true">001</{close_prefix}parameter>'
        f'{prefix}parameter name="integer" string="false">42</{close_prefix}parameter>'
        f'{prefix}parameter name="float" string="false">3.5</{close_prefix}parameter>'
        f'{prefix}parameter name="boolean" string="false">true</{close_prefix}parameter>'
        f'{prefix}parameter name="nothing" string="false">null</{close_prefix}parameter>'
        f'{prefix}parameter name="array" string="false">[1, "two"]</{close_prefix}parameter>'
        f'{prefix}parameter name="object" string="false">{{"nested": false}}'
        f"</{close_prefix}parameter>"
        f"</{close_prefix}invoke>"
        f"</{close_prefix}function_calls>"
    )


@pytest.mark.parametrize("dsml", [True, False])
def test_extract_tool_calls_preserves_json_parameter_types(dsml):
    parser = DeepseekV3_2ToolParser()

    assert parser.extract_tool_calls(_typed_invoke(dsml)) == [
        (
            None,
            "typed_args",
            {
                "text": "001",
                "integer": 42,
                "float": 3.5,
                "boolean": True,
                "nothing": None,
                "array": [1, "two"],
                "object": {"nested": False},
            },
        )
    ]


def test_extract_tool_calls_keeps_malformed_non_string_json_as_text():
    parser = DeepseekV3_2ToolParser()
    model_output = (
        "<｜DSML｜function_calls>"
        '<｜DSML｜invoke name="broken">'
        '<｜DSML｜parameter name="payload" string="false">{"missing": }'
        "</｜DSML｜parameter>"
        "</｜DSML｜invoke>"
        "</｜DSML｜function_calls>"
    )

    assert parser.extract_tool_calls(model_output) == [
        (None, "broken", {"payload": '{"missing": }'})
    ]


def test_extract_tool_calls_streaming_returns_typed_parameters():
    parser = DeepseekV3_2ToolParser()
    incomplete = (
        '<｜DSML｜function_calls><｜DSML｜invoke name="typed_args">'
        '<｜DSML｜parameter name="count" string="false">7'
    )
    complete = incomplete + (
        '</｜DSML｜parameter><｜DSML｜parameter name="label" string="true">007'
        "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜function_calls>"
    )

    assert parser.extract_tool_calls_streaming([], incomplete, incomplete) is None
    assert parser.extract_tool_calls_streaming(
        [incomplete], complete, complete[len(incomplete) :]
    ) == (None, "typed_args", {"count": 7, "label": "007"})


def _stream(parser, history, chunk):
    current = history[-1] + chunk
    result = parser.extract_tool_calls_streaming(history, current, chunk)
    history[-1] = current
    if result is None:
        return []
    return result if isinstance(result, list) else [result]


def _function_calls(*names):
    invokes = "".join(
        f'<｜DSML｜invoke name="{name}">'
        '<｜DSML｜parameter name="city" string="true">杭州</｜DSML｜parameter>'
        "</｜DSML｜invoke>"
        for name in names
    )
    return f"<｜DSML｜function_calls>{invokes}</｜DSML｜function_calls>"


def test_extract_tool_calls_streaming_parser_reused_by_next_request():
    parser = DeepseekV3_2ToolParser()

    def run(text):
        # The model layer starts every request with previous_texts == [""].
        history = [""]
        return [event for char in text for event in _stream(parser, history, char)]

    calls = [(None, "get_weather", {"city": "杭州"})]
    assert run(_function_calls("get_weather")) == calls
    assert "".join(event[0] for event in run("Sunny.")) == "Sunny."
    assert run(_function_calls("get_weather")) == calls


@pytest.mark.parametrize("chunk_size", [1, 7, 64, 1024])
def test_extract_tool_calls_streaming_keeps_interleaved_requests_apart(chunk_size):
    # One parser instance serves every request to the model, so concurrent
    # streams feed it their chunks in turn.
    parser = DeepseekV3_2ToolParser()
    texts = [_function_calls("a1", "a2"), _function_calls("b1", "b2"), "1 < 2."]
    histories = [[""] for _ in texts]
    events = [[] for _ in texts]
    for offset in range(0, max(map(len, texts)), chunk_size):
        for text, history, received in zip(texts, histories, events):
            chunk = text[offset : offset + chunk_size]
            received.extend(_stream(parser, history, chunk))

    assert events[0] == [(None, name, {"city": "杭州"}) for name in ("a1", "a2")]
    assert events[1] == [(None, name, {"city": "杭州"}) for name in ("b1", "b2")]
    assert "".join(event[0] for event in events[2]) == "1 < 2."


def _stream_calls(parser, deltas):
    previous_texts = [""]
    calls = []
    for delta in deltas:
        current_text = previous_texts[-1] + delta
        result = parser.extract_tool_calls_streaming(
            previous_texts, current_text, delta
        )
        previous_texts[-1] = current_text
        if result is None:
            continue
        for event in result if isinstance(result, list) else [result]:
            if event[1] is not None:
                calls.append(event[:3])
    return calls


def test_streaming_long_invoke_in_small_deltas():
    parser = DeepseekV3_2ToolParser()
    code = "print(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)"
    deltas = ["Writing.", "<｜DSML｜function_calls>"]
    for path, content in (("main.py", code), ("b.py", "x = 1")):
        deltas += ['<｜DSML｜invoke name="write_file">']
        deltas += ['<｜DSML｜parameter name="path" string="true">', path]
        deltas += ["</｜DSML｜parameter>"]
        deltas += ['<｜DSML｜parameter name="content" string="true">']
        deltas += [content[i : i + 3] for i in range(0, len(content), 3)]
        deltas += ["</｜DSML｜parameter>", "</｜DSML｜invoke>"]
    deltas += ["</｜DSML｜function_calls>"]

    assert _stream_calls(parser, deltas) == [
        (None, "write_file", {"path": "main.py", "content": code}),
        (None, "write_file", {"path": "b.py", "content": "x = 1"}),
    ]


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather of a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                "required": ["city"],
            },
        },
    }
]
_CALL_BLOCK = re.compile(
    r"<｜DSML｜function_calls>.*?</｜DSML｜function_calls>", re.DOTALL
)


def _catalog_model(name):
    # The catalog's own chat template and tool parser, rendered offline.
    family = match_llm(
        name, model_format="pytorch", quantization="none", download_hub="huggingface"
    )
    assert family is not None
    model = PytorchChatModel("tool-grammar", family, "unused")
    model._tokenizer = None
    model.prepare_parse_tool_calls()
    assert model.tool_parser is not None
    return model


def _prompt(model, messages, tools=_TOOLS, **template_kwargs):
    config = {"chat_template_kwargs": template_kwargs} if template_kwargs else {}
    # _get_full_prompt sets the request's chat context; keep it out of other tests.
    return contextvars.copy_context().run(
        model._get_full_prompt, messages, tools, config
    )


def test_v32_prompt_names_a_call_block_its_parser_reads():
    model = _catalog_model("DeepSeek-V3.2")
    prompt = _prompt(model, [{"role": "user", "content": "Weather in Paris?"}])
    assert any(token in prompt for token in model.tool_parser.tool_call_start_tokens)


def test_v31_prompt_example_still_parses_with_its_parser():
    # V3.1 keeps the token grammar its own parser reads.
    model = _catalog_model("Deepseek-V3.1")
    prompt = _prompt(model, [{"role": "user", "content": "Weather in Paris?"}])
    shown = re.search(
        r"<｜tool▁calls▁begin｜>.*?<｜tool▁calls▁end｜>", prompt, re.DOTALL
    )
    assert shown is not None
    example = shown.group(0).replace("tool_call_arguments", '{"city": "Paris"}')
    example = example.replace("{{additional_tool_calls}}", "")
    assert model.tool_parser.extract_tool_calls(example) == [
        (None, "tool_call_name", {"city": "Paris"})
    ]


@pytest.mark.parametrize("thinking", [False, True])
def test_v32_prompt_example_parses_with_its_parser(thinking):
    model = _catalog_model("DeepSeek-V3.2")
    messages = [{"role": "user", "content": "Weather in Paris?"}]
    prompt = _prompt(model, messages, thinking=thinking)
    assert "<｜tool▁call▁begin｜>" not in prompt
    shown = _CALL_BLOCK.search(prompt)
    assert shown is not None
    # Fill in the placeholders of the call the prompt shows, then parse it.
    example = re.sub(r'string="[^"]*"', 'string="true"', shown.group(0))
    name = re.search(r'invoke name="([^"]+)"', example).group(1)
    param = re.search(r'parameter name="([^"]+)"', example).group(1)
    value = re.search(r'string="true">(.*?)</', example, re.DOTALL).group(1)
    assert model.tool_parser.extract_tool_calls(example) == [
        (None, name, {param: value})
    ]


def _history(arguments, content=None):
    calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }
        for i, name in enumerate(("get_weather", "get_time"))
    ]
    return [
        {"role": "user", "content": "Weather and time in Paris?"},
        {"role": "assistant", "content": content, "tool_calls": calls},
        {"role": "tool", "tool_call_id": "call_0", "content": "sunny"},
        {"role": "tool", "tool_call_id": "call_1", "content": "noon"},
    ]


def _history_calls(prompt):
    history = prompt.split("<｜User｜>", 1)[1]
    shown = _CALL_BLOCK.search(history)
    assert shown is not None
    return history, DeepseekV3_2ToolParser().extract_tool_calls(shown.group(0))


@pytest.mark.parametrize("content", [None, "Checking."])
@pytest.mark.parametrize(
    "arguments",
    [
        {"city": "Paris"},
        {"city": "上海", "days": 3, "units": ["c", "k"], "opts": {"a": 1}, "on": True},
        {"nothing": None},
        {},
    ],
)
def test_v32_history_tool_calls_round_trip_through_its_parser(arguments, content):
    model = _catalog_model("DeepSeek-V3.2")
    prompt = _prompt(model, _history(json.dumps(arguments), content))
    history, calls = _history_calls(prompt)
    assert calls == [
        (None, "get_weather", arguments),
        (None, "get_time", arguments),
    ]
    if content:
        assert history.index(content) < history.index("<｜DSML｜function_calls>")
    # Tool results keep their wrapper.
    assert "<｜tool▁output▁begin｜>sunny<｜tool▁output▁end｜>" in history


@pytest.mark.parametrize(
    "arguments, expected",
    [
        ({"city": "Paris"}, {"city": "Paris"}),
        ("", {}),
        (None, {}),
        ("not json", {"arguments": "not json"}),
    ],
)
def test_v32_history_tool_calls_with_unusual_arguments(arguments, expected):
    model = _catalog_model("DeepSeek-V3.2")
    _, calls = _history_calls(_prompt(model, _history(arguments)))
    assert calls == [(None, "get_weather", expected), (None, "get_time", expected)]


def test_v32_history_tool_calls_without_request_tools():
    # A follow-up request can carry the earlier calls without sending tools again.
    model = _catalog_model("DeepSeek-V3.2")
    prompt = _prompt(model, _history('{"city": "Paris"}'), tools=None)
    assert "## Tools" not in prompt
    _, calls = _history_calls(prompt)
    assert calls == [
        (None, "get_weather", {"city": "Paris"}),
        (None, "get_time", {"city": "Paris"}),
    ]
