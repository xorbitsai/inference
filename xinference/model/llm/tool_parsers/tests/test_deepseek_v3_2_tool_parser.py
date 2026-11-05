# CI: linux

import contextvars
import copy
import json
import re

import pytest

from ...llm_family import match_llm
from ...transformers.core import PytorchChatModel
from ...vllm.core import VLLMChatModel
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


def test_v32_prompt_example_parses_with_its_parser():
    model = _catalog_model("DeepSeek-V3.2")
    prompt = _prompt(model, [{"role": "user", "content": "Weather in Paris?"}])
    assert "<｜tool▁call▁begin｜>" not in prompt
    shown = re.search(
        r'<｜DSML｜function_calls>\n<｜DSML｜invoke name="\$FUNCTION_NAME">.*?'
        r"</｜DSML｜function_calls>",
        prompt,
        re.DOTALL,
    )
    assert shown is not None
    # Fill in the placeholders of the call the instruction shows, then parse it.
    example = shown.group(0).replace("...\n", "").replace("true|false", "true")
    for placeholder, text in (
        ("$FUNCTION_NAME2", "get_time"),
        ("$FUNCTION_NAME", "get_weather"),
        ("$PARAMETER_NAME", "city"),
        ("$PARAMETER_VALUE", "Paris"),
    ):
        example = example.replace(placeholder, text)
    assert model.tool_parser.extract_tool_calls(example) == [
        (None, "get_weather", {"city": "Paris"}),
        (None, "get_time", {}),
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
    # Tool results are replayed as the V3.2 result block.
    assert (
        "<function_results>\n<result>sunny</result>\n<result>noon</result>\n"
        "</function_results>"
    ) in history


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


# Replays through the vLLM prompt path, as VLLMChatModel renders a request. The
# expected prompts are what DeepSeek's official V3.2 encoder (encoding/encoding_dsv32.py
# in deepseek-ai/DeepSeek-V3.2: encode_messages(messages, thinking_mode)) writes for the
# same messages, without its BOS token; request tools go on an empty system message. The
# complete prompt, including whitespace and the continuation after tool results,
# is compared byte for byte.
_EOS = "<｜end▁of▁sentence｜>"
_ASK = "<｜User｜>Weather in Paris?<｜Assistant｜>"
# The encoder's tool instruction (TOOLS_SYSTEM_TEMPLATE), up to the schema lines.
_INSTRUCTION = (
    "\n\n## Tools\n\n"
    "You have access to a set of tools you can use to answer the user's question.\n"
    'You can invoke functions by writing a "<｜DSML｜function_calls>" block like the '
    "following as part of your reply to the user:\n"
    "<｜DSML｜function_calls>\n"
    '<｜DSML｜invoke name="$FUNCTION_NAME">\n'
    '<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE'
    "</｜DSML｜parameter>\n"
    "...\n"
    "</｜DSML｜invoke>\n"
    '<｜DSML｜invoke name="$FUNCTION_NAME2">\n'
    "...\n"
    "</｜DSML｜invoke>\n"
    "</｜DSML｜function_calls>\n\n"
    "String and scalar parameters should be specified as is without any escaping or "
    'quotes, while lists and objects should use JSON format. The "string" attribute '
    'should be set to "true" for string type parameters and "false" for other types '
    "(numbers, booleans, arrays, objects).\n\n"
    "If the thinking_mode is enabled, then after function results you should strongly "
    "consider outputting a thinking block. Here is an example:\n\n"
    "<｜DSML｜function_calls>\n...\n</｜DSML｜function_calls>\n\n"
    "<function_results>\n...\n</function_results>\n\n"
    "<think>...thinking about results</think>\n\n"
    "Here are the functions available in JSONSchema format:\n<functions>\n"
)
_WEATHER_SCHEMA = (
    '{"name": "get_weather", "description": "Get the weather of a city.", '
    '"parameters": {"type": "object", "properties": {"city": {"type": "string"}, '
    '"days": {"type": "integer"}}, "required": ["city"]}}'
)
_TIME_SCHEMA = (
    '{"name": "get_time", "parameters": {"type": "object", "properties": '
    '{"tz": {"type": "string", "description": "城市  时区"}}}}'
)
_TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "parameters": {
            "type": "object",
            "properties": {"tz": {"type": "string", "description": "城市  时区"}},
        },
    },
}
_WEATHER_BLOCK = (
    '\n\n<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n'
    '<｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>\n'
    f"</｜DSML｜invoke>\n</｜DSML｜function_calls>{_EOS}"
)
_TIME_BLOCK = (
    '\n\n<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_time">\n'
    '<｜DSML｜parameter name="tz" string="true">CET</｜DSML｜parameter>\n'
    f"</｜DSML｜invoke>\n</｜DSML｜function_calls>{_EOS}"
)
_BOTH_BLOCK = (
    '\n\n<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n'
    '<｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>\n'
    '</｜DSML｜invoke>\n<｜DSML｜invoke name="get_time">\n'
    '<｜DSML｜parameter name="tz" string="true">CET</｜DSML｜parameter>\n'
    f"</｜DSML｜invoke>\n</｜DSML｜function_calls>{_EOS}"
)
_NO_ARGUMENTS_BLOCK = (
    '\n\n<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_time">\n\n'
    f"</｜DSML｜invoke>\n</｜DSML｜function_calls>{_EOS}"
)
_SUNNY = "\n\n<function_results>\n<result>sunny</result>\n</function_results>\n\n"
_NOON = "\n\n<function_results>\n<result>noon</result>\n</function_results>\n\n"
_SUNNY_AND_NOON = (
    "\n\n<function_results>\n<result>sunny</result>\n<result>noon</result>\n"
    "</function_results>\n\n"
)
_SUNNY_UNFINISHED = "\n\n<function_results>\n<result>sunny</result>"


def _call(name, **arguments):
    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _assistant(*calls, content=None, reasoning=None):
    message = {"role": "assistant", "content": content, "tool_calls": list(calls)}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return message


def _result(text):
    return {"role": "tool", "tool_call_id": "call", "content": text}


_ASKING = {"role": "user", "content": "Weather in Paris?"}
_GET_WEATHER = _call("get_weather", city="Paris")
_GET_TIME = _call("get_time", tz="CET")
_ANSWER = {"role": "assistant", "content": "It is sunny.", "reasoning_content": "Then."}
_REPLAYS = {
    # The review's case: a user request, a get_weather call and its sunny result.
    "round_trip": (
        [_ASKING, _assistant(_GET_WEATHER), _result("sunny")],
        {
            "chat": f"{_ASK}</think>{_WEATHER_BLOCK}{_SUNNY}</think>",
            "thinking": f"{_ASK}<think></think>{_WEATHER_BLOCK}{_SUNNY}<think>",
        },
    ),
    "payload_spaces": (
        [
            {"role": "system", "content": "Be  precise."},
            {"role": "user", "content": "Weather  here?"},
            _assistant(
                _call("get_weather", city="New  York"),
                content="Checking  now.",
                reasoning="First  thought.",
            ),
            _result("mostly  sunny"),
        ],
        {
            mode: "Be  precise.<｜User｜>Weather  here?<｜Assistant｜>"
            + ("<think>First  thought.</think>" if mode == "thinking" else "</think>")
            + 'Checking  now.\n\n<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">New  York</｜DSML｜parameter>\n'
            f"</｜DSML｜invoke>\n</｜DSML｜function_calls>{_EOS}\n\n"
            "<function_results>\n<result>mostly  sunny</result>\n</function_results>\n\n"
            + ("<think>" if mode == "thinking" else "</think>")
            for mode in ("chat", "thinking")
        },
    ),
    "two_results_in_one_block": (
        [
            _ASKING,
            _assistant(_GET_WEATHER, _GET_TIME, reasoning="Two."),
            _result("sunny"),
            _result("noon"),
        ],
        {
            "chat": f"{_ASK}</think>{_BOTH_BLOCK}{_SUNNY_AND_NOON}</think>",
            "thinking": f"{_ASK}<think>Two.</think>{_BOTH_BLOCK}{_SUNNY_AND_NOON}<think>",
        },
    ),
    # The encoder closes the block after as many results as the turn has calls.
    "result_still_missing": (
        [
            _ASKING,
            _assistant(_GET_WEATHER, _GET_TIME, reasoning="Two."),
            _result("sunny"),
        ],
        {
            "chat": f"{_ASK}</think>{_BOTH_BLOCK}{_SUNNY_UNFINISHED}",
            "thinking": f"{_ASK}<think>Two.</think>{_BOTH_BLOCK}{_SUNNY_UNFINISHED}",
        },
    ),
    "second_round_in_the_same_turn": (
        [
            _ASKING,
            _assistant(_GET_WEATHER, content="Checking.", reasoning="First."),
            _result("sunny"),
            _assistant(_GET_TIME, reasoning="Second."),
            _result("noon"),
        ],
        {
            "chat": f"{_ASK}</think>Checking.{_WEATHER_BLOCK}{_SUNNY}</think>"
            f"{_TIME_BLOCK}{_NOON}</think>",
            "thinking": f"{_ASK}<think>First.</think>Checking.{_WEATHER_BLOCK}{_SUNNY}"
            f"<think>Second.</think>{_TIME_BLOCK}{_NOON}<think>",
        },
    ),
    "call_without_arguments": (
        [_ASKING, _assistant(_call("get_time"), reasoning="Clock."), _result("noon")],
        {
            "chat": f"{_ASK}</think>{_NO_ARGUMENTS_BLOCK}{_NOON}</think>",
            "thinking": f"{_ASK}<think>Clock.</think>{_NO_ARGUMENTS_BLOCK}{_NOON}<think>",
        },
    ),
    # An answer right after the results of the current turn closes its reasoning, as
    # the next round of calls does.
    "answer_after_results": (
        [
            _ASKING,
            _assistant(_GET_WEATHER, reasoning="First."),
            _result("sunny"),
            _ANSWER,
        ],
        {
            "chat": f"{_ASK}</think>{_WEATHER_BLOCK}{_SUNNY}</think>It is sunny.{_EOS}",
            "thinking": f"{_ASK}<think>First.</think>{_WEATHER_BLOCK}{_SUNNY}"
            f"<think>Then.</think>It is sunny.{_EOS}",
        },
    ),
    # A finished round followed by a new question: the old reasoning is dropped in both
    # modes, and the old tool result closes with </think> even when thinking is on.
    "finished_round": (
        [
            _ASKING,
            _assistant(_GET_WEATHER, reasoning="First."),
            _result("sunny"),
            _ANSWER,
            {"role": "user", "content": "And Rome?"},
        ],
        {
            "chat": f"{_ASK}</think>{_WEATHER_BLOCK}{_SUNNY}</think>It is sunny.{_EOS}"
            "<｜User｜>And Rome?<｜Assistant｜></think>",
            "thinking": f"{_ASK}</think>{_WEATHER_BLOCK}{_SUNNY}</think>It is sunny.{_EOS}"
            "<｜User｜>And Rome?<｜Assistant｜><think>",
        },
    ),
}


def _stub_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")


async def _replay(messages, thinking, tools=None, tokenizer=None, generation=True):
    family = match_llm(
        "DeepSeek-V3.2",
        model_format="pytorch",
        quantization="none",
        download_hub="huggingface",
    )
    assert family is not None
    model = VLLMChatModel("replay", family, "unused", {})

    async def get_tokenizer(_lora_request):
        return tokenizer

    model._get_tokenizer = get_tokenizer
    kwargs = {"thinking": thinking, "add_generation_prompt": generation}
    config = {"chat_template_kwargs": kwargs}
    if tools:
        config["tools"] = tools
    rendered = []
    if tokenizer is not None:
        apply_chat_template = tokenizer.apply_chat_template

        def render(*args, **kwargs):
            rendered.append(apply_chat_template(*args, **kwargs))
            return rendered[-1]

        tokenizer.apply_chat_template = render
    try:
        prompt, *_ = await model._prepare_chat_inputs(copy.deepcopy(messages), config)
    finally:
        if tokenizer is not None:
            tokenizer.apply_chat_template = apply_chat_template
    if tokenizer is not None:
        assert rendered == [prompt]
    return prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chat", "thinking"])
@pytest.mark.parametrize("name", list(_REPLAYS))
async def test_v32_replay_matches_the_official_encoder(name, mode):
    messages, expected = _REPLAYS[name]
    # The encoder writes no generation prompt after an assistant message.
    last = messages[-1]["role"]
    prompt = await _replay(messages, mode == "thinking", generation=last != "assistant")
    assert prompt == expected[mode]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chat", "thinking"])
async def test_v32_replay_with_request_tools_matches_the_official_encoder(mode):
    # Keep schema whitespace and Unicode, including a tool with no description.
    messages, expected = _REPLAYS["round_trip"]
    prompt = await _replay(messages, mode == "thinking", tools=_TOOLS + [_TIME_TOOL])
    schemas = f"{_WEATHER_SCHEMA}\n{_TIME_SCHEMA}\n</functions>\n"
    assert prompt == _INSTRUCTION + schemas + expected[mode]


@pytest.mark.asyncio
async def test_v32_replay_leaves_a_prefix_after_results_inside_the_thinking():
    # The encoder has no prefix messages. One that follows the results continues the
    # reasoning they opened, as before; it does not close it like an answer does.
    prefix = {"role": "assistant", "content": "It is", "prefix": True}
    prompt = await _replay([*_REPLAYS["round_trip"][0], prefix], True)
    assert prompt.endswith(f"{_SUNNY}<think>It is{_EOS}")


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", [False, True])
async def test_v32_replay_renders_alike_with_a_transformers_tokenizer(thinking):
    messages, _ = _REPLAYS["second_round_in_the_same_turn"]
    tools = _TOOLS + [_TIME_TOOL]
    tokenizer = _stub_tokenizer()
    rendered = []
    apply_chat_template = tokenizer.apply_chat_template

    def spy(*args, **kwargs):
        rendered.append(apply_chat_template(*args, **kwargs))
        return rendered[-1]

    tokenizer.apply_chat_template = spy
    prompt = await _replay(messages, thinking, tools=tools, tokenizer=tokenizer)
    # Rendered by the tokenizer, not by the raw-template fallback.
    assert rendered == [prompt]
    assert prompt == await _replay(messages, thinking, tools=tools)


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("use_tokenizer", [False, True])
@pytest.mark.parametrize("rounds", [1, 2])
async def test_v32_parallel_results_follow_call_ids_in_each_round(
    thinking, use_tokenizer, rounds
):
    messages = [_ASKING]
    expected = _ASK + ("<think>" if thinking else "</think>")
    for index in range(rounds):
        calls = copy.deepcopy([_GET_WEATHER, _GET_TIME])
        for call, name in zip(calls, ("a", "b")):
            call["id"] = f"call_{name}_{index}" if index else f"call_{name}"
        weather, time = ("SUNNY", "NOON") if not index else ("RAINY", "MIDNIGHT")
        messages.extend(
            [
                _assistant(*calls),
                {"role": "tool", "tool_call_id": calls[1]["id"], "content": time},
                {"role": "tool", "tool_call_id": calls[0]["id"], "content": weather},
            ]
        )
        expected += ("</think>" if thinking else "") + _BOTH_BLOCK
        expected += (
            f"\n\n<function_results>\n<result>{weather}</result>\n<result>{time}</result>"
            "\n</function_results>\n\n" + ("<think>" if thinking else "</think>")
        )
    tokenizer = _stub_tokenizer() if use_tokenizer else None
    assert await _replay(messages, thinking, tokenizer=tokenizer) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("use_tokenizer", [False, True])
@pytest.mark.parametrize("reasoning_content", [False, True])
@pytest.mark.parametrize("current_turn", [False, True])
@pytest.mark.parametrize("reply", ["tool_call", "after_results", "plain"])
@pytest.mark.parametrize(
    "reasoning, answer",
    [
        ("Need a live report.", "Let me check."),
        ("Need a live report.", "The literal </think> is quoted."),
        ("", "The literal </think> is quoted."),
    ],
)
async def test_v32_inline_reasoning_round_trips_the_vllm_assistant_message(
    thinking, use_tokenizer, reasoning_content, current_turn, reply, reasoning, answer
):
    family = _catalog_model("DeepSeek-V3.2").model_family
    model = VLLMChatModel("round-trip", family, "unused", {})
    model.prepare_parse_tool_calls()
    model.prepare_parse_reasoning_content(reasoning_content)
    text = f"{reasoning}</think>{answer}"
    if reply == "tool_call":
        text += _function_calls("get_weather").replace("杭州", "Paris")
    # Use vLLM's returned message verbatim, including its generated call ID.
    completion = contextvars.Context().run(
        model._post_process_completion,
        family,
        model.model_uid,
        {"choices": [{"text": text}]},
        _TOOLS,
    )
    assistant = completion["choices"][0]["message"]
    assert assistant["content"] == (
        answer if reasoning_content else f"{reasoning}</think>{answer}"
    )
    assert assistant.get("reasoning_content") == (
        reasoning if reasoning_content else None
    )
    in_turn = thinking and current_turn
    messages = [_ASKING]
    expected = _ASK
    if reply == "after_results":
        messages.extend(
            [_assistant(_GET_WEATHER, reasoning="First."), _result("sunny")]
        )
        expected += (
            ("<think>First.</think>" if in_turn else "</think>")
            + _WEATHER_BLOCK
            + _SUNNY
        )
    expected += f"<think>{reasoning}</think>" if in_turn else "</think>"
    expected += answer
    messages.append(assistant)
    if reply == "tool_call":
        messages.append(
            {
                "role": "tool",
                "tool_call_id": assistant["tool_calls"][0]["id"],
                "content": "sunny",
            }
        )
        expected += _WEATHER_BLOCK + _SUNNY
        expected += "<think>" if in_turn else "</think>"
    else:
        expected += _EOS
    if not current_turn:
        messages.append({"role": "user", "content": "Again"})
        expected += "<｜User｜>Again<｜Assistant｜>" + (
            "<think>" if thinking else "</think>"
        )
    tokenizer = _stub_tokenizer() if use_tokenizer else None
    assert (
        await _replay(
            messages,
            thinking,
            tokenizer=tokenizer,
            generation=messages[-1]["role"] != "assistant",
        )
        == expected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("use_tokenizer", [False, True])
@pytest.mark.parametrize("current_turn", [False, True])
@pytest.mark.parametrize("with_call", [False, True])
async def test_v32_null_reasoning_splits_at_the_first_closing_tag(
    thinking, use_tokenizer, current_turn, with_call
):
    messages = [
        _ASKING,
        _assistant(
            *([_GET_WEATHER] if with_call else []),
            content="<think>Checked.</think>Keep </think> in the answer.",
        ),
    ]
    messages[-1]["reasoning_content"] = None
    in_turn = thinking and current_turn
    expected = _ASK + ("<think>Checked.</think>" if in_turn else "</think>")
    expected += "Keep </think> in the answer."
    if with_call:
        messages.append(_result("sunny"))
        expected += _WEATHER_BLOCK + _SUNNY + ("<think>" if in_turn else "</think>")
    else:
        expected += _EOS
    if not current_turn:
        messages.append({"role": "user", "content": "Again"})
        expected += "<｜User｜>Again<｜Assistant｜>" + (
            "<think>" if thinking else "</think>"
        )
    tokenizer = _stub_tokenizer() if use_tokenizer else None
    assert (
        await _replay(
            messages,
            thinking,
            tokenizer=tokenizer,
            generation=messages[-1]["role"] != "assistant",
        )
        == expected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("tool_calls", ["omitted", None, []])
@pytest.mark.parametrize("current_turn", [False, True])
async def test_v32_history_without_tool_calls_keeps_the_assistant_text(
    tool_calls, thinking, current_turn
):
    # An empty list is an ordinary assistant message: its text stays, and no closing
    # function_calls tag appears without an opening one.
    assistant = {
        "role": "assistant",
        "content": "Hello there.",
        "reasoning_content": "Then.",
    }
    if tool_calls != "omitted":
        assistant["tool_calls"] = tool_calls
    messages = [{"role": "user", "content": "Hi"}, assistant]
    reasoning = "<think>Then.</think>" if thinking and current_turn else "</think>"
    expected = f"<｜User｜>Hi<｜Assistant｜>{reasoning}Hello there.{_EOS}"
    if not current_turn:
        messages.append({"role": "user", "content": "Again"})
        expected += (
            f"<｜User｜>Again<｜Assistant｜>{'<think>' if thinking else '</think>'}"
        )
    prompt = await _replay(messages, thinking, generation=not current_turn)
    assert prompt == expected


@pytest.mark.parametrize("tools", [None, []])
def test_v32_direct_render_without_tools_has_no_instruction(tools):
    model = _catalog_model("DeepSeek-V3.2")
    prompt = model.get_full_context(
        [_ASKING], model.model_family.chat_template, tools=tools, thinking=True
    )
    assert prompt == f"{_ASK}<think>"
