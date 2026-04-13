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
