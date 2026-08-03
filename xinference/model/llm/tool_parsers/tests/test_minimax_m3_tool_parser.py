import pytest

from ..minimax_m3_tool_parser import MiniMaxM3ToolParser


@pytest.fixture
def parser():
    return MiniMaxM3ToolParser()


def test_extract_tool_call_with_m3_xml_arguments(parser):
    output = (
        '<minimax:tool_call><invoke name="search">'
        "<query>weather</query><limit>3</limit><exact>true</exact>"
        "</invoke></minimax:tool_call>"
    )

    assert parser.extract_tool_calls(output) == [
        (None, "search", {"query": "weather", "limit": 3, "exact": True})
    ]


def test_extract_tool_call_with_nested_object_and_array(parser):
    output = (
        '<minimax:tool_call><invoke name="search">'
        "<filters><unit>celsius</unit><range><min>-5</min><max>10</max></range>"
        "</filters><cities><item>Beijing</item><item>Shanghai</item></cities>"
        "</invoke></minimax:tool_call>"
    )

    assert parser.extract_tool_calls(output) == [
        (
            None,
            "search",
            {
                "filters": {"unit": "celsius", "range": {"min": -5, "max": 10}},
                "cities": ["Beijing", "Shanghai"],
            },
        )
    ]


def test_extract_multiple_calls_and_surrounding_text(parser):
    output = (
        "Before "
        '<minimax:tool_call><invoke name="get_weather"><city>Beijing</city></invoke>'
        '<invoke name="get_time"><timezone>UTC+8</timezone></invoke>'
        "</minimax:tool_call> after"
    )

    assert parser.extract_tool_calls(output) == [
        ("Before ", None, None),
        (None, "get_weather", {"city": "Beijing"}),
        (None, "get_time", {"timezone": "UTC+8"}),
        (" after", None, None),
    ]


def test_extract_no_tool_call(parser):
    assert parser.extract_tool_calls("plain response") == [
        ("plain response", None, None)
    ]


def test_extract_preserves_thinking_block(parser):
    output = (
        "<mm:think>Need weather data</mm:think>"
        '<minimax:tool_call><invoke name="get_weather"><city>Beijing</city>'
        "</invoke></minimax:tool_call>"
    )

    assert parser.extract_tool_calls(output) == [
        ("<mm:think>Need weather data</mm:think>", None, None),
        (None, "get_weather", {"city": "Beijing"}),
    ]


def test_extract_legacy_parameter_format(parser):
    output = (
        '<minimax:tool_call><invoke name="get_weather">'
        '<parameter name="city">Beijing</parameter>'
        "</invoke></minimax:tool_call>"
    )

    assert parser.extract_tool_calls(output) == [
        (None, "get_weather", {"city": "Beijing"})
    ]


def test_extract_tool_call_streaming(parser):
    chunks = [
        "<minimax:tool_call>",
        '<invoke name="get_weather">',
        "<city>Beijing</city>",
        "</invoke>",
        "</minimax:tool_call>",
    ]
    previous = []
    current = ""
    results = []

    for chunk in chunks:
        current += chunk
        results.append(parser.extract_tool_calls_streaming(previous, current, chunk))
        previous.append(current)

    assert results == [
        None,
        None,
        None,
        None,
        (None, "get_weather", {"city": "Beijing"}),
    ]


@pytest.mark.parametrize("previous", [[], None])
def test_extract_tool_call_streaming_with_empty_history(parser, previous):
    output = (
        '<minimax:tool_call><invoke name="get_weather"><city>Beijing</city>'
        "</invoke></minimax:tool_call>"
    )

    assert parser.extract_tool_calls_streaming(previous, output, output) == (
        None,
        "get_weather",
        {"city": "Beijing"},
    )


def test_streaming_returns_text_after_completed_call(parser):
    tool_call = (
        '<minimax:tool_call><invoke name="get_weather"><city>Beijing</city>'
        "</invoke></minimax:tool_call>"
    )

    assert parser.extract_tool_calls_streaming(
        [tool_call], tool_call + " done", " done"
    ) == (" done", None, None)
