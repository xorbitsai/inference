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


def test_streaming_yields_text_before_tool_call(parser):
    previous = ["<mm:think>thought"]
    current = "<mm:think>thought</mm:think><minimax:tool_call>"
    delta = "</mm:think><minimax:tool_call>"

    assert parser.extract_tool_calls_streaming(previous, current, delta) == (
        "</mm:think>",
        None,
        None,
    )


def test_content_regex_does_not_match_mismatched_tags(parser):
    output = "<mm:think>thought</minimax:tool_call>"

    assert parser._get_function_calls(output) == [output]


def test_empty_text_has_no_unclosed_tool_call(parser):
    assert parser._has_unclosed_tool_call("") is False


def test_streaming_returns_raw_delta_on_parser_error(parser, monkeypatch):
    def raise_parse_error(_text):
        raise ValueError("unexpected model output")

    monkeypatch.setattr(parser, "_get_function_calls_streaming", raise_parse_error)
    current = "prefix<minimax:tool_call>"

    # Keep the start token in the previous value so this exercises the error
    # fallback instead of the transition-prefix path.
    assert parser.extract_tool_calls_streaming([current], current, "broken delta") == (
        "broken delta",
        None,
        None,
    )
