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


def test_extract_tool_call_escapes_bare_ampersands(parser):
    output = (
        '<minimax:tool_call><invoke name="search">'
        "<query>cats & dogs &amp; birds</query>"
        "<url>https://example.com/?a=1&b=2</url>"
        "</invoke></minimax:tool_call>"
    )

    assert parser.extract_tool_calls(output) == [
        (
            None,
            "search",
            {
                "query": "cats & dogs & birds",
                "url": "https://example.com/?a=1&b=2",
            },
        )
    ]


def test_extract_duplicate_sibling_tags_as_list(parser):
    output = (
        '<minimax:tool_call><invoke name="search">'
        "<city>Beijing</city><city>Shanghai</city><city>Shenzhen</city>"
        "</invoke></minimax:tool_call>"
    )

    assert parser.extract_tool_calls(output) == [
        (None, "search", {"city": ["Beijing", "Shanghai", "Shenzhen"]})
    ]


def test_duplicate_array_tags_remain_separate_arrays(parser):
    output = (
        '<minimax:tool_call><invoke name="search">'
        "<cities><item>Beijing</item></cities>"
        "<cities><item>Shanghai</item></cities>"
        "</invoke></minimax:tool_call>"
    )

    assert parser.extract_tool_calls(output) == [
        (None, "search", {"cities": [["Beijing"], ["Shanghai"]]})
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
        (None, "get_weather", {"city": "Beijing"}, 0),
    ]


def test_extract_multiple_tool_calls_streaming(parser):
    output = (
        '<minimax:tool_call><invoke name="get_weather">'
        '<city>Beijing</city></invoke><invoke name="get_time">'
        "<timezone>UTC+8</timezone></invoke></minimax:tool_call>"
    )

    assert parser.extract_tool_calls_streaming([], output, output) == [
        (None, "get_weather", {"city": "Beijing"}, 0),
        (None, "get_time", {"timezone": "UTC+8"}, 1),
    ]


def test_streaming_keeps_indexes_across_separate_chunks(parser):
    first = (
        '<minimax:tool_call><invoke name="first"><value>1</value>'
        "</invoke></minimax:tool_call>"
    )
    second = (
        '<minimax:tool_call><invoke name="second"><value>2</value>'
        "</invoke></minimax:tool_call>"
    )

    assert parser.extract_tool_calls_streaming([], first, first) == (
        None,
        "first",
        {"value": 1},
        0,
    )
    assert parser.extract_tool_calls_streaming([first], first + second, second) == (
        None,
        "second",
        {"value": 2},
        1,
    )


def test_streaming_preserves_text_between_tool_calls(parser):
    first_call = (
        '<minimax:tool_call><invoke name="first"><value>1</value>'
        "</invoke></minimax:tool_call>"
    )
    second_start = '<minimax:tool_call><invoke name="second">'
    current = first_call + " between " + second_start

    assert parser.extract_tool_calls_streaming(
        [first_call], current, " between " + second_start
    ) == (" between ", None, None)


def test_streaming_preserves_text_after_newly_completed_call(parser):
    previous = '<minimax:tool_call><invoke name="first"><value>1</value>'
    delta = "</invoke></minimax:tool_call> tail"

    assert parser.extract_tool_calls_streaming([previous], previous + delta, delta) == [
        (None, "first", {"value": 1}, 0),
        (" tail", None, None),
    ]


def test_streaming_suppresses_split_tool_call_start(parser):
    previous = ["Before "]
    current = "Before <minimax:tool"

    assert (
        parser.extract_tool_calls_streaming(previous, current, "<minimax:tool") is None
    )

    previous = [current]
    current += '_call><invoke name="get_weather">'
    assert (
        parser.extract_tool_calls_streaming(
            previous, current, '_call><invoke name="get_weather">'
        )
        is None
    )


def test_streaming_releases_partial_start_when_it_is_plain_text(parser):
    previous = ["Before <minimax:tool"]
    current = "Before <minimax:tools are unavailable"

    assert parser.extract_tool_calls_streaming(
        previous, current, "s are unavailable"
    ) == ("<minimax:tools are unavailable", None, None)


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
        0,
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

    monkeypatch.setattr(parser, "extract_tool_calls", raise_parse_error)
    current = "prefix<minimax:tool_call>"

    assert parser.extract_tool_calls_streaming([current], current, "broken delta") == (
        "broken delta",
        None,
        None,
    )
