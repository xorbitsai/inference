import pytest

from ..minimax_tool_parser import MiniMaxToolParser

OUTPUT = (
    "Let me check.\n"
    "<minimax:tool_call>\n"
    '<invoke name="get_weather">\n'
    '<parameter name="city">Hanoi</parameter>\n'
    "</invoke>\n"
    '<invoke name="get_time">\n'
    '<parameter name="zone">UTC</parameter>\n'
    "</invoke>\n"
    "</minimax:tool_call>"
)


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


def test_extract_parallel_invokes():
    assert MiniMaxToolParser().extract_tool_calls(OUTPUT) == [
        ("Let me check.\n", None, None),
        (None, "get_weather", {"city": "Hanoi"}),
        (None, "get_time", {"zone": "UTC"}),
    ]


@pytest.mark.parametrize("size", [1, 7, len(OUTPUT)])
def test_streaming_keeps_every_invoke_of_a_block(size):
    deltas = [OUTPUT[i : i + size] for i in range(0, len(OUTPUT), size)]
    content, calls = _stream(MiniMaxToolParser(), deltas)
    assert content == "Let me check.\n"
    assert calls == [
        ("get_weather", {"city": "Hanoi"}),
        ("get_time", {"zone": "UTC"}),
    ]


def test_streaming_keeps_text_after_the_block():
    deltas = [
        "<minimax:tool_call>",
        '<invoke name="get_weather"><parameter name="city">Hanoi</parameter>',
        "</invoke></minimax:tool_call> Done.",
    ]
    content, calls = _stream(MiniMaxToolParser(), deltas)
    assert content == " Done."
    assert calls == [("get_weather", {"city": "Hanoi"})]


def test_streaming_plain_text_is_unchanged():
    content, calls = _stream(MiniMaxToolParser(), ["Hello", " a < b", " world"])
    assert content == "Hello a < b world"
    assert calls == []


def test_streaming_releases_text_that_only_looked_like_a_tag():
    deltas = ["a <mini", "x <minimax:tool_call>", '<invoke name="f">']
    deltas += ["</invoke></minimax:tool_call>"]
    content, calls = _stream(MiniMaxToolParser(), deltas)
    assert content == "a <minix "
    assert calls == [("f", {})]
