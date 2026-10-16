import pytest

from .. import TOOL_PARSERS


def _calls():
    return (
        '<｜DSML｜ calls>\n<｜DSML｜ invoke name="weather">\n'
        '<｜DSML｜ parameter name="city" string="true">杭州</｜DSML｜ parameter>\n'
        '<｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>\n'
        '</｜DSML｜ invoke>\n<｜DSML｜ invoke name="check">\n'
        '<｜DSML｜ parameter name="enabled" string="false">true</｜DSML｜ parameter>\n'
        '<｜DSML｜ parameter name="data" string="false">{"a": [1, null]}</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )


EXPECTED = [
    (None, "weather", {"city": "杭州", "days": 3}),
    (None, "check", {"enabled": True, "data": {"a": [1, None]}}),
]


def _parser():
    return TOOL_PARSERS["deepseek-v4.1"]()


def test_registration_and_nonstreaming():
    assert _parser().extract_tool_calls(_calls()) == EXPECTED
    assert _parser().extract_tool_calls("hello") == [("hello", None, None)]


@pytest.mark.parametrize("split", range(len(_calls()) + 1))
def test_streaming_all_split_positions(split):
    parser = _parser()
    text = "Checking. " + _calls()
    previous = [""]
    results = []
    for chunk in [text[:split], text[split:]]:
        current = previous[-1] + chunk
        results.extend(
            parser.extract_tool_calls_streaming(previous, current, chunk) or []
        )
        previous[-1] = current
    assert [item for item in results if item[1] is not None] == EXPECTED
    assert "".join(item[0] or "" for item in results) == "Checking. "


def test_streaming_character_chunks_and_reset():
    parser = _parser()
    for _ in range(2):
        previous = [""]
        results = []
        for chunk in _calls():
            current = previous[-1] + chunk
            results.extend(
                parser.extract_tool_calls_streaming(previous, current, chunk) or []
            )
            previous[-1] = current
        assert results == EXPECTED


def test_plain_text_partial_markers():
    parser = _parser()
    text = "a < b; <｜DSML｜ x is text"
    previous = [""]
    results = []
    for chunk in text:
        current = previous[-1] + chunk
        results.extend(
            parser.extract_tool_calls_streaming(previous, current, chunk) or []
        )
        previous[-1] = current
    assert "".join(item[0] or "" for item in results) == text


def test_string_and_empty_arguments():
    text = (
        '<｜DSML｜ calls><｜DSML｜ invoke name="ns::run">'
        '<｜DSML｜ parameter name="value" string="true">001</｜DSML｜ parameter>'
        '</｜DSML｜ invoke><｜DSML｜ invoke name="empty"></｜DSML｜ invoke></｜DSML｜ calls>'
    )
    assert _parser().extract_tool_calls(text) == [
        (None, "ns::run", {"value": "001"}),
        (None, "empty", {}),
    ]


def test_nonstreaming_preserves_surrounding_content():
    assert _parser().extract_tool_calls("Checking. " + _calls() + " Done.") == [
        ("Checking. ", None, None),
        *EXPECTED,
        (" Done.", None, None),
    ]


def test_nonstreaming_preserves_content_between_call_blocks():
    assert _parser().extract_tool_calls(_calls() + "Next. " + _calls()) == [
        *EXPECTED,
        ("Next. ", None, None),
        *EXPECTED,
    ]


def test_nonstreaming_keeps_unparseable_output():
    text = "Checking. <｜DSML｜ calls>broken</｜DSML｜ calls>"
    assert _parser().extract_tool_calls(text) == [(text, None, None)]


def test_v4_parser_registration():
    from ..deepseek_v4_tool_parser import DeepseekV4ToolParser

    assert TOOL_PARSERS["deepseek-v4"] is DeepseekV4ToolParser


def test_interleaved_requests_do_not_skip_completed_invokes():
    parser = _parser()
    opening = "<｜DSML｜ calls>"
    histories = {"a": [""], "b": [""]}

    def feed(request, chunk):
        history = histories[request]
        current = history[-1] + chunk
        result = parser.extract_tool_calls_streaming(history, current, chunk)
        history[-1] = current
        return result

    assert feed("a", opening) is None
    assert feed("b", opening) is None
    for request in ("a", "b"):
        closing = (
            f'<｜DSML｜ invoke name="{request}"></｜DSML｜ invoke>' "</｜DSML｜ calls>"
        )
        assert feed(request, closing) == [(None, request, {})]
        assert feed(request, "") is None


@pytest.mark.parametrize("chunk_size", [1, 7, 64, 1024])
def test_shared_parser_interleaved_streams_match_isolated_results(chunk_size):
    parser = _parser()
    texts = ["First: " + _calls(), "Different length: " + _calls(), "Just a < b."]
    histories = [[""] for _ in texts]
    results = [[] for _ in texts]
    for offset in range(0, max(map(len, texts)), chunk_size):
        for index, text in enumerate(texts):
            chunk = text[offset : offset + chunk_size]
            if not chunk:
                continue
            current = histories[index][-1] + chunk
            results[index].extend(
                parser.extract_tool_calls_streaming(histories[index], current, chunk)
                or []
            )
            histories[index][-1] = current
    for index, prefix in enumerate(("First: ", "Different length: ", "Just a < b.")):
        assert "".join(event[0] or "" for event in results[index]) == prefix
        assert [event for event in results[index] if event[1]] == (
            EXPECTED if index < 2 else []
        )


def test_new_request_does_not_reemit_another_requests_calls():
    parser = _parser()
    text = _calls()
    first_end = text.index("</｜DSML｜ invoke>") + len("</｜DSML｜ invoke>")
    first = text[:first_end]
    assert parser.extract_tool_calls_streaming([""], first, first) == EXPECTED[:1]
    assert parser.extract_tool_calls_streaming([""], "hello", "hello") == [
        ("hello", None, None)
    ]
    assert (
        parser.extract_tool_calls_streaming([first], text, text[first_end:])
        == EXPECTED[1:]
    )
