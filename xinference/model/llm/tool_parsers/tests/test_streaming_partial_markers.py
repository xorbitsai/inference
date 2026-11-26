# CI: linux
import json
import time

import pytest

from ..abstract_tool_parser import find_outside_json_string
from ..deepseek_v3_1_tool_parser import DeepseekV3_1ToolParser
from ..deepseek_v3_2_tool_parser import DeepseekV3_2ToolParser
from ..deepseek_v4_tool_parser import DeepseekV4ToolParser
from ..glm5_tool_parser import Glm5ToolParser
from ..seed_oss_tool_parser import SeedOssToolParser

# Plain text that keeps ending a chunk on "<" or on a longer prefix of a tool
# call tag ("<tool", "<function_calls", "<｜", ...) without ever opening one.
PLAIN_TEXT = (
    "if a < b and b <c then print(a<b); "
    "<tool is not <tool_call, <function_calls is not <function_call>! "
    "<｜ and <｜DSML or <｜tool▁calls are only prefixes."
)

PARSERS = {
    "deepseek-v3.1": (
        DeepseekV3_1ToolParser,
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>"
        '{"location": "Beijing"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
    ),
    "deepseek-v3.2": (
        DeepseekV3_2ToolParser,
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_weather">'
        '<｜DSML｜parameter name="location" string="true">Beijing'
        "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜function_calls>",
    ),
    "deepseek-v4": (
        DeepseekV4ToolParser,
        '<｜DSML｜tool_calls><｜DSML｜invoke name="get_weather">'
        '<｜DSML｜parameter name="location" string="true">Beijing'
        "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>",
    ),
    "deepseek-v3.2-plain": (
        DeepseekV3_2ToolParser,
        '<function_calls><invoke name="get_weather">'
        '<parameter name="location" string="true">Beijing'
        "</parameter></invoke></function_calls>",
    ),
    "deepseek-v4-plain": (
        DeepseekV4ToolParser,
        '<tool_calls><invoke name="get_weather">'
        '<parameter name="location" string="true">Beijing'
        "</parameter></invoke></tool_calls>",
    ),
    "glm5": (
        Glm5ToolParser,
        "<tool_call>get_weather<arg_key>location</arg_key>"
        "<arg_value>Beijing</arg_value></tool_call>",
    ),
    "seed-oss": (
        SeedOssToolParser,
        "<seed:tool_call>\n<function=get_weather>\n"
        "<parameter=location>Beijing</parameter>\n</function>\n</seed:tool_call>",
    ),
}


def _stream(parser, chunks):
    """Feed chunks the way ChatModelMixin._post_process_completion_chunk does."""
    previous_texts = [""]
    contents = []
    tool_calls = []
    for delta_text in chunks:
        current_text = previous_texts[-1] + delta_text
        result = parser.extract_tool_calls_streaming(
            previous_texts, current_text, delta_text
        )
        previous_texts[-1] = current_text
        if result is None:
            continue
        for content, name, args, *_ in result if isinstance(result, list) else [result]:
            if name is not None:
                tool_calls.append((name, args))
            elif content:
                contents.append(content)
    return "".join(contents), tool_calls


def _splits(text):
    yield [text]
    for i in range(1, len(text)):
        yield [text[:i], text[i:]]
    yield list(text)


@pytest.mark.parametrize("parser_name", PARSERS)
def test_streaming_releases_held_text_that_is_not_a_tool_call(parser_name):
    parser_cls, _ = PARSERS[parser_name]
    for chunks in _splits(PLAIN_TEXT):
        content, tool_calls = _stream(parser_cls(), chunks)
        assert content == PLAIN_TEXT, chunks
        assert tool_calls == []


@pytest.mark.parametrize("parser_name", PARSERS)
def test_streaming_keeps_text_before_a_split_tool_call_tag(parser_name):
    parser_cls, tool_call = PARSERS[parser_name]
    prefix = "Checking a < b first. "
    opening_tag = tool_call[: tool_call.index(">") + 1]
    # Each split leaves only part of the opening tag at the end of chunk one,
    # so the parser must hold it back and later decide it is a real tag.
    for split in range(1, len(opening_tag)):
        chunks = [prefix + tool_call[:split], tool_call[split:]]
        content, tool_calls = _stream(parser_cls(), chunks)
        assert content == prefix, chunks
        assert tool_calls == [("get_weather", {"location": "Beijing"})], chunks


@pytest.mark.parametrize("parser_name", PARSERS)
def test_streaming_keeps_text_around_tool_calls(parser_name):
    parser_cls, first = PARSERS[parser_name]
    second = first.replace("Beijing", "Paris")
    text = "A " + first + " B " + second + " C <tag> D"
    expected_calls = [
        ("get_weather", {"location": "Beijing"}),
        ("get_weather", {"location": "Paris"}),
    ]
    for chunks in _splits(text):
        content, tool_calls = _stream(parser_cls(), chunks)
        assert tool_calls == expected_calls, chunks
        assert content == "A  B  C <tag> D", chunks


@pytest.mark.parametrize("parser_name", ["deepseek-v3.2", "deepseek-v4"])
def test_streaming_text_may_mention_the_other_format(parser_name):
    parser_cls, tool_call = PARSERS[parser_name]
    wrapper = "function_calls" if parser_name == "deepseek-v3.2" else "tool_calls"
    text = f"{tool_call} Use <{wrapper}> tags next time."
    for chunks in _splits(text):
        content, tool_calls = _stream(parser_cls(), chunks)
        assert content == f" Use <{wrapper}> tags next time.", chunks
        assert len(tool_calls) == 1, chunks


# The marker that closes the tool calls block of each format.
WRAPPER_END = {
    "deepseek-v3.1": "<｜tool▁calls▁end｜>",
    "deepseek-v3.2": "</｜DSML｜function_calls>",
    "deepseek-v4": "</｜DSML｜tool_calls>",
    "deepseek-v3.2-plain": "</function_calls>",
    "deepseek-v4-plain": "</tool_calls>",
}


# The marker that closes a single call of each format.
CALL_END = {
    "deepseek-v3.1": "<｜tool▁call▁end｜>",
    "deepseek-v3.2": "</｜DSML｜invoke>",
    "deepseek-v4": "</｜DSML｜invoke>",
    "deepseek-v3.2-plain": "</invoke>",
    "deepseek-v4-plain": "</invoke>",
}


def _chunks_of(text, size):
    return [text[i : i + size] for i in range(0, len(text), size)]


@pytest.mark.parametrize("parser_name", WRAPPER_END)
def test_streaming_ignores_block_end_marker_inside_an_argument(parser_name):
    parser_cls, tool_call = PARSERS[parser_name]
    marker = WRAPPER_END[parser_name]
    # The marker that closes the call itself, which a wrapper end may follow
    call_end = CALL_END[parser_name]
    for value in (
        f"example = {marker}; print(example)",
        f"example = {call_end}; print(example)",
        f"example = {call_end}{marker}; print(example)",
    ):
        text = "A " + tool_call.replace("Beijing", value) + " B"
        for chunks in [*_splits(text), _chunks_of(text, 17)]:
            content, tool_calls = _stream(parser_cls(), chunks)
            assert tool_calls == [("get_weather", {"location": value})], chunks
            assert content == "A  B", chunks


@pytest.mark.parametrize("parser_name", ["deepseek-v3.2", "deepseek-v4"])
@pytest.mark.parametrize("plain", (False, True))
def test_streaming_ignores_end_markers_inside_a_json_argument(parser_name, plain):
    name = parser_name + "-plain" if plain else parser_name
    parser_cls, tool_call = PARSERS[name]
    call_end = CALL_END[name]
    marker = WRAPPER_END[name]
    for value in (f"example = {call_end}{marker}; print(example)",):
        # A JSON value ends at its closing tag outside of its strings
        json_call = tool_call.replace('string="true">Beijing', 'string="false">"VALUE"')
        text = "A " + json_call.replace("VALUE", value) + " B"
        for chunks in [*_splits(text), _chunks_of(text, 17)]:
            content, tool_calls = _stream(parser_cls(), chunks)
            assert tool_calls == [("get_weather", {"location": value})], chunks
            assert content == "A  B", chunks


@pytest.mark.parametrize("parser_name", ["deepseek-v3.2", "deepseek-v4"])
@pytest.mark.parametrize("plain", (False, True))
def test_a_call_after_malformed_json_is_kept(parser_name, plain):
    name = parser_name + "-plain" if plain else parser_name
    parser_cls, tool_call = PARSERS[name]
    # The string is never closed, so it must not hide the call that follows
    malformed = tool_call.replace('string="true">Beijing', 'string="false">{"a": "b}')
    valid = tool_call.replace("Beijing", "Paris")
    wrapper_start = valid[: valid.index("<", 1)]
    wrapper_end = WRAPPER_END[name]
    text = "A " + malformed.removesuffix(wrapper_end) + valid[len(wrapper_start) :]
    text += " B"
    expected = ("get_weather", {"location": "Paris"})
    assert expected in [
        (name, args) for _, name, args in parser_cls().extract_tool_calls(text) if name
    ]
    for chunks in [[text], _chunks_of(text, 17), list(text)]:
        content, tool_calls = _stream(parser_cls(), chunks)
        assert expected in tool_calls, chunks
        assert content == "A  B", chunks


def test_find_outside_json_string():
    marker = "<end>"
    # Only a marker outside of a string counts
    assert find_outside_json_string('{"a": "<end>"}<end>', marker, 0) == 14
    assert find_outside_json_string('{"a": "\\"<end>"}<end>', marker, 0) == 16
    # A string that is still open hides the rest of the text
    assert find_outside_json_string('{"a": "x<end>', marker, 0) == -1
    assert find_outside_json_string("no marker at all", marker, 0) == -1
    # An open string cut short by the start of a following call
    text = '{"a": "x<end><call>{"b": 1}<end>'
    assert find_outside_json_string(text, marker, 0, "<call>") == 8
    # A closed string that holds the marker is kept whole
    assert find_outside_json_string('{"a": "x<end>"}<end>', marker, 0, "<call>") == 15


@pytest.mark.parametrize("name", ["deepseek-v3.1", "deepseek-v4"])
def test_a_growing_json_argument_is_not_rescanned_on_every_delta(name):
    parser_cls, tool_call = PARSERS[name]
    # Many string characters and no call end marker until the very end
    value = json.dumps("".join(chr(97 + i % 26) + " " for i in range(16384)))
    if name == "deepseek-v3.1":
        text = tool_call.replace('{"location": "Beijing"}', '{"location": ' + value + "}")
    else:
        text = tool_call.replace(
            'string="true">Beijing', 'string="false">' + value
        )
    chunks = _chunks_of(text, 8)
    started = time.perf_counter()
    content, tool_calls = _stream(parser_cls(), chunks)
    elapsed = time.perf_counter() - started
    assert tool_calls and tool_calls[0][0] == "get_weather"
    assert not content
    # A rescan of the whole value on each delta takes about 5 seconds here
    assert elapsed < 1.5, elapsed
