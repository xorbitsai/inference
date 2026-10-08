import pytest

from ..deepseek_v3_1_tool_parser import DeepseekV3_1ToolParser
from ..deepseek_v3_2_tool_parser import DeepseekV3_2ToolParser
from ..deepseek_v4_tool_parser import DeepseekV42ToolParser
from ..glm5_tool_parser import Glm5ToolParser

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
        DeepseekV42ToolParser,
        '<｜DSML｜tool_calls><｜DSML｜invoke name="get_weather">'
        '<｜DSML｜parameter name="location" string="true">Beijing'
        "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>",
    ),
    "glm5": (
        Glm5ToolParser,
        "<tool_call>get_weather<arg_key>location</arg_key>"
        "<arg_value>Beijing</arg_value></tool_call>",
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
