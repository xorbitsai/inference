# CI: linux
import json
import re

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


NS = "]<]minimax[>["
# Each tag in M3's chat template is preceded by the namespace token.
M3_CALL = (
    f"{NS}<tool_call>\n"
    f'{NS}<invoke name="get_weather">'
    f"{NS}<city>Beijing{NS}</city>"
    f"{NS}<days>3{NS}</days>"
    f"{NS}</invoke>\n"
    f"{NS}</tool_call>"
)
# Only namespaced tags are markup, so raw tags in an argument are text.
M3_CALL_WITH_RAW_INVOKE = (
    f"{NS}<tool_call>\n"
    f'{NS}<invoke name="write_file">'
    f"{NS}<code><invoke>hello</invoke>{NS}</code>"
    f"{NS}</invoke>\n"
    f"{NS}</tool_call>"
)
# A legacy block tag in an argument is text too.
LEGACY_BLOCK = "<minimax:tool_call>hello</minimax:tool_call>"
M3_CALL_WITH_LEGACY_BLOCK = (
    f"{NS}<tool_call>\n"
    f'{NS}<invoke name="write_file">'
    f"{NS}<code>{LEGACY_BLOCK}{NS}</code>"
    f"{NS}</invoke>\n"
    f"{NS}</tool_call>"
)
M3_CALLS = [
    (M3_CALL, ("get_weather", {"city": "Beijing", "days": 3})),
    (M3_CALL_WITH_RAW_INVOKE, ("write_file", {"code": "<invoke>hello</invoke>"})),
    (M3_CALL_WITH_LEGACY_BLOCK, ("write_file", {"code": LEGACY_BLOCK})),
]


def _m3_family():
    from ...llm_family import BUILTIN_LLM_FAMILIES

    return next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "MiniMax-M3")


def test_tool_call_rendered_by_the_m3_template_is_parsed(parser):
    from types import SimpleNamespace

    from ...utils import ChatModelMixin

    family = _m3_family()
    mixin = ChatModelMixin()
    mixin.model_family = SimpleNamespace(
        model_name=family.model_name,
        model_ability=family.model_ability,
        chat_template=family.chat_template,
    )
    arguments = {
        "code": "if a < b and c > d: x = '&lt;' & y",
        "cities": ["A", "B"],
        "markup": '<invoke>hello</invoke> <invoke name="other">x</invoke>',
        "legacy": "<minimax:tool_call>hello</minimax:tool_call>",
    }
    prompt = mixin.get_full_context(
        [
            {"role": "user", "content": "Run it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "run",
                            "arguments": json.dumps(arguments),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "done"},
        ],
        family.chat_template,
        tools=[{"type": "function", "function": {"name": "run"}}],
    )

    start = prompt.index(f'{NS}<tool_call>\n{NS}<invoke name="run">')
    end = prompt.index(f"{NS}</tool_call>", start) + len(f"{NS}</tool_call>")
    assert parser.extract_tool_calls(prompt[start:end]) == [(None, "run", arguments)]


def _m3_model():
    from ...utils import ChatModelMixin

    model = ChatModelMixin()
    model.tool_parser = MiniMaxM3ToolParser()
    model.reasoning_parser = None
    model.model_family = None
    model.model_uid = "test-model"
    return model


def _completion(text, finish_reason):
    return {
        "id": "test",
        "object": "text_completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
    }


@pytest.mark.parametrize("call_text,expected", M3_CALLS)
def test_m3_tool_call_reaches_the_chat_response(call_text, expected):
    result = _m3_model()._post_process_completion(
        None, "test-model", _completion("Checking.\n" + call_text, "stop")
    )
    choice = result["choices"][0]

    assert choice["message"]["content"] == "Checking.\n"
    assert [
        (call["function"]["name"], json.loads(call["function"]["arguments"]))
        for call in choice["message"]["tool_calls"]
    ] == [expected]
    assert choice["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
@pytest.mark.parametrize("call_text,expected", M3_CALLS)
async def test_m3_tool_call_reaches_the_chat_stream(call_text, expected):
    # The namespace and the <tool_call> tags are single tokens and arrive whole.
    pieces = []
    for part in re.split(
        r"(\]<\]minimax\[>\[|</?tool_call>)", "Checking.\n" + call_text
    ):
        if part.startswith((NS, "<tool_call>", "</tool_call>")):
            pieces.append(part)
        else:
            pieces.extend(part[i : i + 3] for i in range(0, len(part), 3))

    async def chunks():
        for piece in pieces:
            yield _completion(piece, None)
        yield _completion("", "stop")

    content, calls, finish_reason = "", [], None
    async for chunk in _m3_model()._async_to_tool_completion_chunks(chunks()):
        for choice in chunk["choices"]:
            content += choice["delta"].get("content") or ""
            calls += [
                (call["function"]["name"], json.loads(call["function"]["arguments"]))
                for call in choice["delta"].get("tool_calls") or []
            ]
            finish_reason = choice["finish_reason"] or finish_reason

    assert content == "Checking.\n"
    assert calls == [expected]
    assert finish_reason == "tool_calls"


# A complete invoke, then output cut off before the outer closing tag.
M3_CALL_CUT_OFF = M3_CALL[: -len(f"{NS}</tool_call>")]


def test_m3_block_cut_off_before_its_closing_tag_is_not_a_call_in_the_response():
    result = _m3_model()._post_process_completion(
        None, "test-model", _completion("Checking.\n" + M3_CALL_CUT_OFF, "length")
    )
    choice = result["choices"][0]

    assert not choice["message"].get("tool_calls")
    assert choice["message"]["content"] == "Checking.\n" + M3_CALL_CUT_OFF
    assert choice["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_m3_block_cut_off_before_its_closing_tag_is_not_a_call_in_the_stream():
    async def chunks():
        yield _completion("Checking.\n", None)
        yield _completion(M3_CALL_CUT_OFF, None)
        yield _completion("", "length")

    content, calls, finish_reason = "", [], None
    async for chunk in _m3_model()._async_to_tool_completion_chunks(chunks()):
        for choice in chunk["choices"]:
            content += choice["delta"].get("content") or ""
            calls += choice["delta"].get("tool_calls") or []
            finish_reason = choice["finish_reason"] or finish_reason

    assert calls == []
    assert finish_reason == "length"
