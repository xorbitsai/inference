# CI: linux
import asyncio

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


def test_streaming_long_call_in_small_deltas():
    code = "\n".join(["print(1)"] * 500)
    deltas = ["Writing.", "<minimax:tool_call>", '<invoke name="write_file">']
    deltas += ['<parameter name="content">']
    deltas += [code[i : i + 3] for i in range(0, len(code), 3)]
    deltas += ["</parameter>", "</invoke>", "</minimax:tool_call>", " Done."]

    content, calls = _stream(MiniMaxToolParser(), deltas)

    assert content == "Writing. Done."
    assert calls == [("write_file", {"content": code})]


# Output that reaches max_tokens after </invoke>, before </minimax:tool_call>.
TRUNCATED = OUTPUT[: -len("</minimax:tool_call>")]


def test_a_block_without_its_closing_tag_stays_text():
    assert MiniMaxToolParser().extract_tool_calls(TRUNCATED) == [
        (TRUNCATED, None, None)
    ]


def _completion(text, finish_reason):
    return {
        "id": "test",
        "object": "text_completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
    }


@pytest.mark.parametrize("stream", [False, True])
def test_a_block_cut_off_before_its_closing_tag_emits_no_call(stream):
    from ...utils import ChatModelMixin

    model = ChatModelMixin()
    model.tool_parser = MiniMaxToolParser()
    model.reasoning_parser = None
    model.model_family = None
    model.model_uid = "test-model"

    if stream:

        async def chunks():
            for i in range(0, len(TRUNCATED), 7):
                yield _completion(TRUNCATED[i : i + 7], None)
            yield _completion("", "length")

        async def collect():
            return [c async for c in model._async_to_tool_completion_chunks(chunks())]

        choices = [c["choices"][0] for c in asyncio.run(collect())]
        content = "".join(c["delta"].get("content") or "" for c in choices)
        calls = [call for c in choices for call in c["delta"].get("tool_calls") or []]
        finish_reason = choices[-1]["finish_reason"]
    else:
        choice = model._post_process_completion(
            None, "test-model", _completion(TRUNCATED, "length")
        )["choices"][0]
        content = choice["message"]["content"]
        calls = choice["message"]["tool_calls"]
        finish_reason = choice["finish_reason"]

    assert calls == []
    assert finish_reason == "length"
    assert content.startswith("Let me check.\n")
