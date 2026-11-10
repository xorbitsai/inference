# CI: linux
import json

import pytest

from ...tool_controls import single_tool_chunks
from ...utils import ChatModelMixin
from .. import TOOL_PARSERS

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "code": {"type": "string"},
                    "line": {"type": "integer"},
                },
            },
        },
    }
]
CODE = '404 "雪"\npath\\file'
ENCODED = json.dumps(CODE, ensure_ascii=False)


def dsml(wrapper, prefix):
    return (
        f'<{prefix}{wrapper}><{prefix}invoke name="write">'
        f'<{prefix}parameter name="code" string="true">',
        f'</{prefix}parameter><{prefix}parameter name="line" string="false">'
        f"7</{prefix}parameter></{prefix}invoke></{prefix}{wrapper}>",
    )


CALLS = {
    "deepseek-v3.2": (*dsml("function_calls", "｜DSML｜"), CODE),
    "deepseek-v4": (*dsml("tool_calls", "｜DSML｜"), CODE),
    "deepseek-v4.1": (*dsml("calls", "｜DSML｜ "), CODE),
    "deepseek-v3.1": (
        '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>write<｜tool▁sep｜>{"code":',
        ',"line":7}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        ENCODED,
    ),
    "deepseek-r1": (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>write\n"
        + chr(96) * 3
        + 'json\n{"code":',
        ',"line":7}\n' + chr(96) * 3 + "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        ENCODED,
    ),
    "deepseek-v3": (
        chr(96) * 3 + 'json\n{"name":"write","parameters":{"code":',
        ',"line":7}}\n' + chr(96) * 3,
        ENCODED,
    ),
    "llama3": ('{"name":"write","parameters":{"code":', ',"line":7}}', ENCODED),
    "qwen": (
        '<tool_call>{"name":"write","arguments":{"code":',
        ',"line":7}}</tool_call>',
        ENCODED,
    ),
    "glm5": (
        "<tool_call>write<arg_key>code</arg_key><arg_value>",
        "</arg_value><arg_key>line</arg_key><arg_value>7</arg_value></tool_call>",
        CODE,
    ),
    "gemma": (
        '<|tool_call>call:write{code:<|"|>',
        '<|"|>,line:7}<tool_call|>',
        CODE,
    ),
    "kimi-k3": (
        '<|open|>tools<|sep|><|open|>call tool="write"<|sep|>'
        '<|open|>argument key="code" type="string"<|sep|>',
        '<|close|>argument<|sep|><|open|>argument key="line" type="integer"<|sep|>'
        "7<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>",
        CODE,
    ),
    "minicpm5": (
        '<function name="write"><param name="code">',
        '</param><param name="line">7</param></function>',
        CODE,
    ),
    "minimax": (
        '<minimax:tool_call><invoke name="write"><parameter name="code">',
        '</parameter><parameter name="line">7</parameter></invoke></minimax:tool_call>',
        CODE,
    ),
    "minimax_m3": (
        "<minimax:tool_call><invoke name='write'><parameter name='code'>",
        "</parameter><line>7</line></invoke></minimax:tool_call>",
        CODE,
    ),
    "seed-oss": (
        "<seed:tool_call><function=write><parameter=code>",
        "</parameter><parameter=line>7</parameter></function></seed:tool_call>",
        CODE,
    ),
}


def feed(model, text, history, state, finish=None):
    chunk = {
        "id": "fixture",
        "model": "fixture",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}],
    }
    return model._post_process_completion_chunk(
        None,
        "fixture",
        chunk,
        previous_texts=history,
        tool_call_state=state,
        tools=TOOLS,
    )


def deltas(chunk):
    if chunk:
        return chunk["choices"][0]["delta"]["tool_calls"]
    return []


def test_matrix_covers_every_registered_parser():
    assert set(CALLS) | {"glm4"} == set(TOOL_PARSERS)


@pytest.mark.parametrize("parser_name", CALLS)
@pytest.mark.parametrize("size", (1, 17))
def test_streams_tool_calls(parser_name, size):
    opening, closing, value = CALLS[parser_name]
    model = ChatModelMixin()
    model.tool_parser = TOOL_PARSERS[parser_name]()
    history, state, calls = [""], {}, []

    def step(text):
        for offset in range(0, len(text), size):
            chunk = feed(model, text[offset : offset + size], history, state)
            if chunk:
                assert not chunk["choices"][0]["delta"]["content"]
            calls.extend(deltas(chunk))

    step(opening + value)
    before_close = "".join(call["function"]["arguments"] for call in calls)
    if parser_name in ("llama3", "minimax_m3"):
        assert not before_close
    else:
        assert "404" in before_close
        with pytest.raises(json.JSONDecodeError):
            json.loads(before_close)

    step(closing)
    assert {call["index"] for call in calls} == {0}
    assert sum("id" in call for call in calls) == 1
    assert sum("name" in call["function"] for call in calls) == 1
    assert calls[0]["function"]["name"] == "write"
    assert json.loads("".join(call["function"]["arguments"] for call in calls)) == {
        "code": CODE,
        "line": 7,
    }
    assert (
        feed(model, "", history, state, "stop")["choices"][0]["finish_reason"]
        == "tool_calls"
    )


@pytest.mark.parametrize("parser_name", CALLS)
def test_truncation_does_not_close_arguments(parser_name):
    opening, _, value = CALLS[parser_name]
    model = ChatModelMixin()
    model.tool_parser = TOOL_PARSERS[parser_name]()
    history, state = [""], {}
    partial = deltas(feed(model, opening + value, history, state))
    assert bool(partial) == (parser_name not in ("llama3", "minimax_m3"))
    chunk = feed(model, "", history, state, "length")
    assert not deltas(chunk)
    assert chunk["choices"][0]["finish_reason"] == "length"


def test_deepseek_streams_long_second_argument():
    opening, closing, value = CALLS["deepseek-v4.1"]
    tag = '<｜DSML｜ parameter name="code" string="true">'
    model = ChatModelMixin()
    model.tool_parser = TOOL_PARSERS["deepseek-v4.1"]()
    history, state = [""], {}
    parts = [
        opening.replace(tag, '<｜DSML｜ parameter name="path" string="true">main.py'),
        "</｜DSML｜ parameter>",
        tag + value,
        closing,
    ]
    events = [deltas(feed(model, part, history, state)) for part in parts]
    assert "main.py" in events[0][0]["function"]["arguments"]
    assert "404" in events[2][0]["function"]["arguments"]
    assert json.loads(
        "".join(call["function"]["arguments"] for batch in events for call in batch)
    ) == {"path": "main.py", "code": CODE, "line": 7}


@pytest.mark.parametrize(
    ("parser_name", "wrapper_start", "wrapper_end"),
    [
        ("deepseek-v4.1", "<｜DSML｜ calls>", "</｜DSML｜ calls>"),
        ("deepseek-r1", "<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>"),
        ("minimax", "<minimax:tool_call>", "</minimax:tool_call>"),
        ("minimax_m3", "<minimax:tool_call>", "</minimax:tool_call>"),
        ("seed-oss", "<seed:tool_call>", "</seed:tool_call>"),
    ],
)
def test_multiple_invokes_in_one_wrapper(parser_name, wrapper_start, wrapper_end):
    opening, closing, value = CALLS[parser_name]
    model = ChatModelMixin()
    model.tool_parser = TOOL_PARSERS[parser_name]()
    history, state, calls = [""], {}, {}
    parts = [
        opening + value,
        closing.removesuffix(wrapper_end),
        opening.removeprefix(wrapper_start) + value,
        closing,
    ]
    for i, part in enumerate(parts):
        events = deltas(feed(model, part, history, state))
        if i in (0, 2) and parser_name != "minimax_m3":
            assert "404" in events[-1]["function"]["arguments"]
            assert events[-1]["index"] == i // 2
        for event in events:
            arguments = calls.setdefault(event["index"], [])
            arguments.append(event["function"]["arguments"])
    assert list(calls) == [0, 1]
    assert [json.loads("".join(parts)) for parts in calls.values()] == [
        {"code": CODE, "line": 7},
        {"code": CODE, "line": 7},
    ]


@pytest.mark.parametrize(
    ("parser_name", "opening", "value", "closing", "expected"),
    [
        (
            "minicpm5",
            '<function name="write"><param name="code"><![CDATA[',
            "  literal </param> ",
            "]]></param></function>",
            "  literal </param> ",
        ),
        (
            "minimax_m3",
            '<minimax:tool_call><invoke name="write"><code>',
            "left &amp; right",
            "</code></invoke></minimax:tool_call>",
            "left & right",
        ),
    ],
)
def test_native_string_wrappers(parser_name, opening, value, closing, expected):
    model = ChatModelMixin()
    model.tool_parser = TOOL_PARSERS[parser_name]()
    history, state, fragments = [""], {}, []
    for char in opening + value:
        fragments.extend(deltas(feed(model, char, history, state)))
    assert bool(fragments) == (parser_name == "minicpm5")
    fragments.extend(deltas(feed(model, closing, history, state)))
    assert json.loads("".join(call["function"]["arguments"] for call in fragments)) == {
        "code": expected
    }


def test_glm4_native_payload_and_plain_text():
    model = ChatModelMixin()
    model.tool_parser = TOOL_PARSERS["glm4"]()
    history, state = [""], {}
    assert (
        feed(model, "hello ", history, state)["choices"][0]["delta"]["content"]
        == "hello "
    )
    assert (
        feed(model, "world", history, state)["choices"][0]["delta"]["content"]
        == "world"
    )
    native = {
        "id": "fixture",
        "name": "write",
        "arguments": '{"code": "404"}',
        "choices": [{"index": 0}],
    }
    (call,) = deltas(
        model._post_process_completion_chunk(
            None,
            "fixture",
            native,
            previous_texts=history,
            tool_call_state=state,
            tools=TOOLS,
        )
    )
    assert json.loads(call["function"]["arguments"]) == {"code": "404"}


def test_deepseek_streams_calls_from_separate_wrappers():
    opening, closing, value = CALLS["deepseek-v4.1"]
    model = ChatModelMixin()
    model.tool_parser = TOOL_PARSERS["deepseek-v4.1"]()
    history, state = [""], {}
    for index in range(2):
        (first,) = deltas(feed(model, opening + value, history, state))
        (last,) = deltas(feed(model, closing, history, state))
        assert first["index"] == last["index"] == index
        assert json.loads(
            first["function"]["arguments"] + last["function"]["arguments"]
        ) == {"code": CODE, "line": 7}


def test_llama_releases_an_ordinary_json_object_at_finish():
    model = ChatModelMixin()
    model.tool_parser = TOOL_PARSERS["llama3"]()
    history, state = [""], {}
    text = '{"city":{"name":"Paris"}}'
    for char in text:
        assert feed(model, char, history, state) is None
    chunk = feed(model, "", history, state, "stop")
    assert chunk["choices"][0]["delta"]["content"] == text
    assert not deltas(chunk)


def test_gemma_streams_string_after_nested_parameter():
    model = ChatModelMixin()
    model.tool_parser = TOOL_PARSERS["gemma"]()
    history, state = [""], {}
    opening = '<|tool_call>call:write{options:{line:7},code:<|"|>'
    (first,) = deltas(feed(model, opening + CODE, history, state))
    assert "404" in first["function"]["arguments"]
    (last,) = deltas(feed(model, '<|"|>}<tool_call|>', history, state))
    assert json.loads(
        first["function"]["arguments"] + last["function"]["arguments"]
    ) == {"options": {"line": 7}, "code": CODE}


@pytest.mark.parametrize("parts", [("a\n", "b\t"), ("a", "\nb\t")])
def test_qwen_literal_control_characters_are_escaped(parts):
    model = ChatModelMixin()
    model.tool_parser = TOOL_PARSERS["qwen"]()
    history, state = [""], {}
    prefix = '<tool_call>{"name":"write","arguments":{"code":"'
    (early,) = deltas(feed(model, prefix + parts[0], history, state))
    (final,) = deltas(feed(model, parts[1] + '"}}</tool_call>', history, state))
    assert json.loads(
        early["function"]["arguments"] + final["function"]["arguments"]
    ) == {"code": "".join(parts)}
    assert (
        feed(model, "", history, state, "stop")["choices"][0]["finish_reason"]
        == "tool_calls"
    )


@pytest.mark.parametrize("parser_name", ["qwen", "deepseek-v3"])
def test_wrong_argument_key_does_not_start_a_tool_call(parser_name):
    model = ChatModelMixin()
    model.tool_parser = TOOL_PARSERS[parser_name]()
    history, state = [""], {}
    opening, closing, value = CALLS[parser_name]
    opening = (
        opening.replace('"arguments"', '"parameters"')
        if parser_name == "qwen"
        else opening.replace('"parameters"', '"arguments"')
    )
    assert not deltas(feed(model, opening + value, history, state))
    assert not deltas(feed(model, closing, history, state))
    assert (
        feed(model, "", history, state, "stop")["choices"][0]["finish_reason"] == "stop"
    )


@pytest.mark.parametrize("suffix_in_same_chunk", [False, True])
def test_llama_python_literal_and_text_after_call(suffix_in_same_chunk):
    model = ChatModelMixin()
    model.tool_parser = TOOL_PARSERS["llama3"]()
    history, state = [""], {}
    suffix = "\nDone."
    first = '{"name":"write","parameters":{"enabled":True,"line":None'
    assert not deltas(feed(model, first, history, state))
    chunk = feed(model, "}}" + (suffix if suffix_in_same_chunk else ""), history, state)
    (call,) = deltas(chunk)
    assert call["index"] == 0
    assert json.loads(call["function"]["arguments"]) == {"enabled": True, "line": None}
    if not suffix_in_same_chunk:
        chunk = feed(model, suffix, history, state)
        assert not deltas(chunk)
    assert chunk["choices"][0]["delta"]["content"] == suffix
    final = feed(model, "", history, state, "stop")
    assert final["choices"][0]["finish_reason"] == "tool_calls"
    assert not deltas(final)


@pytest.mark.parametrize(
    ("parser_name", "tail"),
    [
        ("gemma", "<"),
        ("gemma", "<|"),
        ("deepseek-r1", "<"),
        ("deepseek-v3", chr(96)),
        ("deepseek-v3", chr(96) * 3),
        ("deepseek-v3", chr(96) * 3 + "j"),
    ],
)
@pytest.mark.parametrize("after_call", [False, True])
def test_partial_opening_marker_is_flushed_at_finish(parser_name, tail, after_call):
    model = ChatModelMixin()
    model.tool_parser = TOOL_PARSERS[parser_name]()
    history, state = [""], {}
    if after_call:
        opening, closing, value = CALLS[parser_name]
        assert deltas(feed(model, opening + value + closing, history, state))
    text = feed(model, "text" + tail, history, state)
    assert text["choices"][0]["delta"]["content"] == "text"
    final = feed(model, "", history, state, "stop")
    assert final["choices"][0]["delta"]["content"] == tail


def test_malformed_deepseek_call_cannot_consume_later_call():
    model = ChatModelMixin()
    model.tool_parser = TOOL_PARSERS["deepseek-v3.1"]()
    history, state = [""], {}
    opening, closing, value = CALLS["deepseek-v3.1"]
    (bad,) = deltas(feed(model, opening + value, history, state))
    assert bad["index"] == 0
    close = "<｜tool▁call▁end｜>"
    malformed = ",}" + close
    second = opening.removeprefix("<｜tool▁calls▁begin｜>") + value + closing
    (good,) = deltas(feed(model, malformed + second, history, state))
    assert good["index"] == 1 and good["id"] != bad["id"]
    assert json.loads(good["function"]["arguments"]) == {"code": CODE, "line": 7}
    assert (
        feed(model, "", history, state, "stop")["choices"][0]["finish_reason"]
        == "length"
    )


@pytest.mark.parametrize("raw", ["hello <b>world</b>", "if a<b"])
def test_minimax_m3_literal_angle_bracket_agrees_with_nonstream(raw):
    model = ChatModelMixin()
    model.tool_parser = TOOL_PARSERS["minimax_m3"]()
    history, state = [""], {}
    opening = '<minimax:tool_call><invoke name="write"><code>'
    closing = "</code></invoke></minimax:tool_call>"
    assert not deltas(feed(model, opening + raw.split("<", 1)[0], history, state))
    ((_, expected_name, expected_args),) = model.tool_parser.extract_tool_calls(
        opening + raw + closing
    )
    (call,) = deltas(feed(model, "<" + raw.split("<", 1)[1] + closing, history, state))
    assert call["function"]["name"] == expected_name
    assert json.loads(call["function"]["arguments"]) == expected_args


@pytest.mark.parametrize("parser_name", ["minimax", "minimax_m3", "seed-oss"])
def test_typed_first_invoke_wins_when_parallel_tool_calls_disabled(parser_name):
    model = ChatModelMixin()
    model.tool_parser = TOOL_PARSERS[parser_name]()
    history, state = [""], {}
    opening, closing, value = CALLS[parser_name]
    invoke = (
        "<function=write><parameter=line>1</parameter></function>"
        if parser_name == "seed-oss"
        else '<invoke name="write"><parameter name="line">1</parameter></invoke>'
    )
    pos = opening.index(">") + 1
    parts = [opening[:pos] + invoke + opening[pos:] + value, closing]
    first, last = [feed(model, piece, history, state) for piece in parts]
    assert not deltas(first)
    (filtered,) = single_tool_chunks(chunk for chunk in (last,))
    (call,) = deltas(filtered)
    assert call["index"] == 0
    assert json.loads(call["function"]["arguments"]) == {"line": 1}


def test_r1_preserves_invalid_call_content_next_to_valid_call():
    model = ChatModelMixin()
    model.tool_parser = TOOL_PARSERS["deepseek-r1"]()
    history, state = [""], {}
    opening, closing, value = CALLS["deepseek-r1"]
    wrapper = "<｜tool▁calls▁begin｜>"
    wrapper_end = "<｜tool▁calls▁end｜>"
    malformed = opening + value + ",}\n" + chr(96) * 3 + "<｜tool▁call▁end｜>"
    valid = opening.removeprefix(wrapper) + value + closing
    chunk = feed(model, malformed + valid, history, state)
    (call,) = deltas(chunk)
    assert json.loads(call["function"]["arguments"]) == {"code": CODE, "line": 7}
    assert chunk["choices"][0]["delta"]["content"] == '{"code":' + value + ",}"
    assert wrapper_end in valid


def test_deepseek_v3_deduplicates_by_values_not_only_keys():
    parser = TOOL_PARSERS["deepseek-v3"]()
    opening, closing, _ = CALLS["deepseek-v3"]
    first = opening + '"a"' + closing
    second = opening + '"b"' + closing
    events = parser.extract_tool_calls("\n".join((first, second, first)))
    assert [args["code"] for _, _, args in events] == ["a", "b"]
