import pytest

from ..deepseek_v4_tool_parser import DeepseekV4ToolParser


def _tool_calls(invokes, dsml=True):
    if dsml:
        prefix = "<｜DSML｜"
        close_prefix = "｜DSML｜"
    else:
        prefix = "<"
        close_prefix = ""

    invoke_blocks = []
    for name, params in invokes:
        param_blocks = []
        for param_name, is_string, value in params:
            param_blocks.append(
                f'{prefix}parameter name="{param_name}" '
                f'string="{str(is_string).lower()}">{value}'
                f"</{close_prefix}parameter>"
            )
        invoke_blocks.append(
            f'{prefix}invoke name="{name}">'
            f'{"".join(param_blocks)}</{close_prefix}invoke>'
        )
    return f"{prefix}tool_calls>{''.join(invoke_blocks)}" f"</{close_prefix}tool_calls>"


@pytest.mark.parametrize("dsml", [True, False])
@pytest.mark.parametrize(
    "before, after", [("Checking. ", ""), ("", " Done."), (" \n", "\t")]
)
def test_nonstreaming_preserves_surrounding_content(dsml, before, after):
    block = _tool_calls([("weather", [("city", True, "上海")])], dsml=dsml)
    assert DeepseekV4ToolParser().extract_tool_calls(before + block + after) == [
        *([(before, None, None)] if before else []),
        (None, "weather", {"city": "上海"}),
        *([(after, None, None)] if after else []),
    ]


@pytest.mark.parametrize("between", [" Next. ", " \n"])
def test_nonstreaming_preserves_content_between_call_blocks(between):
    block = _tool_calls([("weather", [("city", True, "上海")])])
    text = block + between + block.replace("上海", "北京")
    assert DeepseekV4ToolParser().extract_tool_calls(text) == [
        (None, "weather", {"city": "上海"}),
        (between, None, None),
        (None, "weather", {"city": "北京"}),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "plain text",
        '<tool_calls><invoke name="weather">',
        "<tool_calls>broken</tool_calls>",
    ],
)
def test_nonstreaming_keeps_unparseable_content(text):
    assert DeepseekV4ToolParser().extract_tool_calls(text) == [(text, None, None)]


@pytest.mark.parametrize("dsml", [True, False])
def test_extract_tool_calls_preserves_json_parameter_types(dsml):
    parser = DeepseekV4ToolParser()
    model_output = _tool_calls(
        [
            (
                "typed_args",
                [
                    ("text", True, "001"),
                    ("integer", False, "42"),
                    ("float", False, "3.5"),
                    ("boolean", False, "true"),
                    ("nothing", False, "null"),
                    ("array", False, '[1, "two"]'),
                    ("object", False, '{"nested": false}'),
                ],
            )
        ],
        dsml=dsml,
    )

    assert parser.extract_tool_calls(model_output) == [
        (
            None,
            "typed_args",
            {
                "text": "001",
                "integer": 42,
                "float": 3.5,
                "boolean": True,
                "nothing": None,
                "array": [1, "two"],
                "object": {"nested": False},
            },
        )
    ]


def test_extract_tool_calls_handles_multiple_invokes_and_parameters():
    parser = DeepseekV4ToolParser()
    model_output = _tool_calls(
        [
            ("first", [("city", True, "杭州"), ("days", False, "3")]),
            ("second", [("enabled", False, "false")]),
        ]
    )

    assert parser.extract_tool_calls(model_output) == [
        (None, "first", {"city": "杭州", "days": 3}),
        (None, "second", {"enabled": False}),
    ]


def test_extract_tool_calls_returns_plain_text_unchanged():
    parser = DeepseekV4ToolParser()
    text = "This is a normal response without tool calls."

    assert parser.extract_tool_calls(text) == [(text, None, None)]


def test_extract_tool_calls_handles_malformed_non_string_json():
    parser = DeepseekV4ToolParser()
    model_output = _tool_calls([("broken", [("payload", False, '{"missing": }')])])

    assert parser.extract_tool_calls(model_output) == [
        (None, "broken", {"payload": '{"missing": }'})
    ]


def test_extract_tool_calls_streaming_waits_then_returns_typed_parameters():
    parser = DeepseekV4ToolParser()
    incomplete = (
        '<｜DSML｜tool_calls><｜DSML｜invoke name="typed_args">'
        '<｜DSML｜parameter name="count" string="false">7'
    )
    complete = incomplete + (
        '</｜DSML｜parameter><｜DSML｜parameter name="label" string="true">007'
        "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>"
    )

    assert parser.extract_tool_calls_streaming([], incomplete, incomplete) is None
    assert parser.extract_tool_calls_streaming(
        [incomplete], complete, complete[len(incomplete) :]
    ) == (None, "typed_args", {"count": 7, "label": "007"})


def test_extract_tool_calls_streaming_returns_multiple_invokes_once_each():
    parser = DeepseekV4ToolParser()
    first = _tool_calls([("first", [("value", False, "1")])])
    second_invoke = (
        '<｜DSML｜invoke name="second">'
        '<｜DSML｜parameter name="value" string="false">2'
        "</｜DSML｜parameter></｜DSML｜invoke>"
    )
    combined = first.replace(
        "</｜DSML｜tool_calls>", second_invoke + "</｜DSML｜tool_calls>"
    )

    assert parser.extract_tool_calls_streaming([], first, first) == (
        None,
        "first",
        {"value": 1},
    )
    assert parser.extract_tool_calls_streaming([first], combined, second_invoke) == (
        None,
        "second",
        {"value": 2},
    )


def _stream(parser, history, chunk):
    current = history[-1] + chunk
    result = parser.extract_tool_calls_streaming(history, current, chunk)
    history[-1] = current
    if result is None:
        return []
    return result if isinstance(result, list) else [result]


@pytest.mark.parametrize("chunk_size", [1, 7, 64, 1024])
def test_extract_tool_calls_streaming_keeps_interleaved_requests_apart(chunk_size):
    # One parser instance serves every request to the model, so concurrent
    # streams feed it their chunks in turn.
    parser = DeepseekV4ToolParser()
    texts = [
        _tool_calls([("a1", [("n", False, "1")]), ("a2", [("n", False, "2")])]),
        _tool_calls([("b1", [("city", True, "杭州")]), ("b2", [])]),
        "1 < 2.",
    ]
    histories = [[""] for _ in texts]
    events = [[] for _ in texts]
    for offset in range(0, max(map(len, texts)), chunk_size):
        for text, history, received in zip(texts, histories, events):
            chunk = text[offset : offset + chunk_size]
            received.extend(_stream(parser, history, chunk))

    assert events[0] == [(None, "a1", {"n": 1}), (None, "a2", {"n": 2})]
    assert events[1] == [(None, "b1", {"city": "杭州"}), (None, "b2", {})]
    assert "".join(event[0] for event in events[2]) == "1 < 2."


def _stream_calls(parser, deltas):
    previous_texts = [""]
    calls = []
    for delta in deltas:
        current_text = previous_texts[-1] + delta
        result = parser.extract_tool_calls_streaming(
            previous_texts, current_text, delta
        )
        previous_texts[-1] = current_text
        if result is None:
            continue
        for event in result if isinstance(result, list) else [result]:
            if event[1] is not None:
                calls.append(event[:3])
    return calls


def test_streaming_long_invoke_in_small_deltas():
    parser = DeepseekV4ToolParser()
    code = "print(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)\nprint(1)"
    deltas = ["Writing.", "<｜DSML｜tool_calls>"]
    for path, content in (("main.py", code), ("b.py", "x = 1")):
        deltas += ['<｜DSML｜invoke name="write_file">']
        deltas += ['<｜DSML｜parameter name="path" string="true">', path]
        deltas += ["</｜DSML｜parameter>"]
        deltas += ['<｜DSML｜parameter name="content" string="true">']
        deltas += [content[i : i + 3] for i in range(0, len(content), 3)]
        deltas += ["</｜DSML｜parameter>", "</｜DSML｜invoke>"]
    deltas += ["</｜DSML｜tool_calls>"]

    assert _stream_calls(parser, deltas) == [
        (None, "write_file", {"path": "main.py", "content": code}),
        (None, "write_file", {"path": "b.py", "content": "x = 1"}),
    ]
