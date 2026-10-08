import pytest

from ..deepseek_v4_tool_parser import DeepseekV42ToolParser


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
def test_extract_tool_calls_preserves_json_parameter_types(dsml):
    parser = DeepseekV42ToolParser()
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
    parser = DeepseekV42ToolParser()
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
    parser = DeepseekV42ToolParser()
    text = "This is a normal response without tool calls."

    assert parser.extract_tool_calls(text) == [(text, None, None)]


def test_extract_tool_calls_handles_malformed_non_string_json():
    parser = DeepseekV42ToolParser()
    model_output = _tool_calls([("broken", [("payload", False, '{"missing": }')])])

    assert parser.extract_tool_calls(model_output) == [
        (None, "broken", {"payload": '{"missing": }'})
    ]


def test_extract_tool_calls_streaming_waits_then_returns_typed_parameters():
    parser = DeepseekV42ToolParser()
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
    parser = DeepseekV42ToolParser()
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
