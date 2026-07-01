from ..glm5_tool_parser import Glm5ToolParser


def test_tool_parser_extract_xml_style_calls():
    parser = Glm5ToolParser()

    test_case = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Beijing</arg_value></tool_call>"

    expected_results = [(None, "get_weather", {"location": "Beijing"})]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results, f"Expected {expected_results}, but got {result}"


def test_tool_parser_preserves_surrounding_text():
    parser = Glm5ToolParser()

    test_case = (
        "Before "
        "<tool_call>get_weather<arg_key>location</arg_key>"
        "<arg_value>Beijing</arg_value></tool_call>"
        " after"
    )

    result = parser.extract_tool_calls(test_case)

    assert result == [
        ("Before ", None, None),
        (None, "get_weather", {"location": "Beijing"}),
        (" after", None, None),
    ]


def test_tool_parser_extract_no_arg_call():
    parser = Glm5ToolParser()

    result = parser.extract_tool_calls("<tool_call>get_time</tool_call>")

    assert result == [(None, "get_time", {})]


def test_tool_parser_extract_json_arg_values():
    parser = Glm5ToolParser()

    test_case = (
        "<tool_call>search<arg_key>query</arg_key><arg_value>Beijing</arg_value>"
        "<arg_key>limit</arg_key><arg_value>3</arg_value>"
        '<arg_key>filters</arg_key><arg_value>{"unit": "celsius"}</arg_value>'
        "</tool_call>"
    )

    result = parser.extract_tool_calls(test_case)

    assert result == [
        (
            None,
            "search",
            {"query": "Beijing", "limit": 3, "filters": {"unit": "celsius"}},
        )
    ]


def test_tool_parser_extract_xml_style_calls_streaming():
    parser = Glm5ToolParser()

    test_cases = [
        ([""], "<tool_call>get_weather", "<tool_call>get_weather"),
        (
            ["<tool_call>get_weather"],
            "<tool_call>get_weather<arg_key>location</arg_key>",
            "<arg_key>location</arg_key>",
        ),
        (
            ["<tool_call>get_weather<arg_key>location</arg_key>"],
            "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Beijing</arg_value>",
            "<arg_value>Beijing</arg_value>",
        ),
        (
            [
                "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Beijing</arg_value>"
            ],
            "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Beijing</arg_value></tool_call>",
            "</tool_call>",
        ),
        (
            [
                "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Beijing</arg_value></tool_call>"
            ],
            "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Beijing</arg_value></tool_call>",
            "",
        ),
    ]

    expected_results = [
        None,
        None,
        None,
        (None, "get_weather", {"location": "Beijing"}),
        None,
    ]

    for i, (previous_text, current_text, delta_text) in enumerate(test_cases):
        result = parser.extract_tool_calls_streaming(
            previous_text, current_text, delta_text
        )
        expected = expected_results[i]

        assert result == expected, f"Case {i} failed: {result} != {expected}"


def test_tool_parser_streaming_preserves_text_after_processed_call():
    parser = Glm5ToolParser()

    previous_text = [
        "<tool_call>get_weather<arg_key>location</arg_key>"
        "<arg_value>Beijing</arg_value></tool_call>"
    ]
    current_text = previous_text[-1] + " done"

    result = parser.extract_tool_calls_streaming(previous_text, current_text, " done")

    assert result == (" done", None, None)


def test_tool_parser_streaming_suppresses_split_start_token():
    parser = Glm5ToolParser()

    result = parser.extract_tool_calls_streaming([""], "<tool", "<tool")

    assert result is None

    result = parser.extract_tool_calls_streaming([""], "Before <tool", "Before <tool")

    assert result == ("Before ", None, None)

    result = parser.extract_tool_calls_streaming(
        ["<tool"],
        "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Beijing</arg_value>",
        "_call>get_weather<arg_key>location</arg_key><arg_value>Beijing</arg_value>",
    )

    assert result is None
