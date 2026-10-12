from ..glm4_tool_parser import Glm4ToolParser


def test_tool_parser_extract_calls():
    parser = Glm4ToolParser()

    test_case = {"name": "get_current_weather", "arguments": {"location": "上海"}}

    expected_results = [(None, "get_current_weather", {"location": "上海"})]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results, f"Expected {expected_results}, but got {result}"


def test_tool_parser_extract_calls_streaming():
    parser = Glm4ToolParser()

    test_case = {"name": "get_current_weather", "arguments": {"location": "上海"}}

    expected_results = (None, "get_current_weather", {"location": "上海"})

    result = parser.extract_tool_calls_streaming(None, test_case, test_case)

    assert result == expected_results, f"Expected {expected_results}, but got {result}"


def test_native_glm_call_and_streamed_json():
    parser = Glm4ToolParser()
    text = 'Read\n{"file_path":"hello.py"}'
    expected = (None, "Read", {"file_path": "hello.py"})
    assert parser.extract_tool_calls(text) == [expected]
    assert (
        parser.extract_tool_calls_streaming(
            ['Read\n{"file_path":'], text, '"hello.py"}'
        )
        == expected
    )
    assert parser.extract_tool_calls_streaming([text], text, "") is None
    assert (
        parser.extract_tool_calls_streaming(
            ["Read\n{"], 'Read\n{"file_path":', '"file_path":'
        )
        is None
    )
    assert parser.extract_tool_calls("ordinary answer") == [
        ("ordinary answer", None, None)
    ]
