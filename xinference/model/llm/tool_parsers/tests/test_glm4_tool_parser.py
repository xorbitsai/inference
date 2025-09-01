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
