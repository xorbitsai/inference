from ..glm4_tool_parser import Glm4ToolParser


def test_tool_parser_extract_calls():
    parser = Glm4ToolParser()

    test_case = '\n```json\n{"name": "get_current_weather", "parameters": {"location": "上海"}}\n```'

    expected_results = [(None, "get_current_weather", {"location": "上海"})]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results, f"Expected {expected_results}, but got {result}"
