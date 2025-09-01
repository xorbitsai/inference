from ..deepseek_v3_tool_parser import DeepseekV3ToolParser


def test_tool_parser_extract_calls_without_thinking():
    parser = DeepseekV3ToolParser()

    test_case = '```json\n{"name": "get_weather_and_time", "parameters": {"location": "Hangzhou"}}\n```'

    expected_results = [(None, "get_weather_and_time", {"location": "Hangzhou"})]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results, f"Expected {expected_results}, but got {result}"
