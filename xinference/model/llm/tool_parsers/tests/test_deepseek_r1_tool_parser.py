from ..deepseek_r1_tool_parser import DeepseekR1ToolParser


def test_tool_parser_extract_calls_without_thinking():
    parser = DeepseekR1ToolParser()

    test_case = '<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n```json\n{"location": "上海"}\n```<｜tool▁call▁end｜>'

    expected_results = [(None, "get_current_weather", {"location": "上海"})]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results, f"Expected {expected_results}, but got {result}"
