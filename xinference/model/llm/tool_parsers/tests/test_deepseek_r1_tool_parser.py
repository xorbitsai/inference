from ..deepseek_r1_tool_parser import DeepseekR1ToolParser


def test_tool_parser_extract_calls_without_thinking():
    parser = DeepseekR1ToolParser()

    test_case = '<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n```json\n{"location": "上海"}\n```<｜tool▁call▁end｜>'

    expected_results = [(None, "get_current_weather", {"location": "上海"})]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results, f"Expected {expected_results}, but got {result}"


def test_tool_parser_extract_parallel_calls():
    parser = DeepseekR1ToolParser()

    test_case = (
        "<｜tool▁calls▁begin｜>"
        '<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n```json\n{"location": "北京"}\n```<｜tool▁call▁end｜>\n'
        '<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n```json\n{"location": "上海"}\n```<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜>"
    )

    expected_results = [
        (None, "get_current_weather", {"location": "北京"}),
        (None, "get_current_weather", {"location": "上海"}),
    ]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results, f"Expected {expected_results}, but got {result}"


def test_tool_parser_extract_calls_with_nested_arguments():
    parser = DeepseekR1ToolParser()

    test_case = (
        "<｜tool▁calls▁begin｜>"
        '<｜tool▁call▁begin｜>function<｜tool▁sep｜>send_email\n```json\n{"to": ["a@example.com", "b@example.com"], "options": {"urgent": true}}\n```<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜>"
    )

    expected_results = [
        (
            None,
            "send_email",
            {"to": ["a@example.com", "b@example.com"], "options": {"urgent": True}},
        )
    ]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results, f"Expected {expected_results}, but got {result}"
