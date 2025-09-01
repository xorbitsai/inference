from ..deepseek_v3_tool_parser import DeepseekV3ToolParser


def test_tool_parser_extract_calls_without_thinking():
    """测试工具解析器的流式提取功能"""
    parser = DeepseekV3ToolParser()

    # 基于真实调试输出的测试用例
    test_case = '```json{"name": "get_weather_and_time","parameters": {"location": "Hangzhou"}}```'

    print(f"\n=== 工具解析器提取测试 ===")

    # 预期结果
    expected_results = [(None, "get_weather_and_time", {"location": "Hangzhou"})]

    result = parser.extract_tool_calls(test_case)

    # 校验结果
    assert result == expected_results
