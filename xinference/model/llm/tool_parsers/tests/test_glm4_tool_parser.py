from ..glm4_tool_parser import Glm4ToolParser

def test_tool_parser_extract_calls():
    """测试工具解析器的流式提取功能"""
    parser = Glm4ToolParser()

    # 基于真实调试输出的测试用例
    test_case = '\n```json\n{\"name\": \"get_current_weather\", \"arguments\": {\"location\": \"上海\"}}\n```'


    # 预期结果
    expected_results = [(None, 'get_current_weather', {'location': '上海'})]

    result = parser.extract_tool_calls(test_case)

    print(result)

    # 校验结果
    assert result == expected_results
