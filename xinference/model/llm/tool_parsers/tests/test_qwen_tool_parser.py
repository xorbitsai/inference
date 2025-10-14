from ..qwen_tool_parser import QwenToolParser


def test_tool_parser_extract_calls_streaming_without_thinking_multi():
    parser = QwenToolParser()

    test_cases = [
        # (previous_texts, current_text, delta_text)
        (["<tool_call>"], "<tool_call>\n", "\n"),
        (["<tool_call>\n"], '<tool_call>\n{"', '{"'),
        (['<tool_call>\n{"'], '<tool_call>\n{"name', "name"),
        (['<tool_call>\n{"name'], '<tool_call>\n{"name":', '":'),
        (['<tool_cal>\n{"name":'], '<tool_call>\n{"name": "', ' "'),
        (['<tool_call>\n{"name": "'], '<tool_call>\n{"name": "get', "get"),
        (
            ['<tool_call>\n{"name": "get'],
            '<tool_call>\n{"name": "get_current',
            "_current",
        ),
        (
            ['<tool_call>\n{"name": "get_current'],
            '<tool_call>\n{"name": "get_current_weather',
            "_weather",
        ),
        (
            ['<tool_call>\n{"name": "get_current_weather'],
            '<tool_call>\n{"name": "get_current_weather",',
            '",',
        ),
        (
            ['<tool_call>\n{"name": "get_current_weather",'],
            '<tool_call>\n{"name": "get_current_weather", "',
            ' "',
        ),
        (
            ['<tool_call>\n{"name": "get_current_weather", "'],
            '<tool_call>\n{"name": "get_current_weather", "arguments',
            "arguments",
        ),
        (
            ['<tool_call>\n{"name": "get_current_weather", "arguments'],
            '<tool_call>\n{"name": "get_current_weather", "arguments":',
            '":',
        ),
        (
            ['<tool_call>\n{"name": "get_current_weather", "arguments":'],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"',
            ' {"',
        ),
        (
            ['<tool_call>\n{"name": "get_current_weather", "arguments": {"'],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location',
            "location",
        ),
        (
            ['<tool_call>\n{"name": "get_current_weather", "arguments": {"location'],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location":',
            '":',
        ),
        (
            ['<tool_call>\n{"name": "get_current_weather", "arguments": {"location":'],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "',
            ' "',
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海',
            "上海",
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n',
            '"}}\n',
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call>',
            "</tool_call>",
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call>'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call>',
            "",
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n',
            "\n",
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"',
            '{"',
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name',
            "name",
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name":',
            '":',
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name":'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "',
            ' "',
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get',
            "get",
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current',
            "_current",
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather',
            "_weather",
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather",',
            '",',
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather",'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "',
            ' "',
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "arguments',
            "arguments",
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "arguments'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "arguments":',
            '":',
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "arguments":'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "arguments": {"',
            ' {"',
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "arguments": {"'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "arguments": {"location',
            "location",
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "arguments": {"location'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "arguments": {"location":',
            '":',
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "arguments": {"location":'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "arguments": {"location": "',
            ' "',
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "arguments": {"location": "'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "arguments": {"location": "北京',
            "北京",
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "arguments": {"location": "北京'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "arguments": {"location": "北京"}}\n',
            '"}}\n',
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "arguments": {"location": "北京"}}\n'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "arguments": {"location": "北京"}}\n</tool_call>',
            "</tool_call>",
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "arguments": {"location": "北京"}}\n</tool_call>'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "arguments": {"location": "北京"}}\n</tool_call>',
            "",
        ),
    ]

    expected_results = [
        None,  # 案例1-5: 普通文本输出
        None,
        None,
        None,
        None,
        None,  # 案例6-24: 工具调用构建中，返回None
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        (None, "get_current_weather", {"location": "上海"}),  # 案例25: 完整工具调用
        None,  # 案例26: 空增量文本
        None,  # 案例1-5: 普通文本输出
        None,
        None,
        None,
        None,
        None,  # 案例6-24: 工具调用构建中，返回None
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        (None, "get_current_weather", {"location": "北京"}),  # 案例25: 完整工具调用
        None,  # 案例26: 空增量文本
    ]

    for i, (previous_texts, current_text, delta_text) in enumerate(test_cases):
        result = parser.extract_tool_calls_streaming(
            previous_texts, current_text, delta_text
        )
        expected = expected_results[i]

        assert result == expected, f"Case {i} failed: {result} != {expected}"


def test_tool_parser_extract_calls_streaming_without_thinking():
    parser = QwenToolParser()

    test_cases = [
        # (previous_texts, current_text, delta_text)
        (["<tool_call>"], "<tool_call>\n", "\n"),
        (["<tool_call>\n"], '<tool_call>\n{"', '{"'),
        (['<tool_call>\n{"'], '<tool_call>\n{"name', "name"),
        (['<tool_call>\n{"name'], '<tool_call>\n{"name":', '":'),
        (['<tool_call>\n{"name":'], '<tool_call>\n{"name": "', ' "'),
        (['<tool_call>\n{"name": "'], '<tool_call>\n{"name": "get', "get"),
        (
            ['<tool_call>\n{"name": "get'],
            '<tool_call>\n{"name": "get_current',
            "_current",
        ),
        (
            ['<tool_call>\n{"name": "get_current'],
            '<tool_call>\n{"name": "get_current_weather',
            "_weather",
        ),
        (
            ['<tool_call>\n{"name": "get_current_weather'],
            '<tool_call>\n{"name": "get_current_weather",',
            '",',
        ),
        (
            ['<tool_call>\n{"name": "get_current_weather",'],
            '<tool_call>\n{"name": "get_current_weather", "',
            ' "',
        ),
        (
            ['<tool_call>\n{"name": "get_current_weather", "'],
            '<tool_call>\n{"name": "get_current_weather", "arguments',
            "arguments",
        ),
        (
            ['<tool_call>\n{"name": "get_current_weather", "arguments'],
            '<tool_call>\n{"name": "get_current_weather", "arguments":',
            '":',
        ),
        (
            ['<tool_call>\n{"name": "get_current_weather", "arguments":'],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"',
            ' {"',
        ),
        (
            ['<tool_call>\n{"name": "get_current_weather", "arguments": {"'],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location',
            "location",
        ),
        (
            ['<tool_call>\n{"name": "get_current_weather", "arguments": {"location'],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location":',
            '":',
        ),
        (
            ['<tool_call>\n{"name": "get_current_weather", "arguments": {"location":'],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "',
            ' "',
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海',
            "上海",
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n',
            '"}}\n',
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call>',
            "</tool_call>",
        ),
        (
            [
                '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call>'
            ],
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call>',
            "",
        ),
    ]

    expected_results = [
        None,  # 案例1-5: 普通文本输出
        None,
        None,
        None,
        None,
        None,  # 案例6-24: 工具调用构建中，返回None
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        (None, "get_current_weather", {"location": "上海"}),  # 案例25: 完整工具调用
        None,  # 案例26: 空增量文本
    ]

    for i, (previous_texts, current_text, delta_text) in enumerate(test_cases):
        result = parser.extract_tool_calls_streaming(
            previous_texts, current_text, delta_text
        )
        expected = expected_results[i]

        assert result == expected, f"Case {i} failed: {result} != {expected}"


def test_tool_parser_extract_calls_streaming_with_thinking():
    parser = QwenToolParser()

    test_cases = [
        # (previous_texts, current_text, delta_text)
        ([""], "<think>", "<think>"),
        (["<think>"], "<think>\n", "\n"),
        (["<think>\n"], "<think>\n好的", "好的"),
        (["<think>\n好的"], "<think>\n好的</think>", "</think>"),
        (["<think>\n好的</think>"], "<think>\n好的</think>\n\n", "\n\n"),
        (
            ["<think>\n好的</think>\n\n"],
            "<think>\n好的</think>\n\n<tool_call>",
            "<tool_call>",
        ),
        (
            ["<think>\n好的</think>\n\n<tool_call>"],
            "<think>\n好的</think>\n\n<tool_call>\n",
            "\n",
        ),
        (
            ["<think>\n好的</think>\n\n<tool_call>\n"],
            '<think>\n好的</think>\n\n<tool_call>\n{"',
            '{"',
        ),
        (
            ['<think>\n好的</think>\n\n<tool_call>\n{"'],
            '<think>\n好的</think>\n\n<tool_call>\n{"name',
            "name",
        ),
        (
            ['<think>\n好的</think>\n\n<tool_call>\n{"name'],
            '<think>\n好的</think>\n\n<tool_call>\n{"name":',
            '":',
        ),
        (
            ['<think>\n好的</think>\n\n<tool_call>\n{"name":'],
            '<think>\n好的</think>\n\n<tool_call>\n{"name": "',
            ' "',
        ),
        (
            ['<think>\n好的</think>\n\n<tool_call>\n{"name": "'],
            '<think>\n好的</think>\n\n<tool_call>\n{"name": "get',
            "get",
        ),
        (
            ['<think>\n好的</think>\n\n<tool_call>\n{"name": "get'],
            '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current',
            "_current",
        ),
        (
            ['<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current'],
            '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather',
            "_weather",
        ),
        (
            ['<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather'],
            '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather",',
            '",',
        ),
        (
            ['<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather",'],
            '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "',
            ' "',
        ),
        (
            ['<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "'],
            '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments',
            "arguments",
        ),
        (
            [
                '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments'
            ],
            '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments":',
            '":',
        ),
        (
            [
                '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments":'
            ],
            '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"',
            ' {"',
        ),
        (
            [
                '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"'
            ],
            '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location',
            "location",
        ),
        (
            [
                '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location'
            ],
            '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location":',
            '":',
        ),
        (
            [
                '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location":'
            ],
            '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "',
            ' "',
        ),
        (
            [
                '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "'
            ],
            '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海',
            "上海",
        ),
        (
            [
                '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海'
            ],
            '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n',
            '"}}\n',
        ),
        (
            [
                '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n'
            ],
            '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call>',
            "</tool_call>",
        ),
        (
            [
                '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call>'
            ],
            '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call>',
            "",
        ),
    ]

    expected_results = [
        ("<think>", None, None),  # 案例1-5: 普通文本输出
        ("\n", None, None),
        ("好的", None, None),
        ("</think>", None, None),
        ("\n\n", None, None),
        None,  # 案例6-24: 工具调用构建中，返回None
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        (None, "get_current_weather", {"location": "上海"}),  # 案例25: 完整工具调用
        None,  # 案例26: 空增量文本
    ]

    for i, (previous_texts, current_text, delta_text) in enumerate(test_cases):
        result = parser.extract_tool_calls_streaming(
            previous_texts, current_text, delta_text
        )
        expected = expected_results[i]

        assert result == expected, f"Case {i} failed: {result} != {expected}"


def test_tool_parser_extract_calls_streaming_with_parser():
    parser = QwenToolParser()

    test_cases = [
        # (previous_texts, current_text, delta_text)
        ([""], "\n\n", "\n\n"),
        (["\n\n"], "\n\n<tool_call>", "<tool_call>"),
        (["\n\n<tool_call>"], "\n\n<tool_call>\n", "\n"),
        (["\n\n<tool_call>\n"], '\n\n<tool_call>\n{"', '{"'),
        (['\n\n<tool_call>\n{"'], '\n\n<tool_call>\n{"name', "name"),
        (['\n\n<tool_call>\n{"name'], '\n\n<tool_call>\n{"name":', '":'),
        (['\n\n<tool_call>\n{"name":'], '\n\n<tool_call>\n{"name": "', ' "'),
        (['\n\n<tool_call>\n{"name": "'], '\n\n<tool_call>\n{"name": "get', "get"),
        (
            ['\n\n<tool_call>\n{"name": "get'],
            '\n\n<tool_call>\n{"name": "get_current',
            "_current",
        ),
        (
            ['\n\n<tool_call>\n{"name": "get_current'],
            '\n\n<tool_call>\n{"name": "get_current_weather',
            "_weather",
        ),
        (
            ['\n\n<tool_call>\n{"name": "get_current_weather'],
            '\n\n<tool_call>\n{"name": "get_current_weather",',
            '",',
        ),
        (
            ['\n\n<tool_call>\n{"name": "get_current_weather",'],
            '\n\n<tool_call>\n{"name": "get_current_weather", "',
            ' "',
        ),
        (
            ['\n\n<tool_call>\n{"name": "get_current_weather", "'],
            '\n\n<tool_call>\n{"name": "get_current_weather", "arguments',
            "arguments",
        ),
        (
            ['\n\n<tool_call>\n{"name": "get_current_weather", "arguments'],
            '\n\n<tool_call>\n{"name": "get_current_weather", "arguments":',
            '":',
        ),
        (
            ['\n\n<tool_call>\n{"name": "get_current_weather", "arguments":'],
            '\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"',
            ' {"',
        ),
        (
            ['\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"'],
            '\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location',
            "location",
        ),
        (
            [
                '\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location'
            ],
            '\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location":',
            '":',
        ),
        (
            [
                '\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location":'
            ],
            '\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "',
            ' "',
        ),
        (
            [
                '\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "'
            ],
            '\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海',
            "上海",
        ),
        (
            [
                '\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海'
            ],
            '\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n',
            '"}}\n',
        ),
        (
            [
                '\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n'
            ],
            '\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call>',
            "</tool_call>",
        ),
        (
            [
                '\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call>'
            ],
            '\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call>',
            "",
        ),
    ]

    expected_results = [
        ("\n\n", None, None),
        None,  # 案例6-24: 工具调用构建中，返回None
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        (None, "get_current_weather", {"location": "上海"}),  # 案例25: 完整工具调用
        None,  # 案例26: 空增量文本
    ]

    for i, (previous_texts, current_text, delta_text) in enumerate(test_cases):
        result = parser.extract_tool_calls_streaming(
            previous_texts, current_text, delta_text
        )
        expected = expected_results[i]

        assert result == expected, f"Case {i} failed: {result} != {expected}"


def test_tool_parser_extract_calls_without_thinking_multi():
    parser = QwenToolParser()

    test_case = '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call><tool_call>\n{"name": "get_current_weather", "arguments": {"location": "北京"}}\n</tool_call>'

    expected_results = [
        (None, "get_current_weather", {"location": "上海"}),
        (None, "get_current_weather", {"location": "北京"}),
    ]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results, f"Case failed: {result} != {expected_results}"


def test_tool_parser_extract_calls_without_thinking():
    parser = QwenToolParser()

    test_case = '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call>'

    expected_results = [(None, "get_current_weather", {"location": "上海"})]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results, f"Case failed: {result} != {expected_results}"


def test_tool_parser_extract_calls_with_thinking():
    parser = QwenToolParser()

    test_case = '<think>\n好的</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call>'

    expected_results = [
        ("<think>\n好的</think>", None, None),
        ("\n\n", None, None),
        (None, "get_current_weather", {"location": "上海"}),
    ]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results


def test_tool_parser_extract_calls_with_parser():
    parser = QwenToolParser()

    test_case = '\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call>'

    expected_results = [
        ("\n\n", None, None),
        (None, "get_current_weather", {"location": "上海"}),
    ]

    result = parser.extract_tool_calls(test_case)

    assert result == expected_results, f"Case failed: {result} != {expected_results}"
