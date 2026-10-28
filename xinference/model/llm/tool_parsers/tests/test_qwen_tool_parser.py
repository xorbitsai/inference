# CI: linux
import json
import re

import pytest

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
        (None, "get_current_weather", None, 0),
        None,
        None,
        None,
        None,
        None,
        None,
        (None, "get_current_weather", {"location": "上海"}, 0),  # 案例25: 完整工具调用
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
        (None, "get_current_weather", None, 1),
        None,
        None,
        None,
        None,
        None,
        None,
        (None, "get_current_weather", {"location": "北京"}, 1),  # 案例25: 完整工具调用
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
        (None, "get_current_weather", None, 0),
        None,
        None,
        None,
        None,
        None,
        None,
        (None, "get_current_weather", {"location": "上海"}, 0),  # 案例25: 完整工具调用
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
        (None, "get_current_weather", None, 0),
        None,
        None,
        None,
        None,
        None,
        None,
        (None, "get_current_weather", {"location": "上海"}, 0),  # 案例25: 完整工具调用
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
        (None, "get_current_weather", None, 0),
        None,
        None,
        None,
        None,
        None,
        None,
        (None, "get_current_weather", {"location": "上海"}, 0),  # 案例25: 完整工具调用
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


@pytest.mark.parametrize(
    "text",
    [
        # Reasoning left in the output when reasoning_content is disabled.
        "The user wants the weather, I will call the tool.\n</think>\n\n",
        # Plain text that merely mentions the Qwen3.5 XML call syntax.
        "Tools are called with <function=get_current_weather>.\n",
    ],
)
def test_tool_parser_extract_calls_keeps_text_outside_tool_call(text, caplog):
    parser = QwenToolParser()

    test_case = (
        text + '<tool_call>\n{"name": "get_current_weather", '
        '"arguments": {"location": "上海"}}\n</tool_call>'
    )

    with caplog.at_level("WARNING"):
        result = parser.extract_tool_calls(test_case)

    assert result == [
        (text, None, None),
        (None, "get_current_weather", {"location": "上海"}),
    ]
    assert not caplog.records


def test_streaming_xml_tool_call_emits_function_name_before_arguments_complete():
    parser = QwenToolParser()

    prefix = "I need current facts.\n"
    function_only = prefix + "<tool_call>\n<function=select_execution_pattern>\n"
    first = function_only + "<parameter=action>\n"
    second = first + "react\n</parameter>\n"
    complete = second + "</function>\n</tool_call>"

    assert (
        parser.extract_tool_calls_streaming(
            [prefix], function_only, function_only[len(prefix) :]
        )
        is None
    )
    assert parser.extract_tool_calls_streaming(
        [function_only], first, first[len(function_only) :]
    ) == (
        None,
        "select_execution_pattern",
        None,
        0,
    )
    assert (
        parser.extract_tool_calls_streaming([first], second, second[len(first) :])
        is None
    )
    assert parser.extract_tool_calls_streaming(
        [second], complete, complete[len(second) :]
    ) == (None, "select_execution_pattern", {"action": "react"}, 0)


def test_streaming_tool_call_preserves_content_before_tag_in_same_delta():
    parser = QwenToolParser()
    prefix = "I should check.\n"
    current = (
        prefix
        + '<tool_call>\n{"name": "web_search", '
        + '"arguments": {"query": "latest news"}}\n</tool_call>'
    )

    assert parser.extract_tool_calls_streaming([""], current, current) == [
        (prefix, None, None),
        (None, "web_search", {"query": "latest news"}, 0),
    ]


def test_streaming_xml_tool_calls_keep_completed_call_before_next_partial_call():
    parser = QwenToolParser()
    first_partial = "analysis\n<tool_call>\n<function=web_search>\n<parameter=query>\n"
    first_complete_second_partial = (
        first_partial
        + "first\n</parameter>\n</function>\n</tool_call>\n"
        + "<tool_call>\n<function=web_search>\n<parameter=query>\n"
    )
    both_complete = (
        first_complete_second_partial
        + "second\n</parameter>\n</function>\n</tool_call>"
    )

    assert parser.extract_tool_calls_streaming(
        ["analysis\n"], first_partial, first_partial[len("analysis\n") :]
    ) == (None, "web_search", None, 0)
    assert parser.extract_tool_calls_streaming(
        [first_partial],
        first_complete_second_partial,
        first_complete_second_partial[len(first_partial) :],
    ) == [
        (None, "web_search", {"query": "first"}, 0),
        (None, "web_search", None, 1),
    ]
    assert parser.extract_tool_calls_streaming(
        [first_complete_second_partial],
        both_complete,
        both_complete[len(first_complete_second_partial) :],
    ) == (None, "web_search", {"query": "second"}, 1)


def test_streaming_tool_calls_keep_inter_call_content_in_generation_order():
    parser = QwenToolParser()
    first_partial = (
        '<tool_call>\n{"name": "web_search", ' '"arguments": {"query": "first"}}\n'
    )
    inter_call_content = " explanation "
    delta = (
        "</tool_call>"
        + inter_call_content
        + '<tool_call>\n{"name": "web_search", "arguments":'
    )
    current = first_partial + delta

    assert parser.extract_tool_calls_streaming([first_partial], current, delta) == [
        (None, "web_search", {"query": "first"}, 0),
        (inter_call_content, None, None),
        (None, "web_search", None, 1),
    ]

    complete_delta = (
        "</tool_call>"
        + inter_call_content
        + '<tool_call>\n{"name": "web_search", '
        + '"arguments": {"query": "second"}}\n</tool_call>'
    )
    complete_current = first_partial + complete_delta

    assert parser.extract_tool_calls_streaming(
        [first_partial], complete_current, complete_delta
    ) == [
        (None, "web_search", {"query": "first"}, 0),
        (inter_call_content, None, None),
        (None, "web_search", {"query": "second"}, 1),
    ]


@pytest.mark.parametrize("malformed_call", ['{"name": "bad"}', "oops"])
def test_streaming_tool_calls_skip_malformed_sibling(malformed_call):
    parser = QwenToolParser()
    current = (
        f"<tool_call>{malformed_call}</tool_call>"
        "<tool_call><function=good><parameter=x>1</parameter>"
        "</function></tool_call>"
    )

    assert parser.extract_tool_calls_streaming([""], current, current) == (
        None,
        "good",
        {"x": 1},
        1,
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "plain text",
        "<tool_call>",
        "<tool_call>\n",
        '<tool_call>\n{"name": "f", "arguments": {}}\n',
        '<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>',
        "a</tool_call>b<tool_call>c",
        "<tool_call>a<tool_call>b</tool_call>c</tool_call>",
        "<tool_call>a</tool_call>\n<tool_call>b</tool_call>\n<tool_call>c\n",
        "<tool_call><tool_call></tool_call></tool_call><tool_call>",
    ],
)
def test_scan_tool_calls_matches_regexes(text):
    parser = QwenToolParser()
    blocks, partial_start = parser._scan_tool_calls(text)

    start_len = len(parser.tool_call_start_token)
    end_len = len(parser.tool_call_end_token)
    assert [
        text[start + start_len : end - end_len] for start, end in blocks
    ] == parser.tool_call_complete_regex.findall(text)

    function_calls = parser._get_function_calls_streaming(text)
    assert parser._last_function_call(text, blocks, partial_start) == (
        function_calls[-1] if function_calls else ""
    )

    tags = [
        (match.start(), match.end(), match.group() == parser.tool_call_start_token)
        for match in re.finditer(r"<tool_call>|</tool_call>", text)
    ]
    assert list(parser._iter_tool_call_tags(text)) == tags


def test_streaming_long_tool_call_in_small_deltas():
    parser = QwenToolParser()
    arguments = {"path": "main.py", "content": "print(1)\n" * 500}
    call = json.dumps({"name": "write_file", "arguments": arguments})
    # The tags are single tokens; the arguments arrive a few characters at a time.
    deltas = ["Let me write it.", "<tool_call>", "\n"]
    deltas += [call[i : i + 3] for i in range(0, len(call), 3)]
    deltas += ["\n", "</tool_call>"]

    events = []
    previous_texts = [""]
    for delta in deltas:
        current_text = previous_texts[-1] + delta
        result = parser.extract_tool_calls_streaming(
            previous_texts, current_text, delta
        )
        previous_texts[-1] = current_text
        if result is not None:
            events.extend(result if isinstance(result, list) else [result])

    assert events == [
        ("Let me write it.", None, None),
        (None, "write_file", None, 0),
        (None, "write_file", arguments, 0),
    ]


def _baichuan_m2():
    from ...llm_family import BUILTIN_LLM_FAMILIES

    return next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "Baichuan-M2")


def test_baichuan_m2_template_tool_calls_parse_with_the_qwen_parser():
    from types import SimpleNamespace

    from ...utils import QWEN_TOOL_CALL_FAMILY, ChatModelMixin
    from .. import TOOL_PARSERS

    family = _baichuan_m2()
    assert "tools" in family.model_ability
    assert family.tool_parser == "qwen"
    assert family.model_name in QWEN_TOOL_CALL_FAMILY

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    messages = [
        {"role": "user", "content": "Weather in Beijing?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Beijing"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "Sunny"},
    ]
    mixin = ChatModelMixin()
    mixin.model_family = SimpleNamespace(
        model_name=family.model_name,
        model_family=None,
        model_ability=family.model_ability,
    )

    # The family's own chat template carries the tool definitions and writes a
    # tool call using the format declared by the template.
    prompt = mixin.get_full_context(messages, family.chat_template, tools=tools)

    call = (
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        "\n</tool_call>"
    )
    assert f"<tools>\n{json.dumps(tools[0])}\n</tools>" in prompt
    assert call in prompt
    assert "<tool_response>\nSunny\n</tool_response>" in prompt
    parser = TOOL_PARSERS[family.tool_parser]()
    assert parser.extract_tool_calls(call) == [
        (None, "get_weather", {"city": "Beijing"})
    ]


def test_baichuan_m2_reasoning_then_tool_call_output():
    from ...reasoning_parser import ReasoningParser
    from ...utils import ChatModelMixin
    from .. import TOOL_PARSERS

    family = _baichuan_m2()
    mixin = ChatModelMixin()
    mixin.tool_parser = TOOL_PARSERS[family.tool_parser]()
    mixin.reasoning_parser = ReasoningParser(
        reasoning_content=True,
        reasoning_start_tag=family.reasoning_start_tag,
        reasoning_end_tag=family.reasoning_end_tag,
        enable_thinking=True,
        auto_insert_start_tag="hybrid" not in family.model_ability,
    )
    completion = {
        "id": "cmpl-1",
        "object": "text_completion",
        "created": 0,
        "model": "Baichuan-M2",
        "choices": [
            {
                "text": (
                    "<think>\nI need the weather, so I call get_weather.\n</think>\n\n"
                    '<tool_call>\n{"name": "get_weather", '
                    '"arguments": {"city": "Beijing"}}\n</tool_call>'
                ),
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }

    choice = mixin._post_process_completion(
        None, model_uid="Baichuan-M2", c=completion
    )["choices"][0]

    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["reasoning_content"] == (
        "\nI need the weather, so I call get_weather.\n"
    )
    (tool_call,) = choice["message"]["tool_calls"]
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": "Beijing"}
