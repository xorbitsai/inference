import json
import logging
import re
from typing import TYPE_CHECKING

from . import register_tool_parser
from .abstract_tool_parser import ToolParser

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


@register_tool_parser("qwen")
class QwenToolParser(ToolParser):
    """
    Qwen 模型的工具解析器实现
    """

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)

        # Sentinel tokens for streaming mode
        self.think_start_token: str = "<think>"
        self.think_end_token: str = "</think>"
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"

        # Regex patterns
        self.think_regex = re.compile("<think>(.*?)</think>", re.DOTALL)
        self.content_regex = r"(<(think|tool_call)>.*?</\2>)"
        self.tool_call_complete_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.tool_call_regex = re.compile(
            r"<tool_call>.*?</tool_call>|<tool_call>.*?$", re.DOTALL)


    def _parse_json_function_call(
            self, function_call_str: str,
    ):
        # Extract function name
        function_calls = self.tool_call_complete_regex.findall(function_call_str)
        if len(function_calls) == 0:
            return function_call_str
        return function_calls[-1]

    
    def _parse_json_function_call_stream(
            self, function_call_str: str,
    ):
        # Extract function name
        function_calls = self.tool_call_complete_regex.findall(function_call_str)
        if len(function_calls) == 0:
            return None
        return function_calls[-1]

    def is_contain_think_end_token(self, model_output: str) -> bool:
        return self.think_end_token in model_output

    def is_contain_think(self, model_output: str) -> bool:
        return self.think_regex.search(model_output) is not None

    def is_contain_tool_call(self, model_output: str) -> bool:
        return self.tool_call_complete_regex.search(model_output) is not None

    def is_contain_tool_call_start_token(self, model_output: str) -> bool:
        return self.tool_call_start_token in model_output

    def is_contain_tool_call_end_token(self, model_output: str) -> bool:
        return self.tool_call_end_token in model_output

    def _get_function_calls(self, model_output: str) -> list[str]:
        # Find all tool calls
        functions_calls = []
        last_end = 0
        for m in re.finditer(self.content_regex, model_output, re.DOTALL):
            if m.start() > last_end:
                functions_calls.append(model_output[last_end : m.start()])
            functions_calls.append(m.group(0))
            last_end = m.end()
        if last_end < len(model_output):
            functions_calls.append(model_output[last_end:])
        return functions_calls

    
    def _get_function_calls_streaming(self, model_output: str) -> list[str]:
        # Find all tool calls
        matched_ranges = self.tool_call_regex.findall(model_output)
        return matched_ranges
                
    
    def extract_tool_calls(self, model_output: str):
        """
        从完整的模型输出中提取工具调用信息
        """
        if self.tool_call_start_token not in model_output:
            return model_output
        try:
            function_calls = self._get_function_calls(model_output)
            if len(function_calls) == 0:
                return None
            
            results = []
            for function_call in function_calls:
                try:
                    function_call = self._parse_json_function_call(function_call)
                    res = json.loads(function_call, strict=False)
                    results.append((None, res["name"], res["arguments"]))
                except Exception as e:
                    logger.error(
                        "Can't parse single qwen tool call output: %s. Error: %s",
                        function_call,
                        e,
                    )
                    results.append((function_call, None, None))
            return results
        except Exception as e:
            logger.error(
                "Can't parse qwen tool call output: %s. Error: %s",
                model_output,
                e,
            )
            return None
    
    
    def _has_unclosed_tool_call(self, text: str) -> bool:
        """
        检查文本中是否有未闭合的tool_call标签
        """
        start_count = text.count(self.tool_call_start_token)
        end_count = text.count(self.tool_call_end_token)
        return start_count > end_count

    def extract_tool_calls_streaming(self, previous_text, current_text: str, 
                                   delta_text: str):
        """
        从流式输出中提取工具调用信息
        """
        try:
            # 如果本轮输出包含<tool_call>开始标记且有未闭合的标签
            if self.is_contain_tool_call_start_token(current_text):
                function_calls = self._get_function_calls_streaming(current_text)
                # 无法解析出完整的tool_call结构
                if self.is_contain_think(function_calls[-1]):
                    return None
                # 如果上一轮的tool_call标签已经闭合，说明这是新的工具调用
                if not self._has_unclosed_tool_call(previous_text[-1]):
                    return None
                # 解析并返回
                function_call = self._parse_json_function_call_stream(function_calls[-1])
                res = json.loads(function_call, strict=False)
                return None, res["name"], res["arguments"]
            else:
                return delta_text, None, None
        except Exception as e:
            return None
