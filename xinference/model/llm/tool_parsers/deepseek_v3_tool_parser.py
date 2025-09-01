import json
import logging
import re
from typing import TYPE_CHECKING

from . import register_tool_parser
from .abstract_tool_parser import ToolParser

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


@register_tool_parser("deepseek-v3")
class DeepseekV3ToolParser(ToolParser):
    """
    Deepseek V3 模型的工具解析器实现
    """

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)

        self.tool_calls_regex = "\s*```json\s*(.*?)\s*```"

    def _parse_json_function_call(
            self, function_call_str: str,
    ):
        # Extract function name
        match = self.json_regex.search(function_call_str)

        if match:
            result = match.group(1)
            return result
        return function_call_str

    
    def _parse_json_function_call_stream(
            self, function_call_str: str,
    ):
        # Extract function name
        match = self.json_regex.search(function_call_str)

        if match:
            result = match.group(1)
            return result
        return None
                
    
    def extract_tool_calls(self, model_output: str):
        """
        从完整的模型输出中提取工具调用信息
        """
        matches = re.findall(self.tool_calls_regex, model_output, re.DOTALL)

        if not matches:
            return [(model_output, None, None)]

        tool_calls = set()  # Used for deduplication
        results = []

        for raw_json in matches:
            func_and_args = None
            try:
                func_and_args = json.loads(raw_json)
                # Convert dictionary to frozenset for deduplication
                arguments_hashable = frozenset(func_and_args["parameters"])
                tool_call_tuple = (
                    None,
                    func_and_args["name"],
                    func_and_args["parameters"],
                )
            except json.JSONDecodeError:
                tool_call_tuple = (
                    raw_json,
                    None,
                    None,
                )  # If parsing fails, treat as raw content
                arguments_hashable = None  # No need for hashing

            # Avoid duplicate entries
            dedup_key = (
                (func_and_args["name"], arguments_hashable)
                if func_and_args is not None
                else (raw_json)
            )
            if dedup_key not in tool_calls:
                tool_calls.add(dedup_key)
                results.append(tool_call_tuple)

        return results


    def extract_tool_calls_streaming(self, previous_text, current_text: str, 
                                   delta_text: str):
        """
        从流式输出中提取工具调用信息
        """
        raise ValueError("Streaming support for tool calls is available only when using Qwen models with vLLM backend or GLM4-chat models without vLLM backend.")
