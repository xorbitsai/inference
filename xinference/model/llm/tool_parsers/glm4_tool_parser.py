import json
import logging
import re
from typing import TYPE_CHECKING

from . import register_tool_parser
from .abstract_tool_parser import ToolParser

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


@register_tool_parser("glm4")
class Glm4ToolParser(ToolParser):
    """
    GLM4 模型的工具解析器实现
    """

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)

        self.json_regex = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)

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
        try:
            json_str = self._parse_json_function_call(model_output)
            parsed_output = json.loads(json_str, strict=False)
            if isinstance(parsed_output, dict):
                try:
                    return [(None, parsed_output["name"], json.loads(parsed_output["arguments"]))]
                except Exception:
                    return [(None, parsed_output["name"], parsed_output["arguments"])]
            else:
                return [(str(model_output), None, None)]
        except (json.JSONDecodeError, KeyError):
            logger.error("Can't parse glm output: %s", model_output)
            return [(str(model_output), None, None)]


    def extract_tool_calls_streaming(self, previous_text, current_text: str, 
                                   delta_text: str):
        """
        从流式输出中提取工具调用信息
        """
        raise ValueError("Streaming support for tool calls is available only when using Qwen models with vLLM backend or GLM4-chat models without vLLM backend.")
