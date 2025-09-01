import json
import logging
import re
from typing import TYPE_CHECKING

from . import register_tool_parser
from .abstract_tool_parser import ToolParser

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


@register_tool_parser("llama3")
class Llama3ToolParser(ToolParser):
    """
    Qwen 模型的工具解析器实现
    """

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)

    
    def extract_tool_calls(self, model_output: str):
        """
        从完整的模型输出中提取工具调用信息
        """
        try:
            data = eval(model_output, {}, {})
            return [(None, data["name"], data["parameters"])]
        except Exception:
            return [(model_output, None, None)]


    def extract_tool_calls_streaming(self, previous_text, current_text: str, 
                                   delta_text: str):
        """
        从流式输出中提取工具调用信息
        """
        raise ValueError("Streaming support for tool calls is available only when using Qwen models with vLLM backend or GLM4-chat models without vLLM backend.")
