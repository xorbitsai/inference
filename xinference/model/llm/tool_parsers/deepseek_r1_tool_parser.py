import json
import logging
import re
from typing import Any, List, Optional, Tuple

from . import register_tool_parser
from .abstract_tool_parser import ToolParser

logger = logging.getLogger(__name__)


@register_tool_parser("deepseek-r1")
class DeepseekR1ToolParser(ToolParser):
    """
    Tool parser implementation for DeepSeek R1 model.

    This parser handles the specific format used by DeepSeek R1 for tool calls,
    which includes special Unicode tokens and JSON-formatted function arguments.
    """

    def __init__(self):
        """
        Initialize the DeepSeek R1 tool parser.
        """
        super().__init__()
        # Regex pattern to match DeepSeek R1 tool call format
        self.tool_calls_regex = (
            r"<\｜tool▁call▁begin｜>function<\｜tool▁sep｜>([^\n]+)\n"
            r"```json\n(.*?)\n```<\｜tool▁call▁end｜>"
        )

    def extract_tool_calls(
        self, model_output: str
    ) -> List[Tuple[Optional[str], Optional[str], Optional[dict]]]:
        """
        Extract tool calls from complete model output.

        Parses the model output to find tool call patterns and extracts
        function names and arguments. Handles JSON parsing errors gracefully
        and deduplicates identical tool calls.

        Args:
            model_output (str): The complete output string from the model.

        Returns:
            List[Tuple[Optional[str], Optional[str], Optional[dict]]]:
            A list of tuples where each tuple contains:
            - content (str or None): Raw content if parsing failed, None if successful
            - function_name (str or None): Name of the function to call
            - arguments (dict or None): Parsed function arguments

        Example:
            >>> parser = DeepseekR1ToolParser()
            >>> output = '<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n```json\n{"location": "上海", "unit": "celsius"}\n```<｜tool▁call▁end｜>'
            >>> result = parser.extract_tool_calls(output)
            >>> print(result)
            [(None, 'get_current_weather', {'location': 'Beijing'})]
        """
        matches = re.findall(self.tool_calls_regex, model_output, re.DOTALL)
        if not matches:
            # No tool calls found, return the original output as content
            return [(model_output, None, None)]

        # Use set for deduplication of identical tool calls
        tool_calls = set()
        results: List[Tuple[Optional[str], Optional[str], Optional[dict]]] = []

        for func_name, raw_json in matches:
            func_and_args = None
            try:
                # Parse JSON arguments
                func_and_args = json.loads(raw_json)
                # Create hashable representation for deduplication
                arguments_hashable = frozenset(func_and_args.items())
                tool_call_tuple = (
                    None,  # No content error
                    func_name,
                    func_and_args,
                )
            except Exception as e:
                # JSON parsing failed, treat as raw content
                logger.warning(
                    f"Failed to parse tool call JSON: {raw_json}, error: {e}"
                )
                tool_call_tuple = (raw_json, None, None)
                arguments_hashable = None

            # Create deduplication key
            dedup_key = (
                (func_name, arguments_hashable)
                if func_and_args is not None
                else raw_json
            )

            # Add to results if not already seen
            if dedup_key not in tool_calls:
                tool_calls.add(dedup_key)
                results.append(tool_call_tuple)

        return results

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ) -> Optional[Any]:
        """
        Extract tool calls from streaming output.

        Currently not supported for DeepSeek R1 model. This method raises
        a ValueError indicating that streaming tool call extraction is only
        available for specific model/backend combinations.

        Args:
            previous_text (List[str]): Previous text chunks from the stream.
            current_text (str): Current accumulated text.
            delta_text (str): New text delta in this chunk.

        Raises:
            ValueError: Always raised as streaming is not supported.

        Note:
            DeepSeek R1 model does not currently support streaming tool call
            extraction. Use extract_tool_calls() with complete output instead.
        """
        raise ValueError(
            "Streaming support for tool calls is available only when using "
            "Qwen models with vLLM backend or GLM4-chat models without vLLM backend. "
            "DeepSeek R1 does not support streaming tool call extraction."
        )
