import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from . import register_tool_parser
from .abstract_tool_parser import ToolParser

logger = logging.getLogger(__name__)


@register_tool_parser("deepseek-v3")
class DeepseekV3ToolParser(ToolParser):
    """
    Tool parser implementation for DeepSeek V3 model.

    This parser handles the specific format used by DeepSeek V3 for tool calls,
    which uses JSON code blocks wrapped in markdown format.

    """

    def __init__(self):
        """
        Initialize the DeepSeek V3 tool parser.
        """
        super().__init__()
        # Regex pattern to match JSON code blocks
        self.tool_calls_regex = r"\s*```json\s*(.*?)\s*```"

    def _parse_json_function_call(
        self,
        function_call_str: str,
    ) -> str:
        """
        Parse JSON function call from string.

        Args:
            function_call_str (str): The function call string to parse.

        Returns:
            str: Parsed result or original string if no match found.

        """
        match = self.tool_calls_regex.search(function_call_str)
        if match:
            result = match.group(1)
            return result
        return function_call_str

    def extract_tool_calls(
        self, model_output: str
    ) -> List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
        """
        Extract tool calls from complete model output.

        Parses the model output to find JSON code blocks containing tool calls
        and extracts function names and parameters. Handles JSON parsing errors
        gracefully and deduplicates identical tool calls.

        Args:
            model_output (str): The complete output string from the model.

        Returns:
            List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
            A list of tuples where each tuple contains:
            - content (str or None): Raw content if parsing failed, None if successful
            - function_name (str or None): Name of the function to call
            - parameters (dict or None): Function parameters

        Example:
            >>> parser = DeepseekV3ToolParser()
            >>> output = '```json\n{"name": "get_weather", "parameters": {"location": "Beijing"}}\n```'
            >>> result = parser.extract_tool_calls(output)
            >>> print(result)
            [(None, 'get_weather', {'location': 'Beijing'})]
        """
        matches = re.findall(self.tool_calls_regex, model_output, re.DOTALL)

        if not matches:
            # No tool calls found, return the original output as content
            return [(model_output, None, None)]

        # Use set for deduplication of identical tool calls
        tool_calls = set()
        results: List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]] = (
            []
        )

        for raw_json in matches:
            func_and_args = None
            try:
                # Parse JSON to extract function call information
                func_and_args = json.loads(raw_json)
                # Convert dictionary to frozenset for deduplication
                arguments_hashable = frozenset(func_and_args["parameters"])
                tool_call_tuple = (
                    None,  # No content error
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

        Currently not supported for DeepSeek V3 model. This method raises
        a ValueError indicating that streaming tool call extraction is only
        available for specific model/backend combinations.

        Args:
            previous_text (List[str]): Previous text chunks from the stream.
            current_text (str): Current accumulated text.
            delta_text (str): New text delta in this chunk.

        Raises:
            ValueError: Always raised as streaming is not supported.
        """
        raise NotImplementedError(
            "Streaming support for tool calls is available only when using "
            "Qwen models with vLLM backend or GLM4-chat models without vLLM backend. "
            "DeepSeek V3 does not support streaming tool call extraction."
        )
