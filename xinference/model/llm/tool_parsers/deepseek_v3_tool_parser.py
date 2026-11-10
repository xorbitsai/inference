import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from . import register_tool_parser
from .abstract_tool_parser import ToolParser
from .argument_stream import ArgumentStart, json_argument_start, stream_delimited_calls

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
            event: Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]
            try:
                call = json.loads(raw_json)
                name = call["name"]
                parameters = call["parameters"]
                if not isinstance(name, str) or not isinstance(parameters, dict):
                    raise ValueError("Expected a named tool with object arguments")
                key: Union[str, Tuple[str, str]] = (
                    name,
                    json.dumps(parameters, sort_keys=True),
                )
                event = (None, name, parameters)
            except (KeyError, TypeError, ValueError):
                key = raw_json
                event = (raw_json, None, None)

            if key not in tool_calls:
                tool_calls.add(key)
                results.append(event)

        return results

    def locate_argument_start(self, text: str) -> Optional[ArgumentStart]:
        # The opening marker also ends with three backticks, so look for
        # closing fences only after a complete opening has been consumed.
        position = 0
        while (start := text.find("```json", position)) != -1:
            start += len("```json")
            end = text.find("```", start)
            if end == -1:
                return json_argument_start(text, start, "parameters")
            position = end + 3
        return None

    def flush_streaming_content(self, text: str) -> Optional[str]:
        events = stream_delimited_calls(self, text, "", "```json", "```", finished=True)
        return events[0][0] if events else None

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ) -> List[Tuple]:
        return stream_delimited_calls(self, current_text, delta_text, "```json", "```")
