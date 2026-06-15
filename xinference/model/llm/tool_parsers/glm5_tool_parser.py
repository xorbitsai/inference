import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from . import register_tool_parser
from .abstract_tool_parser import ToolParser

logger = logging.getLogger(__name__)


@register_tool_parser("glm5")
class Glm5ToolParser(ToolParser):
    """
    Tool parser implementation for GLM5 model.

    This parser handles the specific format used by GLM5 for tool calls,
    which uses XML-like tags (<tool_call>, <arg_key>, <arg_value>).

    """

    def __init__(self):
        """
        Initialize the GLM5 tool parser.
        """
        super().__init__()
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_call_complete_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL
        )
        self.tool_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
        )

    def _parse_xml_function_call(
        self, function_call_str: str
    ) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        function_call_str = function_call_str.strip()
        arg_start = function_call_str.find("<arg_key>")
        function_name = (
            function_call_str[:arg_start].strip()
            if arg_start != -1
            else function_call_str
        )
        if not function_name:
            return function_call_str, None, None

        arguments: Dict[str, Any] = {}
        for key, value in self.tool_arg_regex.findall(function_call_str):
            key = key.strip()
            value = value.strip()
            try:
                value = json.loads(value)
            except Exception:
                pass
            arguments[key] = value

        return None, function_name, arguments

    def _parse_xml_tool_calls(
        self, model_output: str
    ) -> List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
        matches = list(self.tool_call_complete_regex.finditer(model_output))
        if not matches:
            return [(model_output, None, None)]

        results: List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]] = (
            []
        )
        pos = 0
        for match in matches:
            if match.start() > pos:
                results.append((model_output[pos : match.start()], None, None))
            results.append(self._parse_xml_function_call(match.group(1)))
            pos = match.end()
        if pos < len(model_output):
            results.append((model_output[pos:], None, None))
        return results

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
            >>> parser = Glm5ToolParser()
            >>> output = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Beijing</arg_value></tool_call>"
            >>> result = parser.extract_tool_calls(output)
            >>> print(result)
            [(None, 'get_weather', {'location': 'Beijing'})]
        """
        if (
            isinstance(model_output, str)
            and self.tool_call_start_token not in model_output
        ):
            return [(model_output, None, None)]
        if isinstance(model_output, str):
            return self._parse_xml_tool_calls(model_output)
        return [(str(model_output), None, None)]

    def _extract_delta_without_partial_start(
        self, current_text: str, delta_text: str
    ) -> Optional[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
        for i in range(len(self.tool_call_start_token) - 1, 0, -1):
            if current_text.endswith(self.tool_call_start_token[:i]):
                if len(delta_text) > i:
                    return delta_text[:-i], None, None
                return None
        return (delta_text, None, None) if delta_text else None

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ) -> Optional[Any]:
        """
        Extract tool calls from streaming output.

        Args:
            previous_text (List[str]): Previous text chunks from the stream.
            current_text (str): Current accumulated text.
            delta_text (str): New text delta in this chunk.
        """
        if isinstance(current_text, str):
            if self.tool_call_start_token not in current_text:
                return self._extract_delta_without_partial_start(
                    current_text, delta_text
                )
            if current_text.count(self.tool_call_start_token) > current_text.count(
                self.tool_call_end_token
            ):
                return None

            previous = previous_text[-1] if previous_text else ""
            if isinstance(previous, str) and previous.count(
                self.tool_call_end_token
            ) >= current_text.count(self.tool_call_end_token):
                return (delta_text, None, None) if delta_text else None

            results = self._parse_xml_tool_calls(current_text)
            for result in reversed(results):
                if result[1] is not None:
                    return result
            return None
        return (str(current_text), None, None)
