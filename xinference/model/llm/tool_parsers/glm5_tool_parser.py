import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from . import register_tool_parser
from .abstract_tool_parser import ToolParser

logger = logging.getLogger(__name__)

ToolEvent = Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]


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
            if not self._is_string_parameter(function_name, key):
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
        content = self._plain_text_delta(
            current_text, delta_text, [self.tool_call_start_token]
        )
        return (content, None, None) if content else None

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ) -> Optional[Union[ToolEvent, List[ToolEvent]]]:
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

            # Text after an unclosed start tag belongs to a tool call that is
            # still streaming. Emit only what became settled in this chunk:
            # plain text and every tool call block that closed, in order.
            previous = current_text[: len(current_text) - len(delta_text)]
            start = self._settled_length(previous)
            end = self._settled_length(current_text)
            if end <= start:
                return None

            events: List[ToolEvent] = []
            position = start
            for match in self.tool_call_complete_regex.finditer(
                current_text, start, end
            ):
                if match.start() > position:
                    events.append((current_text[position : match.start()], None, None))
                events.append(self._parse_xml_function_call(match.group(1)))
                position = match.end()
            if position < end:
                events.append((current_text[position:end], None, None))
            return events[0] if len(events) == 1 else events
        return (str(current_text), None, None)

    def _settled_length(self, text: str) -> int:
        """Return the length of the prefix of ``text`` that holds no unclosed
        tool call block and no trailing piece of a start tag."""
        # Same result as walking tool_call_complete_regex.finditer(text), but
        # str.find skips the text between tags instead of matching it lazily.
        start_token = self.tool_call_start_token
        end_token = self.tool_call_end_token
        position = 0
        while True:
            open_index = text.find(start_token, position)
            if open_index == -1:
                break
            close_index = text.find(end_token, open_index + len(start_token))
            if close_index == -1:
                return open_index
            position = close_index + len(end_token)
        return len(text) - self._partial_marker_length(
            text, [self.tool_call_start_token]
        )
