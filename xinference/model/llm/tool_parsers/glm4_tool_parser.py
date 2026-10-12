import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from . import register_tool_parser
from .abstract_tool_parser import ToolParser

logger = logging.getLogger(__name__)


@register_tool_parser("glm4")
class Glm4ToolParser(ToolParser):
    """
    Tool parser implementation for GLM4 model.

    This parser handles the specific format used by GLM4 for tool calls,
    which uses JSON code blocks wrapped in markdown format.

    """

    def __init__(self):
        """
        Initialize the GLM4 tool parser.
        """
        super().__init__()
        # Regex pattern to match JSON code blocks
        self.tool_calls_regex = re.compile(r"\s*```json\s*(.*?)\s*```", re.DOTALL)

    @staticmethod
    def _native_call(text: str):
        match = re.match(
            r"^\s*(?:<\|assistant\|>)?([\w.-]+)\s*\n(?:```json\s*)?(\{)", text
        )
        if match is None:
            return None
        try:
            arguments, end = json.JSONDecoder().raw_decode(text[match.start(2) :])
        except ValueError:
            return None
        trailing = text[match.start(2) + end :].strip()
        if trailing not in {"", "```", "<|observation|>"} or not isinstance(
            arguments, dict
        ):
            return None
        return None, match.group(1), arguments

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
            >>> parser = Glm4ToolParser()
            >>> output = {"name": "get_weather", "parameters": {"location": "Beijing"}}
            >>> result = parser.extract_tool_calls(output)
            >>> print(result)
            [(None, 'get_weather', {'location': 'Beijing'})]
        """
        if isinstance(model_output, str):
            call = self._native_call(model_output)
            return [call] if call is not None else [(model_output, None, None)]
        try:
            if isinstance(model_output, dict):
                try:
                    return [
                        (
                            None,
                            model_output["name"],
                            json.loads(model_output["arguments"]),
                        )
                    ]
                except Exception:
                    return [(None, model_output["name"], model_output["arguments"])]
        except KeyError:
            logger.error("Can't parse glm output: %s", model_output)
            return [(str(model_output), None, None)]
        else:
            return [(str(model_output), None, None)]

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ) -> Optional[Any]:
        """
        Extract tool calls from streaming output.

        Hold native tool arguments until a complete JSON object is available,
        and emit each completed call once. Preserve legacy dictionary outputs.

        Args:
            previous_text (List[str]): Previous text chunks from the stream.
            current_text (str): Current accumulated text.
            delta_text (str): New text delta in this chunk.
        """
        if isinstance(current_text, str):
            previous = previous_text[-1] if previous_text else ""
            call = self._native_call(current_text)
            if call is not None:
                return call if self._native_call(previous) is None else None
            # Hold an incomplete native tool call until its JSON is complete.
            if re.match(
                r"^\s*(?:<\|assistant\|>)?[\w.-]+\s*\n(?:```json\s*)?\{", current_text
            ):
                return None
            return (delta_text, None, None) if delta_text else None
        try:
            if isinstance(current_text, dict):
                try:
                    return (
                        None,
                        current_text["name"],
                        json.loads(current_text["arguments"]),
                    )
                except Exception:
                    return (None, current_text["name"], current_text["arguments"])
        except KeyError:
            logger.error("Can't parse glm output: %s", current_text)
            return (str(current_text), None, None)
        else:
            return (str(current_text), None, None)
