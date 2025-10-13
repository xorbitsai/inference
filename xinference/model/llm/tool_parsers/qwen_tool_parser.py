import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from . import register_tool_parser
from .abstract_tool_parser import ToolParser

logger = logging.getLogger(__name__)


@register_tool_parser("qwen")
class QwenToolParser(ToolParser):
    """
    Tool parser implementation for Qwen model.

    This parser handles the specific format used by Qwen for tool calls,
    which uses XML-like tags for both thinking blocks and tool calls.

    """

    def __init__(self):
        """
        Initialize the Qwen tool parser.

        Sets up the XML-like tokens and regex patterns used for parsing
        Qwen model outputs containing thinking blocks and tool calls.
        """
        super().__init__()

        # Sentinel tokens for streaming mode
        self.think_start_token: str = "<think>"
        self.think_end_token: str = "</think>"
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"

        # Regex patterns for parsing different content types
        self.think_regex = re.compile("<think>(.*?)</think>", re.DOTALL)
        self.content_regex = r"(<(think|tool_call)>.*?</\2>)"
        self.tool_call_complete_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL
        )
        self.tool_call_regex = re.compile(
            r"<tool_call>.*?</tool_call>|<tool_call>.*?$", re.DOTALL
        )

    def _parse_json_function_call(
        self,
        function_call_str: str,
    ) -> str:
        """
        Parse JSON function call from string.

        Extracts the JSON content from tool_call XML tags.

        Args:
            function_call_str (str): The function call string to parse.

        Returns:
            str: Extracted JSON string or original string if no match found.
        """
        # First try to find complete tool calls
        function_calls = self.tool_call_complete_regex.findall(function_call_str)
        if len(function_calls) > 0:
            return function_calls[-1]

        # If no complete tool calls found, try to extract from incomplete tool calls
        # Handle cases like <tool_call><tool_call>_city
        if self.tool_call_start_token in function_call_str:
            # Extract content between the last tool_call start token and end of string
            last_start = function_call_str.rfind(self.tool_call_start_token)
            potential_json = function_call_str[
                last_start + len(self.tool_call_start_token) :
            ]
            # Remove any trailing tool_call end tokens
            if self.tool_call_end_token in potential_json:
                potential_json = potential_json.split(self.tool_call_end_token)[0]
            # Clean up any extra whitespace
            potential_json = potential_json.strip()
            if potential_json:
                return potential_json

        return function_call_str

    def _parse_json_function_call_stream(
        self,
        function_call_str: str,
    ) -> Optional[str]:
        """
        Parse JSON function call from streaming string.

        Extracts the JSON content from tool_call XML tags in streaming context.

        Args:
            function_call_str (str): The function call string to parse.

        Returns:
            Optional[str]: Extracted JSON string or None if no complete match found.
        """
        function_calls = self.tool_call_complete_regex.findall(function_call_str)
        if len(function_calls) == 0:
            return None
        return function_calls[-1]

    def is_contain_think_end_token(self, model_output: str) -> bool:
        """
        Check if the model output contains the think end token.

        Args:
            model_output (str): The model output to check.

        Returns:
            bool: True if think end token is present.
        """
        return self.think_end_token in model_output

    def is_contain_think(self, model_output: str) -> bool:
        """
        Check if the model output contains complete thinking blocks.

        Args:
            model_output (str): The model output to check.

        Returns:
            bool: True if complete thinking blocks are present.
        """
        return self.think_regex.search(model_output) is not None

    def is_contain_tool_call(self, model_output: str) -> bool:
        """
        Check if the model output contains complete tool calls.

        Args:
            model_output (str): The model output to check.

        Returns:
            bool: True if complete tool calls are present.
        """
        return self.tool_call_complete_regex.search(model_output) is not None

    def is_contain_tool_call_start_token(self, model_output: str) -> bool:
        """
        Check if the model output contains the tool call start token.

        Args:
            model_output (str): The model output to check.

        Returns:
            bool: True if tool call start token is present.
        """
        return self.tool_call_start_token in model_output

    def is_contain_tool_call_end_token(self, model_output: str) -> bool:
        """
        Check if the model output contains the tool call end token.

        Args:
            model_output (str): The model output to check.

        Returns:
            bool: True if tool call end token is present.
        """
        return self.tool_call_end_token in model_output

    def _get_function_calls(self, model_output: str) -> List[str]:
        """
        Extract all function calls and content blocks from model output.

        Parses the model output to separate thinking blocks, tool calls,
        and regular content into individual components.

        Args:
            model_output (str): The complete model output to parse.

        Returns:
            List[str]: List of content blocks (text, thinking blocks, tool calls).
        """
        functions_calls = []
        last_end = 0
        for m in re.finditer(self.content_regex, model_output, re.DOTALL):
            # Add any text before the current match
            if m.start() > last_end:
                functions_calls.append(model_output[last_end : m.start()])
            # Add the matched content (think or tool_call block)
            functions_calls.append(m.group(0))
            last_end = m.end()
        # Add any remaining text after the last match
        if last_end < len(model_output):
            functions_calls.append(model_output[last_end:])
        return functions_calls

    def _get_function_calls_streaming(self, model_output: str) -> List[str]:
        """
        Extract function calls from streaming model output.

        Finds both complete and incomplete tool calls in streaming context.

        Args:
            model_output (str): The streaming model output to parse.

        Returns:
            List[str]: List of tool call blocks (complete or incomplete).
        """
        matched_ranges = self.tool_call_regex.findall(model_output)
        return matched_ranges

    def extract_tool_calls(
        self, model_output: str
    ) -> List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
        """
        Extract tool calls from complete model output.

        Parses the model output to find tool calls and thinking blocks,
        extracting function names and arguments from JSON content within
        tool_call XML tags.

        Args:
            model_output (str): The complete output string from the model.

        Returns:
            List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
            A list of tuples where each tuple contains:
            - content (str or None): Raw content if parsing failed, None if successful
            - function_name (str or None): Name of the function to call
            - arguments (dict or None): Function arguments

        Example:
            >>> parser = QwenToolParser()
            >>> output = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Beijing"}}\n</tool_call>'
            >>> result = parser.extract_tool_calls(output)
            >>> print(result)
            [(None, 'get_weather', {'location': 'Beijing'})]
        """
        # If no tool call tokens, return original output as content
        if self.tool_call_start_token not in model_output:
            return [(model_output, None, None)]

        try:
            function_calls = self._get_function_calls(model_output)
            if len(function_calls) == 0:
                return [(model_output, None, None)]

            results: List[
                Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]
            ] = []
            for function_call in function_calls:
                try:
                    parsed_json = self._parse_json_function_call(function_call)
                    res = json.loads(parsed_json, strict=False)
                    # Validate that we have the required fields
                    if "name" in res and "arguments" in res:
                        results.append((None, res["name"], res["arguments"]))
                    else:
                        logger.warning(
                            "Invalid tool call format, missing required fields: %s", res
                        )
                        results.append((function_call, None, None))
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
            return [(model_output, None, None)]

    def _has_unclosed_tool_call(self, text: str) -> bool:
        """
        Check if the text has unclosed tool_call tags.

        Counts the number of opening and closing tool_call tags to determine
        if there are any unclosed tool calls in the text.

        Args:
            text (str): The text to check for unclosed tags.

        Returns:
            bool: True if there are unclosed tool_call tags.
        """
        if not text:
            return True
        start_count = text.count(self.tool_call_start_token)
        end_count = text.count(self.tool_call_end_token)
        return start_count > end_count

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ) -> Optional[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
        """
        Extract tool calls from streaming output.

        Processes streaming model output to detect and extract tool calls
        as they are being generated. Handles incomplete tool calls and
        determines when a complete tool call is available.

        Args:
            previous_text (List[str]): Previous text chunks from the stream.
            current_text (str): Current accumulated text.
            delta_text (str): New text delta in this chunk.

        Returns:
            Optional[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
            A tuple containing:
            - content (str or None): Text content or None for tool calls
            - function_name (str or None): Name of the function to call
            - arguments (dict or None): Function arguments
            Returns None if no complete tool call is ready.

        Note:
            This method is designed to work with Qwen's streaming output format
            and handles partial tool calls during generation.
        """
        try:
            # Check if current output contains tool_call start token
            if self.is_contain_tool_call_start_token(current_text):
                function_calls = self._get_function_calls_streaming(current_text)
                # If the last function call contains thinking, it's not a tool call
                if self.is_contain_think(function_calls[-1]):
                    return None
                # If the previous round's tool_call tags are closed, this is a new tool call
                if not self._has_unclosed_tool_call(previous_text[-1]):
                    return None
                # Parse and return
                function_call = self._parse_json_function_call_stream(
                    function_calls[-1]
                )
                if function_call is None:
                    return None
                res = json.loads(function_call, strict=False)
                return None, res["name"], res["arguments"]
            else:
                # Return delta text as regular content
                return (delta_text, None, None)

        except Exception as e:
            logger.error("Error in Qwen streaming tool call extraction: %s", e)
            raise
