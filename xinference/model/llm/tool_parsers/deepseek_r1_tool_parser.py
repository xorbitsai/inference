import json
import logging
import re
from typing import List, Optional, Tuple

from . import register_tool_parser
from .abstract_tool_parser import ToolParser
from .argument_stream import ArgumentStart, open_block, stream_delimited_calls

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

        # Sentinel tokens for streaming mode
        self.think_start_token: str = "<think>"
        self.think_end_token: str = "</think>"
        self.tool_call_start_token: str = "<｜tool▁call▁begin｜>"
        self.tool_call_end_token: str = "<｜tool▁call▁end｜>"

        # Regex pattern to match DeepSeek R1 tool call format
        self.tool_calls_regex = (
            r"<\｜tool▁call▁begin｜>function<\｜tool▁sep｜>([^\n]+)\n"
            r"```json\n(.*?)\n```<\｜tool▁call▁end｜>"
        )

        # Regex pattern to match the entire tool-calls wrapper block.
        # We intentionally do NOT match <think> blocks here so that the
        # "text before" chunk will include both the think block and any
        # narrative text up to the tool calls wrapper, yielding exactly two
        # blocks when there is a single tool calls section:
        # [before_text_including_think, tool_calls_wrapper_block]
        self.content_regex = r"(<\｜tool▁calls▁begin｜>.*?<\｜tool▁calls▁end｜>)"

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
        # If no tool call tokens, return original output as content
        if self.tool_call_start_token not in model_output:
            return [(model_output, None, None)]

        # Get all content blocks (text, thinking blocks, tool calls)
        function_calls = self._get_function_calls(model_output)

        # Use set for deduplication of identical tool calls
        tool_calls = set()
        results: List[Tuple[Optional[str], Optional[str], Optional[dict]]] = []

        for content_block in function_calls:
            # Check if this block is a tool call
            if (
                self.tool_call_start_token in content_block
                and self.tool_call_end_token in content_block
            ):
                # Extract function name and arguments from tool call block
                matches = re.findall(self.tool_calls_regex, content_block, re.DOTALL)
                if not matches:
                    # Malformed tool call, treat as regular content
                    results.append((content_block, None, None))
                    continue

                # A tool calls block may hold several parallel calls
                for func_name, raw_json in matches:
                    func_and_args = None
                    try:
                        # Parse JSON arguments
                        func_and_args = json.loads(raw_json)
                        # Hashable form of the arguments (values may be
                        # lists or objects) for deduplication
                        arguments_hashable = json.dumps(func_and_args, sort_keys=True)
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
            else:
                # This is regular content (text or thinking block), add as-is
                if content_block.strip():  # Only add non-empty content
                    results.append((content_block, None, None))

        return results

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

    _WRAPPER_START = "<｜tool▁calls▁begin｜>"
    _WRAPPER_END = "<｜tool▁calls▁end｜>"
    _JSON_OPEN = re.compile(r"function<｜tool▁sep｜>([^\n]+)\n```json\n\s*(?=\{)")

    def locate_argument_start(self, text: str) -> Optional[ArgumentStart]:
        wrapper = open_block(text, self._WRAPPER_START, self._WRAPPER_END)
        body = open_block(text, self.tool_call_start_token, self.tool_call_end_token)
        if wrapper is None or body is None or body < wrapper:
            return None
        match = self._JSON_OPEN.match(text, body)
        if match is None:
            return None
        preceding = 0
        if self.tool_call_end_token in text[wrapper:body]:
            preceding = sum(
                name is not None
                for _, name, _ in self._parse_streaming_block(text[wrapper:body])
            )
        return ArgumentStart(match[1], match.end(), preceding_calls=preceding)

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ) -> List[Tuple]:
        return stream_delimited_calls(
            self,
            current_text,
            delta_text,
            self._WRAPPER_START,
            self._WRAPPER_END,
            parse_block=self._parse_streaming_block,
        )

    def _parse_streaming_block(self, block: str) -> List[Tuple]:
        events: List[Tuple] = []
        for name, raw in re.findall(self.tool_calls_regex, block, re.DOTALL):
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(arguments, dict):
                events.append((None, name, arguments))
        return events or [(block, None, None)]
