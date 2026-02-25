import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from . import register_tool_parser
from .abstract_tool_parser import ToolParser

logger = logging.getLogger(__name__)


@register_tool_parser("deepseek-v3.1")
class DeepseekV3_1ToolParser(ToolParser):
    """
    Tool parser implementation for DeepSeek V3.1 model.

    This parser handles the specific format used by DeepSeek V3.1 for tool calls,
    which uses <｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜> tokens.

    """

    def __init__(self):
        """
        Initialize the DeepSeek V3.1 tool parser.
        """
        super().__init__()
        # Sentinel tokens for streaming mode
        self.new_tool_calls_regex = re.compile(
            r"(?:<｜tool▁calls▁begin｜>|<｜tool▁call▁begin｜>)(?:<｜tool▁call▁begin｜>)?\s*(.*?)\s*<｜tool▁sep｜>\s*(.*?)\s*<｜tool▁call▁end｜>",
            re.DOTALL,
        )

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
            >>> parser = DeepseekV3_1ToolParser()
            >>> output = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "Beijing"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
            >>> result = parser.extract_tool_calls(output)
            >>> print(result)
            [(None, 'get_weather', {'location': 'Beijing'})]
        """
        new_matches = self.new_tool_calls_regex.findall(model_output)

        if not new_matches:
            # No tool calls found, return the original output as content
            return [(model_output, None, None)]

        # Use set for deduplication of identical tool calls
        tool_calls = set()
        results: List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]] = (
            []
        )

        for name, args_json in new_matches:
            tool_call_tuple: Tuple[
                Optional[str], Optional[str], Optional[Dict[str, Any]]
            ]
            try:
                arguments = json.loads(args_json)
                arguments_hashable = frozenset(arguments)
                tool_call_tuple = (
                    None,
                    name,
                    arguments,
                )
            except json.JSONDecodeError:
                tool_call_tuple = (
                    f"<｜tool▁call▁begin｜>{name}<｜tool▁sep｜>{args_json}<｜tool▁call▁end｜>",
                    None,
                    None,
                )
                arguments_hashable = None

            dedup_key = (
                (name, arguments_hashable)
                if arguments_hashable is not None
                else (tool_call_tuple[0])
            )

            if dedup_key not in tool_calls:
                tool_calls.add(dedup_key)
                results.append(tool_call_tuple)

        return results

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ) -> Optional[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
        """
        Extract tool calls from streaming output.

        Processes streaming model output to detect and extract tool calls
        as they are being generated. Handles incomplete tool calls and
        determines when a complete tool call is available.

        This method also handles cases where the <tool_call> tag is split
        across multiple chunks (e.g., ['<', 'tool', '_call', '>']).

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
            Returns None if no complete tool call is ready yet or if we're
            waiting to see if a tool_call tag is forming.
        """
        try:
            new_pattern_start = "<｜tool▁calls▁begin｜>"

            # New pattern check
            if new_pattern_start in current_text:
                # Calculate tool calls in previous and current text to detect new completions
                full_previous_text = "".join(previous_text)
                previous_tool_calls = self.new_tool_calls_regex.findall(
                    full_previous_text
                )
                current_tool_calls = self.new_tool_calls_regex.findall(current_text)

                if len(current_tool_calls) > len(previous_tool_calls):
                    # A new tool call has been completed
                    last_tool_call = current_tool_calls[-1]
                    name = last_tool_call[0]
                    args_json = last_tool_call[1]
                    try:
                        arguments = json.loads(args_json)
                        return None, name, arguments
                    except json.JSONDecodeError:
                        # If JSON is invalid, we ignore it for now (it might be fixed in next chunks?
                        # No, if regex matched, it means we have end token, so it's final invalid JSON)
                        pass

                # In tool call mode, we suppress output until we find a complete tool call
                return None

            # Check for potential start of new pattern
            for i in range(1, len(new_pattern_start)):
                prefix = new_pattern_start[:i]
                if current_text.endswith(prefix):
                    return None

            # No tool call detected and not forming one, return delta text as regular content
            return (delta_text, None, None)

        except json.JSONDecodeError as e:
            logger.error("JSON decode error in streaming tool call extraction: %s", e)
            return None
        except Exception as e:
            logger.error("Error in DeepSeek V3.1 streaming tool call extraction: %s", e)
            raise
