import json
import logging
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from . import register_tool_parser
from .abstract_tool_parser import ToolParser, find_outside_json_string
from .argument_stream import ArgumentStart, open_block

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
        # The start of a call, up to the separator before its arguments
        self._call_head_regex = re.compile(
            r"(?:<｜tool▁calls▁begin｜>|<｜tool▁call▁begin｜>)(?:<｜tool▁call▁begin｜>)?\s*(.*?)\s*(?=<｜tool▁sep｜>)",
            re.DOTALL,
        )
        self._calls_end_regex = re.compile(r"\s*<｜tool▁calls▁end｜>")

    _SEPARATOR = "<｜tool▁sep｜>"
    _CALL_BEGIN = "<｜tool▁call▁begin｜>"
    _CALL_END = "<｜tool▁call▁end｜>"

    def _call_end(self, text: str, separator: int) -> int:
        """
        Return the index just past the end of the call whose separator token is
        at ``separator``, or -1 while it is still open. A JSON argument ends
        where the JSON does, so a literal call end token inside a string value
        does not end the call.
        """
        close = find_outside_json_string(
            text,
            self._CALL_END,
            separator + len(self._SEPARATOR),
            self._CALL_BEGIN,
        )
        return -1 if close == -1 else close + len(self._CALL_END)

    def _iter_calls(self, text: str) -> Iterator[Tuple[str, str, int, int]]:
        """
        Yield ``(name, arguments, start, end)`` for each complete call in
        ``text``, where ``end`` is just past the call and the end of calls
        token that follows it, if any.
        """
        position = 0
        while (head := self._call_head_regex.search(text, position)) is not None:
            separator = head.end()
            close = self._call_end(text, separator)
            if close == -1:
                return
            arguments = text[separator + len(self._SEPARATOR) : close]
            arguments = arguments[: -len(self._CALL_END)].strip()
            trailing = self._calls_end_regex.match(text, close)
            position = trailing.end() if trailing else close
            yield head[1], arguments, head.start(), position

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
        new_matches = list(self._iter_calls(model_output))

        if not new_matches:
            # No tool calls found, return the original output as content
            return [(model_output, None, None)]

        # Use set for deduplication of identical tool calls
        tool_calls = set()
        results: List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]] = (
            []
        )

        content_start = 0
        for name, args_json, start, end in new_matches:
            content = model_output[content_start:start]
            if content:
                results.append((content, None, None))
            content_start = end
            tool_call_tuple: Tuple[
                Optional[str], Optional[str], Optional[Dict[str, Any]]
            ]
            try:
                arguments = json.loads(args_json)
                # Key on the argument values too, so parallel calls to the same
                # function with different arguments are not merged.
                arguments_hashable = json.dumps(arguments, sort_keys=True)
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

        content = model_output[content_start:]
        if content:
            results.append((content, None, None))
        return results

    _ARGUMENT_OPEN = re.compile(r"([^<]+?)<｜tool▁sep｜>\s*(?=\{)")

    def locate_argument_start(self, text: str) -> Optional[ArgumentStart]:
        wrapper = open_block(text, "<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>")
        body = open_block(text, "<｜tool▁call▁begin｜>", "<｜tool▁call▁end｜>")
        if wrapper is None or body is None or body < wrapper:
            return None
        match = self._ARGUMENT_OPEN.match(text, body)
        if match is None:
            return None
        return ArgumentStart(
            match[1].strip(),
            match.end(),
            json_suffix="<｜tool▁call▁end｜>",
            close_frame=self._close_call,
        )

    @staticmethod
    def _close_call(text: str, start: int) -> int:
        return text.find("<｜tool▁call▁end｜>", start)

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ) -> Optional[
        Union[
            Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]],
            List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]],
        ]
    ]:
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
            A tuple for one event, or a list of tuples when multiple calls
            complete in one engine text delta. Each tuple contains:
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
                # Text outside the tool calls block is content, even when it
                # shares a chunk with a marker
                content = self._text_outside_blocks(
                    current_text,
                    delta_text,
                    new_pattern_start,
                    "<｜tool▁calls▁end｜>",
                    call_start=self._SEPARATOR,
                    call_end=self._call_end,
                )
                content_events = [(content, None, None)] if content else []
                # Calculate tool calls in previous and current text to detect new completions
                full_previous_text = "".join(previous_text)
                # Every match ends with the call end token, so the number of
                # complete calls can only grow when this delta adds one.
                call_end = "<｜tool▁call▁end｜>"
                if current_text.startswith(full_previous_text) and (
                    current_text.find(
                        call_end, max(len(full_previous_text) - len(call_end) + 1, 0)
                    )
                    == -1
                ):
                    return content_events[0] if content_events else None
                previous_tool_calls = list(self._iter_calls(full_previous_text))
                current_tool_calls = list(self._iter_calls(current_text))

                new_tool_calls: List[
                    Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]
                ] = []
                for name, args_json, _, _ in current_tool_calls[
                    len(previous_tool_calls) :
                ]:
                    try:
                        arguments = json.loads(args_json)
                    except json.JSONDecodeError:
                        # A matched call is complete; skip invalid JSON without
                        # losing other calls completed in this delta.
                        continue
                    new_tool_calls.append((None, name, arguments))

                new_tool_calls = content_events + new_tool_calls
                if new_tool_calls:
                    return (
                        new_tool_calls[0]
                        if len(new_tool_calls) == 1
                        else new_tool_calls
                    )

                # In tool call mode, we suppress output until we find a complete tool call
                return None

            # No tool call yet. Hold back a trailing piece that may start the
            # tool calls token, and release it once it turns out to be text.
            content = self._plain_text_delta(
                current_text, delta_text, [new_pattern_start]
            )
            if content is None:
                return None if delta_text else (delta_text, None, None)
            return (content, None, None)

        except json.JSONDecodeError as e:
            logger.error("JSON decode error in streaming tool call extraction: %s", e)
            return None
        except Exception as e:
            logger.error("Error in DeepSeek V3.1 streaming tool call extraction: %s", e)
            raise
