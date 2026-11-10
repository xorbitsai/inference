import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from . import register_tool_parser
from .abstract_tool_parser import ToolParser
from .argument_stream import ArgumentStart, open_block, tagged_string_start

logger = logging.getLogger(__name__)

ToolEvent = Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]


@register_tool_parser("minimax")
class MiniMaxToolParser(ToolParser):
    """
    Tool parser implementation for MiniMax models.

    This parser handles MiniMax tool calls wrapped with <minimax:tool_call>
    tags and <invoke>/<parameter> blocks.
    """

    def __init__(self):
        super().__init__()
        self.think_start_token: str = "<think>"
        self.think_end_token: str = "</think>"
        self.tool_call_start_token: str = "<minimax:tool_call>"
        self.tool_call_end_token: str = "</minimax:tool_call>"

        self.think_regex = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        self.tool_call_complete_regex = re.compile(
            r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL
        )
        self.tool_call_regex = re.compile(
            r"<minimax:tool_call>.*?</minimax:tool_call>|<minimax:tool_call>.*?$",
            re.DOTALL,
        )
        self.invoke_regex = re.compile(
            r"<invoke\s+name=\"([^\"]+)\">(.*?)</invoke>", re.DOTALL
        )
        self.param_regex = re.compile(
            r"<parameter\s+name=\"([^\"]+)\">(.*?)</parameter>", re.DOTALL
        )

    def _parse_param_value(self, value: str, is_string: bool = False) -> Any:
        # A string value sits inline between the tags, so its whitespace is data.
        if is_string:
            return value
        value = value.strip()
        if not value:
            return value
        try:
            return json.loads(value)
        except Exception:
            return value

    def _parse_invoke_calls(self, tool_block: str) -> List[Tuple[str, Dict[str, Any]]]:
        results = []
        for name, body in self.invoke_regex.findall(tool_block):
            args: Dict[str, Any] = {}
            for key, val in self.param_regex.findall(body):
                args[key] = self._parse_param_value(
                    val, self._is_string_parameter(name, key)
                )
            results.append((name, args))
        return results

    def _get_function_calls_streaming(self, model_output: str) -> List[str]:
        matched_ranges = self.tool_call_regex.findall(model_output)
        return matched_ranges

    def is_contain_think(self, model_output: str) -> bool:
        return self.think_regex.search(model_output) is not None

    def _has_unclosed_tool_call(self, text: str) -> bool:
        if not text:
            return True
        start_count = text.count(self.tool_call_start_token)
        end_count = text.count(self.tool_call_end_token)
        return start_count > end_count

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

    def _tool_markup(self, text: str) -> str:
        """Return ``text``, at the same offsets, as tool tags should be read from it."""
        return text

    def _settled_events(
        self, text: str, markup: str, start: int, end: int
    ) -> List[ToolEvent]:
        """Return the plain text and every invoke of each closed block in a range."""
        events: List[ToolEvent] = []
        position = start
        for match in self.tool_call_complete_regex.finditer(markup, start, end):
            if match.start() > position:
                events.append((text[position : match.start()], None, None))
            invokes = self._parse_invoke_calls(match.group(0))
            if invokes:
                events.extend((None, name, args) for name, args in invokes)
            else:
                events.append((match.group(0), None, None))
            position = match.end()
        if position < end:
            events.append((text[position:end], None, None))
        return events

    def extract_tool_calls(
        self, model_output: str
    ) -> List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
        # Only a closed block is a call, as in streaming. A truncated one stays
        # text, so the response keeps its finish reason.
        markup = self._tool_markup(model_output)
        return self._settled_events(model_output, markup, 0, len(model_output)) or [
            (model_output, None, None)
        ]

    _INVOKE_END = "</invoke>"
    _INVOKE_OPEN = re.compile(r'\s*<invoke\s+name="([^"]+)">')
    _PARAMETER_OPEN = re.compile(r'\s*<parameter\s+name="(?P<key>[^"]+)">')

    def locate_argument_start(self, text: str) -> Optional[ArgumentStart]:
        body = open_block(
            self._tool_markup(text),
            self.tool_call_start_token,
            self.tool_call_end_token,
        )
        if body is None:
            return None
        invoke_start: int = body
        preceding = 0
        last_end = text.rfind(self._INVOKE_END, body)
        if last_end != -1:
            for complete in self.invoke_regex.finditer(
                text, body, last_end + len(self._INVOKE_END)
            ):
                invoke_start = complete.end()
                preceding += 1
        match = self._INVOKE_OPEN.match(text, invoke_start)
        if match is None:
            return None
        name = match[1]

        def parse_completed(completed: str) -> Dict[str, Any]:
            return {
                key: self._parse_param_value(
                    value, self._is_string_parameter(name, key)
                )
                for key, value in self.param_regex.findall(completed)
            }

        return tagged_string_start(
            text,
            match.end(),
            name,
            self._PARAMETER_OPEN,
            "</parameter>",
            parse_completed,
            preceding_calls=preceding,
        )

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ) -> Optional[Union[ToolEvent, List[ToolEvent]]]:
        try:
            markup = self._tool_markup(current_text)
            previous = current_text[: len(current_text) - len(delta_text)]
            start = self._settled_length(self._tool_markup(previous))
            end = self._settled_length(markup)
            if self.tool_call_start_token not in markup:
                # Hold back a trailing piece that may start the tool call tag,
                # and release it once it turns out to be text.
                if end <= start:
                    return None if delta_text else (delta_text, None, None)
                return (current_text[start:end], None, None)

            # Text after an unclosed start tag belongs to a tool call block
            # that is still streaming. Emit only what became settled in this
            # chunk: plain text, and every invoke of each block that closed.
            if end <= start:
                return None

            events = self._settled_events(current_text, markup, start, end)
            return events[0] if len(events) == 1 else events
        except Exception as e:
            logger.error("Error in MiniMax streaming tool call extraction: %s", e)
            raise
