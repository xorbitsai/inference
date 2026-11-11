import json
import re
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from . import register_tool_parser
from .abstract_tool_parser import ToolParser
from .argument_stream import ArgumentStart, json_argument_start, stream_delimited_calls


@register_tool_parser("deepseek-v3")
class DeepseekV3ToolParser(ToolParser):
    """
    Tool parser implementation for DeepSeek V3 model.

    This parser handles the specific format used by DeepSeek V3 for tool calls,
    which uses JSON code blocks wrapped in markdown format.

    """

    _OPEN = "```json"
    _CLOSE = "```"
    # Backticks inside a JSON string are data, including an unfinished
    # string. An unescaped newline terminates an invalid JSON string.
    _STRING_OR_FENCE = re.compile(
        r'"(?:[^"\\\n]|\\.)*(?:"|\\?(?=\n|\Z))|(?P<fence>```)'
    )

    @classmethod
    def _find_fence(cls, text: str, start: int) -> int:
        if text.find(cls._CLOSE, start) == -1:
            return -1
        return next(
            (
                match.start()
                for match in cls._STRING_OR_FENCE.finditer(text, start)
                if match["fence"]
            ),
            -1,
        )

    def _parse_block(self, block: str) -> List[Tuple]:
        try:
            call = json.loads(block[len(self._OPEN) : -len(self._CLOSE)])
            name = call["name"]
            parameters = call["parameters"]
            if isinstance(name, str) and name and isinstance(parameters, dict):
                return [(None, name, parameters)]
        except (KeyError, TypeError, ValueError):
            pass
        return [(block, None, None)]

    def extract_tool_calls(
        self, model_output: str
    ) -> List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
        return self._stream_blocks(model_output, model_output, finished=True) or [
            (model_output, None, None)
        ]

    def locate_argument_start(self, text: str) -> Optional[ArgumentStart]:
        position = 0
        prior_names = set()
        while (opening := text.find(self._OPEN, position)) != -1:
            body = opening + len(self._OPEN)
            end = self._find_fence(text, body)
            if end == -1:
                start = json_argument_start(text, body, "parameters")
                # A later call with the same name could be a duplicate. Wait
                # for its full arguments before exposing a new tool index.
                if start is not None and start.name not in prior_names:
                    return replace(
                        start, call_position=opening, close_frame=self._find_fence
                    )
                return None
            position = end + len(self._CLOSE)
            name = self._parse_block(text[opening:position])[0][1]
            if name:
                prior_names.add(name)
        return None

    def _stream_blocks(
        self, text: str, delta: str, *, finished: bool = False
    ) -> List[Tuple]:
        return stream_delimited_calls(
            self,
            text,
            delta,
            self._OPEN,
            self._CLOSE,
            parse_block=self._parse_block,
            finished=finished,
            find_close=self._find_fence,
            deduplicate=True,
        )

    def flush_streaming_content(self, text: str) -> Optional[str]:
        events = self._stream_blocks(text, "", finished=True)
        return events[0][0] if events else None

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ) -> List[Tuple]:
        return self._stream_blocks(current_text, delta_text)
