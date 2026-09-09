# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from . import register_tool_parser
from .abstract_tool_parser import ToolParser


@register_tool_parser("minicpm5")
class MiniCPM5ToolParser(ToolParser):
    """Parse MiniCPM5 XML-style function calls."""

    _FUNCTION_START = "<function"
    _FUNCTION_END = "</function>"
    _CDATA_START = "<![CDATA["
    _CDATA_END = "]]>"
    _FUNCTION_OPEN_RE = re.compile(r"<function\s+name=['\"]([^'\"]+)['\"]\s*>")
    _PARAM_OPEN_RE = re.compile(r"<param\s+name=['\"]([^'\"]+)['\"]\s*>")

    @staticmethod
    def _parse_value(value: str) -> Any:
        value = value.strip()
        if value.startswith("<![CDATA[") and value.endswith("]]>"):
            value = value[9:-3]
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value

    @classmethod
    def _find_closing_tag(cls, text: str, tag: str, start: int) -> Optional[int]:
        closing_tag = f"</{tag}>"
        position = start
        while True:
            closing_position = text.find(closing_tag, position)
            cdata_position = text.find(cls._CDATA_START, position)
            if closing_position == -1:
                return None
            if cdata_position == -1 or closing_position < cdata_position:
                return closing_position
            cdata_end = text.find(
                cls._CDATA_END, cdata_position + len(cls._CDATA_START)
            )
            if cdata_end == -1:
                return None
            position = cdata_end + len(cls._CDATA_END)

    @classmethod
    def _parse_calls(
        cls, model_output: str
    ) -> List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
        results: List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]] = (
            []
        )
        position = 0
        while match := cls._FUNCTION_OPEN_RE.search(model_output, position):
            function_end = cls._find_closing_tag(model_output, "function", match.end())
            if function_end is None:
                break
            if match.start() > position:
                results.append((model_output[position : match.start()], None, None))
            arguments: Dict[str, Any] = {}
            param_position = match.end()
            while param_match := cls._PARAM_OPEN_RE.search(
                model_output, param_position, function_end
            ):
                param_end = cls._find_closing_tag(
                    model_output, "param", param_match.end()
                )
                if param_end is None or param_end > function_end:
                    break
                arguments[param_match.group(1)] = cls._parse_value(
                    model_output[param_match.end() : param_end]
                )
                param_position = param_end + len("</param>")
            results.append((None, match.group(1), arguments))
            position = function_end + len(cls._FUNCTION_END)
        if position < len(model_output):
            results.append((model_output[position:], None, None))
        return results

    def extract_tool_calls(
        self, model_output: str
    ) -> List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
        if (
            not isinstance(model_output, str)
            or self._FUNCTION_START not in model_output
        ):
            return [(str(model_output), None, None)]
        return self._parse_calls(model_output)

    @classmethod
    def _completed_output(cls, text: str) -> str:
        for length in range(len(cls._FUNCTION_START) - 1, 0, -1):
            if text.endswith(cls._FUNCTION_START[:length]):
                text = text[:-length]
                break
        position = 0
        while True:
            function_start = text.find(cls._FUNCTION_START, position)
            if function_start == -1:
                break
            opening_end = text.find(">", function_start + len(cls._FUNCTION_START))
            if opening_end == -1:
                return text[:function_start]
            match = cls._FUNCTION_OPEN_RE.match(text, function_start)
            if match is None:
                position = opening_end + 1
                continue
            function_end = cls._find_closing_tag(text, "function", match.end())
            if function_end is None:
                return text[: match.start()]
            position = function_end + len(cls._FUNCTION_END)
        return text

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ):
        previous = self._completed_output(previous_text[-1] if previous_text else "")
        current = self._completed_output(current_text)
        previous_results = self._parse_calls(previous)
        current_results = self._parse_calls(current)
        previous_call_count = sum(name is not None for _, name, _ in previous_results)
        previous_content_length = sum(
            len(content or "") for content, name, _ in previous_results if name is None
        )
        events: List[Any] = []
        call_count = 0
        current_content_length = 0
        for content, name, arguments in current_results:
            if name is not None:
                if call_count >= previous_call_count:
                    events.append((None, name, arguments, call_count))
                call_count += 1
            elif content:
                start = max(0, previous_content_length - current_content_length)
                if start < len(content):
                    events.append((content[start:], None, None))
                current_content_length += len(content)
        if not events:
            return None
        return events[0] if len(events) == 1 else events
