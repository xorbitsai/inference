import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from . import register_tool_parser
from .abstract_tool_parser import ToolParser

logger = logging.getLogger(__name__)

_OPEN = r"<\|open\|>"
_CLOSE = r"<\|close\|>"
_SEP = r"<\|sep\|>"
_TEXT_UNTIL_SEP = r"(?:(?!" + _SEP + r").)*?"


@register_tool_parser("kimi-k3")
class KimiK3ToolParser(ToolParser):
    """Parse Kimi-K3 XTML response and tools channels."""

    def __init__(self):
        super().__init__()
        self.tools_open = "<|open|>tools<|sep|>"
        self.tools_close = "<|close|>tools<|sep|>"
        self.response_open = "<|open|>response<|sep|>"
        self.response_close = "<|close|>response<|sep|>"
        self.message_close = "<|close|>message<|sep|>"

        self._tools_open_re = re.compile(_OPEN + r"\s*tools\s*" + _SEP)
        self._tools_close_re = re.compile(_CLOSE + r"\s*tools\s*" + _SEP)
        self._response_open_re = re.compile(_OPEN + r"\s*response\s*" + _SEP)
        self._response_close_re = re.compile(_CLOSE + r"\s*response\s*" + _SEP)
        self._message_close_re = re.compile(_CLOSE + r"\s*message\s*" + _SEP)
        self._call_re = re.compile(
            _OPEN
            + r"\s*call\s+(?P<attrs>"
            + _TEXT_UNTIL_SEP
            + r")"
            + _SEP
            + r"(?P<body>.*?)"
            + _CLOSE
            + r"\s*call\s*"
            + _SEP,
            re.DOTALL,
        )
        self._arg_re = re.compile(
            _OPEN
            + r"\s*argument\s+(?P<attrs>"
            + _TEXT_UNTIL_SEP
            + r")"
            + _SEP
            + r"(?P<value>.*?)"
            + _CLOSE
            + r"\s*argument\s*"
            + _SEP,
            re.DOTALL,
        )
        self._attr_re = re.compile(r'(?P<key>\w+)="(?P<value>[^"]*)"')

    def _attrs(self, text: str) -> Dict[str, str]:
        return {
            match["key"]: match["value"]
            .replace("&quot;", '"')
            .replace("&amp;", "&")
            for match in self._attr_re.finditer(text)
        }

    def _decode_call(
        self, attrs: str, body: str
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        tool_name = self._attrs(attrs).get("tool")
        if not tool_name:
            return None
        arguments: Dict[str, Any] = {}
        for match in self._arg_re.finditer(body):
            arg_attrs = self._attrs(match["attrs"])
            key = arg_attrs.get("key")
            if not key:
                continue
            value: Any = match["value"]
            if arg_attrs.get("type", "string") != "string":
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            arguments[key] = value
        return tool_name, arguments

    @staticmethod
    def _hold_partial_marker(text: str, marker: str) -> str:
        for length in range(min(len(text), len(marker) - 1), 0, -1):
            if text.endswith(marker[:length]):
                return text[:-length]
        return text

    def _response_content(self, text: str) -> str:
        open_match = self._response_open_re.search(text)
        if open_match:
            text = text[open_match.end() :]
        close_match = self._response_close_re.search(text)
        if close_match:
            text = text[: close_match.start()]
        else:
            tools_match = self._tools_open_re.search(text)
            if tools_match:
                text = text[: tools_match.start()]
            else:
                text = self._hold_partial_marker(text, self.tools_open)
        text = self._message_close_re.sub("", text)
        return text

    def _parse(
        self, model_output: str
    ) -> Tuple[str, List[Tuple[str, Dict[str, Any]]]]:
        tools_match = self._tools_open_re.search(model_output)
        before_tools = (
            model_output[: tools_match.start()] if tools_match else model_output
        )
        content = self._response_content(before_tools)
        if not tools_match:
            return content, []

        start = tools_match.end()
        close_match = self._tools_close_re.search(model_output, start)
        section = (
            model_output[start : close_match.start()]
            if close_match
            else model_output[start:]
        )
        calls = []
        for match in self._call_re.finditer(section):
            call = self._decode_call(match["attrs"], match["body"])
            if call is not None:
                calls.append(call)
        return content, calls

    def extract_tool_calls(
        self, model_output: str
    ) -> List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
        try:
            content, calls = self._parse(model_output)
            result: List[
                Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]
            ] = []
            if content:
                result.append((content, None, None))
            result.extend((None, name, arguments) for name, arguments in calls)
            return result or [("", None, None)]
        except Exception as exc:
            logger.error("Cannot parse Kimi-K3 tool output: %s", exc)
            return [(model_output, None, None)]

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ):
        try:
            previous = previous_text[-1] if previous_text else ""
            previous_content, previous_calls = self._parse(previous)
            current_content, current_calls = self._parse(current_text)
            events = []
            if current_content.startswith(previous_content):
                content_delta = current_content[len(previous_content) :]
            else:
                content_delta = current_content
            if content_delta:
                events.append((content_delta, None, None))
            for index, (name, arguments) in enumerate(
                current_calls[len(previous_calls) :], start=len(previous_calls)
            ):
                events.append((None, name, arguments, index))
            if not events:
                return None
            return events[0] if len(events) == 1 else events
        except Exception as exc:
            logger.error("Cannot stream Kimi-K3 tool output: %s", exc)
            return (delta_text, None, None)
