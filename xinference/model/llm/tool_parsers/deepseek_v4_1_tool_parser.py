import json
import re
from typing import Any, Dict, List, Optional, Tuple

from . import register_tool_parser
from .abstract_tool_parser import ToolParser
from .argument_stream import ArgumentStart, dsml_argument_start

ToolCall = Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]


@register_tool_parser("deepseek-v4.1")
class DeepseekV41ToolParser(ToolParser):
    """Parse V4.1 DSML calls, whose tag names have a leading space."""

    _START = "<｜DSML｜ calls>"
    _END = "</｜DSML｜ calls>"
    _INVOKE_END = "</｜DSML｜ invoke>"
    _INVOKE = re.compile(
        r'<｜DSML｜ invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜ invoke>',
        re.DOTALL,
    )
    _PARAMETER = re.compile(
        r'<｜DSML｜ parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>'
        r"(.*?)</｜DSML｜ parameter>",
        re.DOTALL,
    )

    def __init__(self):
        self.tool_call_start_tokens = [self._START]

    def _parse_parameters(self, body: str) -> Dict[str, Any]:
        parameters: Dict[str, Any] = {}
        for name, is_string, value in self._PARAMETER.findall(body):
            if is_string == "true":
                parameters[name] = value
            else:
                try:
                    parameters[name] = json.loads(value)
                except json.JSONDecodeError:
                    parameters[name] = value
        return parameters

    def extract_tool_calls(self, model_output: str) -> List[ToolCall]:
        results: List[ToolCall] = []
        blocks = re.finditer(
            re.escape(self._START) + r"(.*?)" + re.escape(self._END),
            model_output,
            re.DOTALL,
        )
        content_start = 0
        for block in blocks:
            invokes = self._INVOKE.findall(block.group(1))
            if not invokes:
                continue
            content = model_output[content_start : block.start()]
            if content:
                results.append((content, None, None))
            for name, body in invokes:
                results.append((None, name, self._parse_parameters(body)))
            content_start = block.end()
        if results:
            content = model_output[content_start:]
            if content:
                results.append((content, None, None))
        return results or [(model_output, None, None)]

    def locate_argument_start(self, text: str) -> Optional[ArgumentStart]:
        return dsml_argument_start(text, "calls", "｜DSML｜ ", self._parse_parameters)

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ) -> Optional[List[ToolCall]]:
        # The parser is shared by concurrent requests. Derive progress solely
        # from this request's accumulated text and delta, never instance state.
        previous_length = len(current_text) - len(delta_text)
        marker_start = current_text.find(self._START)
        if marker_start < 0:
            content = self._plain_text_delta(
                current_text, delta_text, self.tool_call_start_tokens
            )
            return [(content, None, None)] if content else None

        results: List[ToolCall] = []
        # The wrapper can contain several calls, and one response can contain
        # several wrappers. Skip regex work until an invoke can have closed.
        scan_invokes = (
            current_text.find(
                self._INVOKE_END, max(previous_length - len(self._INVOKE_END) + 1, 0)
            )
            != -1
        )
        position = 0
        held = self._partial_marker_length(
            current_text[:previous_length], self.tool_call_start_tokens
        )
        while marker_start != -1:
            content_start = max(position, previous_length - held)
            if marker_start > content_start:
                results.append((current_text[content_start:marker_start], None, None))
            body_start = marker_start + len(self._START)
            body_end = current_text.find(self._END, body_start)
            end = body_end if body_end != -1 else len(current_text)
            if scan_invokes and end > previous_length:
                for invoke in self._INVOKE.finditer(current_text, body_start, end):
                    if invoke.end() > previous_length:
                        name, invoke_body = invoke.groups()
                        results.append(
                            (None, name, self._parse_parameters(invoke_body))
                        )
            if body_end == -1:
                return results or None
            position = body_end + len(self._END)
            marker_start = current_text.find(self._START, position)
        content_start = max(position, previous_length - held)
        content_end = len(current_text) - self._partial_marker_length(
            current_text, self.tool_call_start_tokens
        )
        if content_end > content_start:
            results.append((current_text[content_start:content_end], None, None))
        return results or None
