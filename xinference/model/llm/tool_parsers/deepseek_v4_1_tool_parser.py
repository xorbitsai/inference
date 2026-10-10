import json
import re
from typing import Any, Dict, List, Optional, Tuple

from . import register_tool_parser
from .abstract_tool_parser import ToolParser

ToolCall = Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]


@register_tool_parser("deepseek-v4.1")
class DeepseekV41ToolParser(ToolParser):
    """Parse V4.1 DSML calls, whose tag names have a leading space."""

    _START = "<｜DSML｜ calls>"
    _END = "</｜DSML｜ calls>"
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
        body_start = marker_start + len(self._START)
        if body_start > previous_length:
            # The start tag completed in this delta. Release any preceding
            # text withheld while the tag was still only a partial match.
            old_text = current_text[:previous_length]
            content_start = previous_length - self._partial_marker_length(
                old_text, self.tool_call_start_tokens
            )
            content = current_text[content_start:marker_start]
            if content:
                results.append((content, None, None))

        body_end = current_text.find(self._END, body_start)
        body = current_text[body_start : body_end if body_end >= 0 else None]
        for invoke in self._INVOKE.finditer(body):
            # Emit only invokes whose closing tag arrived in this delta.
            if body_start + invoke.end() > previous_length:
                name, invoke_body = invoke.groups()
                results.append((None, name, self._parse_parameters(invoke_body)))
        return results or None
