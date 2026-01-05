import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from . import register_tool_parser
from .abstract_tool_parser import ToolParser

logger = logging.getLogger(__name__)


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
        self.content_regex = (
            r"(<(?:think|minimax:tool_call)>.*?</(?:think|minimax:tool_call)>)"
        )
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

    def _parse_param_value(self, value: str) -> Any:
        value = value.strip()
        if not value:
            return ""
        try:
            return json.loads(value)
        except Exception:
            return value

    def _parse_invoke_calls(self, tool_block: str) -> List[Tuple[str, Dict[str, Any]]]:
        results = []
        for name, body in self.invoke_regex.findall(tool_block):
            args: Dict[str, Any] = {}
            for key, val in self.param_regex.findall(body):
                args[key] = self._parse_param_value(val)
            results.append((name, args))
        return results

    def _get_function_calls(self, model_output: str) -> List[str]:
        functions_calls = []
        last_end = 0
        for m in re.finditer(self.content_regex, model_output, re.DOTALL):
            if m.start() > last_end:
                functions_calls.append(model_output[last_end : m.start()])
            functions_calls.append(m.group(0))
            last_end = m.end()
        if last_end < len(model_output):
            functions_calls.append(model_output[last_end:])
        return functions_calls

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

    def extract_tool_calls(
        self, model_output: str
    ) -> List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
        if self.tool_call_start_token not in model_output:
            return [(model_output, None, None)]

        results: List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]] = (
            []
        )
        try:
            function_calls = self._get_function_calls(model_output)
            if not function_calls:
                return [(model_output, None, None)]

            for function_call in function_calls:
                if self.tool_call_start_token in function_call:
                    invokes = self._parse_invoke_calls(function_call)
                    if not invokes:
                        results.append((function_call, None, None))
                        continue
                    for name, args in invokes:
                        results.append((None, name, args))
                else:
                    if function_call:
                        results.append((function_call, None, None))
            return results
        except Exception as e:
            logger.error(
                "Can't parse minimax tool call output: %s. Error: %s",
                model_output,
                e,
            )
            return [(model_output, None, None)]

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ) -> Optional[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
        try:
            if self.tool_call_start_token in current_text:
                function_calls = self._get_function_calls_streaming(current_text)
                if not function_calls:
                    return None
                if self.is_contain_think(function_calls[-1]):
                    return None
                if not self._has_unclosed_tool_call(previous_text[-1]):
                    return None
                tool_block = function_calls[-1]
                if self.tool_call_end_token not in tool_block:
                    return None
                invokes = self._parse_invoke_calls(tool_block)
                if not invokes:
                    return None
                name, args = invokes[-1]
                return None, name, args
            return (delta_text, None, None)
        except Exception as e:
            logger.error("Error in MiniMax streaming tool call extraction: %s", e)
            raise
