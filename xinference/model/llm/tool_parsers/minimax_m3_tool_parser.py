import json
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

from . import register_tool_parser
from .abstract_tool_parser import ToolParser

logger = logging.getLogger(__name__)


@register_tool_parser("minimax_m3")
class MiniMaxM3ToolParser(ToolParser):
    """
    Tool parser implementation for MiniMax models.

    This parser handles MiniMax tool calls wrapped with <minimax:tool_call>
    tags and <invoke> blocks containing recursively nested XML arguments.
    """

    def __init__(self):
        super().__init__()
        self.think_start_token: str = "<mm:think>"
        self.think_end_token: str = "</mm:think>"
        self.tool_call_start_token: str = "<minimax:tool_call>"
        self.tool_call_end_token: str = "</minimax:tool_call>"

        self.think_regex = re.compile(r"<mm:think>(.*?)</mm:think>", re.DOTALL)
        self.content_regex = re.compile(
            r"(<mm:think>.*?</mm:think>|"
            r"<minimax:tool_call>.*?</minimax:tool_call>)",
            re.DOTALL,
        )
        self.tool_call_complete_regex = re.compile(
            r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL
        )
        self.tool_call_regex = re.compile(
            r"<minimax:tool_call>.*?</minimax:tool_call>|<minimax:tool_call>.*?$",
            re.DOTALL,
        )
        self.invoke_regex = re.compile(
            r"<invoke\s+name=[\"']([^\"']+)[\"']>(.*?)</invoke>", re.DOTALL
        )
        self.param_regex = re.compile(
            r"<parameter\s+name=[\"']([^\"']+)[\"']>(.*?)</parameter>", re.DOTALL
        )

    def _parse_param_value(self, value: str) -> Any:
        value = value.strip()
        if not value:
            return ""
        try:
            return json.loads(value)
        except Exception:
            return value

    def _parse_xml_element(self, element: ET.Element) -> Any:
        children = list(element)
        if not children:
            return self._parse_param_value(element.text or "")

        if all(child.tag == "item" for child in children):
            return [self._parse_xml_element(child) for child in children]

        result: Dict[str, Any] = {}
        duplicate_tags = set()
        for child in children:
            child_value = self._parse_xml_element(child)
            if child.tag not in result:
                result[child.tag] = child_value
            elif child.tag in duplicate_tags:
                result[child.tag].append(child_value)
            else:
                result[child.tag] = [result[child.tag], child_value]
                duplicate_tags.add(child.tag)
        return result

    def _parse_invoke_args(self, body: str) -> Dict[str, Any]:
        # Model output is often XML-like rather than strictly valid XML. Escape
        # bare ampersands while preserving the five predefined XML entities and
        # numeric character references.
        escaped_body = re.sub(
            r"&(?!amp;|lt;|gt;|quot;|apos;|#[0-9]+;|#x[0-9A-Fa-f]+;)",
            "&amp;",
            body,
        )
        try:
            root = ET.fromstring(f"<root>{escaped_body}</root>")
        except ET.ParseError:
            # Keep compatibility with the legacy MiniMax parameter format if
            # the model emits text that is not valid nested XML.
            return {
                key: self._parse_param_value(value)
                for key, value in self.param_regex.findall(body)
            }

        args: Dict[str, Any] = {}
        duplicate_keys = set()
        for element in root:
            if element.tag == "parameter" and "name" in element.attrib:
                key = element.attrib["name"]
            else:
                key = element.tag
            value = self._parse_xml_element(element)
            if key not in args:
                args[key] = value
            elif key in duplicate_keys:
                args[key].append(value)
            else:
                args[key] = [args[key], value]
                duplicate_keys.add(key)
        return args

    def _parse_invoke_calls(self, tool_block: str) -> List[Tuple[str, Dict[str, Any]]]:
        results = []
        for name, body in self.invoke_regex.findall(tool_block):
            results.append((name, self._parse_invoke_args(body)))
        return results

    def _get_function_calls(self, model_output: str) -> List[str]:
        functions_calls = []
        last_end = 0
        for m in self.content_regex.finditer(model_output):
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
    ) -> Optional[Any]:
        def completed_blocks(
            text: str,
        ) -> List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
            for prefix_length in range(len(self.tool_call_start_token) - 1, 0, -1):
                partial_start = self.tool_call_start_token[:prefix_length]
                if text.endswith(partial_start):
                    text = text[:-prefix_length]
                    break
            if self._has_unclosed_tool_call(text):
                text = text[: text.rfind(self.tool_call_start_token)]
            return self.extract_tool_calls(text)

        try:
            prev_text = previous_text[-1] if previous_text else ""
            previous_blocks = completed_blocks(prev_text)
            current_blocks = completed_blocks(current_text)

            previous_tool_count = sum(
                1 for _, name, _ in previous_blocks if name is not None
            )
            previous_plain_length = sum(
                len(content)
                for content, name, _ in previous_blocks
                if name is None and content
            )

            events: List[Any] = []
            tool_count = 0
            plain_length = 0
            for content, name, args in current_blocks:
                if name is not None:
                    if tool_count >= previous_tool_count:
                        events.append((None, name, args, tool_count))
                    tool_count += 1
                    continue

                if not content:
                    continue
                content_start = max(0, previous_plain_length - plain_length)
                if content_start < len(content):
                    events.append((content[content_start:], None, None))
                plain_length += len(content)

            if not events:
                return None
            if len(events) == 1:
                return events[0]
            return events
        except Exception as e:
            logger.error("Error in MiniMax streaming tool call extraction: %s", e)
            return (delta_text, None, None)
