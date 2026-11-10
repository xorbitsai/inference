import json
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

from . import register_tool_parser
from .abstract_tool_parser import ToolParser

logger = logging.getLogger(__name__)

# MiniMax-M3's chat template writes this token before every tool-call tag, and
# only text that follows it is markup: ]<]minimax[>[<tool_call>, ...<invoke>,
# ...<city>Beijing]<]minimax[>[</city>, ...</tool_call>.
NAMESPACE = "]<]minimax[>["
NAMESPACED_START = NAMESPACE + "<tool_call>"
NAMESPACED_END = NAMESPACE + "</tool_call>"


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
        # In the namespaced format only tags after NAMESPACE are markup, so an
        # argument may hold a raw <invoke>...</invoke> without ending the call.
        self.namespaced_invoke_regex = re.compile(
            re.escape(NAMESPACE)
            + r"<invoke\s+name=[\"']([^\"']+)[\"']>(.*?)"
            + re.escape(NAMESPACE)
            + r"</invoke>",
            re.DOTALL,
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

    def _scan_blocks(self, text: str) -> Tuple[List[str], Optional[int]]:
        """Split text into plain text and blocks, left to right.

        Each block ends at the end tag of its own format, so a tag of the
        other format inside an argument is text. Returns the segments and the
        index where an unclosed block starts, if any.
        """
        pairs = {
            NAMESPACED_START: NAMESPACED_END,
            self.tool_call_start_token: self.tool_call_end_token,
            self.think_start_token: self.think_end_token,
        }
        segments: List[str] = []
        pos = search = 0
        while True:
            found = [(text.find(tag, search), tag) for tag in pairs]
            found = [(index, tag) for index, tag in found if index != -1]
            if not found:
                break
            index, tag = min(found)
            end = text.find(pairs[tag], index + len(tag))
            if end == -1 and tag == self.think_start_token:
                # An unclosed think block is text; keep looking for tool calls.
                search = index + len(tag)
                continue
            if index > pos:
                segments.append(text[pos:index])
            if end == -1:
                return segments, index
            pos = search = end + len(pairs[tag])
            segments.append(text[index:pos])
        if pos < len(text):
            segments.append(text[pos:])
        return segments, None

    def _block_format(self, segment: str) -> Optional[str]:
        """The start tag of the first tool-call block in segment, if any."""
        found = [
            (segment.find(tag), tag)
            for tag in (NAMESPACED_START, self.tool_call_start_token)
            if tag in segment
        ]
        return min(found)[1] if found else None

    @staticmethod
    def _namespace_to_xml(body: str) -> str:
        """Keep the namespaced tags and escape everything else, which is text."""
        parts = body.split(NAMESPACE)
        xml = [escape(parts[0])]
        for part in parts[1:]:
            tag_end = part.find(">") + 1 if part.startswith("<") else 0
            xml.append(part[:tag_end] + escape(part[tag_end:]))
        return "".join(xml)

    def _parse_invoke_args(
        self, name: str, body: str, namespaced: bool = False
    ) -> Dict[str, Any]:
        if namespaced:
            body = self._namespace_to_xml(body)
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
                key: self._parse_param_value(
                    value, self._is_string_parameter(name, key)
                )
                for key, value in self.param_regex.findall(body)
            }

        args: Dict[str, Any] = {}
        duplicate_keys = set()
        for element in root:
            if element.tag == "parameter" and "name" in element.attrib:
                key = element.attrib["name"]
            else:
                key = element.tag
            if len(element) == 0 and self._is_string_parameter(name, key):
                value = self._parse_param_value(element.text or "", is_string=True)
            else:
                value = self._parse_xml_element(element)
            if key not in args:
                args[key] = value
            elif key in duplicate_keys:
                args[key].append(value)
            else:
                args[key] = [args[key], value]
                duplicate_keys.add(key)
        return args

    def _parse_invoke_calls(
        self, tool_block: str, block_format: Optional[str] = None
    ) -> List[Tuple[str, Dict[str, Any]]]:
        results = []
        if block_format is None:
            block_format = self._block_format(tool_block)
        regex = (
            self.namespaced_invoke_regex
            if block_format == NAMESPACED_START
            else self.invoke_regex
        )
        for name, body in regex.findall(tool_block):
            results.append(
                (
                    name,
                    self._parse_invoke_args(
                        name, body, block_format == NAMESPACED_START
                    ),
                )
            )
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
        if self._block_format(model_output) is None:
            return [(model_output, None, None)]

        results: List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]] = (
            []
        )
        try:
            function_calls, unclosed = self._scan_blocks(model_output)
            unclosed_text = None
            if unclosed is not None:
                if model_output.startswith(NAMESPACED_START, unclosed):
                    # An unclosed M3 block is an unfinished turn, not a call.
                    unclosed_text = model_output[unclosed:]
                else:
                    function_calls.append(model_output[unclosed:])
            if not function_calls:
                return [(model_output, None, None)]

            for function_call in function_calls:
                block_format = self._block_format(function_call)
                if block_format is not None:
                    invokes = self._parse_invoke_calls(function_call, block_format)
                    if not invokes:
                        results.append((function_call, None, None))
                        continue
                    for name, args in invokes:
                        results.append((None, name, args))
                else:
                    if function_call:
                        results.append((function_call, None, None))
            if unclosed_text:
                results.append((unclosed_text, None, None))
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
            # Hold back a start tag that may still be arriving. The namespace
            # token arrives whole, before <tool_call>.
            held = 0
            for start in (NAMESPACED_START, self.tool_call_start_token):
                for prefix_length in range(len(start) - 1, held, -1):
                    if text.endswith(start[:prefix_length]):
                        held = prefix_length
                        break
            if held:
                text = text[:-held]
            _, unclosed = self._scan_blocks(text)
            if unclosed is not None:
                text = text[:unclosed]
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
