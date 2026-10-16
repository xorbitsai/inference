import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from . import register_tool_parser
from .abstract_tool_parser import ToolParser

logger = logging.getLogger(__name__)

ToolEvent = Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]


@register_tool_parser("gemma")
class GemmaToolParser(ToolParser):
    """
    Tool parser for Gemma-4 style tool call blocks.

    Gemma emits tool invocations using tokens like:
        <|tool_call>call:get_weather{location:<|"|>Shanghai<|"|>}<tool_call|>
    where strings are wrapped with <|"|> ... <|"|>.
    """

    def __init__(self):
        self.tool_call_start_token = "<|tool_call>"
        self.tool_call_end_token = "<tool_call|>"
        self.string_token = '<|"|>'

        self.tool_call_regex = re.compile(
            r"(<\|tool_call\>.*?<tool_call\|>)", re.DOTALL
        )
        self.call_header_regex = re.compile(r"call\s*:\s*([^{\s]+)", re.IGNORECASE)

    @staticmethod
    def _quote_keys(text: str) -> str:
        pattern = re.compile(r"(?P<prefix>[{,])\s*(?P<key>[A-Za-z0-9_\-]+)\s*:")

        def repl(match: re.Match) -> str:
            prefix = match.group("prefix")
            key = match.group("key")
            return f'{prefix}"{key}":'

        while True:
            new_text, count = pattern.subn(repl, text)
            text = new_text
            if count == 0:
                break
        return text

    def _arguments_to_json(self, arg_block: str) -> str:
        """Rewrite Gemma's argument block as JSON.

        Gemma delimits strings with ``<|"|>`` rather than quoting them, so the
        text between two of those tokens is a literal: it may itself contain
        double quotes, backslashes or newlines, and re-emitting it verbatim
        would produce invalid JSON. Splitting on the delimiter therefore yields
        alternating segments — even indices are structural, odd ones are string
        contents — so each string is re-encoded and bare keys are quoted only in
        the structural text. A value containing something like ``a,b:c`` must
        not be mistaken for a key.
        """
        parts = arg_block.split(self.string_token)
        if len(parts) % 2 == 0:
            # an odd number of delimiters leaves a string open
            raise ValueError("Unterminated string in tool call arguments")
        return "".join(
            (
                json.dumps(part, ensure_ascii=False)
                if index % 2
                else self._quote_keys(part)
            )
            for index, part in enumerate(parts)
        )

    def _parse_arguments(self, arg_block: str) -> Dict[str, Any]:
        cleaned = arg_block.strip()
        if not cleaned:
            return {}
        return json.loads(self._arguments_to_json(cleaned))

    def _parse_tool_call_block(
        self, block: str
    ) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        content = block.strip()
        try:
            # Remove wrapper tokens
            if content.startswith(self.tool_call_start_token):
                content = content[len(self.tool_call_start_token) :]
            if content.endswith(self.tool_call_end_token):
                content = content[: -len(self.tool_call_end_token)]
            content = content.strip()

            match = self.call_header_regex.search(content)
            if not match:
                raise ValueError("Missing call header")
            func_name = match.group(1).strip()

            brace_start = content.find("{", match.end())
            brace_end = content.rfind("}")
            if brace_start == -1 or brace_end == -1 or brace_end < brace_start:
                args = {}
            else:
                args_str = content[brace_start : brace_end + 1]
                args = self._parse_arguments(args_str)
            return (None, func_name, args)
        except Exception as exc:
            logger.warning("Failed to parse Gemma tool call: %s, error: %s", block, exc)
            return (block, None, None)

    def extract_tool_calls(
        self, model_output: str
    ) -> List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
        if self.tool_call_start_token not in model_output:
            return [(model_output, None, None)]

        results: List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]] = (
            []
        )
        last_end = 0
        for match in self.tool_call_regex.finditer(model_output):
            if match.start() > last_end:
                content = model_output[last_end : match.start()]
                if content:
                    results.append((content, None, None))
            block = match.group(0)
            results.append(self._parse_tool_call_block(block))
            last_end = match.end()

        if last_end < len(model_output):
            remainder = model_output[last_end:]
            if remainder:
                results.append((remainder, None, None))

        return results or [(model_output, None, None)]

    def extract_tool_calls_streaming(
        self,
        previous_texts: List[str],
        current_text: str,
        delta_text: str,
    ) -> Optional[Union[ToolEvent, List[ToolEvent]]]:
        if self.tool_call_start_token not in current_text:
            return (delta_text, None, None)

        prev_text = previous_texts[-1] if previous_texts else ""
        # Text after an unclosed start token belongs to a tool call that is
        # still streaming. Everything before it is settled: plain text and
        # complete tool call blocks. Emit only what became settled in this
        # chunk, in order, so that a second tool call is never sent as content
        # and text sharing a chunk with a tag is not dropped.
        start = self._settled_length(prev_text)
        end = self._settled_length(current_text)
        if end <= start:
            return None

        events: List[ToolEvent] = []
        position = start
        for match in self.tool_call_regex.finditer(current_text, start, end):
            if match.start() > position:
                events.append((current_text[position : match.start()], None, None))
            events.append(self._parse_tool_call_block(match.group(0)))
            position = match.end()
        if position < end:
            events.append((current_text[position:end], None, None))
        return events[0] if len(events) == 1 else events

    def _settled_length(self, text: str) -> int:
        """Return the length of the prefix of ``text`` that holds no unclosed
        tool call block."""
        position = 0
        for match in self.tool_call_regex.finditer(text):
            position = match.end()
        open_index = text.find(self.tool_call_start_token, position)
        return len(text) if open_index == -1 else open_index
