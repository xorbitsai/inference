"""Stream confirmed arguments while the existing parser validates completed calls."""

import json
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple

from .abstract_tool_parser import ToolArgumentsDelta, ToolParser, request_tools_var


@dataclass(frozen=True)
class ArgumentStart:
    name: str
    position: int
    prefix: str = ""
    terminator: str = ""
    boundaries: str = ""
    trim: bool = False
    call_position: Optional[int] = None
    preceding_calls: int = 0


class ArgumentReader:
    """Serialize fragments of one call while the original parser validates it."""

    def __init__(self, start: ArgumentStart):
        self.start = start
        self.position = start.position
        self.fragments: List[str] = []
        self.closed = False
        self.depth = 0
        self.in_string = False
        self.escaped = False

    def _json_end(self, text: str) -> int:
        # Scan only new characters and leave the outer object open. The legacy
        # parser validates the complete JSON and the enclosing tool syntax.
        for offset in range(self.position, len(text)):
            char = text[offset]
            if self.in_string:
                if self.escaped:
                    self.escaped = False
                elif char == "\\":
                    self.escaped = True
                elif char == '"':
                    self.in_string = False
            elif char == '"':
                self.in_string = True
            elif char == "{":
                self.depth += 1
            elif char == "}":
                self.depth -= 1
                if self.depth == 0:
                    self.closed = True
                    return offset
        return len(text)

    def _string_end(self, text: str, start: ArgumentStart) -> int:
        end = text.find(start.terminator, self.position)
        if end == -1:
            end = len(text) - ToolParser._partial_marker_length(
                text, (start.terminator,)
            )
        else:
            self.closed = True
        for char in start.boundaries:
            boundary = text.find(char, self.position, end)
            if boundary != -1:
                self.closed = True
                end = boundary
        if start.trim:
            while end > self.position and text[end - 1].isspace():
                end -= 1
        return end

    def finish(self, name: str, arguments: Dict[str, Any], text: str) -> str:
        start = self.start
        if start.name != name:
            raise ValueError("Tool name changed after streaming arguments")
        serialized = "".join(self.fragments)
        if not start.terminator:
            raw = text[start.position :]
            value, length = json.JSONDecoder().raw_decode(raw)
            if value != arguments:
                raise ValueError("Tool arguments changed after streaming")
            return raw[len(serialized) : length]
        final = json.dumps(arguments, ensure_ascii=False)
        if not final.startswith(serialized):
            raise ValueError("Tool arguments changed after streaming")
        return final[len(serialized) :]

    def resume(self, start: ArgumentStart) -> bool:
        # Move to the next confirmed string field in this same invocation.
        if (
            self.start.call_position is None
            or start.call_position != self.start.call_position
            or start.position <= self.start.position
        ):
            return False
        emitted = "".join(self.fragments)
        if not start.prefix.startswith(emitted):
            return False
        self.start = replace(start, prefix=start.prefix[len(emitted) :])
        self.position = start.position
        self.closed = False
        return True

    def read(self, text: str) -> str:
        start = self.start
        if self.closed:
            return ""
        end = (
            self._string_end(text, start) if start.terminator else self._json_end(text)
        )
        if end <= self.position:
            return ""
        fragment = text[self.position : end]
        if start.terminator:
            fragment = json.dumps(fragment, ensure_ascii=False)[1:-1]
        if self.position == start.position:
            fragment = start.prefix + fragment
        self.position = end
        self.fragments.append(fragment)
        return fragment


class ArgumentStream:
    """Associate partial and complete events by call order within one request."""

    def __init__(self) -> None:
        self.index = 0
        self.readers: Dict[int, ArgumentReader] = {}
        self.active: Optional[int] = None

    def advance(
        self,
        parser: ToolParser,
        locate: Callable[[str], Optional[ArgumentStart]],
        previous: List[str],
        text: str,
        delta: str,
        finish_reason: Optional[str],
    ) -> List[Tuple]:
        parsed = parser.extract_tool_calls_streaming(previous, text, delta)
        events: List[Tuple] = []
        for event in parsed if isinstance(parsed, list) else [parsed] if parsed else ():
            content, name, args = event[:3]
            if not name:
                events.append(event)
                continue
            reader = self.readers.pop(self.index, None)
            if reader is not None and reader.fragments:
                args = ToolArgumentsDelta(reader.finish(name, args, text))
            events.append((content, name, args, self.index))
            self.index += 1

        if finish_reason is not None:
            if finish_reason in ("stop", "tool_calls") and any(
                reader.fragments for reader in self.readers.values()
            ):
                raise ValueError("Tool call ended before completing its arguments")
            if not self.index and not self.readers:
                flush = getattr(parser, "flush_streaming_content", None)
                if flush and (content := flush(text)):
                    events.append((content, None, None))
            return events

        reader = self.readers.get(self.active) if self.active is not None else None
        if reader is not None and not reader.closed:
            fragment = reader.read(text)
            if fragment:
                events.append(
                    (None, reader.start.name, ToolArgumentsDelta(fragment), self.active)
                )
            if not reader.closed:
                return events

        start = locate(text)
        if start is None or not requested_tool(start.name):
            return events
        index = self.index + start.preceding_calls
        existing = self.readers.get(index)
        if existing is not None:
            if not existing.resume(start):
                return events
            reader = existing
        else:
            reader = self.readers[index] = ArgumentReader(start)
        self.active = index
        fragment = reader.read(text)
        if fragment:
            events.append((None, start.name, ToolArgumentsDelta(fragment), index))
        return events


def tagged_string_start(
    text: str,
    position: int,
    name: str,
    parameter: Pattern,
    terminator: str,
    parse_completed: Callable[[str], Dict[str, Any]],
    *,
    explicitly_typed: bool = False,
    boundaries: str = "",
    trim: bool = False,
    preceding_calls: int = 0,
) -> Optional[ArgumentStart]:
    """Find the open string parameter and serialize already completed fields."""
    body_start = position
    while match := parameter.match(text, position):
        tag = match.groupdict().get("tag")
        key = (match["key"] or tag or "").strip()
        close = f"</{tag}>" if tag else terminator
        value_close = close
        value_trim = trim
        value_boundaries = boundaries
        start = match.end()
        if trim:
            while start < len(text) and text[start].isspace():
                start += 1
            cdata_open = "<![CDATA["
            if text.startswith(cdata_open, start):
                start += len(cdata_open)
                value_close = "]]>"
                value_trim = False
                value_boundaries = ""
            elif cdata_open.startswith(text[start : start + len(cdata_open)]):
                return None
        end = text.find(value_close, start)
        if end != -1 and value_close != close:
            end = text.find(close, end + len(value_close))
            if end == -1:
                return None
        if end != -1:
            position = end + len(close)
            continue
        if explicitly_typed:
            if match["type"] not in (None, "string", "true"):
                return None
        elif not ToolParser._is_string_parameter(name, key):
            return None
        arguments = parse_completed(text[body_start:position])
        if key in arguments:
            return None
        arguments[key] = ""
        return ArgumentStart(
            name,
            start,
            json.dumps(arguments, ensure_ascii=False)[:-2],
            value_close,
            value_boundaries,
            value_trim,
            body_start,
            preceding_calls,
        )
    return None


def open_block(text: str, opening: str, closing: str) -> Optional[int]:
    """Return the body offset of the last unclosed block."""
    position = text.rfind(opening)
    if position < 0:
        return None
    position += len(opening)
    return position if text.find(closing, position) == -1 else None


_JSON_HEADER = re.compile(
    r'\s*\{\s*"name"\s*:\s*("(?:[^"\\]|\\.)*")\s*,\s*'
    r'"(?:arguments|parameters)"\s*:\s*(?=\{)'
)


def json_argument_start(text: str, position: int) -> Optional[ArgumentStart]:
    match = _JSON_HEADER.match(text, position)
    if match:
        try:
            return ArgumentStart(json.loads(match[1]), match.end())
        except json.JSONDecodeError:
            pass
    return None


def dsml_argument_start(
    text: str,
    wrapper: str,
    prefix: str,
    parse_parameters: Callable[[str], Dict[str, Any]],
) -> Optional[ArgumentStart]:
    body = open_block(text, f"<{prefix}{wrapper}>", f"</{prefix}{wrapper}>")
    if body is None:
        return None
    invoke_end = f"</{prefix}invoke>"
    last_end = text.rfind(invoke_end, body)
    if last_end != -1:
        body = last_end + len(invoke_end)
    invoke = re.compile(rf'\s*<{re.escape(prefix)}invoke\s+name="([^"]+)"\s*>')
    match = invoke.match(text, body)
    if match is None:
        return None
    parameter = re.compile(
        rf'\s*<{re.escape(prefix)}parameter\s+name="(?P<key>[^"]+)"'
        r'\s+string="(?P<type>true|false)"\s*>'
    )
    return tagged_string_start(
        text,
        match.end(),
        match[1],
        parameter,
        f"</{prefix}parameter>",
        parse_parameters,
        explicitly_typed=True,
    )


def stream_delimited_calls(
    parser: ToolParser,
    text: str,
    delta: str,
    opening: str,
    closing: str,
    *,
    parse_block: Optional[Callable[[str], List[Tuple]]] = None,
) -> List[Tuple]:
    """Stream text around delimited blocks and parse each newly closed block."""
    previous_end = len(text) - len(delta)
    position = 0
    events: List[Tuple] = []
    while (start := text.find(opening, position)) != -1:
        content_start = max(position, previous_end)
        if position <= previous_end:
            content_start -= min(
                previous_end - position,
                parser._partial_marker_length(text[:previous_end], (opening,)),
            )
        if start > content_start:
            events.append((text[content_start:start], None, None))
        end = text.find(closing, start + len(opening))
        if end == -1:
            return events
        position = end + len(closing)
        if position > previous_end:
            parse = parse_block or parser.extract_tool_calls
            events.extend(parse(text[start:position]))
    if position < len(text):
        content = text[max(position, previous_end) :]
        if position < previous_end:
            held = min(
                previous_end - position,
                parser._partial_marker_length(text[:previous_end], (opening,)),
            )
            content = text[previous_end - held : previous_end] + content
        held = parser._partial_marker_length(content, (opening,))
        if len(content) > held:
            events.append((content[: len(content) - held], None, None))
    return events


def requested_tool(name: str) -> bool:
    for tool in request_tools_var.get() or ():
        function = tool.get("function", tool) if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name") == name:
            return True
    return False
