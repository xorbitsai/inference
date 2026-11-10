"""Stream confirmed arguments while the existing parser validates completed calls."""

import json
import logging
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple

from .abstract_tool_parser import ToolArgumentsDelta, ToolParser, request_tools_var

logger = logging.getLogger(__name__)


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
    allow_control_chars: bool = False
    json_suffix: str = ""


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

    def _read_json(
        self, text: str, end: int, *, final: bool = False
    ) -> Tuple[int, str]:
        # Scan only new characters and leave the outer object open. The legacy
        # parser validates the complete JSON and the enclosing tool syntax.
        start = self.position
        pieces = []
        for offset in range(start, end):
            char = text[offset]
            if self.in_string:
                if self.escaped:
                    self.escaped = False
                elif char == "\\":
                    self.escaped = True
                elif char == '"':
                    self.in_string = False
                elif self.start.allow_control_chars and ord(char) < 0x20:
                    pieces.append(text[start:offset])
                    pieces.append(json.dumps(char)[1:-1])
                    start = offset + 1
            elif char == '"':
                self.in_string = True
            elif char == "{":
                self.depth += 1
            elif char == "}":
                self.depth -= 1
                if self.depth == 0 and not final:
                    self.closed = True
                    end = offset
                    break
        pieces.append(text[start:end])
        return end, "".join(pieces)

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

    def finish(self, name: str, arguments: Dict[str, Any], text: str) -> Optional[str]:
        start = self.start
        if start.name != name:
            return None
        if not start.terminator:
            raw = text[start.position :]
            try:
                decoder = json.JSONDecoder(strict=not start.allow_control_chars)
                value, length = decoder.raw_decode(raw)
            except json.JSONDecodeError:
                return None
            if value != arguments or (
                start.json_suffix
                and not raw[length:].lstrip().startswith(start.json_suffix)
            ):
                return None
            end = start.position + length
            if self.closed:
                return text[self.position : end]
            return self._read_json(text, end, final=True)[1]
        final = json.dumps(arguments, ensure_ascii=False)
        serialized = "".join(self.fragments)
        return final[len(serialized) :] if final.startswith(serialized) else None

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
        if start.terminator:
            end = self._string_end(text, start)
            fragment = json.dumps(text[self.position : end], ensure_ascii=False)[1:-1]
        else:
            end, fragment = self._read_json(text, len(text))
        if end <= self.position:
            return ""
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
        self.incomplete = False

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
            while (reader := self.readers.pop(self.index, None)) is not None:
                if not reader.fragments:
                    break
                fragment = (
                    reader.finish(name, args, text) if isinstance(args, dict) else None
                )
                if fragment is not None:
                    args = ToolArgumentsDelta(fragment)
                    break
                # The legacy parser can skip malformed calls. Keep the
                # abandoned index reserved and deliver later valid calls with
                # fresh indices. Never send a closing brace for an invalid call.
                self.incomplete = True
                logger.debug("Tool call %d could not be reconciled", self.index)
                self.index += 1
            events.append((content, name, args, self.index))
            self.index += 1

        if finish_reason is not None:
            self.incomplete |= any(reader.fragments for reader in self.readers.values())
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
        # A later invoke must never appear before the first one. The
        # single-tool stream filters on the first index it sees.
        if any(
            not self.readers.get(i) or not self.readers[i].fragments
            for i in range(self.index, index)
        ):
            return events
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
        key = match["key"].strip()
        close = terminator
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
    r'"(arguments|parameters)"\s*:\s*(?=\{)'
)


def json_argument_start(
    text: str, position: int, key: str, *, allow_control_chars: bool = False
) -> Optional[ArgumentStart]:
    match = _JSON_HEADER.match(text, position)
    if match and match[2] == key:
        try:
            return ArgumentStart(
                json.loads(match[1]),
                match.end(),
                allow_control_chars=allow_control_chars,
            )
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
    finished: bool = False,
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
        held = 0 if finished else parser._partial_marker_length(content, (opening,))
        if len(content) > held:
            events.append((content[: len(content) - held], None, None))
    return events


def requested_tool(name: str) -> bool:
    for tool in request_tools_var.get() or ():
        function = tool.get("function", tool) if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name") == name:
            return True
    return False
