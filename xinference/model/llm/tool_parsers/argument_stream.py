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
    close_frame: Optional[Callable[[str, int], int]] = None


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
        self.wire_index: Optional[int] = None

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
        self.next_wire_index = 0
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
                if content:
                    # A rejected frame is sometimes returned as plain text.
                    # Retire that reader before considering a valid sibling,
                    # even if both events arrived in the same engine chunk.
                    for index, rejected_reader in tuple(self.readers.items()):
                        rejected = rejected_reader.start
                        position = rejected.call_position
                        if (
                            position is not None
                            and rejected.close_frame is not None
                            and position + len(content) > rejected.position
                            and text.startswith(content, position)
                            and rejected.close_frame(text, rejected.position) != -1
                        ):
                            self.incomplete |= rejected_reader.wire_index is not None
                            del self.readers[index]
                events.append(event)
                continue
            # A source call consumes exactly its own reader. A failed match
            # cannot consume the reader for the next call in the same block.
            reader = self.readers.pop(self.index, None)
            wire_index = None
            if reader is not None and reader.wire_index is not None:
                fragment = (
                    reader.finish(name, args, text) if isinstance(args, dict) else None
                )
                if fragment is not None:
                    args = ToolArgumentsDelta(fragment)
                    wire_index = reader.wire_index
                else:
                    self.incomplete = True
                    logger.debug("Tool call %d could not be reconciled", self.index)
            if wire_index is None:
                wire_index = self.next_wire_index
                self.next_wire_index += 1
            events.append((content, name, args, wire_index))
            self.index += 1

        # Some parsers return rejected frames as text, or skip them entirely.
        # Once such a frame closes, its reader must not absorb the next frame.
        for index, pending_reader in tuple(self.readers.items()):
            close = pending_reader.start.close_frame
            if close is not None and close(text, pending_reader.start.position) != -1:
                self.incomplete |= pending_reader.wire_index is not None
                del self.readers[index]

        if finish_reason is not None:
            self.incomplete |= any(
                reader.wire_index is not None for reader in self.readers.values()
            )
            return events

        reader = self.readers.get(self.active) if self.active is not None else None
        if reader is not None and not reader.closed:
            fragment = reader.read(text)
            if fragment:
                events.append(
                    (
                        None,
                        reader.start.name,
                        ToolArgumentsDelta(fragment),
                        self._wire_index(reader),
                    )
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
            i not in self.readers or self.readers[i].wire_index is None
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
            events.append(
                (
                    None,
                    start.name,
                    ToolArgumentsDelta(fragment),
                    self._wire_index(reader),
                )
            )
        return events

    def _wire_index(self, reader: ArgumentReader) -> int:
        if reader.wire_index is None:
            reader.wire_index = self.next_wire_index
            self.next_wire_index += 1
        return reader.wire_index


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
    close_pattern: Optional[Pattern] = None,
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
        if close_pattern is not None:
            close_match = close_pattern.search(text, start)
            if close_match:
                position = close_match.end()
                continue
        end = text.find(value_close, start) if close_pattern is None else -1
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
    find_close: Optional[Callable[[str, int], int]] = None,
    deduplicate: bool = False,
) -> List[Tuple]:
    """Stream text around delimited blocks and parse each newly closed block."""
    previous_end = len(text) - len(delta)
    position = 0
    events: List[Tuple] = []
    seen = set()
    earlier_blocks: List[Tuple[int, int]] = []
    parse = parse_block or parser.extract_tool_calls
    while (start := text.find(opening, position)) != -1:
        content_start = max(position, previous_end)
        if position <= previous_end:
            content_start -= min(
                previous_end - position,
                parser._partial_marker_length(text[:previous_end], (opening,)),
            )
        if start > content_start:
            events.append((text[content_start:start], None, None))
        body = start + len(opening)
        end = find_close(text, body) if find_close else text.find(closing, body)
        if end == -1:
            if finished and not previous_end:
                events.append((text[start:], None, None))
            return events
        position = end + len(closing)
        if position <= previous_end:
            if deduplicate:
                earlier_blocks.append((start, position))
            continue
        for event in parse(text[start:position]):
            if deduplicate and event[1]:
                for earlier_start, earlier_end in earlier_blocks:
                    for _, name, args in parse(text[earlier_start:earlier_end]):
                        if name:
                            seen.add((name, json.dumps(args, sort_keys=True)))
                earlier_blocks.clear()
                key = (event[1], json.dumps(event[2], sort_keys=True))
                if key in seen:
                    continue
                seen.add(key)
            events.append(event)
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
