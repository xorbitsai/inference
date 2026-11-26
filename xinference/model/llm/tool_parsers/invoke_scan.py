"""Find the end of a DeepSeek invoke or call by reading its arguments."""

import json
import re
from typing import Iterator, Optional, Tuple

from .abstract_tool_parser import find_outside_json_string

_DSML = "<｜DSML｜"


def _tags(dsml: bool) -> Tuple[re.Pattern, re.Pattern, str, str, str]:
    """
    Return the invoke open, parameter open, parameter close and invoke close
    patterns and the text that starts an invoke.
    """
    prefix = _DSML if dsml else "<"
    close = "</" + prefix[1:]
    return (
        re.compile(rf'{re.escape(prefix)}invoke\s+name="([^"]+)"\s*>'),
        re.compile(
            rf'\s*{re.escape(prefix)}parameter\s+name="[^"]+"\s+string="(true|false)"\s*>'
        ),
        f"{close}parameter>",
        f"{close}invoke>",
        f"{prefix}invoke",
    )


_TAGS = {dsml: _tags(dsml) for dsml in (True, False)}


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def invoke_end(text: str, opening: int, dsml: bool, final: bool = False) -> int:
    """
    Return the index just past the invoke that starts at ``opening``, or -1
    while it is still open.

    The parameters are read one by one: a value ends at its closing tag, which
    for a JSON value is the first one outside a JSON string, so a literal
    ``</invoke>`` inside a value does not end the invoke. Text that is not a parameter falls
    back to the first closing tag. With ``final`` the text is complete, so a JSON
    string still open at its end was cut short and the first closing tag counts.
    """
    open_tag, parameter_open, parameter_close, invoke_close, invoke_start = _TAGS[dsml]
    head = open_tag.match(text, opening)
    if head is None:
        close = text.find(invoke_close, opening + 1)
        return -1 if close == -1 else close + len(invoke_close)
    position = head.end()
    while True:
        parameter = parameter_open.match(text, position)
        if parameter is None:
            close = text.find(invoke_close, position)
            return -1 if close == -1 else close + len(invoke_close)
        value = parameter.end()
        if parameter[1] == "false":
            # A JSON value may hold the closing tag inside one of its strings
            # (a string cut short before the next invoke does not hide it)
            close = find_outside_json_string(
                text, parameter_close, value, invoke_start, final
            )
            if final and close != -1 and not _is_json(text[value:close]):
                # An odd quote moved the boundary: the value is not JSON, so
                # its first closing tag ends it
                close = text.find(parameter_close, value)
        else:
            close = text.find(parameter_close, value)
        if close == -1:
            return -1
        position = close + len(parameter_close)


def iter_invokes(
    text: str,
    dsml: bool,
    start: int = 0,
    stop: Optional[int] = None,
    final: bool = False,
) -> Iterator[Tuple[str, str, int]]:
    """
    Yield ``(name, body, end)`` for each complete invoke in ``text[start:stop]``,
    where ``body`` is what lies between the open and the closing tag and
    ``end`` is the index just past the closing tag. ``final`` is as in
    ``invoke_end``.
    """
    open_tag, _, _, invoke_close, _ = _TAGS[dsml]
    text = text if stop is None else text[:stop]
    position = start
    while (head := open_tag.search(text, position)) is not None:
        end = invoke_end(text, head.start(), dsml, final)
        if end == -1:
            return
        yield head[1], text[head.end() : end - len(invoke_close)], end
        position = end
