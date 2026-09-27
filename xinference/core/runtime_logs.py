"""Bounded reads of the current Xinference application log."""

import os
import re
from typing import Dict, Optional, Tuple

from ..deploy.utils import get_log_file

MAX_RUNTIME_LOG_BYTES = 256 * 1024


def _cursor(stat: os.stat_result, offset: int) -> str:
    return f"{stat.st_dev}:{stat.st_ino}:{offset}"


def _parse_cursor(value: str) -> Optional[Tuple[int, int, int]]:
    try:
        device, inode, offset = (int(part) for part in value.split(":"))
        if min(device, inode, offset) < 0:
            return None
        return device, inode, offset
    except (AttributeError, TypeError, ValueError):
        return None


def read_runtime_log(cursor: str = "") -> Dict[str, object]:
    """Read the latest chunk, or continue from a cursor after rotation."""
    path = get_log_file("runtime")
    try:
        current = os.stat(path)
    except FileNotFoundError:
        return {"text": "", "cursor": "", "has_more": False, "reset": False}

    position = _parse_cursor(cursor) if cursor else None
    read_path = path
    reset = bool(cursor and position is None)

    if position and (position[0], position[1]) != (current.st_dev, current.st_ino):
        # A rotated file retains its inode. Drain it before reading the new file.
        directory, name = os.path.split(path)
        with os.scandir(directory) as entries:
            for entry in entries:
                if not entry.name.startswith(name + ".") or not entry.is_file(
                    follow_symlinks=False
                ):
                    continue
                suffix = entry.name[len(name) + 1 :]
                if not re.fullmatch(r"(?:\d{4}-\d{2}-\d{2}(?:\.\d+)?|\d+)", suffix):
                    continue
                try:
                    stat = entry.stat()
                except FileNotFoundError:
                    continue
                if (stat.st_dev, stat.st_ino) == position[:2]:
                    read_path = entry.path
                    break
            else:
                position = None
                reset = True

    try:
        with open(read_path, "rb") as stream:
            stat = os.fstat(stream.fileno())
            if position and position[2] <= stat.st_size:
                start = position[2]
            else:
                start = max(0, stat.st_size - MAX_RUNTIME_LOG_BYTES)
                reset = bool(cursor)

            stream.seek(start)
            if start and (not position or reset):
                # Initial tail starts at a line boundary when possible.
                stream.readline(MAX_RUNTIME_LOG_BYTES)
            data = stream.read(MAX_RUNTIME_LOG_BYTES)
            offset = stream.tell()
    except FileNotFoundError:
        return {"text": "", "cursor": "", "has_more": False, "reset": True}

    if read_path != path and offset >= stat.st_size:
        # On the next poll, resume at the start of the new active file.
        next_cursor = _cursor(current, 0)
        has_more = True
    else:
        next_cursor = _cursor(stat, offset)
        has_more = offset < stat.st_size

    return {
        "text": data.decode("utf-8", errors="replace"),
        "cursor": next_cursor,
        "has_more": has_more,
        "reset": reset,
    }
