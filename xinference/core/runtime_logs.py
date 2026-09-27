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
    archives: list[tuple[tuple[int, str, int, int], str, os.stat_result]] = []
    archive_index = None

    if position and (position[0], position[1]) != (current.st_dev, current.st_ino):
        # Drain retained archives in rotation order before the active file.
        directory, name = os.path.split(path)
        with os.scandir(directory) as entries:
            for entry in entries:
                if not entry.name.startswith(name + ".") or not entry.is_file(
                    follow_symlinks=False
                ):
                    continue
                suffix = entry.name[len(name) + 1 :]
                date_match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})(?:\.(\d+))?", suffix)
                if suffix.isdigit():
                    sort_key = (0, "", 0, -int(suffix))
                elif date_match:
                    date, sequence = date_match.groups()
                    sort_key = (
                        1,
                        date,
                        0 if sequence else 1,
                        int(sequence) if sequence else 0,
                    )
                else:
                    continue
                try:
                    stat = entry.stat()
                except OSError:
                    continue
                archives.append((sort_key, entry.path, stat))
        archives.sort(key=lambda archive: archive[0])
        for index, (_, archive_path, archive_stat) in enumerate(archives):
            if (archive_stat.st_dev, archive_stat.st_ino) == position[:2]:
                read_path = archive_path
                archive_index = index
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
        # Resume from the next retained archive, then continue into the active file.
        if archive_index is not None and archive_index + 1 < len(archives):
            next_stat = archives[archive_index + 1][2]
        else:
            next_stat = current
        next_cursor = _cursor(next_stat, 0)
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
