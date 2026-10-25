# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""A bounded, same-host negative directory for immutable snapshot addresses."""

import fcntl
import mmap
import os
import tempfile
import uuid
from array import array
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Tuple

_BYTES = 64 * 1024
_HEADER = 32


def _indices(key: int) -> Tuple[int, int]:
    return key % (_BYTES * 8), ((key >> 32) ^ (key * 0x9E3779B1)) % (_BYTES * 8)


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


class SharedBlockDirectory:
    def __init__(self):
        # Some bundled Linux Python builds omit os.memfd_create. TemporaryFile
        # on tmpfs is anonymous (or unlinked immediately) and works there too.
        self._file = tempfile.TemporaryFile(
            dir="/dev/shm", prefix="xinference-xavier-directory-"
        )
        self._fd = self._file.fileno()
        try:
            os.ftruncate(self._fd, _HEADER + _BYTES)
            self._data = mmap.mmap(self._fd, _HEADER + _BYTES)
        except BaseException:
            self._file.close()
            raise
        self._token = uuid.uuid4().bytes
        self._data[:16] = self._token
        self._data[16] = 1
        self._counts = array("I", [0]) * (_BYTES * 8)

    def metadata(self) -> dict:
        return {
            "boot_id": _boot_id(),
            "path": f"/proc/{os.getpid()}/fd/{self._fd}",
            "token": self._token,
        }

    @contextmanager
    def update(self) -> Iterator[None]:
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)

    def add(self, key: int) -> None:
        for index in _indices(key):
            self._counts[index] += 1
            if self._counts[index] == 1:
                offset = _HEADER + (index >> 3)
                self._data[offset] |= 1 << (index & 7)

    def remove(self, key: int) -> None:
        for index in _indices(key):
            self._counts[index] -= 1
            if not self._counts[index]:
                offset = _HEADER + (index >> 3)
                self._data[offset] &= ~(1 << (index & 7))

    def close(self) -> None:
        if self._fd >= 0:
            with self.update():
                self._data[16] = 0
            self._data.close()
            self._file.close()
            self._fd = -1


class LocalBlockDirectory:
    @classmethod
    def attach(cls, metadata: Optional[dict]) -> Optional["LocalBlockDirectory"]:
        if metadata is None:
            return None
        try:
            if metadata["boot_id"] != _boot_id():
                return None
            return cls(metadata)
        except (OSError, ValueError):
            return None

    def __init__(self, metadata: dict):
        self._path = metadata["path"]
        self.valid = True
        self._fd = os.open(self._path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            self._data = mmap.mmap(self._fd, _HEADER + _BYTES, access=mmap.ACCESS_READ)
            if self._data[:16] != metadata["token"]:
                raise ValueError("Directory owner changed")
            stat = os.fstat(self._fd)
            self._identity = (stat.st_dev, stat.st_ino)
        except BaseException:
            if hasattr(self, "_data"):
                self._data.close()
            os.close(self._fd)
            raise

    def contains(self, key: int) -> Optional[bool]:
        # Independent read-only opens have independent flock ownership. Never
        # block EngineCore behind publication; a busy writer uses the RPC path.
        try:
            fcntl.flock(self._fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        try:
            try:
                stat = os.stat(self._path)
            except OSError:
                self.valid = False
                return None
            if (stat.st_dev, stat.st_ino) != self._identity or not self._data[16]:
                self.valid = False
                return None
            return all(
                self._data[_HEADER + (index >> 3)] & (1 << (index & 7))
                for index in _indices(key)
            )
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)

    def close(self) -> None:
        if self._fd >= 0:
            self._data.close()
            os.close(self._fd)
            self._fd = -1
