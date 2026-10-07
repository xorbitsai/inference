# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Bounded, leased same-host receive buffers; remote actors use normal RPCs."""

import mmap
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple

import torch

from .request_transfer import MAX_REQUEST_BYTES

_HEADER = 64


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


class SharedReadBuffer:
    def __init__(self, capacity: int = MAX_REQUEST_BYTES):
        if not 0 < capacity <= MAX_REQUEST_BYTES:
            raise ValueError("Invalid shared receive capacity")
        self.capacity = capacity
        self.token = uuid.uuid4().bytes
        self._file = tempfile.TemporaryFile(
            dir="/dev/shm", prefix="xinference-xavier-read-"
        )
        try:
            os.ftruncate(self._file.fileno(), _HEADER + capacity)
            self._data = mmap.mmap(self._file.fileno(), _HEADER + capacity)
            self._data[:16] = self.token
            self._data[16] = 1
            self._payload: Optional[torch.Tensor] = torch.frombuffer(
                self._data, dtype=torch.uint8, offset=_HEADER
            )
        except BaseException:
            if hasattr(self, "_data"):
                self._data.close()
            self._file.close()
            raise
        self._lease: Optional[bytes] = None

    def metadata(self) -> dict:
        return {
            "boot_id": _boot_id(),
            "path": f"/proc/{os.getpid()}/fd/{self._file.fileno()}",
            "token": self.token,
            "capacity": self.capacity,
        }

    def acquire(self, nbytes: int) -> Optional[Tuple[bytes, torch.Tensor]]:
        if self._lease is not None or self._payload is None:
            return None
        if not 0 < nbytes <= self.capacity:
            return None
        self._lease = uuid.uuid4().bytes
        self._data[24:40] = self._lease
        return self._lease, self._payload[:nbytes]

    def release(self, lease: bytes) -> None:
        if lease == self._lease and not self._file.closed:
            self._lease = None
            self._data[24:40] = bytes(16)

    def close(self) -> None:
        if not self._file.closed:
            self._data[16] = 0
            self._payload = None
            self._data.close()
            self._file.close()


class LocalReadBuffer:
    @classmethod
    def attach(cls, metadata: Optional[dict]) -> Optional["LocalReadBuffer"]:
        if metadata is None:
            return None
        try:
            if metadata["boot_id"] != _boot_id():
                return None
            return cls(metadata)
        except (OSError, ValueError):
            return None

    def __init__(self, metadata: dict):
        capacity = metadata["capacity"]
        if not 0 < capacity <= MAX_REQUEST_BYTES:
            raise ValueError("Invalid shared receive capacity")
        self.capacity = capacity
        self.token = metadata["token"]
        self._path = metadata["path"]
        self._fd = os.open(self._path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            stat = os.fstat(self._fd)
            if stat.st_size != _HEADER + capacity:
                raise ValueError("Shared receive buffer size changed")
            self._identity = (stat.st_dev, stat.st_ino)
            # Private writable views let torch read without a read-only-buffer
            # warning. Consumer writes cannot modify the actor's shared payload.
            self._data = mmap.mmap(
                self._fd, _HEADER + capacity, access=mmap.ACCESS_COPY
            )
            if self._data[:16] != self.token or not self._data[16]:
                raise ValueError("Shared receive owner changed")
            self._payload: Optional[torch.Tensor] = torch.frombuffer(
                self._data, dtype=torch.uint8, offset=_HEADER
            )
        except BaseException:
            if hasattr(self, "_data"):
                self._data.close()
            os.close(self._fd)
            raise

    def valid(self) -> bool:
        if self._fd < 0:
            return False
        try:
            stat = os.stat(self._path)
        except OSError:
            return False
        return (
            (stat.st_dev, stat.st_ino) == self._identity
            and self._data[:16] == self.token
            and bool(self._data[16])
        )

    def copy(self, lease: bytes, nbytes: int, *, pin_memory: bool) -> torch.Tensor:
        if (
            not self.valid()
            or self._data[24:40] != lease
            or not 0 < nbytes <= self.capacity
        ):
            raise RuntimeError("Shared receive lease or owner changed")
        assert self._payload is not None
        payload = self._payload[:nbytes]
        # The actor can reuse its receive slot immediately after the caller's
        # release RPC. H2D must own an independent pinned allocation first.
        if pin_memory:
            return payload.pin_memory()
        return torch.from_numpy(payload.numpy().copy())

    def close(self) -> None:
        if self._fd >= 0:
            self._payload = None
            self._data.close()
            os.close(self._fd)
            self._fd = -1
