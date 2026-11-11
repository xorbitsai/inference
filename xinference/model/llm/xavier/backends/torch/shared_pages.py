# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Fixed-size Xavier pages in one shared-memory file.

The cache actor owns the file and every piece of metadata: which key lives in
which slot, LRU order, pins and pending writes. Engines on the same host map
the file and copy page bytes directly between GPU memory and their slots, so
the actor never carries page payloads. Engines elsewhere use the actor's
ordinary put/get RPCs, which read and write the same slots.
"""

import os
import shutil
import struct
import tempfile
import time
import uuid
from collections import OrderedDict
from typing import Dict, List, Optional, Set

import torch

MAGIC = b"XINFXAVIERPAGES1"
HEADER_BYTES = 4096
# A writer that neither commits nor aborts within this time has died.
_WRITE_TIMEOUT = 120.0


def shared_directory() -> str:
    """Where page files live: tmpfs when the host has one."""
    return "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()


def _header(nonce: bytes, page_bytes: int, slots: int) -> bytes:
    return MAGIC + nonce + struct.pack("<QQ", page_bytes, slots)


class SharedPageArena:
    """Native snapshot store whose pages live in a shared-memory file."""

    def __init__(self, directory: str, name: str, page_bytes: int, capacity: int):
        if page_bytes <= 0 or capacity <= 0:
            raise ValueError("Shared Xavier pages need a positive size and capacity")
        nbytes = HEADER_BYTES + page_bytes * capacity
        if shutil.disk_usage(directory).free < nbytes:
            raise OSError(f"{directory} cannot hold {nbytes} bytes of Xavier pages")
        self.path = os.path.join(directory, name)
        self.nonce = uuid.uuid4().bytes
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            # Reserve tmpfs pages now: a lazily grown file faults with SIGBUS
            # in whichever process first touches a page that does not fit.
            if hasattr(os, "posix_fallocate"):
                os.posix_fallocate(fd, 0, nbytes)
            else:  # macOS has no fallocate; used by local tests only.
                os.ftruncate(fd, nbytes)
            os.pwrite(fd, _header(self.nonce, page_bytes, capacity), 0)
        except BaseException:
            os.close(fd)
            os.unlink(self.path)
            raise
        os.close(fd)
        self.page_bytes = page_bytes
        self.capacity = capacity
        self.pages = map_pages(self.path, self.nonce, page_bytes, capacity)
        self.slots: OrderedDict[int, int] = OrderedDict()  # ready keys, LRU
        self.writing: Dict[int, tuple] = {}  # key -> (slot, deadline)
        self.free: List[int] = list(range(capacity - 1, -1, -1))
        self.ready: Set[int] = set()
        self.leases: Dict[str, Set[int]] = {}
        self.evicted: Set[int] = set()
        # Release-handoff bookkeeping in the cache actor expects these names.
        self.blocks = self.slots
        self.logical_dtypes: Dict[int, dict] = {}

    def info(self) -> dict:
        return dict(
            path=self.path,
            nonce=self.nonce.hex(),
            page_bytes=self.page_bytes,
            slots=self.capacity,
        )

    def _pinned(self) -> Set[int]:
        return set().union(*self.leases.values()) if self.leases else set()

    def _take_slot(self, pinned: Set[int]) -> Optional[int]:
        now = time.monotonic()
        for key, (slot, deadline) in list(self.writing.items()):
            if deadline <= now:
                del self.writing[key]
                self.free.append(slot)
        if self.free:
            return self.free.pop()
        for key in self.slots:
            if key not in pinned:
                slot = self.slots.pop(key)
                self.ready.discard(key)
                self.evicted.add(key)
                return slot
        return None

    def allocate(self, keys: List[int]) -> List[Optional[int]]:
        """Reserve a slot per new key; -1 marks a stored key, None no room."""
        pinned = self._pinned()
        deadline = time.monotonic() + _WRITE_TIMEOUT
        result: List[Optional[int]] = []
        for key in keys:
            if key in self.ready:
                self.slots.move_to_end(key)
                result.append(-1)
            elif key in self.writing:
                result.append(None)  # Another writer is storing it now.
            else:
                slot = self._take_slot(pinned)
                if slot is not None:
                    self.writing[key] = (slot, deadline)
                result.append(slot)
        return result

    def commit(self, keys: List[int], slots: List[int]) -> int:
        """Publish written slots; slots no longer owned by the writer are lost."""
        published = 0
        for key, slot in zip(keys, slots):
            if self.writing.get(key, (None,))[0] != slot:
                continue
            del self.writing[key]
            self.slots[key] = slot
            self.ready.add(key)
            published += 1
        return published

    def abort(self, keys: List[int], slots: List[int]) -> None:
        for key, slot in zip(keys, slots):
            if self.writing.get(key, (None,))[0] == slot:
                del self.writing[key]
                self.free.append(slot)

    def locate(self, keys: List[int]) -> List[int]:
        return [self.slots[key] for key in keys]

    # SnapshotBackend methods used by XavierCacheActor's RPC paths.

    def ready_keys(self, keys: List[int]) -> List[int]:
        ready = []
        for key in dict.fromkeys(keys):
            if key in self.ready:
                ready.append(key)
                self.slots.move_to_end(key)
        return ready

    def stage(self, layer, keys, tensors, logical_dtype=None):
        for key, tensor in zip(keys, tensors):
            (slot,) = self.allocate([key])
            if slot is None or slot < 0:
                continue
            self.pages[slot].copy_(tensor.reshape(-1).view(torch.uint8))
            self.commit([key], [slot])

    def stage_packed_blocks(self, keys, packed, layers, *, retain_storage=False):
        raise NotImplementedError("Shared Xavier pages store whole pages only")

    def can_stage(self, keys: Set[int]) -> bool:
        return len(keys | self._pinned()) <= self.capacity

    async def prefetch(self, keys: List[int]) -> bool:
        return set(keys).issubset(self.ready)

    def has_layer(self, layer, keys, logical_dtype=None) -> bool:
        return layer == "page" and all(key in self.ready for key in keys)

    def release_consumer(self, rank: int):
        for lease in list(self.leases):
            if lease.startswith(f"{rank}:"):
                self.release(lease)

    def publish(self, keys: List[int], layers: Set[str]) -> List[int]:
        return [key for key in keys if key in self.ready]

    def read(self, layer: str, keys: List[int]) -> torch.Tensor:
        return torch.stack([self.pages[self.slots[key]] for key in keys]).view(
            torch.float16
        )

    def reserve(self, lease: str, keys: List[int]) -> bool:
        if not set(keys).issubset(self.ready):
            return False
        self.leases.setdefault(lease, set()).update(keys)
        for key in keys:
            self.slots.move_to_end(key)
        return True

    def release(self, lease: str):
        self.leases.pop(lease, None)

    def take_evicted(self) -> List[int]:
        evicted = list(self.evicted)
        self.evicted.clear()
        return evicted

    def get_stats(self) -> dict:
        return {
            "backend": "native",
            "capacity": self.capacity,
            "pages": len(self.ready),
            "shared_memory": True,
        }

    async def close(self):
        self.pages = None
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass


def map_pages(path: str, nonce: bytes, page_bytes: int, slots: int) -> torch.Tensor:
    """Map a page file created by this deployment's cache actor."""
    nbytes = HEADER_BYTES + page_bytes * slots
    with open(path, "rb") as file:
        header = file.read(len(_header(nonce, page_bytes, slots)))
    if header != _header(nonce, page_bytes, slots) or os.path.getsize(path) < nbytes:
        raise ValueError("Xavier shared pages belong to another cache")
    mapped = torch.from_file(path, shared=True, size=nbytes, dtype=torch.uint8)
    return mapped[HEADER_BYTES:].view(slots, page_bytes)
