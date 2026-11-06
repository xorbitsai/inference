# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Snapshot backend boundary used by Xavier's engine and transport adapters."""

from typing import Dict, List, Optional, Protocol, Set, Tuple
from uuid import uuid4

import torch


class SnapshotBackend(Protocol):
    capacity: int

    def stage(self, layer, keys, tensors, logical_dtype=None): ...

    def stage_packed_blocks(self, keys, packed, layers, *, retain_storage=False): ...

    def publish(self, keys: List[int], layers: Set[str]) -> List[int]: ...

    def ready_keys(self, keys: List[int]) -> List[int]: ...

    def can_stage(self, keys: Set[int]) -> bool: ...

    async def prefetch(self, keys: List[int]) -> bool: ...

    def reserve(self, lease: str, keys: List[int]) -> bool: ...

    def release(self, lease: str): ...

    def release_consumer(self, rank: int): ...

    def has_layer(self, layer, keys, logical_dtype=None) -> bool: ...

    def take_evicted(self) -> List[int]: ...

    def read(self, layer: str, keys: List[int]) -> torch.Tensor: ...

    async def close(self): ...

    def get_stats(self) -> dict: ...


async def read_page_snapshots(
    store: SnapshotBackend, keys: List[int]
) -> List[Optional[torch.Tensor]]:
    """Return owned pages, bounding each RPC's copies by the staging capacity."""
    candidates = store.ready_keys(keys)[: store.capacity]
    if not candidates:
        return [None] * len(keys)
    # Leave headroom for LMCache's existing L1 objects and asynchronous writes.
    # Returned pages are independent copies; each read lease can be released
    # before requesting the next batch. Total copies remain capacity-bounded.
    batch_size = (
        max(1, store.capacity // 8)
        if store.get_stats()["backend"] == "lmcache"
        else len(candidates)
    )
    pages = {}
    for offset in range(0, len(candidates), batch_size):
        batch = candidates[offset : offset + batch_size]
        if not await store.prefetch(batch):
            continue
        lease = uuid4().hex
        if not store.reserve(lease, batch):
            continue
        try:
            pages.update({key: store.read("page", [key])[0] for key in batch})
        finally:
            store.release(lease)
    return [pages.get(key) for key in keys]


def create_snapshot_backend(
    capacity: int,
    name: str = "native",
    config: Optional[dict] = None,
    *,
    namespace: str = "",
    schema: Optional[Dict[str, Tuple[tuple, torch.dtype]]] = None,
) -> SnapshotBackend:
    from ..config import normalize_cache_backend
    from .snapshot import KVSnapshotStore

    name, options = normalize_cache_backend(name, config)
    if name == "native":
        return KVSnapshotStore(capacity)
    from ..lmcache import LMCacheSnapshotBackend

    return LMCacheSnapshotBackend(capacity, options, namespace, schema)
