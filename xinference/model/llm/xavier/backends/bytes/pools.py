# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Byte-bounded pages for engines whose host cache spans several pools.

Hybrid and MLA models (DeepSeek-V4, for example) offload one prefix page as
several objects: compressed KV, indexer rows, sliding-window state and so on.
Each pool has its own page size, and a page may be stored for one pool but not
another (sliding-window state covers only the tail of a prefix). The engine
owns the hit policy; this store only keeps immutable bytes per (pool, key).
"""

from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ...contract import fingerprint_metadata

MAX_POOL_ENTRIES = 4096

_PoolKey = Tuple[str, str]


def _require_schema(schema: Mapping[str, Mapping[str, Any]]) -> Dict[str, int]:
    if not isinstance(schema, Mapping) or not schema:
        raise ValueError("Xavier pool schema must name at least one pool")
    sizes = {}
    for name, entry in schema.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(entry, Mapping)
            or type(entry.get("page_bytes")) is not int
            or entry["page_bytes"] < 0
        ):
            raise ValueError("Invalid Xavier pool schema")
        sizes[name] = entry["page_bytes"]
    return sizes


class PoolPageStore:
    """LRU pages keyed by (pool, key) under one byte budget.

    Chains are touched head-last so eviction removes tails before heads, which
    keeps the stored prefixes usable for longest-prefix lookups.
    """

    def __init__(self, capacity_bytes: int):
        if type(capacity_bytes) is not int or capacity_bytes <= 0:
            raise ValueError("Xavier cache capacity must be a positive integer")
        self.capacity_bytes = capacity_bytes
        self.namespace: Optional[str] = None
        self._page_bytes: Dict[str, int] = {}
        self._pages: "OrderedDict[_PoolKey, bytes]" = OrderedDict()
        # Which replica wrote each page, for transfer accounting and logs.
        self._writers: Dict[_PoolKey, str] = {}
        self.used_bytes = 0
        self.counts = dict(
            stored_pages=0,
            read_pages=0,
            missed_pages=0,
            evicted_pages=0,
        )

    def configure(
        self, identity: Mapping, schema: Mapping[str, Mapping[str, Any]]
    ) -> str:
        sizes = _require_schema(schema)
        namespace = fingerprint_metadata({"identity": identity, "pools": schema})
        if self.namespace is not None:
            if namespace != self.namespace:
                raise ValueError(
                    "Xavier pool layout or model identity differs between replicas"
                )
            return namespace
        if max(sizes.values()) > self.capacity_bytes:
            raise ValueError("Xavier CPU cache budget cannot hold one complete page")
        self.namespace = namespace
        self._page_bytes = sizes
        return namespace

    def _entries(
        self, namespace: str, entries: Sequence[Sequence[str]]
    ) -> List[_PoolKey]:
        if self.namespace is None or namespace != self.namespace:
            raise ValueError("Unregistered or incompatible Xavier cache namespace")
        if not isinstance(entries, (list, tuple)) or len(entries) > MAX_POOL_ENTRIES:
            raise ValueError("Invalid Xavier pool page batch")
        result = []
        for entry in entries:
            if (
                not isinstance(entry, (list, tuple))
                or len(entry) != 2
                or entry[0] not in self._page_bytes
                or not isinstance(entry[1], str)
                or not entry[1]
            ):
                raise ValueError("Invalid Xavier pool page key")
            result.append((entry[0], entry[1]))
        return result

    def _evict_for(self, nbytes: int, keep: set) -> bool:
        while self.used_bytes + nbytes > self.capacity_bytes:
            victim = next((key for key in self._pages if key not in keep), None)
            if victim is None:
                return False
            self.used_bytes -= len(self._pages.pop(victim))
            self._writers.pop(victim, None)
            self.counts["evicted_pages"] += 1
        return True

    def put(
        self,
        namespace: str,
        entries: Sequence[Sequence[str]],
        pages: Sequence[bytes],
        writer: str = "",
    ) -> List[bool]:
        keys = self._entries(namespace, entries)
        if len(keys) != len(pages) or any(
            type(page) is not bytes or len(page) != self._page_bytes[key[0]]
            for key, page in zip(keys, pages)
        ):
            raise ValueError("Xavier pool page geometry differs from its schema")
        batch = set(keys)
        # Admit in chain order, so a full cache keeps the head rather than an
        # unusable tail, then insert tails first so the head is most recent.
        admitted: List[Tuple[_PoolKey, Optional[bytes]]] = []
        seen = set()
        pending = 0
        for key, page in zip(keys, pages):
            if key in seen:
                continue
            seen.add(key)
            if key in self._pages:
                admitted.append((key, None))
                continue
            if not self._evict_for(pending + len(page), batch):
                break
            pending += len(page)
            admitted.append((key, page))
        for key, stored in reversed(admitted):
            if stored is None:
                self._pages.move_to_end(key)
                continue
            self._pages[key] = stored
            self._writers[key] = writer
            self.used_bytes += len(stored)
            self.counts["stored_pages"] += 1
        return [key in self._pages for key in keys]

    def get(self, namespace: str, entries: Sequence[Sequence[str]]) -> dict:
        """Return {"pages": [bytes | None], "writers": {replica: pages}}."""
        keys = self._entries(namespace, entries)
        pages = [self._pages.get(key) for key in keys]
        writers: Dict[str, int] = {}
        for key, page in reversed(list(zip(keys, pages))):
            if page is not None:
                self._pages.move_to_end(key)
                writer = self._writers.get(key, "")
                writers[writer] = writers.get(writer, 0) + 1
        hits = sum(page is not None for page in pages)
        self.counts["read_pages"] += hits
        self.counts["missed_pages"] += len(pages) - hits
        return {"pages": pages, "writers": writers}

    def contains(self, namespace: str, entries: Sequence[Sequence[str]]) -> List[bool]:
        keys = self._entries(namespace, entries)
        return [key in self._pages for key in keys]

    def get_stats(self) -> dict:
        return {
            **self.counts,
            "namespace": self.namespace,
            "pages": len(self._pages),
            "used_bytes": self.used_bytes,
            "capacity_bytes": self.capacity_bytes,
            "pools": dict(self._page_bytes),
        }
