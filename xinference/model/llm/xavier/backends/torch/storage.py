# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Bounded supervisor-owned CPU snapshots for external engine cache adapters."""

import hashlib
import time
import uuid
from typing import Optional

import torch
import xoscar as xo

from ...contract import KVCacheContract, fingerprint_metadata
from .snapshot import KVSnapshotStore


class XavierCacheActor(xo.StatelessActor):
    @classmethod
    def default_uid(cls):
        return "xavier-cache"

    def __init__(self, capacity_bytes: int):
        super().__init__()
        if type(capacity_bytes) is not int or capacity_bytes <= 0:
            raise ValueError("Xavier CPU cache capacity must be a positive integer")
        self.capacity_bytes = capacity_bytes
        self._namespace: Optional[str] = None
        self._page_elements = 0
        self._store: Optional[KVSnapshotStore] = None
        self._counts = dict(
            stored_pages=0, read_pages=0, missed_pages=0, evicted_pages=0
        )
        self._handoffs: dict[str, tuple[list[int], float, bool]] = {}

    def configure(self, contract: dict, storage_format: dict) -> str:
        contract_obj = KVCacheContract.from_dict(contract)
        if storage_format.get("layout") != "layer_first":
            raise ValueError("Unsupported Xavier storage page layout")
        namespace = fingerprint_metadata(
            {"contract": contract, "format": storage_format}
        )
        if self._namespace is not None:
            if namespace != self._namespace:
                raise ValueError("Xavier cache namespace differs between replicas")
            return namespace
        page_bytes = contract_obj.layer_nbytes * contract_obj.num_layers
        if self.capacity_bytes < page_bytes:
            raise ValueError("Xavier CPU cache budget cannot hold one complete page")
        self._page_elements = page_bytes // 2
        self._store = KVSnapshotStore(self.capacity_bytes // page_bytes)
        self._namespace = namespace
        return namespace

    def _keys(self, namespace, keys, limit=128):
        if self._namespace is None or namespace != self._namespace:
            raise ValueError("Unregistered or incompatible Xavier cache namespace")
        if len(keys) > limit or any(
            not isinstance(key, str) or not key for key in keys
        ):
            raise ValueError("Invalid Xavier page keys or batch size")
        return [
            int.from_bytes(
                hashlib.sha256((namespace + "\0" + key).encode()).digest(), "big"
            )
            for key in keys
        ]

    def _expire_handoffs(self):
        for ticket, (_, deadline, _) in list(self._handoffs.items()):
            if deadline <= time.monotonic():
                self.release_handoff(ticket)

    def reserve_handoff(self, namespace, keys):
        """Atomically pin a complete prompt prefix before P acknowledges D."""
        self._expire_handoffs()
        storage_keys = self._keys(namespace, keys, limit=4096)
        if len(storage_keys) > self._store.capacity:
            raise ValueError("Xavier CPU cache budget cannot hold this PD prefix")
        ticket = uuid.uuid4().hex
        if not self._store.reserve(ticket, storage_keys):
            return None
        # D releases on completion/cancellation; abandoned router processes are
        # bounded even when no explicit cleanup RPC can reach the store.
        self._handoffs[ticket] = (storage_keys, time.monotonic() + 300, False)
        return ticket

    def prepare_handoff(self, namespace, keys):
        """Reserve capacity before P writes, including pages not yet present."""
        self._expire_handoffs()
        storage_keys = self._keys(namespace, keys, limit=4096)
        pinned = (
            set().union(*self._store.leases.values()) if self._store.leases else set()
        )
        if len(set(storage_keys)) > self._store.capacity:
            raise ValueError("Xavier CPU cache budget cannot hold this PD prefix")
        if len(pinned.union(storage_keys)) > self._store.capacity:
            raise RuntimeError("Xavier CPU cache is full of active PD handoffs")
        ticket = uuid.uuid4().hex
        self._store.leases[ticket] = set(storage_keys)
        self._handoffs[ticket] = (storage_keys, time.monotonic() + 300, True)
        return ticket

    def validate_handoff(self, namespace, keys, ticket):
        self._expire_handoffs()
        storage_keys = self._keys(namespace, keys, limit=4096)
        handoff = self._handoffs.get(ticket)
        return (
            handoff is not None
            and handoff[0] == storage_keys
            and set(storage_keys).issubset(self._store.ready)
        )

    def release_handoff(self, ticket):
        handoff = self._handoffs.pop(ticket, None)
        if self._store is not None:
            self._store.release(ticket)
            if handoff and handoff[2]:
                pinned = (
                    set().union(*self._store.leases.values())
                    if self._store.leases
                    else set()
                )
                for key in handoff[0]:
                    if key not in pinned:
                        self._store.blocks.pop(key, None)
                        self._store.logical_dtypes.pop(key, None)
                        self._store.ready.discard(key)

    def put(self, namespace, keys, pages):
        self._expire_handoffs()
        storage_keys = self._keys(namespace, keys)
        if len(storage_keys) != len(pages):
            raise ValueError("Xavier page and key counts differ")
        for page in pages:
            if (
                page.device.type != "cpu"
                or page.dtype != torch.uint8
                or page.ndim != 1
                or page.numel() != self._page_elements * 2
                or not page.is_contiguous()
                or page.storage_offset() % 2
            ):
                raise ValueError(
                    "Xavier page geometry or dtype differs from its contract"
                )
        results = []
        for key, page in zip(storage_keys, pages):
            self._store.stage("page", [key], page.view(torch.float16).unsqueeze(0))
            results.append(bool(self._store.publish([key], {"page"})))
        self._counts["stored_pages"] += sum(results)
        self._counts["evicted_pages"] += len(self._store.evicted)
        self._store.evicted.clear()
        return results

    def get(self, namespace, keys):
        storage_keys = self._keys(namespace, keys)
        pages = []
        for key in storage_keys:
            if key in self._store.ready:
                pages.append(self._store.read("page", [key])[0].view(torch.uint8))
                self._store.blocks.move_to_end(key)
            else:
                pages.append(None)
        self._counts["read_pages"] += sum(page is not None for page in pages)
        self._counts["missed_pages"] += sum(page is None for page in pages)
        # read() owns each response, so eviction cannot invalidate an in-flight RPC.
        return pages

    def exists(self, namespace, keys):
        storage_keys = self._keys(namespace, keys)
        for index, key in enumerate(storage_keys):
            if key not in self._store.ready:
                return index
        return len(storage_keys)

    def get_stats(self):
        self._expire_handoffs()
        return {
            **self._counts,
            "namespace": self._namespace,
            "pages": len(self._store.ready) if self._store else 0,
            "capacity_bytes": self.capacity_bytes,
            "active_handoffs": len(self._handoffs),
            "used_bytes": (
                len(self._store.blocks) * self._page_elements * 2 if self._store else 0
            ),
        }
