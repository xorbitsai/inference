# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Canonical, historical prefix pages in LMCache, separate from live P/D KV."""

import re
from typing import Optional

import torch

from ...contract import KVCacheContract
from ..torch.backend import create_snapshot_backend
from ..torch.storage import XavierCacheActor


class XavierHistoryCacheActor(XavierCacheActor):
    """Expose owned canonical bytes while LMCache manages CPU and L2 storage."""

    def __init__(self, cache_backend: dict):
        if cache_backend.get("name") != "lmcache":
            raise ValueError("Xavier history requires the LMCache backend")
        super().__init__(cache_backend["config"]["cpu_bytes"], cache_backend)
        self._wire_namespace: Optional[str] = None

    # The bytes RPC returns capabilities rather than the native Torch namespace.
    def configure(self, contract: dict) -> dict:  # type: ignore[override]
        metadata = KVCacheContract.from_dict(contract)
        super().configure(
            contract,
            {
                "layout": "layer_first",
                "physical_layout": "canonical-layer-kv-token-head-dimension",
                "hash_format": "sglang-storage-chain-v1",
                "version": 1,
            },
        )
        self._wire_namespace = metadata.fingerprint
        assert self._store is not None
        return {
            "namespace": self._wire_namespace,
            "capacity_pages": min(4096, self._store.get_stats()["directory_capacity"]),
            "max_keys": 4096,
        }

    def configure_pages(self, namespace: str, page_bytes: int) -> dict:
        """Opaque engine-owned chunks, including complete hybrid-state groups.

        The engine supplies a fingerprint containing its physical layout and
        checkpoint identity. Xavier transports the bytes without interpreting
        attention or recurrent state.
        """
        if (
            not isinstance(namespace, str)
            or not re.fullmatch(r"[0-9a-f]{64}", namespace)
            or type(page_bytes) is not int
            or page_bytes <= 0
            or page_bytes % 2
        ):
            raise ValueError("Invalid Xavier opaque page geometry or namespace")
        if self._namespace is not None:
            if (
                self._namespace != namespace
                or self._wire_namespace != namespace
                or self._page_elements * 2 != page_bytes
            ):
                raise ValueError("Xavier cache namespace differs between replicas")
        else:
            self._page_elements = page_bytes // 2
            self._store = create_snapshot_backend(
                self.capacity_bytes // page_bytes,
                self._backend_name,
                self._backend_options,
                namespace=f"{self._deployment}-{namespace[:16]}",
                schema={"page": ((self._page_elements,), torch.float16)},
            )
            self._namespace = self._wire_namespace = namespace
        assert self._store is not None
        return {
            "namespace": namespace,
            "capacity_pages": min(4096, self._store.get_stats()["directory_capacity"]),
            "max_keys": 4096,
        }

    def _keys(self, namespace, keys, limit=4096):
        if self._wire_namespace is None or namespace != self._wire_namespace:
            raise ValueError("Unregistered or incompatible Xavier history namespace")
        if (
            not isinstance(keys, (list, tuple))
            or len(keys) > limit
            or any(
                not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key)
                for key in keys
            )
            or len(set(keys)) != len(keys)
        ):
            raise ValueError("Invalid Xavier history page keys")
        return super()._keys(self._namespace, keys, limit)

    def put(self, namespace, keys, pages, start=0):
        self._keys(namespace, keys)
        page_bytes = self._page_elements * 2
        if (
            type(start) is not int
            or not 0 <= start <= len(keys)
            or len(keys) - start != len(pages)
            or any(type(page) is not bytes or len(page) != page_bytes for page in pages)
        ):
            raise ValueError("Xavier history page geometry differs from its contract")
        # Only the missing suffix owns a new snapshot. Refresh prefix metadata
        # without reading its payload back from CPU memory or external storage.
        self._store.ready_keys(self._keys(namespace, list(reversed(keys[:start]))))
        carriers = [
            torch.frombuffer(bytearray(page), dtype=torch.uint8) for page in pages
        ]
        return super().put(namespace, keys[start:], carriers)

    async def get(self, namespace, keys):
        self._keys(namespace, keys)
        result = []
        # Disk history may exceed the transient CPU staging capacity. Read
        # bounded chunks rather than silently truncating a valid long prefix.
        batch_size = max(
            1, min(self._store.capacity, 16 * 1024**2 // (self._page_elements * 2))
        )
        for offset in range(0, len(keys), batch_size):
            pages = await super().get(namespace, keys[offset : offset + batch_size])
            for page in pages:
                if page is None:
                    return result
                result.append(page.numpy().tobytes())
        return result

    def contains(self, namespace, keys):
        storage_keys = self._keys(namespace, keys)
        ready = set(self._store.ready_keys(storage_keys))
        return [key in ready for key in storage_keys]

    async def verified_prefix(self, namespace: str, keys: list[str]) -> int:
        # The LMCache directory is optimistic. Only the background writer needs
        # to confirm residency before skipping publication; reuse the bounded
        # read path to reconcile L2 eviction without transferring payloads by RPC.
        self._keys(namespace, keys)
        candidates = len(keys)
        if not candidates:
            return 0
        assert self._store is not None
        first_missing = candidates
        # Reconcile the entire candidate suffix as well: a later missing page
        # must not retain an optimistic directory entry that skips its rewrite.
        for start in range(0, candidates, self._store.capacity):
            pages = await super().get(
                namespace, keys[start : min(candidates, start + self._store.capacity)]
            )
            for index, page in enumerate(pages):
                if page is None:
                    first_missing = min(first_missing, start + index)
        return first_missing

    def get_stats(self):
        return {**super().get_stats(), "namespace": self._wire_namespace}
