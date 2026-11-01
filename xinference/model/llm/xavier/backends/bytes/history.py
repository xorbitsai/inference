# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Canonical, historical prefix pages in LMCache, separate from live P/D KV."""

import re
from typing import Optional

import torch

from ...contract import KVCacheContract
from ..torch.storage import XavierCacheActor


class XavierHistoryCacheActor(XavierCacheActor):
    """Expose owned canonical bytes while LMCache manages CPU and L2 storage."""

    def __init__(self, cache_backend: dict):
        if cache_backend.get("name") != "lmcache":
            raise ValueError("Xavier P/D history requires the LMCache backend")
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
            "capacity_pages": self._store.capacity,
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
        pages = await super().get(namespace, keys)
        result = []
        for page in pages:
            if page is None:
                break
            result.append(page.numpy().tobytes())
        return result

    def get_stats(self):
        return {**super().get_stats(), "namespace": self._wire_namespace}
