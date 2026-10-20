# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Optional LMCache storage with bounded Xavier staging and leased read copies.

LMCache owns its DRAM pool, disk I/O and eviction. Xavier retains only partial
exports and in-flight reads, independently of recyclable engine GPU pages.
"""

import asyncio
import hashlib
import logging
import math
import re
import shutil
import sys
import threading
import time
from collections import OrderedDict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from packaging.version import Version

from ..contract import fingerprint_metadata
from .torch.snapshot import KVSnapshotStore

logger = logging.getLogger(__name__)


def check_lmcache_available() -> None:
    if sys.platform != "linux":
        raise RuntimeError("The Xavier LMCache backend requires Linux")
    try:
        installed = Version(version("lmcache"))
    except PackageNotFoundError as error:
        raise RuntimeError(
            "Install lmcache>=0.5.6,<0.6 in the Xavier transfer actor environment"
        ) from error
    if not Version("0.5.6") <= installed < Version("0.6"):
        raise RuntimeError("The Xavier LMCache backend requires lmcache>=0.5.6,<0.6")
    try:
        from lmcache.lmcache_fs import LMCacheFSClient  # noqa: F401
        from lmcache.v1.distributed.storage_manager import StorageManager  # noqa: F401
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Xavier LMCache requires a compatible LMCache build with the native FS extension"
        ) from error


class _LMCacheStorage:
    """The version-specific LMCache API is confined to this adapter."""

    def __init__(self, config: dict, namespace: str, page_bytes: int):
        check_lmcache_available()
        from lmcache.v1.distributed.api import (
            GroupedObjectKeys,
            MemoryLayoutDesc,
            ObjectKey,
            PrefetchTaskSpec,
        )
        from lmcache.v1.distributed.config import (
            EvictionConfig,
            L1ManagerConfig,
            L1MemoryManagerConfig,
            StorageManagerConfig,
        )
        from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
        from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
            FSNativeL2AdapterConfig,
        )
        from lmcache.v1.distributed.storage_manager import StorageManager

        self._key_type = ObjectKey
        self._group_type = GroupedObjectKeys
        self._spec_type = PrefetchTaskSpec
        self._namespace = namespace
        self._layout = MemoryLayoutDesc([torch.Size([page_bytes])], [torch.uint8])
        self._closed = threading.Event()
        # Each actor owns an independent LMCache instance and disk directory.
        # Deployment IDs prevent reuse after relaunch with different weights.
        self._disk_path = Path(config["disk_path"]) / namespace
        disk = FSNativeL2AdapterConfig(
            base_path=str(self._disk_path),
            max_capacity_gb=config["disk_bytes"] / (1 << 30),
        )
        disk.eviction_config = EvictionConfig("LRU")
        self.manager = StorageManager(
            StorageManagerConfig(
                l1_manager_config=L1ManagerConfig(
                    L1MemoryManagerConfig(
                        size_in_bytes=config["cpu_bytes"],
                        use_lazy=False,
                        shm_name="",
                    )
                ),
                eviction_config=EvictionConfig("LRU"),
                l2_adapter_config=L2AdaptersConfig([disk]),
                prefetch_policy="retain",
            )
        )

    def _key(self, key: int):
        digest = hashlib.sha256(str(key).encode("ascii")).digest()
        return self._key_type(chunk_hash=digest, model_name=self._namespace, kv_rank=0)

    def store(self, key: int, page: torch.Tensor) -> bool:
        object_key = self._key(key)
        reserved = self.manager.reserve_write([object_key], self._layout)
        obj = reserved.get(object_key)
        if obj is None:
            return False
        # Both tensors are flat CPU uint8 with exactly the negotiated page size.
        obj.tensor.view(torch.uint8).reshape(-1).copy_(page)
        self.manager.finish_write([object_key])
        return True

    def load(
        self, keys: List[int]
    ) -> Tuple[Dict[int, torch.Tensor], Set[int], int, int]:
        object_keys = [self._key(key) for key in keys]
        spec = self._spec_type(
            [self._group_type(object_keys, 0, self._layout)], fetching_policy="full"
        )
        handle = self.manager.submit_prefetch_task(spec)
        result = None
        while not self._closed.is_set():
            if self.manager.wait_prefetch_status(handle, 0.1):
                result = self.manager.query_prefetch_status(handle)
                if result is not None:
                    break
        if result is None:
            return {}, set(), 0, 0
        indices = result.hit_cells[0].get_indices_list()
        hits = [object_keys[i] for i in indices]
        pages = {}
        if hits:
            # The context releases successful locks on failure. On normal exit
            # release them explicitly, after copying out of LMCache ownership.
            with self.manager.read_prefetched_results(hits) as objects:
                if objects is not None:
                    pages = {
                        keys[i]: obj.tensor.view(torch.uint8).reshape(-1).clone()
                        for i, obj in zip(indices, objects)
                    }
            if objects is not None:
                self.manager.finish_read_prefetched(hits)
        found = result.found_cells
        absent = (
            set(keys) - {keys[i] for i in found[0].get_indices_list()}
            if found is not None
            else set()
        )
        return pages, absent, result.l1_hit_count, result.l2_hit_count

    def stop(self) -> None:
        self._closed.set()

    def close(self) -> None:
        self.manager.close()
        # The directory is private to this deployment/replica and cannot be
        # rediscovered after relaunch. Remove only that owned cache directory.
        if self._disk_path.exists():
            shutil.rmtree(self._disk_path)


class LMCacheSnapshotBackend(KVSnapshotStore):
    def __init__(
        self,
        capacity: int,
        config: dict,
        namespace: str,
        schema: Optional[Dict[str, Tuple[tuple, torch.dtype]]],
    ):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", namespace) or not schema:
            raise ValueError(
                "Xavier LMCache requires a deployment namespace and KV schema"
            )
        self._schema = dict(sorted(schema.items()))
        self._page_bytes = sum(
            math.prod(shape) * dtype.itemsize for shape, dtype in self._schema.values()
        )
        if self._page_bytes <= 0:
            raise ValueError("Xavier LMCache requires nonempty KV pages")
        allocation_bytes = (self._page_bytes + 4095) // 4096 * 4096
        if min(config["cpu_bytes"], config["disk_bytes"]) < allocation_bytes:
            raise ValueError("Xavier LMCache budgets cannot hold one aligned KV page")
        # Staging/leased copies are transient and have their own bounded budget.
        super().__init__(min(capacity, config["cpu_bytes"] // allocation_bytes))
        self._config = dict(config)
        self._known: OrderedDict[int, None] = OrderedDict()
        self._directory_capacity = (
            config["cpu_bytes"] + config["disk_bytes"]
        ) // allocation_bytes
        identity = fingerprint_metadata(
            {
                name: [list(shape), str(dtype)]
                for name, (shape, dtype) in self._schema.items()
            }
        )
        self._storage = _LMCacheStorage(
            config, namespace + "-" + identity[:16], self._page_bytes
        )
        self._pending: Optional[asyncio.Task] = None
        self._closing = False
        self._counts = dict(
            stored_pages=0, l1_hits=0, l2_hits=0, misses=0, timeouts=0, read_seconds=0.0
        )

    def _discard_resident(self, key: int) -> None:
        self.blocks.pop(key, None)
        self.logical_dtypes.pop(key, None)
        self.ready.discard(key)

    def _evict_one(self, pinned):
        evicted = super()._evict_one(pinned)
        # Eviction of a transient read copy does not evict the LMCache object.
        self.evicted.difference_update(self._known)
        return evicted

    def ready_keys(self, keys: List[int]) -> List[int]:
        ready = []
        for key in dict.fromkeys(keys):
            if key in self._known:
                ready.append(key)
                self._known.move_to_end(key)
        return ready

    def stage_packed_blocks(self, keys, packed, layers, *, retain_storage=False):
        # A transient page must release its allocation independently on publish.
        super().stage_packed_blocks(keys, packed, layers, retain_storage=False)

    def publish(self, keys: List[int], layers: Set[str]) -> List[int]:
        if layers != self._schema.keys():
            raise ValueError("Xavier LMCache publication differs from its KV schema")
        accepted = self.ready_keys(keys)
        for key in super().publish(keys, layers):
            parts = []
            for layer, (shape, dtype) in self._schema.items():
                tensor = self.blocks[key][layer]
                if (
                    tuple(tensor.shape) != tuple(shape)
                    or self.logical_dtypes[key][layer] != dtype
                ):
                    raise ValueError(
                        "Xavier LMCache payload differs from its KV schema"
                    )
                parts.append(tensor.reshape(-1).view(torch.uint8))
            if key in self._known or self._storage.store(key, torch.cat(parts)):
                if key not in self._known:
                    self._counts["stored_pages"] += 1
                self._known[key] = None
                self._known.move_to_end(key)
                if key not in accepted:
                    accepted.append(key)
            if not any(key in pinned for pinned in self.leases.values()):
                self._discard_resident(key)
        # Bound the optimistic directory independently of LMCache's actual
        # residency. A stale candidate becomes a miss during prefetch/reserve.
        while len(self._known) > self._directory_capacity:
            victim = next(
                (
                    k
                    for k in self._known
                    if not any(k in v for v in self.leases.values())
                ),
                None,
            )
            if victim is None:
                break
            del self._known[victim]
            self.evicted.add(victim)
        return [key for key in accepted if key in self._known]

    async def prefetch(self, keys: List[int]) -> bool:
        if self._closing or not self.can_stage(set(keys)):
            return False
        missing = [key for key in dict.fromkeys(keys) if key not in self.ready]
        if not missing:
            return True
        if not set(missing).issubset(self._known):
            return False
        # Bound abandoned or timed-out I/O to one job per actor. The job still
        # consumes the LMCache result and releases locks after caller cancellation.
        if self._pending is not None and not self._pending.done():
            return False
        started = time.monotonic()
        task = self._pending = asyncio.create_task(
            asyncio.to_thread(self._storage.load, missing)
        )

        def completed(future):
            if self._pending is future:
                self._pending = None
            if not future.cancelled():
                future.exception()

        task.add_done_callback(completed)
        try:
            pages, absent, l1_hits, l2_hits = await asyncio.wait_for(
                asyncio.shield(task), self._config["read_timeout"]
            )
        except asyncio.TimeoutError:
            self._counts["timeouts"] += 1
            return False
        except Exception:
            logger.warning(
                "Xavier LMCache prefetch failed; recomputing locally", exc_info=True
            )
            return False
        finally:
            self._counts["read_seconds"] += time.monotonic() - started
        self._counts["l1_hits"] += l1_hits
        self._counts["l2_hits"] += l2_hits
        self._counts["misses"] += len(missing) - len(pages)
        for key in absent:
            self._known.pop(key, None)
            self.evicted.add(key)
        if len(pages) != len(missing) or not self.can_stage(set(keys)):
            return False
        if any(
            page.dtype != torch.uint8
            or page.ndim != 1
            or page.numel() != self._page_bytes
            or page.device.type != "cpu"
            or not page.is_contiguous()
            for page in pages.values()
        ):
            raise ValueError("Invalid Xavier LMCache page")
        reservation = f"_prefetch:{id(task)}"
        self.leases[reservation] = set(keys)
        try:
            for key, page in pages.items():
                offset = 0
                for layer, (shape, dtype) in self._schema.items():
                    nbytes = math.prod(shape) * dtype.itemsize
                    carrier = torch.float16 if dtype == torch.bfloat16 else dtype
                    tensor = page[offset : offset + nbytes].view(carrier).reshape(shape)
                    super().stage(
                        layer, [key], tensor.unsqueeze(0), logical_dtype=dtype
                    )
                    offset += nbytes
                super().publish([key], set(self._schema))
        except BaseException:
            for key in missing:
                self._discard_resident(key)
            raise
        finally:
            self.leases.pop(reservation)
        return set(keys).issubset(self.ready)

    def release(self, lease: str) -> None:
        super().release(lease)
        pinned = set().union(*self.leases.values())
        for key in list(self.ready - pinned):
            self._discard_resident(key)

    def get_stats(self) -> dict:
        return {
            **self._counts,
            "backend": "lmcache",
            "pages": len(self._known),
            "resident_pages": len(self.ready),
            "capacity": self.capacity,
        }

    async def close(self) -> None:
        self._closing = True
        self._storage.stop()
        try:
            if self._pending is not None:
                await asyncio.shield(self._pending)
        finally:
            await asyncio.to_thread(self._storage.close)
