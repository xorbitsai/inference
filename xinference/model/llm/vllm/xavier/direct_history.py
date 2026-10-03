# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Bounded GPU history retained after request-scoped handoff."""

import asyncio
import logging
import time

import torch

from .request_transfer import LayerRead, batch_reads
from .tiered_snapshot import TieredKVSnapshotStore

logger = logging.getLogger(__name__)


class GPUHistoryStore(TieredKVSnapshotStore):
    """Independent, immutable GPU blocks; eviction never performs CPU copies."""

    def __init__(self, budget, block_bytes, device):
        super().__init__(1, budget, block_bytes, device)
        self.capacity = self.gpu_capacity
        self.cpu_capacity = 0

    def _admit(self, key, pinned):
        if not self.gpu_capacity:
            self.metrics["skipped"] += 1
            return False
        if len(self.blocks) >= self.gpu_capacity:
            victim = next((k for k in self.blocks if k not in pinned), None)
            if victim is None:
                self.metrics["skipped"] += 1
                return False
            self._drop(victim)
            # No external directory in this phase: lookup is local to the P.
            self.evicted.clear()
        self.blocks[key] = {}
        self.logical_dtypes[key] = {}
        self._block_sizes[key] = 0
        self.tiers[key] = "gpu"
        self._gpu_lru[key] = None
        self.counts["gpu"] += 1
        return True


class DirectHistoryMixin:
    def _init_history(self, budget):
        block_bytes = sum(c[0].numel() * c.element_size() for c in self.caches.values())
        self.history = (
            GPUHistoryStore(budget, block_bytes, self.device) if budget else None
        )
        self._history_task = None
        self._history_closing = False
        self._history_leases = {}
        self._history_reading = set()
        self.history_pending_bytes = min(budget, 64 * 1024 * 1024)
        # A soft admission deadline, not a cancellation timeout: an in-flight
        # CUDA chunk always drains before the engine can reuse its blocks.
        self.history_retention_seconds = 0.01
        self.metrics.update(
            history_saved_blocks=0,
            history_skipped_requests=0,
            history_failures=0,
            history_hit_blocks=0,
            history_loaded_requests=0,
        )

    def _expire_history_leases(self):
        now = time.monotonic()
        for lease, deadline in list(self._history_leases.items()):
            if deadline <= now and lease not in self._history_reading:
                self.release_history(lease)

    def reserve_history(self, lease, keys):
        if self.closing or self.history is None:
            return []
        self._expire_history_leases()
        if lease in self._history_leases:
            raise ValueError("Duplicate history lease")
        matched = []
        for key in keys:
            if key not in self.history.ready:
                break
            matched.append(key)
        if matched:
            assert self.history.reserve(lease, matched)
            self._history_leases[lease] = time.monotonic() + 120
        return matched

    def release_history(self, lease):
        if lease not in self._history_reading:
            self._history_leases.pop(lease, None)
            if self.history is not None:
                self.history.release(lease)

    def _schedule_history(self, ticket, state):
        if state.retaining:
            return True
        if state.retention_attempted:
            return False
        state.retention_attempted = True
        if self._history_closing or self.history is None or not state.hashes:
            return False
        if state.held_blocks * self.history.block_bytes > self.history_pending_bytes:
            self.metrics["history_skipped_requests"] += 1
            return False
        if self._history_task is not None and not self._history_task.done():
            self.metrics["history_skipped_requests"] += 1
            return False
        limit = min(
            self.history.gpu_capacity,
            self.history_pending_bytes // self.history.block_bytes,
        )
        candidates = []
        for block, key in state.hashes.items():
            if key in self.history.ready:
                self.history.touch(key)
            elif len(candidates) < limit:
                candidates.append((key, block))
        if not candidates:
            return False
        state.retaining = True
        deadline = time.monotonic() + self.history_retention_seconds
        task = self._history_task = asyncio.create_task(
            self._retain_history(ticket, state, candidates, deadline)
        )
        self.tasks.add(task)

        def done(future):
            self.tasks.discard(future)
            if not future.cancelled() and future.exception() is not None:
                exc = future.exception()
                logger.error(
                    "GPU history retention failed",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

        task.add_done_callback(done)
        return True

    async def _retain_history(self, ticket, state, candidates, deadline):
        pending = []
        try:
            # At most one writer and one bounded chunk. Yield between chunks;
            # never queue the writer ahead of an already active handoff or load.
            count = max(1, min(32, self.slab_bytes // self.history.block_bytes))
            for start in range(0, len(candidates), count):
                while self.send_lock.locked() or self.recv_lock.locked():
                    if self._history_closing or time.monotonic() >= deadline:
                        return
                    await asyncio.sleep(0.001)
                if self._history_closing or time.monotonic() >= deadline:
                    return
                async with self.send_lock:
                    self._expire_history_leases()
                    pairs = [
                        (k, b)
                        for k, b in candidates[start : start + count]
                        if k not in self.history.ready
                    ]
                    if not pairs:
                        continue
                    pending, ids = map(list, zip(*pairs))
                    index = self._index_tensor(state.indices, ids)
                    layers = {
                        name: cache.index_select(0, index)
                        for name, cache in self.caches.items()
                    }
                    self.history.stage_blocks(pending, layers)
                    await asyncio.to_thread(torch.cuda.synchronize, self.device)
                    available = self.history.publish(pending, set(self.caches))
                    self.metrics["history_saved_blocks"] += len(available)
                    pending = []
                    del layers, index
                await asyncio.sleep(0)
        except Exception:
            self.metrics["history_failures"] += 1
            logger.warning("Skipping failed GPU history retention", exc_info=True)
        finally:
            # Original engine slots stay owned even after D finishes, until all
            # optional snapshot reads have drained. Never publish partial data.
            await asyncio.to_thread(torch.cuda.synchronize, self.device)
            for key in pending:
                if key in self.history.blocks and key not in self.history.ready:
                    self.history._drop(key)
            self.history.evicted.clear()
            state.retaining = False
            self.release_direct(ticket)

    async def load_history(self, requests, leases):
        async with self.recv_lock:
            for ranks, lease in zip(requests, leases):
                indices = {}
                try:
                    if self.history is None or lease not in self._history_leases:
                        raise RuntimeError("GPU history lease expired")
                    self._history_reading.add(lease)
                    reads = []
                    for layers in ranks.values():
                        for name, mapping in layers.items():
                            cache = self.caches[name]
                            if not set(mapping).issubset(self.history.leases[lease]):
                                raise ValueError("History read outside lease")
                            reads.append(
                                LayerRead(
                                    name,
                                    list(mapping),
                                    list(mapping.values()),
                                    tuple(cache.shape[1:]),
                                    cache.dtype,
                                )
                            )
                    for batch in batch_reads(reads, max_bytes=self.slab_bytes):
                        for read in batch:
                            blocks = self.history.read(
                                read.layer, read.keys, self.device
                            )
                            if (
                                blocks.dtype != read.dtype
                                or tuple(blocks.shape[1:]) != read.block_shape
                            ):
                                raise ValueError("GPU history layout mismatch")
                            self.caches[read.layer][
                                self._index_tensor(indices, read.destinations)
                            ] = blocks
                        await asyncio.to_thread(torch.cuda.synchronize, self.device)
                    self.metrics["history_hit_blocks"] += len(
                        self.history.leases[lease]
                    )
                    self.metrics["history_loaded_requests"] += 1
                finally:
                    await asyncio.to_thread(torch.cuda.synchronize, self.device)
                    indices.clear()
                    self._history_reading.discard(lease)
                    self.release_history(lease)
