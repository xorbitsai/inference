# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Bounded GPU/CPU history retained after request-scoped handoff."""

import asyncio
import logging
import time
from collections import OrderedDict

import torch

from .request_transfer import LayerRead, batch_reads
from .tiered_snapshot import TieredKVSnapshotStore

logger = logging.getLogger(__name__)


class HistoryStore(TieredKVSnapshotStore):
    """Independent GPU-first history with bounded CPU overflow."""

    def __init__(self, budget, block_bytes, device, cpu_capacity=0):
        super().__init__(max(1, cpu_capacity), budget, block_bytes, device)
        self.cpu_capacity = cpu_capacity
        self.capacity = self.gpu_capacity + cpu_capacity

    def reserve(self, lease, keys):
        if not set(keys).issubset(self.ready):
            return False
        self.leases.setdefault(lease, set()).update(keys)
        # Prefix heads outlive their tails. Reservation is not a cache hit:
        # scheduling can retry or abort without ever restoring any blocks.
        for key in reversed(keys):
            self.touch(key)
        return True

    def read(self, layer, keys, device=None):
        # Batch same-tier blocks before H2D instead of synchronizing once per
        # block. load_history partitions reads by tier and holds their leases.
        values = [self.blocks[key][layer] for key in keys]
        target = torch.device("cpu") if device is None else device
        if all(value.device == values[0].device for value in values):
            return torch.stack(values).to(target).contiguous()
        return super().read(layer, keys, device)


class DirectHistoryMixin:
    def _init_history(self, budget):
        block_bytes = sum(c[0].numel() * c.element_size() for c in self.caches.values())
        cpu_capacity = self.store.cpu_capacity
        self.history = (
            HistoryStore(budget, block_bytes, self.device, cpu_capacity)
            if budget
            else None
        )
        self._history_task = None
        self._history_closing = False
        self._history_leases = {}
        self._history_reading = set()
        # Metadata only. A second completed request demonstrates reuse before
        # new content can displace existing GPU history. Empty slots need no
        # probation, preserving first-request retention while there is room.
        self._history_probation: OrderedDict[int, None] = OrderedDict()
        self._history_probation_limit = min(
            8192, 2 * self.history.capacity if self.history else 0
        )
        self.history_chunk_bytes = 2 * 1024 * 1024
        self.history_fill_chunk_bytes = 8 * 1024 * 1024
        self.history_pending_bytes = min(
            self.history.capacity * block_bytes if self.history else 0, 64 * 1024 * 1024
        )
        # A soft admission deadline, not a cancellation timeout: an in-flight
        # CUDA chunk always drains before the engine can reuse its blocks.
        self.history_retention_seconds = 0.01
        self.metrics.update(
            history_saved_blocks=0,
            history_skipped_requests=0,
            history_failures=0,
            history_hit_blocks=0,
            history_cpu_hit_blocks=0,
            history_gpu_hit_blocks=0,
            history_loaded_requests=0,
            history_admission_rejected_blocks=0,
            history_admission_reused_blocks=0,
            history_save_chunks=0,
            history_capacity_limited_blocks=0,
            history_deadline_dropped_blocks=0,
            history_closing_dropped_blocks=0,
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
            if not self.history.reserve(lease, matched):
                return []
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
        limit = min(
            self.history.capacity,
            self.history_pending_bytes // self.history.block_bytes,
        )
        candidates = []
        free = max(0, self.history.capacity - len(self.history.blocks))
        prefix = []
        blocked = None
        for block, key in state.hashes.items():
            if blocked is not None:
                if key not in self.history.ready:
                    self.metrics[blocked] += 1
                continue
            if len(prefix) >= self.history.capacity or (
                key not in self.history.ready and len(candidates) >= limit
            ):
                blocked = "history_capacity_limited_blocks"
                if key not in self.history.ready:
                    self.metrics[blocked] += 1
            elif key in self.history.ready:
                prefix.append(key)
            elif free or key in self._history_probation:
                candidates.append((key, block))
                prefix.append(key)
                free = max(0, free - 1)
                if key in self._history_probation:
                    self.metrics["history_admission_reused_blocks"] += 1
            else:
                blocked = "history_admission_rejected_blocks"
                self.metrics[blocked] += 1
        # Observe the full request, but keep heads newest in both probation and
        # cache LRU. A rejected head must not admit unreachable suffix blocks.
        for key in reversed(list(state.hashes.values())):
            if key in self.history.ready:
                self.history.touch(key)
                self._history_probation.pop(key, None)
            else:
                self._history_probation[key] = None
                self._history_probation.move_to_end(key)
                while len(self._history_probation) > self._history_probation_limit:
                    self._history_probation.popitem(last=False)
        if not candidates:
            return False
        # Observe completed requests even when the single writer is busy, but
        # never queue another writer or extend engine block ownership for it.
        if self._history_task is not None and not self._history_task.done():
            self.metrics["history_skipped_requests"] += 1
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
                    exc_info=exc,
                )

        task.add_done_callback(done)
        return True

    async def _retain_history(self, ticket, state, candidates, deadline):
        pending = []
        prefix_keys = list(state.hashes.values())[: self.history.capacity]
        try:
            # At most one writer and one bounded chunk. Yield between chunks;
            # never queue the writer ahead of an already active handoff or load.
            start = 0
            while start < len(candidates):
                # Filling unused capacity can amortize per-layer gathers;
                # replacements use smaller chunks to limit lock hold time.
                free = self.history.gpu_capacity - self.history.counts["gpu"]
                chunk_bytes = (
                    self.history_fill_chunk_bytes
                    if free > 0
                    else self.history_chunk_bytes
                )
                count = max(
                    1,
                    min(
                        32,
                        min(self.slab_bytes, chunk_bytes) // self.history.block_bytes,
                    ),
                )
                if free > 0:
                    count = min(count, free)
                chunk = candidates[start : start + count]
                while True:
                    if self._history_closing or time.monotonic() >= deadline:
                        reason = "closing" if self._history_closing else "deadline"
                        self.metrics[f"history_{reason}_dropped_blocks"] += (
                            len(candidates) - start
                        )
                        return
                    if not self.send_lock.locked() and not self.recv_lock.locked():
                        break
                    await asyncio.sleep(0.001)
                start += len(chunk)
                async with self.send_lock:
                    self._expire_history_leases()
                    pairs = [(k, b) for k, b in chunk if k not in self.history.ready]
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
                    self.metrics["history_save_chunks"] += 1
                    for key in available:
                        self._history_probation.pop(key, None)
                    for key in reversed(prefix_keys):
                        if key in self.history.ready:
                            self.history.touch(key)
                    complete = len(available) == len(pending)
                    if not complete:
                        self.metrics["history_capacity_limited_blocks"] += (
                            len(pending) - len(available) + len(candidates) - start
                        )
                    pending = []
                    del layers, index
                    if not complete:
                        # Capacity may be pinned by another reader. Do not
                        # continue past a prefix we could not publish.
                        return
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
        failed = False
        try:
            await self._load_history(requests, leases)
        except BaseException:
            failed = True
            raise
        finally:
            errors = []
            for lease in leases:
                try:
                    self.release_history(lease)
                except BaseException as error:
                    errors.append(error)
            if errors and not failed:
                raise errors[0]
            for cleanup_error in errors:
                logger.warning("History lease cleanup failed", exc_info=cleanup_error)

    async def _load_history(self, requests, leases):
        async with self.recv_lock:
            for ranks, lease in zip(requests, leases):
                indices = {}
                try:
                    if self.history is None or lease not in self._history_leases:
                        raise RuntimeError("GPU history lease expired")
                    self._history_reading.add(lease)
                    reads = []
                    written = set()
                    for layers in ranks.values():
                        for name, mapping in layers.items():
                            cache = self.caches[name]
                            if not set(mapping).issubset(self.history.leases[lease]):
                                raise ValueError("History read outside lease")
                            for tier in ("gpu", "cpu"):
                                pairs = [
                                    (key, dest)
                                    for key, dest in mapping.items()
                                    if self.history.tiers[key] == tier
                                ]
                                if pairs:
                                    keys, destinations = map(list, zip(*pairs))
                                    reads.append(
                                        LayerRead(
                                            name,
                                            keys,
                                            destinations,
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
                            written.update(read.keys)
                        await asyncio.to_thread(torch.cuda.synchronize, self.device)
                    hits = {"gpu": 0, "cpu": 0}
                    for key in written:
                        tier = self.history.tiers[key]
                        hits[tier] += 1
                        self.metrics[f"history_{tier}_hit_blocks"] += 1
                        self.history.metrics[f"{tier}_hits"] += 1
                    self.metrics["history_hit_blocks"] += len(written)
                    self.metrics["history_loaded_requests"] += 1
                    logger.debug(
                        "Restored Xavier history: blocks=%s gpu=%s cpu=%s",
                        len(written),
                        hits["gpu"],
                        hits["cpu"],
                    )
                finally:
                    await asyncio.to_thread(torch.cuda.synchronize, self.device)
                    indices.clear()
                    self._history_reading.discard(lease)
