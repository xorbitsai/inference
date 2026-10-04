# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Request-scoped GPU handoff with independent tiered history."""

import asyncio
import logging
import time
from dataclasses import dataclass, field

import torch
import xoscar as xo

from .direct_history import DirectHistoryMixin
from .gpu_transfer import GPUTransfer
from .request_transfer import LayerRead, batch_reads, unpack_reads

logger = logging.getLogger(__name__)


@dataclass
class DirectRequest:
    request_id: str
    blocks: set[int]
    deadline: float
    claimed: bool = False
    reading: bool = False
    released: bool = False
    indices: dict[tuple[int, ...], torch.Tensor] = field(default_factory=dict)
    hashes: dict[int, int] = field(default_factory=dict)
    retaining: bool = False
    retention_attempted: bool = False


class DirectGPUTransfer(DirectHistoryMixin, GPUTransfer):
    def __init__(self, actor, caches, budget, *, slab_bytes: int | None = None):
        super().__init__(actor, caches, 0, slab_bytes=slab_bytes)
        self.direct_requests: dict[str, DirectRequest] = {}
        self.finished_sending: set[str] = set()
        self.metrics.update(
            direct_registered=0, direct_finished=0, direct_expired=0, index_uploads=0
        )

        self._init_history(budget)

    def _index_tensor(
        self, indices: dict[tuple[int, ...], torch.Tensor], values: list[int]
    ) -> torch.Tensor:
        # Reuse across layers and slabs within one request. Constructing a CUDA
        # tensor from a Python list otherwise synchronizes an H2D copy each time.
        key = tuple(values)
        if key not in indices:
            indices[key] = torch.tensor(values, dtype=torch.long, device=self.device)
            self.metrics["index_uploads"] += 1
        return indices[key]

    async def _close(self):
        self._history_closing = True
        if self._history_task is not None:
            await asyncio.gather(self._history_task, return_exceptions=True)
        if self.history is not None:
            self.metrics["history"] = self.history.stats()
        await super()._close()
        if self.history is not None:
            for key in list(self.history.blocks):
                self.history._drop(key)
            self.history.leases.clear()
            self.history.evicted.clear()
        self._history_leases.clear()
        self._history_probation.clear()
        for state in self.direct_requests.values():
            state.indices.clear()
        self.direct_requests.clear()
        self.finished_sending.clear()

    def register_direct(self, ticket, request_id, blocks, hashes=None):
        if self.closing or ticket in self.direct_requests:
            raise ValueError("Invalid direct handoff registration")
        if not blocks or any(
            block < 0 or block >= len(cache)
            for cache in self.caches.values()
            for block in blocks
        ):
            raise ValueError("Invalid direct KV block IDs")
        if hashes is not None and len(hashes) != len(blocks):
            raise ValueError("History hashes and engine blocks differ")
        self.direct_requests[ticket] = DirectRequest(
            request_id,
            set(blocks),
            time.monotonic() + 120,
            hashes=dict(zip(blocks, hashes or [])),
        )
        self.metrics["direct_registered"] += 1

    def _live_direct(self, ticket):
        state = self.direct_requests.get(ticket)
        if state is None or state.released:
            return None
        if not state.reading and time.monotonic() >= state.deadline:
            self.metrics["direct_expired"] += 1
            state.retention_attempted = True
            self.release_direct(ticket)
            return None
        return state

    def claim_direct(self, ticket):
        state = self._live_direct(ticket)
        if state is None:
            return False
        # Reclaim abandoned claims even when D dies before submitting a load.
        # A scheduling retry or a completed slab refreshes this idle lease.
        state.claimed = True
        state.deadline = time.monotonic() + 600
        return True

    def abandon_direct(self, ticket):
        state = self.direct_requests.get(ticket)
        if state is not None and not state.claimed:
            self.release_direct(ticket)

    def release_direct(self, ticket):
        state = self.direct_requests.get(ticket)
        if state is None:
            return
        state.released = True
        if not state.reading:
            if self._schedule_history(ticket, state):
                return
            state.indices.clear()
            self.finished_sending.add(state.request_id)
            del self.direct_requests[ticket]
            self.metrics["direct_finished"] += 1

    def poll_direct(self):
        for ticket in list(self.direct_requests):
            self._live_direct(ticket)
        result = self.finished_sending
        self.finished_sending = set()
        return result

    async def send_direct(self, ticket, reads, remote_ref, slab_bytes):
        async with self.send_lock:
            state = self._live_direct(ticket)
            if state is None:
                return False
            if slab_bytes not in self.send_buffers:
                raise ValueError("Direct peer transfer slab sizes differ")
            if sum(read.nbytes for read in reads) > slab_bytes:
                raise ValueError("Direct transfer exceeds slab capacity")
            for read in reads:
                cache = self.caches[read.layer]
                if (
                    not set(read.keys) <= state.blocks
                    or cache.dtype != read.dtype
                    or tuple(cache.shape[1:]) != read.block_shape
                ):
                    raise ValueError("Direct KV source layout or block IDs differ")
            state.reading = True
            state.deadline = time.monotonic() + 600
            gather_fenced = False
            try:
                offset = 0
                for read in reads:
                    end = offset + read.nbytes
                    target = (
                        self.send_buffer[offset:end]
                        .view(read.dtype)
                        .reshape(len(read.keys), *read.block_shape)
                    )
                    cache = self.caches[read.layer]
                    torch.index_select(
                        cache,
                        0,
                        self._index_tensor(state.indices, read.keys),
                        out=target,
                    )
                    offset = end
                await asyncio.to_thread(torch.cuda.synchronize, self.device)
                gather_fenced = True
                await xo.copy_to([self.send_buffers[slab_bytes]], [remote_ref])
                self.metrics["wire_bytes"] += slab_bytes
                self.metrics["useful_bytes"] += offset
            finally:
                # Failed/cancelled gathers may still read engine slots. A
                # completed gather has already drained those reads; copy_to
                # owns only the slab and drains its transfer before returning.
                if not gather_fenced:
                    await asyncio.to_thread(torch.cuda.synchronize, self.device)
                state.reading = False
                state.deadline = time.monotonic() + 600
                if state.released:
                    self.release_direct(ticket)
            return True

    async def load_direct(self, requests, tickets):
        failed = False
        try:
            return await self._load_direct(requests, tickets)
        except BaseException:
            failed = True
            raise
        finally:

            results = await asyncio.gather(
                *(
                    self.actor.release_remote_direct_gpu_v1(rank, ticket)
                    for ranks, ticket in zip(requests, tickets)
                    for rank in ranks
                ),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    if not failed:
                        raise result
                    logger.warning("Direct ticket cleanup failed", exc_info=result)

    async def _load_direct(self, requests, tickets):
        self.metrics["load_calls"] += 1
        self.metrics["load_requests"] += len(requests)
        invalid_blocks = set()
        async with self.recv_lock:
            for ranks, ticket in zip(requests, tickets):
                if len(ranks) != 1:
                    raise ValueError("Direct handoff requires one producer")
                rank, layers = next(iter(ranks.items()))
                sender = await xo.actor_ref(
                    address=self.actor._world_addresses[rank],
                    uid=f"{self.actor.default_uid()}-{rank}",
                )
                indices: dict[tuple[int, ...], torch.Tensor] = {}
                scatter_pending = False
                try:
                    reads = [
                        LayerRead(
                            name,
                            list(mapping),
                            list(mapping.values()),
                            tuple(self.caches[name].shape[1:]),
                            self.caches[name].dtype,
                        )
                        for name, mapping in layers.items()
                        if mapping
                    ]
                    for batch in batch_reads(reads, max_bytes=self.slab_bytes):
                        size = sum(read.nbytes for read in batch)
                        slab_bytes = min(n for n in self.recv_refs if n >= size)
                        available = await sender.send_direct_gpu_v1(
                            ticket, batch, self.recv_refs[slab_bytes], slab_bytes
                        )
                        if not available:
                            # Never expose a partial request as valid KV. vLLM's
                            # load-error callback recomputes all its destinations.
                            invalid_blocks.update(
                                dest
                                for mapping in layers.values()
                                for dest in mapping.values()
                            )
                            break
                        scatter_pending = True
                        for read, blocks in unpack_reads(
                            self.recv_buffer[:size], batch
                        ):
                            cache = self.caches[read.layer]
                            cache[self._index_tensor(indices, read.destinations)] = (
                                blocks
                            )
                        await asyncio.to_thread(torch.cuda.synchronize, self.device)
                        scatter_pending = False
                        self.metrics["gpu_batches"] += 1
                finally:
                    # Every successful batch already fenced its writes. Fence
                    # partial/error batches before releasing destination slots.
                    if scatter_pending:
                        await asyncio.to_thread(torch.cuda.synchronize, self.device)
                    indices.clear()
        return invalid_blocks
