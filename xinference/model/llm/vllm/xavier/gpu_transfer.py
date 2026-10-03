# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Local CUDA IPC caches and persistent xoscar NIXL transfer buffers."""

import asyncio
import logging
from importlib.metadata import version
from typing import Dict

import torch
import xoscar as xo
from packaging.version import Version

from .request_transfer import (
    MAX_REQUEST_BLOCKS,
    MAX_REQUEST_BYTES,
    LayerRead,
    batch_reads,
    unpack_reads,
)
from .tiered_snapshot import TieredKVSnapshotStore

logger = logging.getLogger(__name__)


async def finish_before_cancel(task):
    """Keep ownership until completion, including after repeated cancellation."""
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        # Observe the result without replacing the caller's cancellation.
        if not task.cancelled():
            task.exception()
        raise


class GPUTransfer:
    def __init__(self, actor, caches: Dict[str, torch.Tensor], budget: int):
        if Version(version("xoscar")) < Version("0.11.1"):
            raise RuntimeError("Xavier GPU transfer requires xoscar[nixl]>=0.11.1")
        if not actor.address.startswith("nixl://"):
            raise RuntimeError("Xavier GPU transfer requires a NIXL actor pool")
        if not caches or any(not cache.is_cuda for cache in caches.values()):
            raise ValueError("Xavier GPU transfer requires registered CUDA KV caches")
        devices = {cache.device for cache in caches.values()}
        if len(devices) != 1:
            raise ValueError("Xavier GPU transfer requires one CUDA device per replica")
        self.actor, self.caches = actor, caches
        self.device = next(iter(devices))
        sizes = [cache[0].numel() * cache.element_size() for cache in caches.values()]
        self.store = TieredKVSnapshotStore(
            actor._snapshot_store.capacity, budget, sum(sizes), self.device
        )
        # Fixed slabs avoid registration churn. Separate directions allow hybrids
        # to serve a peer while they themselves are waiting for incoming data.
        self.slab_bytes = max(
            max(sizes), min(MAX_REQUEST_BYTES, sum(sizes) * MAX_REQUEST_BLOCKS)
        )
        self.send_buffer = torch.zeros(
            self.slab_bytes, dtype=torch.uint8, device=self.device
        )
        self.recv_buffer = torch.zeros_like(self.send_buffer)
        self.recv_ref = xo.buffer_ref(actor.address, self.recv_buffer)
        # Keep stable views for small warm-cache transfers. Reuse their buffer
        # identities, rather than creating and registering a slice per request.
        small_bytes = min(self.slab_bytes, 256 * 1024)
        self.send_buffers = {self.slab_bytes: self.send_buffer}
        self.recv_buffers = {self.slab_bytes: self.recv_buffer}
        self.recv_refs = {self.slab_bytes: self.recv_ref}
        if small_bytes < self.slab_bytes:
            self.send_buffers[small_bytes] = self.send_buffer[:small_bytes]
            self.recv_buffers[small_bytes] = self.recv_buffer[:small_bytes]
            self.recv_refs[small_bytes] = xo.buffer_ref(
                actor.address, self.recv_buffers[small_bytes]
            )
        self.send_lock, self.recv_lock = asyncio.Lock(), asyncio.Lock()
        self.tasks: set[asyncio.Task] = set()
        self.closing = False
        self.metrics = dict(
            gpu_batches=0,
            cpu_batches=0,
            wire_bytes=0,
            useful_bytes=0,
            load_calls=0,
            load_requests=0,
        )

    async def run(self, function, *args):
        if self.closing:
            raise RuntimeError("Xavier GPU transfer is shutting down")
        task = asyncio.create_task(function(*args))
        self.tasks.add(task)

        def completed(future):
            self.tasks.discard(future)
            if not future.cancelled():
                future.exception()  # Observe failures even after repeated cancellation.

        task.add_done_callback(completed)
        return await finish_before_cancel(task)

    async def load_requests_with_leases(self, requests, leases):
        try:
            await self.load_requests(requests)
        finally:
            results = await asyncio.gather(
                *(
                    self.actor.release_remote_blocks_v1(lease, ranks)
                    for lease, ranks in leases
                    if lease
                ),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result

    async def close(self):
        task = getattr(self, "_close_task", None)
        if task is None:
            self.closing = True
            task = self._close_task = asyncio.create_task(self._close())
        return await finish_before_cancel(task)

    async def _close(self):
        if self.tasks:
            await asyncio.gather(*list(self.tasks), return_exceptions=True)
        try:
            await asyncio.to_thread(torch.cuda.synchronize, self.device)
            logger.info(
                "Xavier GPU cache stats: %s",
                dict(self.metrics, cache=self.store.stats()),
            )
        finally:
            self.recv_refs.clear()
            self.recv_buffers.clear()
            self.send_buffers.clear()
            self.recv_ref = None
            self.caches.clear()

    async def stage(self, entries):
        if self.closing:
            raise RuntimeError("Xavier GPU transfer is shutting down")
        keys = {key for layers in entries for ids, _ in layers.values() for key in ids}
        failed = False
        copied = False
        try:
            for layers in entries:
                for layer, (block_keys, ids) in layers.items():
                    if all(
                        key in self.store.ready
                        and layer in self.store.blocks.get(key, {})
                        for key in block_keys
                    ):
                        for key in block_keys:
                            self.store.blocks.move_to_end(key)
                        continue
                    cache = self.caches[layer]
                    blocks = cache.index_select(
                        0, torch.tensor(ids, device=cache.device)
                    )
                    self.store.stage(layer, block_keys, blocks)
                    copied = True
        except Exception:
            failed = True
            logger.warning(
                "Failed to stage Xavier GPU snapshots; skipping new snapshots",
                exc_info=True,
            )
        finally:
            # Even a failed gather/copy may have queued reads of EngineCore's
            # slots. Drain them before returning ownership to EngineCore.
            try:
                if copied or failed:
                    await asyncio.to_thread(torch.cuda.synchronize, self.device)
            except BaseException:
                failed = True
                raise
            finally:
                if failed:
                    for key in keys:
                        if key in self.store.blocks and key not in self.store.ready:
                            self.store._drop(key)

    def locations(self, reads):
        result = {}
        for read in reads:
            for key in read.keys:
                if (
                    key not in self.store.ready
                    or read.layer not in self.store.blocks[key]
                ):
                    raise KeyError("Requested Xavier snapshot is not published")
                tensor = self.store.blocks[key][read.layer]
                if (
                    tensor.dtype != read.dtype
                    or tuple(tensor.shape) != read.block_shape
                ):
                    raise ValueError("Xavier source and destination KV layouts differ")
                result[key] = self.store.tiers[key]
        return result

    async def send(self, reads, remote_ref, slab_bytes):
        async with self.send_lock:
            if slab_bytes not in self.send_buffers:
                raise ValueError("Xavier peer transfer slab sizes differ")
            locations = self.locations(reads)
            if any(tier != "gpu" for tier in locations.values()):
                raise ValueError("GPU transfer requested for a CPU snapshot")
            size = sum(read.nbytes for read in reads)
            if size > slab_bytes:
                raise ValueError("Xavier GPU batch exceeds transfer slab")
            offset = 0
            for read in reads:
                end = offset + read.nbytes
                target = (
                    self.send_buffer[offset:end]
                    .view(read.dtype)
                    .reshape(len(read.keys), *read.block_shape)
                )
                torch.stack(
                    [self.store.blocks[key][read.layer] for key in read.keys],
                    out=target,
                )
                offset = end
            await asyncio.to_thread(torch.cuda.synchronize, self.device)
            await xo.copy_to([self.send_buffers[slab_bytes]], [remote_ref])
            self.metrics["wire_bytes"] += slab_bytes
            self.metrics["useful_bytes"] += size

    async def load(self, ranks):
        return await self.load_requests([ranks])

    async def load_requests(self, requests):
        from .transfer import TransferActor

        # Index by destination, not hash: one cached block may feed multiple
        # requests' slots. Reject conflicting writes before any transfer starts.
        ranks = {}
        owners = {}
        request_bytes = {}
        for request in requests:
            for rank, layers in request.items():
                size = sum(
                    len(mapping)
                    * self.caches[name][0].numel()
                    * self.caches[name].element_size()
                    for name, mapping in layers.items()
                )
                request_bytes[rank] = max(request_bytes.get(rank, 0), size)
                for name, mapping in layers.items():
                    for key, destination in mapping.items():
                        target = (name, destination)
                        if target in owners:
                            if owners[target] != key:
                                raise ValueError("Conflicting Xavier KV destinations")
                            continue
                        owners[target] = key
                        ranks.setdefault(rank, {}).setdefault(name, {})[
                            destination
                        ] = key

        self.metrics["load_calls"] += 1
        self.metrics["load_requests"] += len(requests)
        async with self.recv_lock:
            for rank, layers in ranks.items():
                sender = await xo.actor_ref(
                    address=self.actor._world_addresses[rank],
                    uid=f"{TransferActor.default_uid()}-{rank}",
                )
                reads = [
                    LayerRead(
                        name,
                        list(mapping.values()),
                        list(mapping),
                        tuple(self.caches[name].shape[1:]),
                        self.caches[name].dtype,
                    )
                    for name, mapping in layers.items()
                    if mapping
                ]
                locations = await sender.gpu_snapshot_locations_v1(reads)
                for tier in ("gpu", "cpu"):
                    selected = []
                    for read in reads:
                        pairs = [
                            (key, dest)
                            for key, dest in zip(read.keys, read.destinations)
                            if locations[key] == tier
                        ]
                        if pairs:
                            selected.append(
                                LayerRead(
                                    read.layer,
                                    [p[0] for p in pairs],
                                    [p[1] for p in pairs],
                                    read.block_shape,
                                    read.dtype,
                                )
                            )
                    # Combining small requests must not turn each transfer back
                    # into a padded full-slab copy. Large requests retain the
                    # original batch bound to avoid fragmenting cold prefills.
                    limit = self.slab_bytes
                    if tier == "gpu":
                        limit = min(
                            (n for n in self.recv_refs if n >= request_bytes[rank]),
                            default=self.slab_bytes,
                        )
                    for batch in batch_reads(selected, max_bytes=limit):
                        size = sum(read.nbytes for read in batch)
                        if tier == "gpu":
                            slab_bytes = min(n for n in self.recv_refs if n >= size)
                            await sender.send_gpu_request_v1(
                                batch, self.recv_refs[slab_bytes], slab_bytes
                            )
                            payload = self.recv_buffer[:size]
                        else:
                            payload = await self.actor.read_request_blocks_v1(
                                rank, batch
                            )
                        for read, blocks in unpack_reads(payload, batch):
                            cache = self.caches[read.layer]
                            cache[
                                torch.tensor(read.destinations, device=cache.device)
                            ] = blocks.to(cache.device, non_blocking=True)
                        # Both producer's IPC ownership and receiver slab reuse
                        # require the cache writes to finish before acknowledging.
                        await asyncio.to_thread(torch.cuda.synchronize, self.device)
                        self.metrics[tier + "_batches"] += 1


class GPUTransferMixin:
    def map_gpu_caches_v1(self, descriptors, budget):
        from torch.multiprocessing.reductions import rebuild_cuda_tensor

        if getattr(self, "_gpu_transfer", None) is not None:
            raise RuntimeError("Xavier GPU caches already registered")
        caches = {
            name: rebuild_cuda_tensor(*desc) for name, desc in descriptors.items()
        }
        runtime = GPUTransfer(self, caches, budget)
        self._gpu_transfer = runtime
        self._snapshot_store = runtime.store

    async def stage_gpu_requests_v1(self, entries):
        runtime = self._gpu_transfer
        return await runtime.run(runtime.stage, entries)

    def gpu_snapshot_locations_v1(self, reads):
        return self._gpu_transfer.locations(reads)

    async def send_gpu_request_v1(self, reads, remote_ref, slab_bytes):
        runtime = self._gpu_transfer
        return await runtime.run(runtime.send, reads, remote_ref, slab_bytes)

    async def load_gpu_request_v1(self, ranks):
        runtime = self._gpu_transfer
        return await runtime.run(runtime.load, ranks)

    async def load_gpu_requests_v1(self, requests, leases=()):
        runtime = self._gpu_transfer
        # The actor loop advances independently of EngineCore. Keep both writes
        # and lease cleanup inside the protected task before returning readiness.
        return await runtime.run(runtime.load_requests_with_leases, requests, leases)

    async def close_gpu_caches_v1(self):
        runtime = getattr(self, "_gpu_transfer", None)
        if runtime is not None:
            await runtime.close()
