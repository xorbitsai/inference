# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""GPU transfer of tensor-parallel shard rows between replicas over xoscar NIXL.

Every TP rank of a replica packs its part of a block into one row. The CPU
snapshot store remains the authority on which blocks exist and which are
leased. This runtime keeps a bounded GPU copy of recent rows on each rank's
device and moves rows between the matching ranks of two replicas.

An xoscar NIXL channel runs one copy at a time and keeps only the registration
of its most recent copy. One copy therefore carries the persistent slab of every
shard: the registration stays warm across requests. Gathers, scatters and
synchronization stay per device, on threads, so the actor loop never waits on a
GPU and one rank's device never waits for another's.

Rows missing from the GPU copy are left to the caller, which reads them from the
CPU snapshot store as before.
"""

import asyncio
import logging
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import xoscar as xo

from .gpu_transfer import finish_before_cancel

logger = logging.getLogger(__name__)

# xoscar retains the registration of one copy up to this size per channel.
REGISTRATION_BYTES = 256 * 1024 * 1024
# Rows per shard and round. Larger prefixes take several rounds.
MAX_SLAB_ROWS = 1024
# Small copies use a prefix view of the slab, kept registered while repeated.
SMALL_SLAB_BYTES = 4 * 1024 * 1024
# All TP ranks start a request's load in the same engine step.
RENDEZVOUS_SECONDS = 5.0
PREWARM_ATTEMPTS = 60
PREWARM_INTERVAL_SECONDS = 10.0

# (cache name, block shape, dtype, offset in the row, bytes)
Layout = Sequence[Tuple[str, Tuple[int, ...], torch.dtype, int, int]]


class ShardGPUUnavailable(RuntimeError):
    """The GPU path cannot serve this request; the caller reads from CPU."""


def _layout_key(layout: Layout):
    return [
        (name, tuple(shape), str(dtype), start, n)
        for name, shape, dtype, start, n in layout
    ]


class _Shard:
    """One TP rank's device: mapped engine caches, GPU row copy and slabs."""

    def __init__(
        self,
        caches: Dict[str, torch.Tensor],
        layout: Layout,
        row_bytes: int,
        budget: int,
    ):
        devices = {cache.device for cache in caches.values()}
        if len(devices) != 1:
            raise ValueError("A Xavier TP shard must use one device")
        if row_bytes <= 0 or sum(entry[4] for entry in layout) != row_bytes:
            raise ValueError("Xavier TP shard layout does not match its row size")
        self.device = next(iter(devices))
        self.caches = caches
        self.layout = list(layout)
        self.row_bytes = row_bytes
        self.capacity = max(0, budget) // row_bytes
        self.arena = (
            torch.empty(
                (self.capacity, row_bytes), dtype=torch.uint8, device=self.device
            )
            if self.capacity
            else None
        )
        self.slots: "OrderedDict[int, int]" = OrderedDict()
        self.free = list(range(self.capacity - 1, -1, -1))
        self.lock = asyncio.Lock()
        self.stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        # Filled once every shard has registered; slab sizes depend on them all.
        self.send: List[torch.Tensor] = []
        self.recv: List[torch.Tensor] = []
        # One thread per device: its CUDA state is set up once, here, rather
        # than on the first requests that happen to land on a pool thread.
        self.executor = ThreadPoolExecutor(1, thread_name_prefix="xavier-tp-gpu")
        self.executor.submit(self._call, lambda: None).result()

    def _call(self, function, *args):
        if self.stream is None:
            return function(*args)
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            result = function(*args)
        self.stream.synchronize()
        return result

    async def run(self, function, *args):
        """Run device work on this shard's stream; return once it completed."""
        future = asyncio.get_running_loop().run_in_executor(
            self.executor, self._call, function, *args
        )
        # A cancelled caller must not release buffers the device still uses.
        return await finish_before_cancel(future)

    def close(self):
        self.executor.shutdown(wait=True)
        self.slots.clear()
        self.free.clear()
        self.arena = None
        self.caches = {}
        self.send = []
        self.recv = []


@dataclass
class _Rendezvous:
    num_shards: int
    ready: asyncio.Future
    parts: Dict[int, Tuple[List[int], Dict[str, Dict[int, int]]]] = field(
        default_factory=dict
    )
    task: Optional[asyncio.Task] = None
    timer: Optional[asyncio.TimerHandle] = None


class ShardGPUTransfer:
    def __init__(self, actor, num_shards: int):
        if num_shards < 2:
            raise ValueError("Xavier shard GPU transfer requires TP>1")
        if not str(actor.address).startswith("nixl://"):
            raise RuntimeError("Xavier TP GPU transfer requires a NIXL actor pool")
        self.actor = actor
        self.num_shards = num_shards
        self.shards: Dict[int, _Shard] = {}
        # Slab classes: index 0 is the full slab, later ones are prefix views.
        self.class_rows: List[List[int]] = []
        self.recv_refs: List[List[Any]] = []
        self.send_lock = asyncio.Lock()
        self.recv_lock = asyncio.Lock()
        self.rendezvous: Dict[Tuple[str, int], _Rendezvous] = {}
        self.failed: "OrderedDict[Tuple[str, int], None]" = OrderedDict()
        self.small_streak: Dict[int, int] = {}
        self.tasks: set = set()
        self.closing = False
        self.prewarmed: Dict[int, bool] = {}
        self.metrics: Dict[str, Any] = dict(
            admitted_rows=0,
            evicted_rows=0,
            sent_rows=0,
            sent_copies=0,
            loaded_rows=0,
            missed_rows=0,
            load_rounds=0,
            wire_bytes=0,
            useful_bytes=0,
            rendezvous_timeouts=0,
            failed_loads=0,
            last_failure="",
            gather_ms=0.0,
            copy_ms=0.0,
            peer_ms=0.0,
            scatter_ms=0.0,
            load_ms=0.0,
        )
        # Per-load timings, for diagnosing the transfer path.
        self.recent: deque = deque(maxlen=64)

    # Registration ------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return (
            not self.closing
            and len(self.shards) == self.num_shards
            and bool(self.class_rows)
        )

    def register(self, shard, caches, layout, row_bytes, budget) -> bool:
        if self.closing:
            raise RuntimeError("Xavier shard GPU transfer is shutting down")
        if not 0 <= shard < self.num_shards:
            raise ValueError("Invalid Xavier TP shard")
        previous = self.shards.get(shard)
        if previous is not None:
            if previous.row_bytes != row_bytes or _layout_key(
                previous.layout
            ) != _layout_key(layout):
                raise ValueError("Xavier TP shard registration changed")
            return self.ready  # A worker retried after a lost reply.
        self.shards[shard] = _Shard(caches, layout, row_bytes, budget)
        if len(self.shards) == self.num_shards:
            self._allocate_slabs()
            logger.info(
                "Xavier TP GPU transfer ready: shards=%s, row_bytes=%s, "
                "cached_rows=%s, slab_rows=%s",
                self.num_shards,
                [self.shards[i].row_bytes for i in range(self.num_shards)],
                [self.shards[i].capacity for i in range(self.num_shards)],
                self.class_rows,
            )
            self._spawn(self._prewarm())
        return self.ready

    def _allocate_slabs(self):
        shards = [self.shards[i] for i in range(self.num_shards)]
        per_shard = REGISTRATION_BYTES // self.num_shards
        full = [max(1, min(MAX_SLAB_ROWS, per_shard // s.row_bytes)) for s in shards]
        small = [
            min(f, max(1, SMALL_SLAB_BYTES // s.row_bytes))
            for f, s in zip(full, shards)
        ]
        classes = [full] if small == full else [full, small]
        for i, shard in enumerate(shards):
            nbytes = full[i] * shard.row_bytes
            shard.send = [torch.zeros(nbytes, dtype=torch.uint8, device=shard.device)]
            shard.recv = [torch.zeros(nbytes, dtype=torch.uint8, device=shard.device)]
            # Persistent views keep their identity, and with it the registration.
            for counts in classes[1:]:
                count = counts[i] * shard.row_bytes
                shard.send.append(shard.send[0][:count])
                shard.recv.append(shard.recv[0][:count])
        self.class_rows = classes
        address = self.actor.address
        self.recv_refs = [
            [xo.buffer_ref(address, shard.recv[c]) for shard in shards]
            for c in range(len(classes))
        ]

    # Producer ----------------------------------------------------------------

    async def admit(self, shard_id: int, keys: List[int], rows: List[torch.Tensor]):
        """Copy rows the CPU store just staged into this shard's GPU copy."""
        shard = self.shards.get(shard_id)
        if self.closing or shard is None or shard.arena is None or not keys:
            return
        async with shard.lock:
            if shard.arena is None:
                return
            new: List[Tuple[int, int, torch.Tensor]] = []
            pending = set()
            for key, row in zip(keys, rows):
                if key in shard.slots:
                    shard.slots.move_to_end(key)
                elif key not in pending and len(new) < shard.capacity:
                    pending.add(key)
                    new.append((key, -1, row))
            if not new:
                return
            assigned = []
            for key, _, row in new:
                if shard.free:
                    slot = shard.free.pop()
                else:
                    _, slot = shard.slots.popitem(last=False)
                    self.metrics["evicted_rows"] += 1
                assigned.append((key, slot, row))
            index = torch.tensor([slot for _, slot, _ in assigned], dtype=torch.long)
            host = torch.stack(
                [row.view(torch.uint8).reshape(-1) for _, _, row in assigned]
            )
            if host.shape[1] != shard.row_bytes:
                shard.free.extend(slot for _, slot, _ in assigned)
                raise ValueError("Xavier TP shard row size changed")

            def upload():
                shard.arena.index_copy_(
                    0, index.to(shard.device), host.to(shard.device)
                )

            try:
                await shard.run(upload)
            except BaseException:
                shard.free.extend(slot for _, slot, _ in assigned)
                raise
            for key, slot, _ in assigned:
                shard.slots[key] = slot
            self.metrics["admitted_rows"] += len(assigned)

    def admit_in_background(self, shard_id: int, keys, rows):
        if self.closing or shard_id not in self.shards:
            return
        if self.shards[shard_id].arena is None:
            return

        async def admit():
            try:
                await self.admit(shard_id, list(keys), list(rows))
            except Exception:
                logger.warning("Failed to cache Xavier TP rows on GPU", exc_info=True)

        self._spawn(admit())

    async def _gather(self, shard: _Shard, keys: List[int], target: torch.Tensor):
        async with shard.lock:
            hits = [key for key in keys if key in shard.slots]
            if not hits:
                return []
            for key in hits:
                shard.slots.move_to_end(key)
            index = torch.tensor([shard.slots[key] for key in hits], dtype=torch.long)
            out = target[: len(hits) * shard.row_bytes].view(len(hits), shard.row_bytes)

            def gather():
                torch.index_select(shard.arena, 0, index.to(shard.device), out=out)

            await shard.run(gather)
            return hits

    async def send(self, keys: List[List[int]], refs: List[Any], slab_class: int):
        """Write this replica's cached rows into a peer's slabs, one copy."""
        if not self.ready:
            raise ShardGPUUnavailable("Xavier TP GPU transfer is not ready")
        if len(keys) != self.num_shards or len(refs) != self.num_shards:
            raise ValueError("Xavier TP GPU transfer shard count differs")
        if not 0 <= slab_class < len(self.class_rows):
            raise ValueError("Xavier TP GPU slab classes differ")
        rows = self.class_rows[slab_class]
        if any(len(k) > r for k, r in zip(keys, rows)):
            raise ValueError("Xavier TP GPU batch exceeds the slab")
        async with self.send_lock:
            shards = [self.shards[i] for i in range(self.num_shards)]
            started = time.monotonic()
            hits = await asyncio.gather(
                *(
                    self._gather(shard, shard_keys, shard.send[slab_class])
                    for shard, shard_keys in zip(shards, keys)
                )
            )
            gathered = time.monotonic()
            self.metrics["gather_ms"] += (gathered - started) * 1000
            if any(hits):
                buffers = [shard.send[slab_class] for shard in shards]
                await xo.copy_to(buffers, list(refs))
                self.metrics["copy_ms"] += (time.monotonic() - gathered) * 1000
                self.metrics["sent_copies"] += 1
                self.metrics["sent_rows"] += sum(len(h) for h in hits)
                self.metrics["wire_bytes"] += sum(b.numel() for b in buffers)
                self.metrics["useful_bytes"] += sum(
                    len(h) * s.row_bytes for h, s in zip(hits, shards)
                )
            return [list(h) for h in hits]

    async def warm(self, refs: List[Any], slab_class: int):
        """Copy the slabs once so that the first request finds a warm channel."""
        if not self.ready:
            raise ShardGPUUnavailable("Xavier TP GPU transfer is not ready")
        async with self.send_lock:
            await xo.copy_to(
                [self.shards[i].send[slab_class] for i in range(self.num_shards)],
                list(refs),
            )

    # Consumer ----------------------------------------------------------------

    async def load(
        self,
        shard: int,
        lease: str,
        from_rank: int,
        keys: List[int],
        mappings: Dict[str, Dict[int, int]],
    ) -> List[int]:
        """Load this shard's rows of one lease; return the keys now in place.

        The ranks of a replica call this for the same lease in the same engine
        step. The first call waits for the others, then one copy per round
        carries every shard's rows. Keys that are not returned were not
        written and must be read from the CPU snapshot store.
        """
        if not self.ready or not keys:
            return []
        key = (lease, from_rank)
        if key in self.failed:
            return []
        entry = self.rendezvous.get(key)
        if entry is None:
            loop = asyncio.get_running_loop()
            entry = _Rendezvous(self.num_shards, loop.create_future())
            entry.timer = loop.call_later(RENDEZVOUS_SECONDS, self._expire, key)
            self.rendezvous[key] = entry
        if entry.task is not None or shard in entry.parts or entry.ready.done():
            return []
        entry.parts[shard] = (list(keys), mappings)
        if len(entry.parts) == entry.num_shards:
            if entry.timer is not None:
                entry.timer.cancel()
            entry.task = self._spawn(self._transfer(from_rank, entry.parts))
            entry.ready.set_result(True)
            entry.task.add_done_callback(lambda _: self.rendezvous.pop(key, None))
        try:
            started = await asyncio.shield(entry.ready)
        except asyncio.CancelledError:
            if entry.task is None:
                self._fail(key, "cancelled before every TP rank arrived")
            else:
                # The copy may still write this rank's caches: wait for it.
                await finish_before_cancel(entry.task)
            raise
        if not started or entry.task is None:
            return []
        loaded = await finish_before_cancel(entry.task)
        return loaded.get(shard, [])

    def _expire(self, key):
        entry = self.rendezvous.get(key)
        if entry is not None and entry.task is None:
            self.metrics["rendezvous_timeouts"] += 1
            self._fail(key, "TP ranks did not all request the lease")

    def _fail(self, key, reason):
        entry = self.rendezvous.pop(key, None)
        self.failed[key] = None
        while len(self.failed) > 4096:
            self.failed.popitem(last=False)
        if entry is not None:
            if entry.timer is not None:
                entry.timer.cancel()
            if not entry.ready.done():
                entry.ready.set_result(False)
        logger.debug("Xavier TP GPU load of %s skipped: %s", key, reason)

    def _select_class(self, from_rank: int, counts: List[int]) -> int:
        fitting = [
            c
            for c, rows in enumerate(self.class_rows)
            if all(n <= r for n, r in zip(counts, rows))
        ]
        candidate = fitting[-1]
        if candidate == 0:
            self.small_streak.pop(from_rank, None)
            return 0
        # xoscar keeps one registration per channel. Switch to the small view
        # only for repeated small copies, so mixed traffic does not alternate.
        streak = min(2, self.small_streak.get(from_rank, 0) + 1)
        self.small_streak[from_rank] = streak
        return candidate if streak == 2 else 0

    async def _transfer(self, from_rank, parts) -> Dict[int, List[int]]:
        started = time.monotonic()
        loaded: Dict[int, List[int]] = {shard: [] for shard in parts}
        peer_ms = scatter_ms = 0.0
        classes: List[int] = []
        try:
            peer = await xo.actor_ref(
                address=self.actor._world_addresses[from_rank],
                uid=f"{self.actor.default_uid()}-{from_rank}",
            )
            full = self.class_rows[0]
            rounds = max(
                (len(parts[s][0]) + full[s] - 1) // full[s]
                for s in range(self.num_shards)
            )
            async with self.recv_lock:
                for r in range(rounds):
                    chunk = [
                        parts[s][0][r * full[s] : (r + 1) * full[s]]
                        for s in range(self.num_shards)
                    ]
                    slab_class = self._select_class(from_rank, [len(c) for c in chunk])
                    classes.append(slab_class)
                    sent = time.monotonic()
                    hits = await peer.send_gpu_shards_v1(
                        chunk, self.recv_refs[slab_class], slab_class
                    )
                    scattered = time.monotonic()
                    peer_ms += (scattered - sent) * 1000
                    if len(hits) != self.num_shards or any(
                        not set(h) <= set(c) for h, c in zip(hits, chunk)
                    ):
                        raise RuntimeError("Xavier TP GPU peer returned other rows")
                    await asyncio.gather(
                        *(
                            self._scatter(
                                self.shards[s], hits[s], parts[s][1], slab_class
                            )
                            for s in range(self.num_shards)
                            if hits[s]
                        )
                    )
                    scatter_ms += (time.monotonic() - scattered) * 1000
                    for s in range(self.num_shards):
                        loaded[s].extend(hits[s])
                    self.metrics["load_rounds"] += 1
        except Exception as error:
            # Rows written before the failure are complete: each round's
            # scatter finished before the next started. Report only those.
            self.metrics["failed_loads"] += 1
            self.metrics["last_failure"] = repr(error)[:300]
            logger.warning(
                "Xavier TP GPU load from rank %s failed; reading from CPU",
                from_rank,
                exc_info=True,
            )
        total = sum(len(parts[s][0]) for s in parts)
        done = sum(len(v) for v in loaded.values())
        elapsed = (time.monotonic() - started) * 1000
        self.metrics["loaded_rows"] += done
        self.metrics["missed_rows"] += total - done
        self.metrics["peer_ms"] += peer_ms
        self.metrics["scatter_ms"] += scatter_ms
        self.metrics["load_ms"] += elapsed
        self.recent.append(
            dict(
                from_rank=from_rank,
                rows=done,
                requested=total,
                classes=classes,
                peer_ms=peer_ms,
                scatter_ms=scatter_ms,
                ms=elapsed,
            )
        )
        logger.debug(
            "Xavier TP GPU load from rank %s: %s/%s rows in %.2f ms",
            from_rank,
            done,
            total,
            elapsed,
        )
        return loaded

    async def _scatter(self, shard: _Shard, keys, mappings, slab_class):
        payload = shard.recv[slab_class][: len(keys) * shard.row_bytes].view(
            len(keys), shard.row_bytes
        )
        writes = []
        for name, shape, dtype, start, nbytes in shard.layout:
            mapping = mappings.get(name, {})
            selected = [i for i, key in enumerate(keys) if key in mapping]
            if not selected:
                continue
            writes.append(
                (
                    shard.caches[name],
                    torch.tensor(selected, dtype=torch.long),
                    torch.tensor(
                        [mapping[keys[i]] for i in selected], dtype=torch.long
                    ),
                    shape,
                    dtype,
                    start,
                    nbytes,
                )
            )

        def scatter():
            for cache, rows, destinations, shape, dtype, start, nbytes in writes:
                blocks = payload[:, start : start + nbytes]
                if len(rows) != payload.shape[0]:
                    blocks = blocks[rows.to(shard.device)]
                cache[destinations.to(shard.device)] = (
                    blocks.contiguous().view(dtype).view(len(rows), *shape)
                )

        await shard.run(scatter)

    # Lifecycle ---------------------------------------------------------------

    def _spawn(self, coroutine) -> asyncio.Task:
        task = asyncio.ensure_future(coroutine)
        self.tasks.add(task)

        def done(future):
            self.tasks.discard(future)
            if not future.cancelled():
                future.exception()

        task.add_done_callback(done)
        return task

    async def _prewarm(self):
        """Open the channel from every peer before the first request needs it."""
        own = getattr(self.actor, "_rank", None)
        peers = [
            rank
            for rank in range(len(self.actor._world_addresses))
            if rank != own and rank != 0
        ]
        for _ in range(PREWARM_ATTEMPTS):
            if self.closing:
                return
            for rank in peers:
                if self.prewarmed.get(rank):
                    continue
                try:
                    peer = await xo.actor_ref(
                        address=self.actor._world_addresses[rank],
                        uid=f"{self.actor.default_uid()}-{rank}",
                    )
                    started = time.monotonic()
                    async with self.recv_lock:
                        await peer.warm_gpu_shards_v1(self.recv_refs[0], 0)
                    self.prewarmed[rank] = True
                    logger.info(
                        "Xavier TP GPU channel from rank %s warmed in %.1f ms",
                        rank,
                        (time.monotonic() - started) * 1000,
                    )
                except Exception:
                    logger.debug(
                        "Xavier TP GPU peer %s is not ready yet", rank, exc_info=True
                    )
            if all(self.prewarmed.get(rank) for rank in peers):
                return
            await asyncio.sleep(PREWARM_INTERVAL_SECONDS)

    def stats(self) -> dict:
        return dict(
            self.metrics,
            ready=self.ready,
            registered=sorted(self.shards),
            cached_rows={i: len(s.slots) for i, s in self.shards.items()},
            capacity_rows={i: s.capacity for i, s in self.shards.items()},
            slab_rows=self.class_rows,
            prewarmed=dict(self.prewarmed),
            pending_rendezvous=len(self.rendezvous),
            recent=list(self.recent),
        )

    async def close(self):
        self.closing = True
        for key in list(self.rendezvous):
            self._fail(key, "shutting down")
        if self.tasks:
            await asyncio.gather(*list(self.tasks), return_exceptions=True)
        logger.info("Xavier TP GPU transfer stats: %s", self.stats())
        for shard in self.shards.values():
            if shard.stream is not None:
                await asyncio.to_thread(shard.stream.synchronize)
            shard.close()
        self.recv_refs = []
        self.class_rows = []


class ShardGPUTransferMixin:
    """TransferActor endpoints; the runtime exists once a TP rank registers."""

    def register_gpu_shard_v1(
        self, shard, num_shards, descriptors, layout, row_bytes, budget
    ):
        from torch.multiprocessing.reductions import rebuild_cuda_tensor

        runtime = getattr(self, "_shard_gpu_transfer", None)
        if runtime is None:
            runtime = self._shard_gpu_transfer = ShardGPUTransfer(self, num_shards)
        elif runtime.num_shards != num_shards:
            raise ValueError("Xavier TP GPU transfer shard count changed")
        if shard in runtime.shards:
            return runtime.register(
                shard, runtime.shards[shard].caches, layout, row_bytes, budget
            )
        caches = {
            name: rebuild_cuda_tensor(*descriptor)
            for name, descriptor in descriptors.items()
        }
        return runtime.register(shard, caches, layout, row_bytes, budget)

    def _ready_shard_gpu_transfer(self) -> ShardGPUTransfer:
        runtime = getattr(self, "_shard_gpu_transfer", None)
        if runtime is None or not runtime.ready:
            raise ShardGPUUnavailable("Xavier TP GPU transfer is not ready")
        return runtime

    async def send_gpu_shards_v1(self, keys, refs, slab_class):
        return await self._ready_shard_gpu_transfer().send(keys, refs, slab_class)

    async def warm_gpu_shards_v1(self, refs, slab_class):
        return await self._ready_shard_gpu_transfer().warm(refs, slab_class)

    async def load_gpu_shard_v1(self, shard, lease, from_rank, keys, mappings):
        runtime = getattr(self, "_shard_gpu_transfer", None)
        if runtime is None:
            return []
        return await runtime.load(shard, lease, from_rank, keys, mappings)

    def get_gpu_shard_stats_v1(self):
        runtime = getattr(self, "_shard_gpu_transfer", None)
        return runtime.stats() if runtime is not None else {}

    async def close_gpu_shards_v1(self):
        runtime = getattr(self, "_shard_gpu_transfer", None)
        if runtime is not None:
            await runtime.close()
