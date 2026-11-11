# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""TRT-LLM connector for prefix sharing and Xavier P/D handoffs.

Reads acquire owned CPU snapshots before reporting a hit. Eviction between
lookup and GPU load therefore cannot turn an admitted prefix into missing KV.
P/D loads suspend their request until GPU writes finish, allowing unrelated
decode work to continue. Shared CPU prefix loads remain synchronous. Chunking,
overlap scheduling, and eviction/replay are disabled.
"""

import asyncio
import collections
import ctypes
import json
import logging
import os
import queue
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field

import torch
import xoscar as xo

try:
    from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
        KvCacheConnectorScheduler,
        KvCacheConnectorWorker,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {
        "tensorrt_llm",
        "tensorrt_llm._torch",
        "tensorrt_llm._torch.pyexecutor",
        "tensorrt_llm._torch.pyexecutor.connectors",
        "tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector",
    }:
        raise
    from tensorrt_llm._torch.pyexecutor.kv_cache_connector import (
        KvCacheConnectorScheduler,
        KvCacheConnectorWorker,
    )

from ..xavier.actor_loop import acquire_actor_loop, release_actor_loop
from ..xavier.contract import KVCacheContract, build_prefix_keys, fingerprint_metadata
from .pd import request_state
from .xavier import CONFIG_ENV, WORKER_CONFIG_ENV

logger = logging.getLogger(__name__)
_clients: weakref.WeakSet = weakref.WeakSet()
_publishers: weakref.WeakSet = weakref.WeakSet()
# One store RPC carries at most this many page keys.
_KEY_BATCH = 128
# Host bytes each rank may hold for publications still being uploaded.
_PUBLISH_BUDGET = 2 * 1024**3
# GPU bytes each rank may hold for pages still being copied to shared memory.
_STAGING_BUDGET = 512 * 1024**2
# Longest a rank trusts that a page it published is still stored. Mapped
# shared pages expire hints at the first eviction; this bounds the rest.
_HINT_TTL = 60.0


def shard_key(key: str, rank: int, tensor_parallel_size: int) -> str:
    """Name one tensor-parallel rank's share of a page; TP1 keys are unchanged."""
    return key if tensor_parallel_size == 1 else f"{key}@tp{rank}"


def _local_rank() -> int:
    try:
        import tensorrt_llm

        return int(tensorrt_llm.mpi_rank())
    except Exception:
        return 0


def _page_lock(pages: torch.Tensor) -> None:
    """Let the GPU copy shared pages with DMA; copies work unlocked as well."""
    error = torch.cuda.cudart().cudaHostRegister(
        pages.data_ptr(), pages.numel(), 1  # cudaHostRegisterPortable
    )
    if int(error) != 0:
        logger.warning("Could not page-lock Xavier shared pages: error %d", error)
        # The runtime keeps a failed registration as its last error, which the
        # next unrelated CUDA call would report. Clear it.
        try:
            ctypes.CDLL(None).cudaGetLastError()
        except (AttributeError, OSError):
            pass


def _load_config() -> dict:
    # Tensor-parallel ranks run in MPI workers that receive only TRTLLM* names.
    raw = os.environ.get(WORKER_CONFIG_ENV) or os.environ[CONFIG_ENV]
    return json.loads(raw)


class XavierClient:
    def __init__(self, config=None):
        self.config = _load_config() if config is None else dict(config)
        self.contract = KVCacheContract.from_dict(self.config["contract"])
        self.pd_gpu = self.config.get("role") in ("prefill", "decode")
        self.tensor_parallel_size = int(self.config.get("tensor_parallel_size", 1))
        # Shared-memory pages, when the cache actor runs on this host.
        self.shared = None
        self._pages = None
        self._evictions = None
        # Identifies this client's pending shared-memory writes in the store.
        self.writer = uuid.uuid4().hex
        self._writes = 0
        self._writes_lock = threading.Lock()
        self._ref = None
        self._closed = False
        ready = threading.Event()

        def run():
            self._loop = acquire_actor_loop()
            ready.set()
            try:
                self._loop.run_forever()
            finally:
                release_actor_loop(self._loop)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        ready.wait()
        _clients.add(self)
        try:
            if self.pd_gpu:
                self.namespace = fingerprint_metadata(
                    dict(
                        contract=self.contract.to_dict(),
                        engine="trt-llm",
                        version=self.config.get("engine_version", "1.2.1"),
                        layout="trt-block-layer-kv-packed-gpu-v1",
                    )
                )
                if self.config.get("heterogeneous"):
                    self.namespace = self.contract.fingerprint
                    self.rpc("configure", self.namespace, self.contract.to_dict())
                else:
                    self.rpc("configure", self.namespace)
                self.capacity_bytes = 0
            elif self.config.get("shared_prefix"):
                self.namespace = self.rpc("configure", self.contract.to_dict())[
                    "namespace"
                ]
                self.capacity_bytes = self.rpc("get_stats")["capacity_bytes"]
            else:
                storage_format = {
                    "engine": "trt-llm",
                    "version": self.config.get("engine_version", "1.2.1"),
                    "layout": "layer_first",
                    "physical_layout": "block-layer-kv-packed",
                    "hash_format": "xavier-prefix-v1",
                }
                if self.tensor_parallel_size > 1:
                    # The contract describes one rank's KV heads. Each page is
                    # stored once per rank, so differing TP layouts never mix.
                    storage_format.update(
                        tensor_parallel_size=self.tensor_parallel_size,
                        shard="kv-heads",
                    )
                from .compat import supports_tensor_parallel

                if supports_tensor_parallel(storage_format["version"]):
                    storage_format["shared_memory"] = True
                self.namespace = self.rpc(
                    "configure", self.contract.to_dict(), storage_format
                )
                self.capacity_bytes = self.rpc("get_stats")["capacity_bytes"]
                if storage_format.get("shared_memory"):
                    self.shared = self.rpc("shared_pages", self.namespace)
        except BaseException:
            self.close()
            raise

    async def directory(self):
        if self._ref is None:
            self._ref = await xo.actor_ref(
                address=self.config["address"], uid=self.config["uid"]
            )
        return self._ref

    def submit(self, coroutine):
        if self._closed:
            raise RuntimeError("TRT-LLM Xavier client is closed")

        async def invoke():
            return await coroutine

        return asyncio.run_coroutine_threadsafe(invoke(), self._loop)

    def call(self, coroutine):
        # GPU operations must drain before their caller can return engine slots.
        # Do not cancel a transfer future when a wall-clock timeout expires.
        return self.submit(coroutine).result()

    def rpc(self, method, *args, **kwargs):
        if self._closed:
            raise RuntimeError("TRT-LLM Xavier client is closed")

        async def invoke():
            return await getattr(await self.directory(), method)(*args, **kwargs)

        timeout = max(10, self.config.get("read_timeout", 0) + 1)
        future = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(invoke(), timeout=timeout), self._loop
        )
        try:
            return future.result(timeout=timeout + 1)
        except BaseException:
            future.cancel()
            raise

    def read_prefix(self, keys):
        pages = []
        page_bytes = self.contract.layer_nbytes * self.contract.num_layers
        batch_size = (
            max(1, min(128, 32 * 1024**2 // page_bytes))
            if self.config.get("shared_prefix")
            else 128
        )
        try:
            for offset in range(0, len(keys), batch_size):
                batch_keys = keys[offset : offset + batch_size]
                batch = self.rpc("get", self.namespace, batch_keys)
                for page in batch:
                    if page is None:
                        return pages
                    expected = self.contract.layer_nbytes * self.contract.num_layers
                    if self.config.get("shared_prefix"):
                        if type(page) is not bytes or len(page) != expected:
                            raise ValueError("Invalid shared-prefix KV page")
                        pages.append(page)
                        continue
                    if (
                        page.dtype != torch.uint8
                        or page.device.type != "cpu"
                        or page.numel() != expected
                    ):
                        raise ValueError("TRT-LLM Xavier returned invalid KV page data")
                    pages.append(page.clone())
                # The bytes store ends a response at the first missing page.
                if len(batch) < len(batch_keys):
                    break
        except Exception:
            # Nothing has been admitted yet: remote loss can safely be a miss.
            logger.warning(
                "TRT-LLM Xavier lookup failed; recomputing prefix", exc_info=True
            )
            return []
        return pages

    def stored_prefix(self, keys):
        """Count leading keys whose pages are all present in the store."""
        count = 0
        for offset in range(0, len(keys), _KEY_BATCH):
            batch = keys[offset : offset + _KEY_BATCH]
            found = self.rpc("exists", self.namespace, batch)
            count += found
            if found < len(batch):
                break
        return count

    def pages(self):
        """Map this deployment's shared pages, or return None to use RPC."""
        if self.shared is None:
            return None
        if self._pages is None:
            from ..xavier.backends.torch.shared_pages import (
                evictions_of,
                map_file,
                pages_of,
            )

            info = self.shared
            try:
                mapped = map_file(
                    info["path"],
                    bytes.fromhex(info["nonce"]),
                    info["page_bytes"],
                    info["slots"],
                )
            except (OSError, ValueError):
                # The cache actor runs on another host or in another container.
                logger.info("Xavier shared pages are not mapped; using actor RPC")
                self.shared = None
                return None
            pages = pages_of(mapped, info["page_bytes"], info["slots"])
            if torch.cuda.is_available() and info["path"].startswith("/dev/shm/"):
                _page_lock(pages)
            self._evictions = evictions_of(mapped)
            self._pages = pages
        return self._pages

    def evictions(self):
        """The store's eviction count, read from shared memory, or None."""
        return None if self.pages() is None else int(self._evictions[0])

    def writing(self, delta):
        with self._writes_lock:
            self._writes += delta

    def pin_sharded_prefix(self, keys):
        """Pin the leading pages present on every rank.

        Returns (pages, ticket, slots); slots lists each page's shared-memory
        slot per rank, or is None when ranks read through actor RPC.
        """
        tp = self.tensor_parallel_size
        keys = keys[: 4096 // tp]
        sharded = [shard_key(key, rank, tp) for key in keys for rank in range(tp)]
        try:
            pages = self.stored_prefix(sharded) // tp
            while pages:
                # Pages can be evicted between the lookup and the pin.
                if self.shared is not None:
                    pinned = self.rpc(
                        "reserve_shared", self.namespace, sharded[: pages * tp]
                    )
                    if pinned is not None:
                        slots = pinned["slots"]
                        return (
                            pages,
                            pinned["ticket"],
                            [slots[i * tp : (i + 1) * tp] for i in range(pages)],
                        )
                else:
                    ticket = self.rpc(
                        "reserve_handoff", self.namespace, sharded[: pages * tp]
                    )
                    if ticket is not None:
                        return pages, ticket, None
                pages = self.stored_prefix(sharded[: pages * tp]) // tp
        except Exception:
            # Nothing has been admitted yet: remote loss can safely be a miss.
            logger.warning(
                "TRT-LLM Xavier lookup failed; recomputing prefix", exc_info=True
            )
        return 0, None, None

    def release(self, ticket):
        try:
            self.rpc("release_handoff", ticket)
        except Exception:
            # The store also expires abandoned pins.
            logger.warning("Could not release TRT-LLM Xavier pages", exc_info=True)

    def read_pinned(self, keys):
        """Read pages that the scheduler pinned for this step."""
        pages = []
        expected = self.contract.layer_nbytes * self.contract.num_layers
        for offset in range(0, len(keys), _KEY_BATCH):
            batch = self.rpc("get", self.namespace, keys[offset : offset + _KEY_BATCH])
            for page in batch:
                if (
                    page is None
                    or page.dtype != torch.uint8
                    or page.device.type != "cpu"
                    or page.numel() != expected
                ):
                    raise RuntimeError("A pinned TRT-LLM Xavier page is unavailable")
                pages.append(page)
        return pages

    def close(self):
        if not self._closed:
            if self.shared is not None and self._writes == 0:
                # No copy into this client's slots is pending: return the
                # slots of writes that never completed.
                try:
                    self.rpc("release_writer", self.namespace, self.writer)
                except Exception:
                    logger.warning(
                        "Could not release Xavier write slots", exc_info=True
                    )
            self._closed = True
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=6)
            if self._thread.is_alive():
                logger.warning("Timed out stopping TRT-LLM Xavier RPC thread")


def close_clients():
    from .gpu import close_service

    close_service()
    for publisher in list(_publishers):
        publisher.close()
    for client in list(_clients):
        client.close()


class _StoredHints:
    """Pages this rank knows are stored, so it need not publish them again.

    A hint is added only after the store acknowledged the page: a commit, an
    allocation reporting it stored, a successful upload, or an import of a
    pinned page. It records the store's eviction count read while the page was
    known to be stored, and is valid only while that count is unchanged and
    for at most ``ttl`` seconds. Any eviction therefore makes the page
    eligible for publication again.
    """

    def __init__(self, limit, ttl=_HINT_TTL):
        self.limit = max(1, limit)
        self.ttl = ttl
        self._lock = threading.Lock()
        self._keys: collections.OrderedDict = collections.OrderedDict()
        # Latest eviction count reported through actor RPC.
        self.reported = None

    def add(self, keys, evictions):
        now = time.monotonic()
        with self._lock:
            for key in keys:
                self._keys[key] = (evictions, now)
                self._keys.move_to_end(key)
            while len(self._keys) > self.limit:
                self._keys.popitem(last=False)

    def report(self, evictions):
        with self._lock:
            if self.reported is None or evictions > self.reported:
                self.reported = evictions

    def unknown(self, keys, evictions):
        """Return the keys without a valid hint."""
        expired = time.monotonic() - self.ttl
        with self._lock:
            missing = []
            for key in keys:
                hint = self._keys.get(key)
                if hint is None or hint[0] != evictions or hint[1] < expired:
                    self._keys.pop(key, None)
                    missing.append(key)
            return missing


class _Publisher:
    """Upload staged pages to the shared store off the engine thread.

    The engine thread copies pages into host memory before TRT can reuse their
    slots; only the store RPC runs here. Publication is best effort: pages
    beyond the in-flight budget are skipped and later requests recompute them.
    """

    def __init__(self, client, budget=_PUBLISH_BUDGET, hints=None):
        self.client = client
        self.budget = budget
        self.hints = hints
        self.in_flight = 0
        self._idle = threading.Condition()
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, pool, saves):
        """Stage leading pages within the budget; return how many were staged."""
        page_bytes = pool[0].numel() * pool.element_size()
        with self._idle:
            count = min(len(saves), (self.budget - self.in_flight) // page_bytes)
            if count <= 0:
                return 0
            self.in_flight += count * page_bytes
        try:
            pin = pool.is_cuda
            pages = []
            for _, slot in saves[:count]:
                page = torch.empty(page_bytes, dtype=torch.uint8, pin_memory=pin)
                page.copy_(pool[slot].view(torch.uint8).reshape(-1), non_blocking=pin)
                pages.append(page)
            if pin:
                torch.cuda.current_stream(pool.device).synchronize()
        except BaseException:
            self._done(count * page_bytes)
            raise
        self._queue.put(([key for key, _ in saves[:count]], pages))
        return count

    def _done(self, nbytes):
        with self._idle:
            self.in_flight -= nbytes
            self._idle.notify_all()

    def _put(self, keys, pages):
        for offset in range(0, len(keys), _KEY_BATCH):
            batch = keys[offset : offset + _KEY_BATCH]
            evictions = self.client.rpc("eviction_count", self.client.namespace)
            # Another replica may have stored this prefix meanwhile.
            stored = self.client.rpc("exists", self.client.namespace, batch)
            acknowledged = batch[:stored]
            if stored < len(batch):
                results = self.client.rpc(
                    "put",
                    self.client.namespace,
                    batch[stored:],
                    pages[offset + stored : offset + len(batch)],
                )
                acknowledged += [k for k, ok in zip(batch[stored:], results) if ok]
            if self.hints is not None:
                self.hints.report(evictions)
                self.hints.add(acknowledged, evictions)

    def _run(self):
        while (item := self._queue.get()) is not None:
            keys, pages = item
            try:
                self._put(keys, pages)
            except Exception:
                # Publishing is optional; failure leaves subsequent requests cold.
                logger.warning("TRT-LLM Xavier publication failed", exc_info=True)
            finally:
                self._done(sum(page.numel() for page in pages))

    def flush(self, timeout=None):
        with self._idle:
            return self._idle.wait_for(lambda: self.in_flight == 0, timeout)

    def close(self):
        self._queue.put(None)
        self._thread.join(timeout=5)


def _store_shared(client, keys, copy, synchronize, hints=None):
    """Allocate shared-memory slots, copy page i into its slot, then commit.

    Returns the allocation: a slot, -1 for pages already stored, or None when
    the store has no room. Slots return to the store only by commit, by an
    abort after every copy into them drained, or by the writer's release.
    """
    pages = client.pages()
    evictions = client.evictions()
    client.writing(1)
    try:
        slots = []
        for offset in range(0, len(keys), 4096):
            slots += client.rpc(
                "allocate_shared",
                client.namespace,
                keys[offset : offset + 4096],
                client.writer,
            )
        written = [
            (i, slot) for i, slot in enumerate(slots) if slot is not None and slot >= 0
        ]
        try:
            for index, slot in written:
                copy(index, pages[slot])
            synchronize()
        except BaseException:
            try:
                synchronize()
            except BaseException:
                # A copy may still be in flight: keep the slots reserved.
                logger.warning(
                    "Could not drain Xavier shared-memory writes", exc_info=True
                )
            else:
                _commit_shared(client, keys, written, False)
            raise
        _commit_shared(client, keys, written, True)
        if hints is not None:
            hints.add(
                [key for key, slot in zip(keys, slots) if slot is not None], evictions
            )
        return slots
    finally:
        client.writing(-1)


def _commit_shared(client, keys, written, ok):
    if not written:
        return
    try:
        client.rpc(
            "commit_shared",
            client.namespace,
            [keys[index] for index, _ in written],
            [slot for _, slot in written],
            ok,
        )
    except Exception:
        # The slots stay with this writer until it releases them on shutdown.
        logger.warning("TRT-LLM Xavier publication failed", exc_info=True)
        if ok:
            raise


class _StagedWriter:
    """Publish pages without waiting for host copies on the engine thread.

    The engine stream copies new pages into GPU staging rows, ordered before
    any later kernel can reuse their pool slots. A background thread copies
    the rows into shared memory on its own stream and commits them. When the
    rows are exhausted, callers copy the remaining pages synchronously.
    """

    def __init__(self, client, pool, budget=_STAGING_BUDGET, hints=None):
        page_bytes = pool[0].numel() * pool.element_size()
        rows = budget // page_bytes
        if rows < 1:
            raise ValueError("Xavier staging budget is smaller than one page")
        self.client = client
        self.hints = hints
        self.staging = torch.empty(
            (rows, page_bytes), dtype=torch.uint8, device=pool.device
        )
        self.free = list(range(rows))
        self._idle = threading.Condition()
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, pool, stream, saves):
        with self._idle:
            count = min(len(saves), len(self.free))
            rows = [self.free.pop() for _ in range(count)]
        if not count:
            return 0
        with torch.cuda.stream(stream):
            for row, (_, slot) in zip(rows, saves):
                self.staging[row].copy_(
                    pool[slot].view(torch.uint8).reshape(-1), non_blocking=True
                )
            event = torch.cuda.Event()
            event.record(stream)
        self._queue.put(([key for key, _ in saves[:count]], rows, event))
        return count

    def _run(self):
        torch.cuda.set_device(self.staging.device)
        side = torch.cuda.Stream(self.staging.device)
        while (item := self._queue.get()) is not None:
            keys, rows, event = item
            try:
                with torch.cuda.stream(side):
                    side.wait_event(event)

                    def copy(index, target):
                        target.copy_(self.staging[rows[index]], non_blocking=True)

                    _store_shared(self.client, keys, copy, side.synchronize, self.hints)
            except Exception:
                # Publishing is optional; failure leaves subsequent requests cold.
                logger.warning("TRT-LLM Xavier publication failed", exc_info=True)
            finally:
                side.synchronize()
                with self._idle:
                    self.free.extend(rows)
                    self._idle.notify_all()

    def flush(self, timeout=None):
        total = len(self.staging)
        with self._idle:
            return self._idle.wait_for(lambda: len(self.free) == total, timeout)

    def close(self):
        self._queue.put(None)
        self._thread.join(timeout=5)


@dataclass
class XavierMetadata:
    loads: list = field(default_factory=list)
    saves: list = field(default_factory=list)
    pd_loads: list = field(default_factory=list)
    pd_saves: list = field(default_factory=list)
    # (slot, page key, shared-memory slot per rank or None): every rank
    # copies its own share of each pinned page.
    shard_loads: list = field(default_factory=list)


class XavierConnectorScheduler(KvCacheConnectorScheduler):
    def __init__(self, llm_args):
        super().__init__(llm_args)
        self.client = XavierClient()
        self.contract = self.client.contract
        self._history = (
            XavierClient(
                {
                    **self.client.config["prefix_cache"],
                    "contract": self.contract.to_dict(),
                }
            )
            if getattr(self.client, "config", {}).get("role") == "prefill"
            and self.client.config.get("prefix_cache")
            else None
        )
        self._pending = {}
        self._allocations = {}
        self._pd_requests = {}
        self._pd_sources = {}
        self._pd_destinations = {}
        # Pins for sharded loads, released once every rank has copied them.
        self._loading_pins = {}

    def _forget_pending(self, request_id):
        pending = self._pending.pop(request_id, None)
        if pending is not None and len(pending) == 4:
            self.client.release(pending[2])

    def _keys(self, tokens):
        if self.client.config.get("shared_prefix") or self._history is not None:
            from ..xavier.prefix import prefix_keys

            return prefix_keys(tokens, self.contract.block_size)
        return [key.digest for key in build_prefix_keys(self.contract, tokens)]

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        request_id = request.request_id
        self._forget_pending(request_id)
        state = request_state(request)
        if state is not None:
            if tuple(request.get_tokens(0)) != state.tokens:
                raise ValueError("TRT-LLM P/D scheduler prompt differs")
            self._pd_requests[request_id] = state
            if state.role == "prefill":
                self._pd_sources[request_id] = state
                state.engine_request_id = request_id
                if self._history is not None:
                    return self._match_prefix(
                        request, num_computed_tokens, self._history
                    )
                return 0, False
            if num_computed_tokens:
                raise ValueError("TRT-LLM P/D requires an isolated destination prefix")
            state.engine_request_id = request_id
            self._pd_destinations[request_id] = state
            count = max(len(state.tokens) - 1, 0)
            return count, bool(count)
        if self.client.pd_gpu:
            return 0, False  # TRT initialization requests do not share history.
        return self._match_prefix(request, num_computed_tokens, self.client)

    def _match_prefix(self, request, num_computed_tokens, client):
        block = self.contract.block_size
        if num_computed_tokens % block:
            return 0, False
        tokens = list(request.get_tokens(0))
        # Leave at least the final prompt token for local logits computation.
        keys = self._keys(tokens[: max(len(tokens) - 1, 0)])
        start = num_computed_tokens // block
        page_bytes = self.contract.layer_nbytes * self.contract.num_layers
        if client.tensor_parallel_size > 1 or client.shared is not None:
            # Each rank reads its own shards: pin pages here, copy them later.
            count, ticket, slots = client.pin_sharded_prefix(keys[start:])
            if count:
                self._pending[request.request_id] = (
                    start,
                    keys[start : start + count],
                    ticket,
                    slots,
                )
            return count * block, False
        pending_bytes = sum(
            len(pending[1]) * page_bytes for pending in self._pending.values()
        )
        available = max(client.capacity_bytes - pending_bytes, 0) // page_bytes
        pages = client.read_prefix(keys[start : start + available])
        if pages:
            self._pending[request.request_id] = (start, pages)
        return len(pages) * block, False

    def update_state_after_alloc(self, request, block_ids):
        state = self._pd_requests.get(request.request_id)
        self._allocations[request.request_id] = (
            (
                []
                if state is not None
                and not (state.role == "prefill" and self._history is not None)
                else self._keys(list(request.get_tokens(0)))
            ),
            list(block_ids),
        )

    def build_connector_meta(self, scheduler_output):
        # Ranks copy pinned pages before the forward pass whose collectives
        # complete the previous step, so those pins are no longer needed.
        for ticket in self._loading_pins.values():
            self.client.release(ticket)
        self._loading_pins.clear()
        metadata = XavierMetadata()
        # TRT omits newly suspended async loads from new_requests. Allocated
        # decode requests must still reach the worker to start their transfer.
        async_ids = {
            rid
            for rid, state in self._pd_requests.items()
            if state.role == "decode" and state.num_pages and rid in self._allocations
        }
        for rid in async_ids:
            state = self._pd_requests.pop(rid)
            _, blocks = self._allocations.pop(rid)
            if len(blocks) < state.num_pages:
                raise ValueError("TRT-LLM allocated too few P/D KV blocks")
            metadata.pd_loads.append((state, blocks[: state.num_pages]))
        for request in scheduler_output.new_requests:
            if request.request_id in async_ids:
                continue
            request_id = request.request_id
            pending = self._pending.pop(request_id, None)
            keys, blocks = self._allocations.pop(request_id, ([], []))
            if not blocks:
                blocks = list(request.new_block_ids)
            state = self._pd_requests.pop(request_id, None)
            if state is not None:
                if len(blocks) < state.num_pages:
                    raise ValueError("TRT-LLM allocated too few P/D KV blocks")
                if state.role == "prefill":
                    metadata.pd_saves.append((state, blocks[: state.num_pages]))
                    if self._history is not None:
                        imported = 0
                        if pending is not None:
                            start, pages = pending
                            if start + len(pages) > len(blocks):
                                raise ValueError(
                                    "TRT-LLM allocated too few history blocks"
                                )
                            metadata.loads.extend(
                                zip(blocks[start : start + len(pages)], pages)
                            )
                            imported = len(pages) * self.contract.block_size
                        complete = (
                            request.computed_position
                            + imported
                            + request.num_scheduled_tokens
                        ) // self.contract.block_size
                        metadata.saves.extend(zip(keys[:complete], blocks[:complete]))
                else:
                    metadata.pd_loads.append((state, blocks[: state.num_pages]))
                continue
            if self.client.pd_gpu:
                continue
            imported = 0
            if pending is not None:
                start, pages = pending[:2]
                if start + len(pages) > len(blocks):
                    if len(pending) == 4:
                        self.client.release(pending[2])
                    raise ValueError(
                        "TRT-LLM allocated too few blocks for Xavier prefix"
                    )
                targets = blocks[start : start + len(pages)]
                if len(pending) == 4:
                    slots = pending[3] or [None] * len(pages)
                    metadata.shard_loads.extend(zip(targets, pages, slots))
                    self._loading_pins[request_id] = pending[2]
                else:
                    metadata.loads.extend(zip(targets, pages))
                imported = len(pages) * self.contract.block_size
            complete = (
                request.computed_position + imported + request.num_scheduled_tokens
            ) // self.contract.block_size
            metadata.saves.extend(zip(keys[:complete], blocks[:complete]))
        return metadata

    def request_finished(self, request, cache_block_ids):
        self._forget_pending(request.request_id)
        ticket = self._loading_pins.pop(request.request_id, None)
        if ticket is not None:
            self.client.release(ticket)
        self._allocations.pop(request.request_id, None)
        self._pd_requests.pop(request.request_id, None)
        destination = self._pd_destinations.pop(request.request_id, None)
        if destination is not None:
            destination.cancelled = True
            # Cancellation is exceptional: block this callback until all writes
            # drain. TRT cannot free/reassign these slots before we return.
            if destination.load_future is not None:
                try:
                    destination.load_future.result()
                except Exception:
                    pass  # The failed transfer has already drained its writes.
        state = self._pd_sources.pop(request.request_id, None)
        # TRT pins these blocks until the worker returns this ID from
        # get_finished. A completed Python output does not end the KV lifetime.
        return bool(state is not None and state.registered)


class XavierConnectorWorker(KvCacheConnectorWorker):
    def __init__(self, llm_args):
        super().__init__(llm_args)
        self.client = XavierClient()
        self.contract = self.client.contract
        self._history = (
            XavierClient(
                {
                    **self.client.config["prefix_cache"],
                    "contract": self.contract.to_dict(),
                }
            )
            if getattr(self.client, "config", {}).get("role") == "prefill"
            and self.client.config.get("prefix_cache")
            else None
        )
        self.pool = None
        self.rank = _local_rank()
        self._publisher = None
        self._writer = None
        self._writer_failed = False
        # Pages this rank knows are stored. Skipping them needs no round trip
        # to the cache actor, which may be busy receiving uploads.
        self._hints = _StoredHints(
            self.client.capacity_bytes
            // (self.contract.num_layers * self.contract.layer_nbytes)
            if getattr(self.client, "capacity_bytes", 0)
            else 1
        )
        self._waiting_saves = set()
        self._finished_saves = set()
        self._loading = {}
        self._waiting_loads = set()
        self.load_failed = None

    def register_kv_caches(self, kv_cache_tensor):
        contract = self.contract
        expected = (
            contract.num_layers,
            2,
            contract.block_size * contract.num_kv_heads * contract.head_dim,
        )
        dtype = getattr(torch, contract.logical_dtype)
        if (
            kv_cache_tensor.dtype != dtype
            or tuple(kv_cache_tensor.shape[1:]) != expected
        ):
            raise ValueError(
                f"TRT-LLM GPU KV pool {tuple(kv_cache_tensor.shape)} "
                f"({kv_cache_tensor.dtype}) differs from Xavier's contract "
                f"(blocks, {expected}), {dtype}"
            )
        if not kv_cache_tensor.is_contiguous():
            raise ValueError("TRT-LLM Xavier requires a contiguous KV pool")
        self.pool = kv_cache_tensor
        # Map and page-lock shared pages while the engine loads; locking
        # gigabytes takes seconds and must not delay the first request.
        self.client.pages()
        if self.client.pd_gpu:
            from .gpu import register_service

            register_service(self.client, kv_cache_tensor)

    def _validate_slots(self, slots):
        if self.pool is None or any(
            type(slot) is not int or slot < 0 or slot >= len(self.pool)
            for slot in slots
        ):
            raise ValueError("TRT-LLM Xavier received invalid KV page slots")

    def start_load_kv(self, stream):
        metadata = self.get_connector_meta()
        if metadata is None:
            return
        self._validate_slots(
            [slot for slot, _ in metadata.loads]
            + [slot for slot, _, _ in metadata.shard_loads]
            + [slot for _, slots in metadata.pd_loads for slot in slots]
        )
        if metadata.shard_loads:
            tp = self.client.tensor_parallel_size
            keys = [shard_key(key, self.rank, tp) for _, key, _ in metadata.shard_loads]
            shared = self.client.pages()
            if shared is not None and all(
                slots is not None for _, _, slots in metadata.shard_loads
            ):
                # Pinned pages: copy each rank's slot straight into the pool.
                pages = [
                    shared[slots[self.rank]] for _, _, slots in metadata.shard_loads
                ]
            else:
                pages = self.client.read_pinned(keys)
            for (slot, _, _), page in zip(metadata.shard_loads, pages):
                self.pool[slot].copy_(
                    page.view(self.pool.dtype).reshape(self.pool.shape[1:]),
                    non_blocking=self.pool.is_cuda,
                )
            if self.pool.is_cuda:
                torch.cuda.current_stream(self.pool.device).synchronize()
            # The pin kept these pages stored while this rank copied them.
            self._hints.add(keys, self._evictions())
            if self.rank == 0:
                try:
                    self.client.rpc(
                        "record_import",
                        self.client.namespace,
                        len(metadata.shard_loads),
                    )
                except Exception:
                    logger.warning(
                        "Could not record TRT-LLM Xavier imports", exc_info=True
                    )
        if metadata.loads and (
            self.client.config.get("shared_prefix") or self._history is not None
        ):
            from ..xavier.backends.torch.prefix import unpack_pages
            from .cross_engine import canonical_trt_views

            unpack_pages(
                canonical_trt_views(self.pool, self.contract),
                [slot for slot, _ in metadata.loads],
                [page for _, page in metadata.loads],
                self.contract,
            )
        else:
            for slot, page in metadata.loads:
                self.pool[slot].copy_(
                    page.view(self.pool.dtype).reshape(self.pool.shape[1:]),
                    non_blocking=False,
                )
        for state, blocks in metadata.pd_loads:
            from .gpu import service

            if state.num_pages:
                state.load_future = service().submit("load", state.room, blocks)
                self._loading[state.engine_request_id] = state
            else:
                service().rpc("load", state.room, blocks)
        if metadata.loads:
            # Copy completion, not a lookup, is the evidence for imported KV.
            try:
                cache_client = self._history or self.client
                cache_client.rpc(
                    "record_import", cache_client.namespace, len(metadata.loads)
                )
            except Exception:
                logger.warning("Could not record TRT-LLM Xavier imports", exc_info=True)

    def _evictions(self):
        evictions = self.client.evictions()
        return self._hints.reported if evictions is None else evictions

    def _write_shared(self, saves):
        """Copy new pages from the pool straight into shared-memory slots."""
        keys = [key for key, _ in saves]
        pool = self.pool

        def copy(index, target):
            source = pool[saves[index][1]].view(torch.uint8).reshape(-1)
            target.copy_(source, non_blocking=pool.is_cuda)

        def synchronize():
            if pool.is_cuda:
                torch.cuda.current_stream(pool.device).synchronize()

        try:
            _store_shared(self.client, keys, copy, synchronize, self._hints)
        except Exception:
            logger.warning("TRT-LLM Xavier publication failed", exc_info=True)

    def _staged_writer(self):
        if self._writer is None and not self._writer_failed:
            try:
                self._writer = _StagedWriter(self.client, self.pool, hints=self._hints)
            except (RuntimeError, ValueError):
                # No GPU memory for staging: copy synchronously instead.
                logger.warning("No GPU staging for Xavier publication", exc_info=True)
                self._writer_failed = True
            else:
                _publishers.add(self._writer)
        return self._writer

    def wait_for_layer_load(self, layer_idx, stream):
        pass  # Async requests resume only after get_finished confirms GPU writes.

    def save_kv_layer(self, layer_idx, stream):
        pass  # wait_for_save publishes only after the entire forward pass.

    def wait_for_save(self, stream):
        metadata = self.get_connector_meta()
        if metadata is None or not (metadata.saves or metadata.pd_saves):
            return
        self._validate_slots([slot for _, slot in metadata.saves])
        stream.synchronize()
        for state, blocks in metadata.pd_saves:
            from .gpu import service

            if state.cancelled:
                continue
            try:
                service().rpc("publish", state.room, state.engine_request_id, blocks)
            except Exception:
                if not state.cancelled:
                    raise
                # Cleanup can tombstone/drop a room while publish is in flight.
                # This cancellation must not escape TRT's batch-wide callback.
                continue
            state.registered = bool(blocks)
        if not metadata.saves:
            return
        cache_client = self._history or self.client
        shared_prefix = cache_client.config.get("shared_prefix")
        page_bytes = self.contract.num_layers * self.contract.layer_nbytes
        batch_size = (
            max(1, min(128, 32 * 1024**2 // page_bytes)) if shared_prefix else 128
        )
        saves = metadata.saves
        if not shared_prefix:
            tp = self.client.tensor_parallel_size
            # Imported and repeated prefixes are already stored. Skip them
            # instead of copying every page to the host again.
            saves = [(shard_key(key, self.rank, tp), slot) for key, slot in saves]
            missing = set(
                self._hints.unknown([key for key, _ in saves], self._evictions())
            )
            saves = [(key, slot) for key, slot in saves if key in missing]
            if saves and self.client.pages() is not None:
                staged = 0
                if self.pool.is_cuda and isinstance(stream, torch.cuda.Stream):
                    writer = self._staged_writer()
                    if writer is not None:
                        staged = writer.submit(self.pool, stream, saves)
                if staged < len(saves):
                    self._write_shared(saves[staged:])
            elif saves:
                if self._publisher is None:
                    self._publisher = _Publisher(self.client, hints=self._hints)
                    _publishers.add(self._publisher)
                self._publisher.submit(self.pool, saves)
            return
        start = 0
        # Request prefixes are causal: keeping their first occurrence also
        # keeps every head before its suffix, including overlapping requests.
        slots_by_key = {}
        for key, slot in saves:
            slots_by_key.setdefault(key, slot)
        saves = list(slots_by_key.items())[
            : min(cache_client.capacity_bytes // page_bytes, 4096)
        ]
        keys = [key for key, _ in saves]
        try:
            start = cache_client.rpc(
                ("pending_prefix" if self._history is not None else "exists"),
                cache_client.namespace,
                keys,
            )
        except Exception:
            logger.warning("TRT-LLM Xavier publication failed", exc_info=True)
            return
        from ..xavier.backends.torch.prefix import pack_pages
        from .cross_engine import canonical_trt_views

        for offset in range(start, len(saves), batch_size):
            batch = saves[offset : offset + batch_size]
            try:
                pages = pack_pages(
                    canonical_trt_views(self.pool, self.contract),
                    [slot for _, slot in batch],
                    self.contract,
                )
                cache_client.rpc(
                    "put",
                    cache_client.namespace,
                    keys[: offset + len(batch)],
                    pages,
                    offset,
                )
            except Exception:
                # Publishing is optional; failure leaves subsequent requests cold.
                logger.warning("TRT-LLM Xavier publication failed", exc_info=True)
                break

    def get_finished(self, finished_gen_req_ids, started_loading_req_ids):
        if not self.client.pd_gpu:
            return [], []
        self._waiting_loads.update(started_loading_req_ids)
        loaded = []
        for rid, state in list(self._loading.items()):
            if state.cancelled:
                if state.load_future.done():
                    self._loading.pop(rid)
                    self._waiting_loads.discard(rid)
                continue
            if rid not in self._waiting_loads or not state.load_future.done():
                continue
            self._loading.pop(rid)
            self._waiting_loads.discard(rid)
            try:
                state.load_future.result()
            except Exception as error:
                if self.load_failed is None:
                    raise  # Never resume a request with missing/partial KV.
                self.load_failed(rid, error)
            else:
                state.imported_tokens = len(state.tokens) - 1
                loaded.append(rid)
        self._waiting_saves.update(finished_gen_req_ids)
        if self._waiting_saves:
            from .gpu import service

            self._finished_saves.update(service().rpc("poll_direct_gpu_v1"))
        finished = self._finished_saves & self._waiting_saves
        self._waiting_saves.difference_update(finished)
        self._finished_saves.difference_update(finished)
        return list(finished), loaded
