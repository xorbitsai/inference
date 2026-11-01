# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""TRT-LLM 1.2.1 connector for prefix sharing and Xavier P/D handoffs.

Reads acquire owned CPU snapshots before reporting a hit. Eviction between
lookup and GPU load therefore cannot turn an admitted prefix into missing KV.
P/D loads suspend their request until GPU writes finish, allowing unrelated
decode work to continue. Shared CPU prefix loads remain synchronous. Chunking,
overlap scheduling, and eviction/replay are disabled.
"""

import asyncio
import json
import logging
import os
import threading
import weakref
from dataclasses import dataclass, field

import torch
import xoscar as xo
from tensorrt_llm._torch.pyexecutor.kv_cache_connector import (
    KvCacheConnectorScheduler,
    KvCacheConnectorWorker,
)

from ..xavier.actor_loop import acquire_actor_loop, release_actor_loop
from ..xavier.contract import KVCacheContract, build_prefix_keys, fingerprint_metadata
from .pd import request_state
from .xavier import CONFIG_ENV

logger = logging.getLogger(__name__)
_clients: weakref.WeakSet = weakref.WeakSet()


class XavierClient:
    def __init__(self, config=None):
        self.config = (
            json.loads(os.environ[CONFIG_ENV]) if config is None else dict(config)
        )
        self.contract = KVCacheContract.from_dict(self.config["contract"])
        self.pd_gpu = self.config.get("role") in ("prefill", "decode")
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
                        version="1.2.1",
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
                self.namespace = self.rpc(
                    "configure",
                    self.contract.to_dict(),
                    {
                        "engine": "trt-llm",
                        "version": "1.2.1",
                        "layout": "layer_first",
                        "physical_layout": "block-layer-kv-packed",
                        "hash_format": "xavier-prefix-v1",
                    },
                )
                self.capacity_bytes = self.rpc("get_stats")["capacity_bytes"]
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

    def close(self):
        if not self._closed:
            self._closed = True
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=6)
            if self._thread.is_alive():
                logger.warning("Timed out stopping TRT-LLM Xavier RPC thread")


def close_clients():
    from .gpu import close_service

    close_service()
    for client in list(_clients):
        client.close()


@dataclass
class XavierMetadata:
    loads: list = field(default_factory=list)
    saves: list = field(default_factory=list)
    pd_loads: list = field(default_factory=list)
    pd_saves: list = field(default_factory=list)


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

    def _keys(self, tokens):
        if self.client.config.get("shared_prefix") or self._history is not None:
            from ..xavier.prefix import prefix_keys

            return prefix_keys(tokens, self.contract.block_size)
        return [key.digest for key in build_prefix_keys(self.contract, tokens)]

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        request_id = request.request_id
        self._pending.pop(request_id, None)
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
        pending_bytes = sum(
            len(pages) * page_bytes for _, pages in self._pending.values()
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
                start, pages = pending
                if start + len(pages) > len(blocks):
                    raise ValueError(
                        "TRT-LLM allocated too few blocks for Xavier prefix"
                    )
                metadata.loads.extend(zip(blocks[start : start + len(pages)], pages))
                imported = len(pages) * self.contract.block_size
            complete = (
                request.computed_position + imported + request.num_scheduled_tokens
            ) // self.contract.block_size
            metadata.saves.extend(zip(keys[:complete], blocks[:complete]))
        return metadata

    def request_finished(self, request, cache_block_ids):
        self._pending.pop(request.request_id, None)
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
            + [slot for _, slots in metadata.pd_loads for slot in slots]
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
        start = 0
        if shared_prefix:
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
        for offset in range(start, len(saves), batch_size):
            batch = saves[offset : offset + batch_size]
            try:
                if shared_prefix:
                    from ..xavier.backends.torch.prefix import pack_pages
                    from .cross_engine import canonical_trt_views

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
                else:
                    pages = [
                        self.pool[slot]
                        .detach()
                        .cpu()
                        .contiguous()
                        .view(torch.uint8)
                        .reshape(-1)
                        .clone()
                        for _, slot in batch
                    ]
                    self.client.rpc(
                        "put", self.client.namespace, [key for key, _ in batch], pages
                    )
            except Exception:
                # Publishing is optional; failure leaves subsequent requests cold.
                logger.warning("TRT-LLM Xavier publication failed", exc_info=True)
                if shared_prefix:
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
