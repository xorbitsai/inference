# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""TRT-LLM 1.2.1 connector for same-engine, full-page prefix sharing.

Reads acquire owned CPU snapshots before reporting a hit. Eviction between
lookup and GPU load therefore cannot turn an admitted prefix into missing KV.
The first implementation uses synchronous transfers and disables chunking,
overlap, and eviction/replay in the engine scheduler.
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
from ..xavier.contract import KVCacheContract, build_prefix_keys
from .pd import request_state
from .xavier import CONFIG_ENV

logger = logging.getLogger(__name__)
_clients: weakref.WeakSet = weakref.WeakSet()


class XavierClient:
    def __init__(self):
        self.config = json.loads(os.environ[CONFIG_ENV])
        self.contract = KVCacheContract.from_dict(self.config["contract"])
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

    def rpc(self, method, *args):
        if self._closed:
            raise RuntimeError("TRT-LLM Xavier client is closed")

        async def invoke():
            if self._ref is None:
                self._ref = await xo.actor_ref(
                    address=self.config["address"], uid=self.config["uid"]
                )
            return await getattr(self._ref, method)(*args)

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
        try:
            for offset in range(0, len(keys), 128):
                batch = self.rpc("get", self.namespace, keys[offset : offset + 128])
                for page in batch:
                    if page is None:
                        return pages
                    expected = self.contract.layer_nbytes * self.contract.num_layers
                    if (
                        page.dtype != torch.uint8
                        or page.device.type != "cpu"
                        or page.numel() != expected
                    ):
                        raise ValueError("TRT-LLM Xavier returned invalid KV page data")
                    pages.append(page.clone())
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
    for client in list(_clients):
        client.close()


@dataclass
class XavierMetadata:
    loads: list = field(default_factory=list)
    saves: list = field(default_factory=list)
    pd_loads: list = field(default_factory=list)


class XavierConnectorScheduler(KvCacheConnectorScheduler):
    def __init__(self, llm_args):
        super().__init__(llm_args)
        self.client = XavierClient()
        self.contract = self.client.contract
        self._pending = {}
        self._allocations = {}
        self._pd_requests = {}

    def _keys(self, tokens):
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
                return 0, False
            if num_computed_tokens:
                raise ValueError("TRT-LLM P/D requires an isolated destination prefix")
            return max(len(state.tokens) - 1, 0), False
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
        available = max(self.client.capacity_bytes - pending_bytes, 0) // page_bytes
        pages = self.client.read_prefix(keys[start : start + available])
        if pages:
            self._pending[request_id] = (start, pages)
        return len(pages) * block, False

    def update_state_after_alloc(self, request, block_ids):
        state = self._pd_requests.get(request.request_id)
        self._allocations[request.request_id] = (
            (
                state.keys
                if state is not None
                else self._keys(list(request.get_tokens(0)))
            ),
            list(block_ids),
        )

    def build_connector_meta(self, scheduler_output):
        metadata = XavierMetadata()
        for request in scheduler_output.new_requests:
            request_id = request.request_id
            pending = self._pending.pop(request_id, None)
            keys, blocks = self._allocations.pop(request_id, ([], []))
            if not blocks:
                blocks = list(request.new_block_ids)
            state = self._pd_requests.pop(request_id, None)
            if state is not None:
                if len(blocks) < len(state.keys):
                    raise ValueError("TRT-LLM allocated too few P/D KV blocks")
                if state.role == "prefill":
                    metadata.saves.extend(zip(state.keys, blocks))
                else:
                    metadata.loads.extend(zip(blocks, state.pages))
                    metadata.pd_loads.append(state)
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
        return False


class XavierConnectorWorker(KvCacheConnectorWorker):
    def __init__(self, llm_args):
        super().__init__(llm_args)
        self.client = XavierClient()
        self.contract = self.client.contract
        self.pool = None

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
        self._validate_slots([slot for slot, _ in metadata.loads])
        for slot, page in metadata.loads:
            self.pool[slot].copy_(
                page.view(self.pool.dtype).reshape(self.pool.shape[1:]),
                non_blocking=False,
            )
        for state in metadata.pd_loads:
            state.imported_tokens = max(len(state.tokens) - 1, 0)
        if metadata.loads:
            # Copy completion, not a lookup, is the evidence for imported KV.
            try:
                self.client.rpc(
                    "record_import", self.client.namespace, len(metadata.loads)
                )
            except Exception:
                logger.warning("Could not record TRT-LLM Xavier imports", exc_info=True)

    def wait_for_layer_load(self, layer_idx, stream):
        pass  # start_load_kv completes all copies synchronously.

    def save_kv_layer(self, layer_idx, stream):
        pass  # wait_for_save publishes only after the entire forward pass.

    def wait_for_save(self, stream):
        metadata = self.get_connector_meta()
        if metadata is None or not metadata.saves:
            return
        self._validate_slots([slot for _, slot in metadata.saves])
        stream.synchronize()
        for offset in range(0, len(metadata.saves), 128):
            batch = metadata.saves[offset : offset + 128]
            keys = [key for key, _ in batch]
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
            try:
                self.client.rpc("put", self.client.namespace, keys, pages)
            except Exception:
                # Publishing is optional; failure leaves subsequent requests cold.
                logger.warning("TRT-LLM Xavier publication failed", exc_info=True)

    def get_finished(self, finished_gen_req_ids, started_loading_req_ids):
        return [], []
