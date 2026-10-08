# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""SGLang HiCache read/write adapter backed by Xavier's shared CPU snapshots."""

import asyncio
import importlib.metadata
import logging
import threading

import torch
import xoscar as xo
from sglang.srt.mem_cache.hicache_storage import HiCacheStorage

from ...xavier.actor_loop import acquire_actor_loop, release_actor_loop
from ...xavier.contract import KVCacheContract

logger = logging.getLogger(__name__)


class XavierHiCacheStorage(HiCacheStorage):
    def __init__(self, storage_config, kwargs=None):
        if (
            any(
                getattr(storage_config, name) != 1
                for name in ("tp_size", "pp_size", "attn_cp_size")
            )
            or storage_config.is_mla_model
        ):
            raise ValueError("SGLang Xavier requires unsharded full-attention KV")
        self._config = dict(storage_config.extra_config)
        self._contract = KVCacheContract.from_dict(self._config["contract"])
        self._namespace = None
        self._closed = False
        self._rpc_loop = None
        self._rpc_thread = None
        self._rpc_ref = None
        self._rpc_lock = threading.Lock()

    def _rpc(self, method, *args):
        if self._closed:
            raise RuntimeError("Xavier storage is closed")
        with self._rpc_lock:
            if self._closed:
                raise RuntimeError("Xavier storage is closed")
            if self._rpc_thread is None:
                ready = threading.Event()

                def run():
                    loop = self._rpc_loop = acquire_actor_loop()
                    ready.set()
                    try:
                        loop.run_forever()
                    finally:
                        release_actor_loop(loop)

                self._rpc_thread = threading.Thread(target=run, daemon=True)
                self._rpc_thread.start()
                ready.wait()
            loop = self._rpc_loop

        async def invoke():
            if self._rpc_ref is None:
                self._rpc_ref = await xo.actor_ref(
                    address=self._config["address"], uid=self._config["uid"]
                )
            return await getattr(self._rpc_ref, method)(*args)

        with self._rpc_lock:
            if self._closed:
                raise RuntimeError("Xavier storage is closed")
            future = asyncio.run_coroutine_threadsafe(
                asyncio.wait_for(invoke(), timeout=10), loop
            )
        try:
            return future.result(timeout=11)
        except BaseException:
            future.cancel()
            raise

    def register_mem_pool_host(self, pool):
        super().register_mem_pool_host(pool)
        contract = self._contract
        if (
            pool.dtype != torch.float16
            or pool.page_size != contract.block_size
            or pool.layer_num != contract.num_layers
            or pool.head_num != contract.num_kv_heads
            or pool.head_dim != contract.head_dim
            or pool.layout != "layer_first"
        ):
            raise ValueError("SGLang host KV pool differs from Xavier's contract")
        self._namespace = self._rpc(
            "configure",
            contract.to_dict(),
            {
                "engine": "sglang",
                "version": importlib.metadata.version("sglang"),
                "layout": pool.layout,
                "hash_format": "sglang-storage-chain-v1",
            },
        )

    def _request(self, method, keys, fallback, *args):
        result = 0 if method == "exists" else []
        for offset in range(0, len(keys), 128):
            batch = keys[offset : offset + 128]
            batch_args = [arg[offset : offset + 128] for arg in args]
            try:
                value = self._rpc(method, self._namespace, batch, *batch_args)
            except Exception:
                logger.warning(
                    "Xavier storage %s failed; treating pages as misses",
                    method,
                    exc_info=True,
                )
                value = 0 if method == "exists" else fallback[offset : offset + 128]
            if method == "exists":
                result += value
                if value != len(batch):
                    break
            else:
                result.extend(value)
        return result

    def get(self, key, target_location=None, target_sizes=None):
        page = self._request("get", [key], [None])[0]
        if page is not None:
            page = page.view(torch.float16)
        if page is not None and target_location is not None:
            target_location.copy_(page.reshape(target_location.shape))
        return page

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        if target_locations is not None or target_sizes is not None:
            raise ValueError("Xavier storage uses the HiCache v1 host-pool interface")
        return [
            page.view(torch.float16) if page is not None else None
            for page in self._request("get", keys, [None] * len(keys))
        ]

    def set(self, key, value=None, target_location=None, target_sizes=None):
        return self.batch_set([key], [value])

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None):
        if values is None or len(keys) != len(values):
            return False
        if any(page.dtype != torch.float16 for page in values):
            raise ValueError("Xavier requires FP16 host pages")
        pages = [
            page.detach().cpu().contiguous().view(torch.uint8).reshape(-1).clone()
            for page in values
        ]
        return all(self._request("put", keys, [False] * len(keys), pages))

    def exists(self, key):
        return self.batch_exists([key]) == 1

    def batch_exists(self, keys, extra_info=None):
        return self._request("exists", keys, 0)

    def _offsets(self, keys, host_indices):
        if (
            host_indices.ndim != 1
            or host_indices.numel() != len(keys) * self._contract.block_size
        ):
            raise ValueError("HiCache host index count differs from page keys")
        page_size = self._contract.block_size
        indices = host_indices.tolist()
        offsets = indices[::page_size]
        if host_indices.dtype not in (torch.int32, torch.int64) or any(
            offset < 0
            or offset % page_size
            or indices[index * page_size : (index + 1) * page_size]
            != list(range(offset, offset + page_size))
            for index, offset in enumerate(offsets)
        ):
            raise ValueError(
                "HiCache host indices must contain aligned contiguous pages"
            )
        return offsets

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        offsets = self._offsets(keys, host_indices)
        pages = [
            self.mem_pool_host.get_data_page(index, flat=True).clone().view(torch.uint8)
            for index in offsets
        ]
        return self._request("put", keys, [False] * len(keys), pages)

    def batch_get_v1(self, keys, host_indices, extra_info=None):
        offsets = self._offsets(keys, host_indices)
        pages = self._request("get", keys, [None] * len(keys))
        results = []
        for index, page in zip(offsets, pages):
            if page is not None and not self._closed:
                self.mem_pool_host.set_from_flat_data_page(
                    index, page.view(torch.float16)
                )
                results.append(True)
            else:
                results.append(False)
        return results

    def get_stats(self):
        return self._rpc("get_stats")

    def close(self):
        with self._rpc_lock:
            if self._closed:
                return
            self._closed = True
            if self._rpc_thread is None:
                return
            self._rpc_loop.call_soon_threadsafe(self._rpc_loop.stop)
        self._rpc_thread.join(timeout=6)
        if self._rpc_thread.is_alive():
            logger.warning("Timed out stopping Xavier HiCache RPC thread")
