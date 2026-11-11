# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""SGLang HiCache read/write adapter backed by Xavier's shared page storage."""

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

# SGLang names the anchor pool "kv". A logical anchor (DeepSeek-V4) holds no
# bytes; its zero-byte page marks a prefix whose side pools were all written.
_KV = "kv"
# Hashes TP-sharded pools per rank even for MLA models (SGLang's Mamba state).
_RANK_SHARDED_POOLS = ("mamba",)
_MAX_RPC_ENTRIES = 1024


class _V1Transfer:
    def __init__(self, name, keys, host_indices):
        self.name, self.keys, self.host_indices = name, keys, host_indices


class XavierHiCacheStorage(HiCacheStorage):
    def __init__(self, storage_config, kwargs=None):
        self._config = dict(storage_config.extra_config)
        self._pools = self._config.get("pools")
        if self._pools:
            # SGLang's HiCache controller picks the writer: one rank for
            # replicated MLA pools, every rank for TP-sharded pools.
            if any(
                getattr(storage_config, name, 1) != 1
                for name in ("pp_size", "attn_cp_size", "dcp_size")
            ):
                raise ValueError(
                    "SGLang Xavier supports tensor parallelism without PP, CP or DCP"
                )
            self._is_mla = bool(storage_config.is_mla_model)
            self._tp_rank = storage_config.tp_rank
            self._tp_size = storage_config.tp_size
            self._contract = None
        elif (
            any(
                getattr(storage_config, name) != 1
                for name in ("tp_size", "pp_size", "attn_cp_size")
            )
            or storage_config.is_mla_model
        ):
            raise ValueError("SGLang Xavier requires unsharded full-attention KV")
        else:
            self._contract = KVCacheContract.from_dict(self._config["contract"])
        self._layouts = {}
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
            timeout = max(10, self._config.get("read_timeout", 0) + 1)
            future = asyncio.run_coroutine_threadsafe(
                asyncio.wait_for(invoke(), timeout=timeout), loop
            )
        try:
            return future.result(timeout=timeout + 1)
        except BaseException:
            future.cancel()
            raise

    def register_mem_pool_host(self, pool):
        super().register_mem_pool_host(pool)
        if self._pools:
            self._layouts[_KV] = self._layout(_KV, pool)
            return
        contract = self._contract
        if (
            pool.dtype != getattr(torch, contract.logical_dtype)
            or pool.page_size != contract.block_size
            or pool.layer_num != contract.num_layers
            or pool.head_num != contract.num_kv_heads
            or pool.head_dim != contract.head_dim
            or pool.layout != "layer_first"
        ):
            raise ValueError("SGLang host KV pool differs from Xavier's contract")
        if self._config.get("shared_prefix"):
            from sglang.srt.mem_cache.utils import get_storage_hash_str

            from ...xavier.prefix import prefix_keys

            class Tokens(list):
                extra_key = None
                cache_salt = "xavier-format-probe"

            tokens = Tokens(range(contract.block_size))
            if get_storage_hash_str(
                tokens, page_size=contract.block_size
            ) != prefix_keys(tokens, contract.block_size, tokens.cache_salt):
                raise ValueError(
                    "SGLang storage hashes differ from Xavier's shared-prefix protocol"
                )
            self._namespace = self._rpc("configure", contract.to_dict())["namespace"]
            return
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
                if self._config.get("shared_prefix") and method == "put":
                    batch_args[0] = [self._to_canonical(page) for page in batch_args[0]]
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
                if self._config.get("shared_prefix") and method == "get":
                    value = [
                        self._from_canonical(page) if page is not None else None
                        for page in value
                    ]
                    value += [None] * (len(batch) - len(value))
                result.extend(value)
        return result

    def _to_canonical(self, page):
        c = self._contract
        return (
            page.view(getattr(torch, c.logical_dtype))
            .reshape(2, c.num_layers, c.block_size, c.num_kv_heads, c.head_dim)
            .permute(1, 0, 2, 3, 4)
            .contiguous()
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )

    def _from_canonical(self, page):
        c = self._contract
        if type(page) is not bytes or len(page) != c.layer_nbytes * c.num_layers:
            raise ValueError("Invalid shared-prefix KV page")
        return (
            torch.frombuffer(bytearray(page), dtype=torch.uint8)
            .reshape(c.num_layers, 2, c.block_size, c.num_kv_heads, c.head_dim, 2)
            .permute(1, 0, 2, 3, 4, 5)
            .contiguous()
            .reshape(-1)
        )

    def get(self, key, target_location=None, target_sizes=None):
        page = self._request("get", [key], [None])[0]
        if page is not None:
            page = page.view(getattr(torch, self._contract.logical_dtype))
        if page is not None and target_location is not None:
            target_location.copy_(page.reshape(target_location.shape))
        return page

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        if target_locations is not None or target_sizes is not None:
            raise ValueError("Xavier storage uses the HiCache v1 host-pool interface")
        return [
            (
                page.view(getattr(torch, self._contract.logical_dtype))
                if page is not None
                else None
            )
            for page in self._request("get", keys, [None] * len(keys))
        ]

    def set(self, key, value=None, target_location=None, target_sizes=None):
        return self.batch_set([key], [value])

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None):
        if values is None or len(keys) != len(values):
            return False
        if any(
            page.dtype != getattr(torch, self._contract.logical_dtype)
            for page in values
        ):
            raise ValueError(
                f"Xavier requires {self._contract.logical_dtype} host pages"
            )
        pages = [
            page.detach().cpu().contiguous().view(torch.uint8).reshape(-1).clone()
            for page in values
        ]
        return all(self._request("put", keys, [False] * len(keys), pages))

    def exists(self, key):
        return self.batch_exists([key]) == 1

    def batch_exists(self, keys, extra_info=None):
        if self._pools:
            return self._pool_exists(keys)
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
        if self._pools:
            return self._write_pool(_KV, self.mem_pool_host, keys, host_indices)
        offsets = self._offsets(keys, host_indices)
        pages = [
            self.mem_pool_host.get_data_page(index, flat=True).clone().view(torch.uint8)
            for index in offsets
        ]
        return self._request("put", keys, [False] * len(keys), pages)

    def batch_get_v1(self, keys, host_indices, extra_info=None):
        if self._pools:
            return self._batch_io_v2(
                [_V1Transfer(_KV, keys, host_indices)], write=False
            )[_KV]
        offsets = self._offsets(keys, host_indices)
        pages = self._request("get", keys, [None] * len(keys))
        results = []
        for index, page in zip(offsets, pages):
            if page is not None and not self._closed:
                self.mem_pool_host.set_from_flat_data_page(
                    index, page.view(getattr(torch, self._contract.logical_dtype))
                )
                results.append(True)
            else:
                results.append(False)
        return results

    # Multi-pool mode: MLA and hybrid models, any tensor-parallel size.

    @staticmethod
    def _layout(name, pool):
        page = pool.get_dummy_flat_data_page()
        return {
            "page_bytes": page.numel() * page.element_size(),
            "dtype": str(page.dtype).removeprefix("torch."),
            "page_size": int(getattr(pool, "page_size", 1) or 1),
            "layer_num": int(getattr(pool, "layer_num", 0) or 0),
            "layout": getattr(pool, "layout", None),
            "class": type(pool).__name__,
        }

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        super().register_mem_host_pool_v2(host_pool, host_pool_name)
        if not self._pools:
            raise ValueError("SGLang Xavier needs multi-pool mode for this model")
        name = str(host_pool_name)
        if name != _KV or host_pool is not getattr(self, "mem_pool_host", None):
            self._layouts[name] = self._layout(name, host_pool)

    def on_host_pools_registered(self):
        if not self._pools:
            return
        identity = {
            **self._config["identity"],
            "engine": "sglang",
            "version": importlib.metadata.version("sglang"),
            # MLA pages are replicated across TP ranks; sharded pages are not.
            "tp_size": None if self._is_mla else self._tp_size,
            "hash_format": "sglang-storage-chain-v1",
        }
        self._namespace = self._rpc("configure_pools", identity, self._layouts)
        logger.info(
            "Xavier shared HiCache ready on TP rank %d/%d: %s",
            self._tp_rank,
            self._tp_size,
            ", ".join(
                f"{name}={layout['page_bytes']}B"
                for name, layout in sorted(self._layouts.items())
            ),
        )

    def _entry(self, pool, key):
        pool = str(pool)
        if not self._is_mla or pool in _RANK_SHARDED_POOLS:
            key = f"{key}@tp{self._tp_rank}"
        return [pool, key]

    def _pool_rpc(self, method, entries, pages=None):
        """Batched RPC; a failure turns the whole request into misses."""
        results = []
        writers = {}
        try:
            for offset in range(0, len(entries), _MAX_RPC_ENTRIES):
                batch = entries[offset : offset + _MAX_RPC_ENTRIES]
                args = (self._namespace, batch)
                if pages is not None:
                    args += (
                        pages[offset : offset + _MAX_RPC_ENTRIES],
                        self._config.get("replica", ""),
                    )
                value = self._rpc(method, *args)
                if method == "get_pool_pages":
                    for writer, count in value["writers"].items():
                        writers[writer] = writers.get(writer, 0) + count
                    value = value["pages"]
                if len(value) != len(batch):
                    raise ValueError("Xavier returned a short pool page batch")
                results.extend(value)
        except Exception:
            logger.warning(
                "Xavier storage %s failed; treating pages as misses",
                method,
                exc_info=True,
            )
            return [None if method == "get_pool_pages" else False] * len(entries), {}
        return results, writers

    def _present(self, pool, keys):
        if not keys:
            return []
        present, _ = self._pool_rpc(
            "contains_pool_pages", [self._entry(pool, key) for key in keys]
        )
        return [bool(value) for value in present]

    def _pool_exists(self, keys):
        present = self._present(_KV, keys)
        return present.index(False) if False in present else len(present)

    def _host_offsets(self, pool, keys, host_indices):
        page_size = int(getattr(pool, "page_size", 1) or 1)
        if host_indices is None or host_indices.numel() != len(keys) * page_size:
            return None
        return [int(host_indices[i * page_size].item()) for i in range(len(keys))]

    def _write_pool(self, name, pool, keys, host_indices):
        offsets = self._host_offsets(pool, keys, host_indices)
        if offsets is None:
            logger.error("Xavier %s host indices differ from page keys", name)
            return [False] * len(keys)
        if self._layouts[str(name)]["page_bytes"] == 0:
            pages = [b""] * len(keys)
        else:
            pages = [
                pool.get_data_page(offset, flat=True)
                .contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
                for offset in offsets
            ]
        stored, _ = self._pool_rpc(
            "put_pool_pages", [self._entry(name, key) for key in keys], pages
        )
        return [bool(value) for value in stored]

    def _read_pool(self, name, pool, keys, host_indices):
        offsets = self._host_offsets(pool, keys, host_indices)
        if offsets is None:
            logger.error("Xavier %s host indices differ from page keys", name)
            return [False] * len(keys), {}, 0
        layout = self._layouts[str(name)]
        pages, writers = self._pool_rpc(
            "get_pool_pages", [self._entry(name, key) for key in keys]
        )
        results = []
        nbytes = 0
        dtype = getattr(torch, layout["dtype"])
        for offset, page in zip(offsets, pages):
            if page is None or self._closed or len(page) != layout["page_bytes"]:
                results.append(False)
                continue
            if layout["page_bytes"]:
                pool.set_from_flat_data_page(
                    offset,
                    torch.frombuffer(bytearray(page), dtype=torch.uint8).view(dtype),
                )
                nbytes += len(page)
            results.append(True)
        return results, writers, nbytes

    def _registered(self, name):
        name = str(name)
        if name == _KV:
            return self.mem_pool_host
        pool = getattr(self, "registered_pools", {}).get(name)
        if pool is None:
            for key, value in getattr(self, "registered_pools", {}).items():
                if str(key) == name:
                    return value
            raise ValueError(f"Unregistered Xavier host pool: {name}")
        return pool

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        from sglang.srt.mem_cache.hicache_storage import (
            PoolHitPolicy,
            PoolTransferResult,
        )

        kv_pages = self._pool_exists(keys)
        hit_count = {_KV: kv_pages} if kv_pages else {}
        restorable = list(range(1, kv_pages + 1))
        for transfer in pool_transfers or []:
            if not restorable:
                break
            present = self._present(transfer.name, keys[:kv_pages])
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = present.index(False) if False in present else kv_pages
                allowed = set(range(1, boundary + 1))
            elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                allowed = {
                    length
                    for length in range(1, kv_pages + 1)
                    if all(present[max(0, length - trailing) : length])
                }
                boundary = max(allowed, default=0)
            else:
                raise ValueError(
                    f"Xavier does not support pool hit policy {transfer.hit_policy}"
                )
            if boundary:
                hit_count[transfer.name] = boundary
            restorable = [length for length in restorable if length in allowed]
        return PoolTransferResult(
            restorable[-1] if restorable else 0, hit_count, restorable
        )

    def _batch_io_v2(self, transfers, write):
        results = {}
        writers = {}
        nbytes = 0
        for transfer in transfers:
            pool = self._registered(transfer.name)
            keys = list(transfer.keys or [])
            if write:
                results[transfer.name] = self._write_pool(
                    transfer.name, pool, keys, transfer.host_indices
                )
                continue
            ok, read_writers, read_bytes = self._read_pool(
                transfer.name, pool, keys, transfer.host_indices
            )
            results[transfer.name] = ok
            nbytes += read_bytes
            for writer, count in read_writers.items():
                writers[writer] = writers.get(writer, 0) + count
        if not write and nbytes:
            self._log_restore(results, writers, nbytes)
        return results

    def _log_restore(self, results, writers, nbytes):
        log = logger.info if self._tp_rank == 0 else logger.debug
        log(
            "Xavier restored %s (%d bytes) on TP rank %d of replica %s from "
            "pages written by %s",
            ", ".join(f"{name}={sum(ok)}" for name, ok in results.items()),
            nbytes,
            self._tp_rank,
            self._config.get("replica", "?"),
            ", ".join(f"{name or '?'}:{count}" for name, count in writers.items()),
        )

    def batch_get_v2(self, transfers, extra_info=None):
        return self._batch_io_v2(transfers, write=False)

    def batch_set_v2(self, transfers, extra_info=None):
        return self._batch_io_v2(transfers, write=True)

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
