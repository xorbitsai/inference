# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Ordinary vLLM requests exchanging canonical prefix pages with other engines."""

import logging
import time
from dataclasses import dataclass, field

import xoscar as xo

from ...xavier.backends.torch.pd import canonical_vllm_views
from ...xavier.backends.torch.prefix import pack_pages, unpack_pages
from ...xavier.contract import KVCacheContract
from ...xavier.prefix import prefix_keys
from .v1_connector import XavierConnector, XavierConnectorMetadata, XavierKVSchema

logger = logging.getLogger(__name__)
PREFIX_RPC_TIMEOUT = 1.0
PREFIX_RPC_BACKOFF = 5.0


@dataclass
class PrefixMetadata(XavierConnectorMetadata):
    prefix_loads: list = field(default_factory=list)


class PrefixConnector(XavierConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self.contract = KVCacheContract.from_dict(self._xavier_config["contract"])
        if len(kv_cache_config.kv_cache_groups) != 1 or self._has_recurrent_cache:
            raise ValueError(
                "Cross-engine prefix caching requires one full-attention KV group"
            )
        self._prefix_ref = None
        self._prefix_retry_at = 0.0
        self._prefix_pending = {}
        self._prefix_allocated = {}
        self._prefix_views = {}

    async def _prefix_rpc(self, method, *args):
        import asyncio

        if time.monotonic() < self._prefix_retry_at:
            raise ConnectionError("Xavier prefix storage is in backoff")

        async def invoke():
            if self._prefix_ref is None:
                self._prefix_ref = await xo.actor_ref(
                    address=self._xavier_config["address"],
                    uid=self._xavier_config["uid"],
                )
                await self._prefix_ref.configure(self.contract.to_dict())
            return await getattr(self._prefix_ref, method)(*args)

        try:
            return await asyncio.wait_for(invoke(), timeout=PREFIX_RPC_TIMEOUT)
        except Exception:
            self._prefix_ref = None
            self._prefix_retry_at = time.monotonic() + PREFIX_RPC_BACKOFF
            raise

    def _rpc(self, method, *args):
        return self._call(self._prefix_rpc(method, *args))

    def register_kv_caches(self, kv_caches):
        self._registered_kv_caches = dict(kv_caches)
        self._prefix_views = canonical_vllm_views(
            kv_caches, self._num_cache_blocks, self.contract
        )
        self._kv_schema = XavierKVSchema(
            self._block_size,
            {
                name: (tuple(t.shape[1:]), t.dtype)
                for name, t in self._prefix_views.items()
            },
            self._cache_dtype,
        )
        self._rpc("get_stats")  # Fail launch if peers disagree about model semantics.

    def _build_xavier_hashes(self, token_ids, cache_salt=None):
        return [
            (int(key, 16), i)
            for i, key in enumerate(
                prefix_keys(token_ids, self._block_size, cache_salt)
            )
        ]

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        rid = request.request_id
        salt = getattr(request, "cache_salt", None)
        previous_salt = self._request_cache_salts.get(rid)
        self._request_cache_salts[rid] = salt
        if num_computed_tokens % self._block_size:
            self._prefix_pending.pop(rid, None)
            return 0, False
        start = num_computed_tokens // self._block_size
        pending = self._prefix_pending.get(rid)
        if pending is not None and pending[0] == start and previous_salt == salt:
            return len(pending[1]) * self._block_size, False
        self._prefix_pending.pop(rid, None)
        keys = prefix_keys(
            request.prompt_token_ids[:-1],
            self._block_size,
            getattr(request, "cache_salt", None),
        )
        page_bytes = self.contract.num_layers * self.contract.layer_nbytes
        pending_bytes = sum(
            len(pages) * page_bytes
            for _, pages in [
                *self._prefix_pending.values(),
                *self._prefix_allocated.values(),
            ]
        )
        available = (
            max(self._xavier_config["capacity_bytes"] - pending_bytes, 0) // page_bytes
        )
        try:
            pages = (
                self._rpc(
                    "get",
                    self.contract.fingerprint,
                    keys[start : start + min(available, 4096)],
                )
                if available and keys[start:]
                else []
            )
            if any(type(p) is not bytes or len(p) != page_bytes for p in pages):
                raise ValueError("Invalid shared-prefix KV page")
        except Exception:
            logger.warning(
                "Xavier prefix lookup failed; recomputing locally", exc_info=True
            )
            pages = []
        if pages:
            self._prefix_pending[rid] = (start, pages)
        return len(pages) * self._block_size, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self._request_cache_salts[request.request_id] = getattr(
            request, "cache_salt", None
        )
        pending = self._prefix_pending.pop(request.request_id, None)
        if pending is not None and num_external_tokens:
            start, pages = pending
            ids = list(blocks.get_block_ids()[0])
            if num_external_tokens != len(pages) * self._block_size or start + len(
                pages
            ) > len(ids):
                raise ValueError("vLLM allocated too few shared-prefix slots")
            self._prefix_allocated[request.request_id] = (
                ids[start : start + len(pages)],
                pages,
            )

    def build_connector_meta(self, scheduler_output):
        meta = PrefixMetadata()
        self._build_store_meta(scheduler_output, meta)
        scheduled = {req.req_id for req in scheduler_output.scheduled_new_reqs}
        scheduled.update(scheduler_output.scheduled_cached_reqs.req_ids)
        for rid in scheduled:
            load = self._prefix_allocated.pop(rid, None)
            if load is not None:
                meta.prefix_loads.append(load)
        return meta

    def start_load_kv(self, forward_context, **kwargs):
        for slots, pages in self._get_connector_metadata().prefix_loads:
            unpack_pages(self._prefix_views, slots, pages, self.contract)
            try:
                self._rpc("record_import", self.contract.fingerprint, len(pages))
            except Exception:
                logger.warning("Could not record Xavier prefix import", exc_info=True)

    def wait_for_save(self):
        for request in self._get_connector_metadata().store_requests:
            page_bytes = self.contract.num_layers * self.contract.layer_nbytes
            batch_size = max(1, min(128, 32 * 1024**2 // page_bytes))
            count = min(
                len(request.block_ids),
                self._xavier_config["capacity_bytes"] // page_bytes,
                4096,
            )
            keys = [f"{key:064x}" for key in request.block_hashes[:count]]
            try:
                start = self._rpc("exists", self.contract.fingerprint, keys)
            except Exception:
                logger.warning("Xavier prefix publication failed", exc_info=True)
                return
            for offset in range(start, count, batch_size):
                end = min(offset + batch_size, count)
                slots = request.block_ids[offset:end]
                try:
                    pages = pack_pages(self._prefix_views, slots, self.contract)
                    self._rpc(
                        "put", self.contract.fingerprint, keys[:end], pages, offset
                    )
                except Exception:
                    logger.warning("Xavier prefix publication failed", exc_info=True)
                    break

    def get_finished(self, finished_req_ids):
        return None, None

    def request_finished(self, request, block_ids, block_ids_by_group=None):
        rid = request.request_id
        self._prefix_pending.pop(rid, None)
        self._prefix_allocated.pop(rid, None)
        self._request_cache_salts.pop(rid, None)
        self._chunked_prefill.pop(rid, None)
        return False, None
