# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Share vLLM-owned offload chunks without interpreting KV or recurrent state."""

import asyncio
import hashlib
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np
import xoscar as xo
from vllm import __version__ as VLLM_VERSION
from vllm.model_executor.layers.mamba.mamba_utils import get_conv_state_layout
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import JobResult, SecondaryTierManager, TransferJob
from vllm.v1.metrics.cache_hit_source import CacheHitSource

from ...xavier.actor_loop import acquire_actor_loop, release_actor_loop
from ...xavier.contract import fingerprint_metadata

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec
    from vllm.v1.kv_offload.tiering.backpressure import BackpressureDetector

logger = logging.getLogger(__name__)
_RPC_TIMEOUT = 5
_MAX_BATCH_BYTES = 16 * 1024**2


class _XavierLookup(AsyncLookupManager):
    def __init__(self, tier: "XavierOffloadTier"):
        self._tier = tier
        super().__init__(tier.tier_type)

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        return self._tier._executor.submit(
            self._tier._contains, keys, req_context
        ).result()


class XavierOffloadTier(SecondaryTierManager):
    medium = Medium.CPU
    cache_hit_source = CacheHitSource.P2P

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        address: str,
        uid: bytes,
        model_hash: str,
        state_dtypes: list[str],
        backpressure_detector: Optional["BackpressureDetector"] = None,
    ) -> None:
        super().__init__(
            offloading_spec, primary_kv_view, tier_type, backpressure_detector
        )
        self.locality = Locality.REMOTE
        self._address = address
        self._uid = uid
        # One chunk occupies a row in the primary tier. Keep group IDs in
        # OffloadKey and include the actual host layout in the namespace.
        self._rows = np.asarray(primary_kv_view)
        self._page_bytes = self._rows.shape[1]
        mapper = FileMapper.from_offloading_spec("", offloading_spec)
        self._namespace = fingerprint_metadata(
            {
                "vllm": VLLM_VERSION,
                "model": model_hash,
                "layout": mapper.get_run_config(),
                "cache_layout": offloading_spec.config.kv_cache_layout,
                "worker_bytes": offloading_spec.config.worker_kv_bytes_per_block,
                "chunk_bytes": self._page_bytes,
                "state_dtypes": state_dtypes,
                "conv_state_layout": get_conv_state_layout(),
            }
        )
        self._batch_size = max(1, min(128, _MAX_BATCH_BYTES // self._page_bytes))
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="xavier-kv"
        )
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ref = None
        self._jobs: dict[Future[JobResult], TransferJob] = {}
        self._lookup = _XavierLookup(self)

    def _rpc(self, method: str, *args):
        # Actor refs and clients belong to the executor thread and its loop.
        async def call():
            if self._ref is None:
                ref = await xo.actor_ref(address=self._address, uid=self._uid)
                await ref.configure_pages(self._namespace, self._page_bytes)
                self._ref = ref
            return await getattr(self._ref, method)(
                *((self._namespace, *args) if method != "bind_cache_session" else args)
            )

        if self._loop is None:
            self._loop = acquire_actor_loop()
        try:
            return self._loop.run_until_complete(asyncio.wait_for(call(), _RPC_TIMEOUT))
        except Exception:
            self._ref = None
            raise

    @staticmethod
    def _key(key: OffloadKey) -> str:
        # The actor uses SHA-256 keys. OffloadKey also includes the group ID.
        return hashlib.sha256(key).hexdigest()

    def _contains(
        self, keys: list[OffloadKey], req_context: Optional[ReqContext] = None
    ) -> list[bool]:
        from ...xavier.cache_lifecycle import CACHE_SESSION_ARG

        session = (
            (req_context.kv_transfer_params or {}).get(CACHE_SESSION_ARG)
            if req_context
            else None
        )
        if isinstance(session, dict):
            try:
                self._rpc(
                    "bind_cache_session",
                    session,
                    [self._key(key) for key in keys[:4096]],
                )
            except Exception:
                logger.debug("Agent offload binding unavailable", exc_info=True)
        hits = []
        for start in range(0, len(keys), 4096):
            hits.extend(
                self._rpc(
                    "contains", [self._key(k) for k in keys[start : start + 4096]]
                )
            )
        return hits

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        found = self._lookup.lookup(key, req_context)
        if found is None:
            return LookupResult.RETRY
        return LookupResult.HIT if found else LookupResult.MISS

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup.cleanup(req_context.req_id)

    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup.flush()

    def submit_store(self, job_metadata: TransferJob) -> None:
        self._jobs[self._executor.submit(self._transfer, job_metadata)] = job_metadata

    submit_load = submit_store

    def _transfer(self, job: TransferJob) -> JobResult:
        started = time.monotonic()
        keys = [self._key(key) for key in job.keys]
        transferred = 0
        try:
            for start in range(0, len(keys), self._batch_size):
                end = start + self._batch_size
                batch = keys[start:end]
                rows = job.chunk_ids[start:end]
                if job.is_promotion:
                    pages = self._rpc("get", batch)
                    if len(pages) != len(batch):
                        # An eviction after lookup makes this promotion a miss.
                        # vLLM keeps the primary rows private until completion.
                        break
                    for row, page in zip(rows, pages):
                        self._rows[row] = np.frombuffer(page, dtype=np.uint8)
                else:
                    published = self._rpc(
                        "put", batch, [self._rows[row].tobytes() for row in rows]
                    )
                    if not all(published):
                        break
                transferred += len(batch)
        except Exception:
            logger.warning("Xavier offload transfer failed", exc_info=True)
        return JobResult(
            job_id=job.job_id,
            success=transferred == len(keys),
            transfer_time=time.monotonic() - started,
            transfer_bytes=transferred * self._page_bytes,
        )

    def get_finished_jobs(self) -> Iterable[JobResult]:
        for future in list(self._jobs):
            if future.done():
                job = self._jobs.pop(future)
                result = future.result()
                if job.is_promotion and not result.success:
                    self._lookup.mark_miss(job.keys)
                yield result

    def has_pending_work(self) -> bool:
        return bool(self._jobs)

    def drain_jobs(self) -> None:
        wait(self._jobs)

    def shutdown(self) -> None:
        self._lookup.shutdown()
        self.drain_jobs()
        if self._loop is not None:
            self._executor.submit(release_actor_loop, self._loop).result()
        self._executor.shutdown()
        # Release the NumPy view before vLLM unmaps the primary cache.
        del self._rows
