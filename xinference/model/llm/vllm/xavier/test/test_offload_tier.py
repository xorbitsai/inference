# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest


def _make_tier(rows, address, uid):
    from ..offload_tier import XavierOffloadTier

    config = SimpleNamespace(
        model=SimpleNamespace(name="hybrid", dtype="bfloat16"),
        groups=(
            SimpleNamespace(tokens_per_block=16, layer_names=("attention",)),
            SimpleNamespace(tokens_per_block=64, layer_names=("recurrent",)),
        ),
        cache=SimpleNamespace(tokens_per_hash=16),
        parallel=SimpleNamespace(tp_size=1, pp_size=1, dcp_size=1, pcp_size=1, rank=0),
        canonical_layout=False,
        kv_cache_layout="NHD",
        worker_kv_bytes_per_block=rows.shape[1],
    )
    return XavierOffloadTier(
        SimpleNamespace(config=config),
        memoryview(rows),
        "test",
        address,
        uid,
        "model",
        ["auto", "auto"],
    )


def _transfer(tier, keys, ids, promote, context):
    from vllm.v1.kv_offload.tiering.base import TransferJob

    job = TransferJob(0, keys, np.array(ids), promote, context)
    if promote:
        tier.submit_load(job)
    else:
        tier.submit_store(job)
    tier.drain_jobs()
    return next(iter(tier.get_finished_jobs()))


@pytest.fixture
def offload_runtime():
    pytest.importorskip("vllm.v1.kv_offload.tiering.base")
    from vllm.v1.kv_offload.base import ReqContext, make_offload_key

    from ....xavier.backends.bytes.storage import XavierBytesCacheActor

    cache = XavierBytesCacheActor(16)
    rows = np.arange(32, dtype=np.uint8).reshape(4, 8)
    tier = _make_tier(rows, "unused", b"cache")
    cache.configure_pages(tier._namespace, rows.shape[1])
    tier._rpc = lambda method, *args: getattr(cache, method)(tier._namespace, *args)
    keys = [make_offload_key(b"same-prefix", group) for group in (0, 1)]
    context = ReqContext(req_id="request")

    def transfer(ids, promote, keys=keys):
        return _transfer(tier, keys, ids, promote, context)

    try:
        yield tier, rows, keys, context, transfer
    finally:
        tier.shutdown()


def test_offload_preserves_distinct_groups_for_same_prefix(offload_runtime):
    tier, rows, keys, _, transfer = offload_runtime
    expected = rows[:2].copy()
    assert transfer([0, 1], promote=False).success
    assert tier._contains(keys) == [True, True]

    rows[:2] = 0
    assert transfer([2, 3], promote=True).success
    np.testing.assert_array_equal(rows[2:], expected)


def test_eviction_during_promotion_does_not_publish_rows(offload_runtime):
    tier, rows, keys, _, transfer = offload_runtime
    assert transfer([0], promote=False, keys=keys[:1]).success
    assert tier._contains(keys) == [True, False]

    before = rows[2:].copy()
    result = transfer([2, 3], promote=True)
    assert not result.success
    np.testing.assert_array_equal(rows[2:], before)


@pytest.mark.asyncio
async def test_offload_republishes_partially_evicted_disk_prefix(tmp_path):
    import xoscar as xo
    from xoscar.backends.allocate_strategy import ProcessIndex

    from ....xavier.tests.test_backends import _require_real_lmcache
    from ....xavier.tests.utils import DiskHistoryCacheActor

    _require_real_lmcache()
    pytest.importorskip("vllm.v1.kv_offload.tiering.base")
    from vllm.v1.kv_offload.base import ReqContext, make_offload_key

    page_bytes = 2 << 20
    rows = np.zeros((4, page_bytes), dtype=np.uint8)
    rows[0].fill(17)
    rows[1].fill(29)
    other_rows = rows.copy()
    keys = [make_offload_key(b"same-prefix", group) for group in (0, 1)]
    context = ReqContext(req_id="request")
    pool = await xo.create_actor_pool(
        "127.0.0.1", n_process=1, subprocess_start_method="spawn"
    )
    async with pool:
        address, uid = pool.external_address, b"offload-disk-republish"
        actor = await xo.create_actor(
            DiskHistoryCacheActor,
            dict(
                name="lmcache",
                namespace="offload-disk-republish",
                config=dict(
                    cpu_bytes=16 << 20,
                    disk_bytes=64 << 20,
                    disk_path=str(tmp_path),
                    read_timeout=10,
                ),
            ),
            address=address,
            allocate_strategy=ProcessIndex(1),
            uid=uid,
        )
        writer = _make_tier(rows, actor.address, uid)
        reader = _make_tier(other_rows, actor.address, uid)
        try:
            assert (
                await asyncio.to_thread(_transfer, writer, keys, [0, 1], False, context)
            ).success
            await actor.evict_l2_pages(writer._namespace, [writer._key(keys[0])])
            stored = await actor.get_stats()
            assert await actor.contains(
                writer._namespace, [writer._key(k) for k in keys]
            ) == [True, True]

            # Two replicas may export the recomputed page before either
            # attempts a read that would reconcile Xavier's directory.
            republished = await asyncio.gather(
                asyncio.to_thread(_transfer, writer, keys, [0, 1], False, context),
                asyncio.to_thread(_transfer, reader, keys, [0, 1], False, context),
            )
            assert all(result.success for result in republished)
            stats = await actor.get_stats()
            assert (
                stats["backend_stats"]["stored_pages"]
                > stored["backend_stats"]["stored_pages"]
            )
            await actor.force_disk_reads()
            result = await asyncio.to_thread(
                _transfer, reader, keys, [2, 3], True, context
            )
            assert result.success
            np.testing.assert_array_equal(other_rows[2:], rows[:2])
            assert (await actor.get_stats())["backend_stats"]["l2_hits"] == 2
        finally:
            await asyncio.to_thread(writer.shutdown)
            await asyncio.to_thread(reader.shutdown)
            await xo.destroy_actor(actor)
