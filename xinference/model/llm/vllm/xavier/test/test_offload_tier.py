# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def offload_runtime():
    pytest.importorskip("vllm.v1.kv_offload.tiering.base")
    from vllm.v1.kv_offload.base import ReqContext, make_offload_key
    from vllm.v1.kv_offload.tiering.base import TransferJob

    from ....xavier.backends.bytes.storage import XavierBytesCacheActor
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
        worker_kv_bytes_per_block=8,
    )
    cache = XavierBytesCacheActor(16)
    rows = np.arange(32, dtype=np.uint8).reshape(4, 8)
    tier = XavierOffloadTier(
        SimpleNamespace(config=config),
        memoryview(rows),
        "test",
        "unused",
        b"cache",
        "model",
        ["auto", "auto"],
    )
    cache.configure_pages(tier._namespace, rows.shape[1])
    tier._rpc = lambda method, *args: getattr(cache, method)(tier._namespace, *args)
    keys = [make_offload_key(b"same-prefix", group) for group in (0, 1)]
    context = ReqContext(req_id="request")

    def transfer(ids, promote, keys=keys):
        job = TransferJob(0, keys, np.array(ids), promote, context)
        if promote:
            tier.submit_load(job)
        else:
            tier.submit_store(job)
        tier.drain_jobs()
        return next(iter(tier.get_finished_jobs()))

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
