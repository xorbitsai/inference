# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
import threading

import pytest
import torch

from ..direct_history import GPUHistoryStore
from .test_direct_handoff import direct_runtime


def history_runtime(monkeypatch):
    r = direct_runtime(monkeypatch)
    r._init_history(32)
    r.history_retention_seconds = 10
    r.caches["K"].copy_(torch.arange(16).reshape(8, 2))
    return r


@pytest.mark.asyncio
async def test_snapshot_waits_for_fence_and_survives_engine_reuse(monkeypatch):
    r = history_runtime(monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def fence(*args):
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(torch.cuda, "synchronize", fence)
    original = r.caches["K"][[1, 2]].clone()
    r.register_direct("t", "p", [1, 2], [101, 102])
    r.release_direct("t")
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        assert not r.history.ready
        assert not r.poll_direct()
        assert r.direct_requests["t"].retaining
    finally:
        release.set()
    await r._history_task
    assert r.poll_direct() == {"p"}
    r.caches["K"].zero_()
    assert r.reserve_history("h", [101, 102, 999]) == [101, 102]
    await r.load_history([{0: {"K": {101: 5, 102: 3}}}], ["h"])
    assert torch.equal(r.caches["K"][[5, 3]], original)
    assert not r.history.leases and not r._history_leases
    assert r.metrics["history_loaded_requests"] == 1
    assert r.history.stats()["cpu_blocks"] == 0


@pytest.mark.asyncio
async def test_busy_writer_skips_next_request_without_backlog(monkeypatch):
    r = history_runtime(monkeypatch)
    await r.send_lock.acquire()
    r.register_direct("a", "a", [1], [101])
    r.release_direct("a")
    writer = r._history_task
    r.register_direct("b", "b", [2], [102])
    r.release_direct("b")
    assert r.poll_direct() == {"b"}
    assert r._history_task is writer
    assert r.metrics["history_skipped_requests"] == 1
    r.send_lock.release()
    await writer
    assert r.poll_direct() == {"a"}


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["held_budget", "deadline", "expired"])
async def test_optional_retention_releases_engine_when_skipped(monkeypatch, reason):
    r = history_runtime(monkeypatch)
    r.register_direct(
        "t", "p", [1], [101], held_blocks=9 if reason == "held_budget" else 1
    )
    if reason == "deadline":
        r.history_retention_seconds = 0
    if reason == "expired":
        r.direct_requests["t"].deadline = 0
        assert r.poll_direct() == {"p"}
    else:
        r.release_direct("t")
        if r._history_task:
            await r._history_task
        assert r.poll_direct() == {"p"}
    assert not r.history.blocks and not r.direct_requests


def test_gpu_eviction_respects_lease_without_cpu_demotion():
    store = GPUHistoryStore(8, 4, torch.device("cpu"))

    def save(keys):
        store.stage_blocks(keys, {"K": torch.ones(len(keys), 2, dtype=torch.bfloat16)})
        store.publish(keys, {"K"})

    save([1, 2])
    assert store.reserve("h", [1])
    save([3])
    assert store.ready == {1, 3}
    assert store.reserve("j", [3])
    save([4])
    assert store.ready == {1, 3}
    assert store.stats()["demotions"] == store.stats()["cpu_blocks"] == 0
    assert store.stats()["gpu_reserved_bytes"] == 8


@pytest.mark.asyncio
async def test_failed_staging_never_publishes_partial_snapshot(monkeypatch):
    r = history_runtime(monkeypatch)
    original = r.history.stage_blocks

    def fail(keys, layers):
        original(keys, layers)
        raise RuntimeError("staging failed")

    monkeypatch.setattr(r.history, "stage_blocks", fail)
    r.register_direct("t", "p", [1, 2], [101, 102])
    r.release_direct("t")
    await r._history_task
    assert not r.history.ready and not r.history.blocks
    assert r.poll_direct() == {"p"}
    assert r.metrics["history_failures"] == 1


@pytest.mark.asyncio
async def test_invalid_history_read_releases_lease(monkeypatch):
    r = history_runtime(monkeypatch)
    r.register_direct("t", "p", [1], [101])
    r.release_direct("t")
    await r._history_task
    r.reserve_history("h", [101])
    with pytest.raises(ValueError, match="outside lease"):
        await r.load_history([{0: {"K": {102: 5}}}], ["h"])
    assert not r._history_leases and not r.history.leases


def test_producer_history_maps_prefix_after_local_hit(connector):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    connector._direct_test = connector._history_enabled = True
    actor = SimpleNamespace(
        reserve_direct_history_v1=AsyncMock(), release_direct_history_v1=AsyncMock()
    )
    request = SimpleNamespace(request_id="p", prompt_token_ids=list(range(65)))
    keys = [
        key for key, _ in connector._build_xavier_hashes(request.prompt_token_ids[:64])
    ]
    actor.reserve_direct_history_v1.return_value = keys[1:3]
    connector._get_transfer_ref = AsyncMock(return_value=actor)
    assert connector.get_num_new_matched_tokens(request, 16) == (32, True)
    load = connector._requests_need_load["p"]
    actor.reserve_direct_history_v1.assert_awaited_once_with(load.lease, keys[1:])
    connector.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda: [[4, 5, 6, 7]]), 32
    )
    assert connector._requests_need_load["p"].local_transfers_by_group == {
        0: {connector._rank: {keys[1]: 5, keys[2]: 6}}
    }


def test_unallocated_history_abort_releases_lease(connector, monkeypatch):
    import sys
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.request",
        SimpleNamespace(RequestStatus=SimpleNamespace(FINISHED_ABORTED="aborted")),
    )
    connector._direct_test = connector._history_enabled = True
    actor = SimpleNamespace(
        reserve_direct_history_v1=AsyncMock(return_value=[101]),
        release_direct_history_v1=AsyncMock(),
    )
    connector._get_transfer_ref = AsyncMock(return_value=actor)
    request = SimpleNamespace(
        request_id="p",
        prompt_token_ids=list(range(33)),
        status="aborted",
        kv_transfer_params={"do_remote_decode": True},
    )
    assert connector.get_num_new_matched_tokens(request, 0) == (16, True)
    lease = connector._requests_need_load["p"].lease
    assert connector.request_finished(request, []) == (False, None)
    actor.release_direct_history_v1.assert_awaited_once_with(lease)
    assert not connector._requests_need_load


@pytest.mark.asyncio
async def test_close_stops_waiting_writer_and_reclaims_snapshots(monkeypatch):
    r = history_runtime(monkeypatch)
    await r.recv_lock.acquire()
    r.register_direct("t", "p", [1], [101])
    r.release_direct("t")
    await r.close()
    assert not r.direct_requests and not r.history.blocks
    assert not r._history_leases
    r.recv_lock.release()


@pytest.mark.asyncio
async def test_cancelled_restore_holds_lease_until_cuda_drains(monkeypatch):
    r = history_runtime(monkeypatch)
    r.register_direct("t", "p", [1], [101])
    r.release_direct("t")
    await r._history_task
    r.reserve_history("h", [101])
    entered, release = threading.Event(), threading.Event()

    def fence(*args):
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(torch.cuda, "synchronize", fence)
    task = asyncio.create_task(r.run(r.load_history, [{0: {"K": {101: 5}}}], ["h"]))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        r._history_leases["h"] = 0
        r._expire_history_leases()
        r.release_history("h")
        assert "h" in r.history.leases and not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not r.history.leases and not r._history_leases
    assert r.caches["K"][5].tolist() == [2, 3]


def test_local_prefix_partial_tail_does_not_query_history(connector):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    connector._direct_test = connector._history_enabled = True
    connector._get_transfer_ref = AsyncMock()
    request = SimpleNamespace(request_id="p", prompt_token_ids=list(range(41)))
    assert connector.get_num_new_matched_tokens(request, 32) == (0, False)
    connector._get_transfer_ref.assert_not_called()
    assert not connector._requests_need_load


@pytest.mark.asyncio
async def test_full_history_preserves_hot_content_until_new_content_repeats(
    monkeypatch,
):
    r = history_runtime(monkeypatch)
    r._init_history(8)
    r.history_retention_seconds = 10

    async def finish(ticket, blocks, hashes):
        r.register_direct(ticket, ticket, blocks, hashes)
        r.release_direct(ticket)
        if r._history_task:
            await r._history_task
        assert r.poll_direct() == {ticket}

    await finish("warm", [1, 2], [101, 102])
    assert r.history.ready == {101, 102}
    await finish("cold", [3, 4], [103, 104])
    assert r.history.ready == {101, 102}
    assert r.metrics["history_admission_rejected_blocks"] == 2
    await finish("repeat", [3, 4], [103, 104])
    assert r.history.ready == {103, 104}
    assert r.reserve_history("hit", [103, 104]) == [103, 104]
    assert not r._history_probation
    assert r.metrics["history_saved_blocks"] == 4
    assert r.history.stats()["cpu_blocks"] == 0


@pytest.mark.asyncio
async def test_probation_is_bounded_and_forgets_old_cold_content(monkeypatch):
    r = history_runtime(monkeypatch)
    r._init_history(4)
    r.history_retention_seconds = 10
    for key in (100, 101, 102, 103, 101):
        r.register_direct(str(key), str(key), [1], [key])
        r.release_direct(str(key))
        if r._history_task:
            await r._history_task
        r.poll_direct()
    assert r.history.ready == {100}
    assert list(r._history_probation) == [103, 101]
    await r.close()
    assert not r._history_probation


@pytest.mark.asyncio
async def test_snapshot_chunks_respect_byte_budget(monkeypatch):
    r = history_runtime(monkeypatch)
    r.history_chunk_bytes = r.history_fill_chunk_bytes = 8
    r.register_direct("t", "p", [1, 2, 3, 4, 5], [101, 102, 103, 104, 105])
    original = r.history.stage_blocks
    sizes = []

    def stage(keys, layers):
        sizes.append(sum(t.numel() * t.element_size() for t in layers.values()))
        return original(keys, layers)

    monkeypatch.setattr(r.history, "stage_blocks", stage)
    r.release_direct("t")
    await r._history_task
    assert sizes == [8, 8, 4]
    assert r.metrics["history_save_chunks"] == 3
    assert r.history.ready == {101, 102, 103, 104, 105}
    assert r.poll_direct() == {"p"}


@pytest.mark.asyncio
async def test_fill_batches_grow_but_replacement_batches_stay_small(monkeypatch):
    r = history_runtime(monkeypatch)
    r._init_history(20)
    r.history_retention_seconds = 10
    r.history_fill_chunk_bytes = 16
    r.history_chunk_bytes = 8
    sizes = []
    original = r.history.stage_blocks

    def stage(keys, layers):
        sizes.append(len(keys))
        return original(keys, layers)

    monkeypatch.setattr(r.history, "stage_blocks", stage)
    for ticket, hashes in (
        ("fill", list(range(101, 106))),
        ("observe", list(range(201, 206))),
        ("replace", list(range(201, 206))),
    ):
        r.register_direct(ticket, ticket, [1, 2, 3, 4, 5], hashes)
        r.release_direct(ticket)
        if r._history_task:
            await r._history_task
        assert r.poll_direct() == {ticket}
    assert sizes == [4, 1, 2, 2, 1]
    assert r.history.ready == set(range(201, 206))
