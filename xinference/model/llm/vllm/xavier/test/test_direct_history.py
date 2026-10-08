# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
import threading

import pytest
import torch

from ..direct_history import HistoryStore
from .test_direct_handoff import direct_runtime


def history_runtime(monkeypatch):
    r = direct_runtime(monkeypatch)
    r.store.cpu_capacity = 0
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
@pytest.mark.parametrize("reason", ["deadline", "expired"])
async def test_optional_retention_releases_engine_when_skipped(monkeypatch, reason):
    r = history_runtime(monkeypatch)
    r.register_direct("t", "p", [1], [101])
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
    store = HistoryStore(8, 4, torch.device("cpu"))

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

    connector._direct_handoff = connector._history_enabled = True
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
    connector._direct_handoff = connector._history_enabled = True
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

    connector._direct_handoff = connector._history_enabled = True
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
    r.store.cpu_capacity = 0
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
    r.store.cpu_capacity = 0
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
    r.store.cpu_capacity = 0
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


@pytest.mark.parametrize("matched_blocks", [1, 161, 277, 278])
def test_fixed_history_prefix_length_and_destination_mapping(connector, matched_blocks):
    """Keep the 4441-token divergence reproduction's exact prefix boundaries."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    connector._direct_handoff = connector._history_enabled = True
    request = SimpleNamespace(request_id="p", prompt_token_ids=list(range(4441)))
    keys = [
        key for key, _ in connector._build_xavier_hashes(request.prompt_token_ids[:-1])
    ]
    actor = SimpleNamespace(
        reserve_direct_history_v1=AsyncMock(return_value=keys[:matched_blocks])
    )
    connector._get_transfer_ref = AsyncMock(return_value=actor)
    expected = min(matched_blocks * 16, 4440)
    assert connector.get_num_new_matched_tokens(request, 0) == (expected, True)
    # Non-contiguous physical allocation must preserve logical token order.
    destinations = list(reversed(range(278)))
    connector.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda: [destinations]), expected
    )
    load = connector._requests_need_load["p"]
    assert load.local_transfers_by_group == {
        0: {connector._rank: dict(zip(keys[:matched_blocks], destinations))}
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("prefix_blocks", [1, 161, 277, 278])
@pytest.mark.parametrize("gpu_blocks", [5, 278])
async def test_fixed_prefix_restore_preserves_raw_bits(
    monkeypatch, device, prefix_blocks, gpu_blocks
):
    """Real CUDA fences when available; CPU CI exercises the same history path."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    synchronize = torch.cuda.synchronize
    r = direct_runtime(monkeypatch)
    if device == "cuda":
        monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    r.device = torch.device(device)
    # 24 layer tensors, each including both K and V, with arbitrary BF16 bits.
    generator = torch.Generator().manual_seed(42)
    r.caches = {
        str(layer): torch.randint(
            -32768, 32768, (280, 2, 16, 2), dtype=torch.int16, generator=generator
        )
        .view(torch.bfloat16)
        .to(device)
        for layer in range(24)
    }
    originals = {
        name: cache[:prefix_blocks].clone() for name, cache in r.caches.items()
    }
    block_bytes = sum(
        cache[0].numel() * cache.element_size() for cache in r.caches.values()
    )
    r.store.cpu_capacity = 278
    r._init_history(gpu_blocks * block_bytes)
    r.history_retention_seconds = 30
    r.slab_bytes = 8 * 1024 * 1024
    keys = list(range(1000, 1000 + prefix_blocks))
    r.register_direct("save", "request", list(range(prefix_blocks)), keys)
    r.release_direct("save")
    await r._history_task
    assert r.poll_direct() == {"request"}
    assert r.metrics["history_failures"] == 0
    # Engine reuses every original slot before restoring to a reversed mapping.
    for cache in r.caches.values():
        cache.zero_()
    assert r.reserve_history("restore", keys + [-1]) == keys
    destinations = list(reversed(range(2, prefix_blocks + 2)))
    await r.load_history(
        [{0: {name: dict(zip(keys, destinations)) for name in r.caches}}], ["restore"]
    )
    for name, cache in r.caches.items():
        assert torch.equal(
            cache[destinations].view(torch.int16), originals[name].view(torch.int16)
        )
        assert not cache[:2].count_nonzero()
    assert not r.history.leases and not r._history_leases
    assert r.history.stats()["cpu_blocks"] == max(0, prefix_blocks - gpu_blocks)
    assert r.metrics["history_cpu_hit_blocks"] == max(0, prefix_blocks - gpu_blocks)


@pytest.mark.asyncio
@pytest.mark.parametrize("device", ["cpu", "cuda"])
async def test_history_spills_to_cpu_and_restores_mixed_prefix(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    synchronize = torch.cuda.synchronize
    r = direct_runtime(monkeypatch)
    if device == "cuda":
        monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    r.device = torch.device(device)
    r.caches = {
        name: (torch.arange(64, dtype=torch.int16).reshape(8, 2, 4) + shift)
        .view(torch.bfloat16)
        .to(device)
        for name, shift in [("K", 32620), ("V", -32768)]
    }
    block_bytes = 32
    r.store.cpu_capacity = 2
    r._init_history(2 * block_bytes)
    r.history_retention_seconds = 10
    original = {name: cache[:4].clone() for name, cache in r.caches.items()}
    r.register_direct("save", "p", [0, 1, 2, 3], [100, 101, 102, 103])
    r.release_direct("save")
    await r._history_task
    assert r.poll_direct() == {"p"}
    assert r.history.tiers[100] == "gpu"
    assert r.history.counts == {"gpu": 2, "cpu": 2}
    assert r.history.stats()["demotions"] == 2
    assert r.reserve_history("read", [100, 101, 102, 103, 999]) == [100, 101, 102, 103]
    for cache in r.caches.values():
        cache.zero_()
    await r.load_history(
        [{0: {name: {100: 7, 101: 4, 102: 6, 103: 5} for name in r.caches}}], ["read"]
    )
    for name, cache in r.caches.items():
        assert torch.equal(
            cache[[7, 4, 6, 5]].view(torch.int16), original[name].view(torch.int16)
        )
        assert not cache[:4].count_nonzero()
    assert r.metrics["history_cpu_hit_blocks"] == 2
    assert r.metrics["history_gpu_hit_blocks"] == 2
    assert not r.history.leases and not r._history_leases


def test_history_eviction_only_after_both_tiers_full_and_preserves_leases():
    store = HistoryStore(8, 4, torch.device("cpu"), cpu_capacity=2)

    def save(key):
        store.stage_blocks(
            [key], {"K": torch.tensor([[key, -key]], dtype=torch.bfloat16)}
        )
        store.publish([key], {"K"})

    for key in [1, 2, 3, 4]:
        save(key)
    assert store.ready == {1, 2, 3, 4}
    assert store.reserve("reader", [1, 3])
    save(5)
    assert store.ready == {1, 3, 4, 5}
    assert store.tiers[1] == "cpu" and store.tiers[3] == "gpu"
    assert store.counts == {"gpu": 2, "cpu": 2}
    assert store.reserve("all", list(store.ready))
    save(6)
    assert store.ready == {1, 3, 4, 5}


@pytest.mark.asyncio
async def test_long_request_retains_bounded_leading_prefix(monkeypatch):
    r = history_runtime(monkeypatch)
    # Simulate a request larger than the retention cap, without allocating 8B KV.
    r.history_pending_bytes = 2 * r.history.block_bytes
    r.register_direct("t", "p", [0, 1, 2, 3], [100, 101, 102, 103])
    r.release_direct("t")
    await r._history_task
    assert r.history.ready == {100, 101}
    assert r.metrics["history_capacity_limited_blocks"] == 2
    assert not r.metrics["history_skipped_requests"]
    assert r.poll_direct() == {"p"}


@pytest.mark.asyncio
@pytest.mark.parametrize("closing", [False, True])
async def test_abandoned_retention_counts_remaining_candidates(monkeypatch, closing):
    r = history_runtime(monkeypatch)
    r.register_direct("t", "p", [1, 2], [101, 102])
    r.history_retention_seconds = 0
    r.release_direct("t")
    r._history_closing = closing
    await r._history_task
    reason = "closing" if closing else "deadline"
    assert r.metrics[f"history_{reason}_dropped_blocks"] == 2
    assert r.poll_direct() == {"p"}


@pytest.mark.asyncio
async def test_history_batch_failure_releases_all_leases_preserving_error(monkeypatch):
    r = history_runtime(monkeypatch)
    r.register_direct("t", "p", [1, 2], [101, 102])
    r.release_direct("t")
    await r._history_task
    assert r.reserve_history("a", [101]) == [101]
    assert r.reserve_history("b", [102]) == [102]
    original = r.release_history

    def release(lease):
        original(lease)
        if lease == "a":
            raise RuntimeError("cleanup error")

    monkeypatch.setattr(r, "release_history", release)
    with pytest.raises(ValueError, match="outside lease"):
        await r.load_history([{0: {"K": {999: 4}}}, {0: {"K": {102: 5}}}], ["a", "b"])
    assert not r.history.leases and not r._history_leases


@pytest.mark.asyncio
async def test_prefix_heads_survive_new_admission_and_reservation(monkeypatch):
    r = history_runtime(monkeypatch)
    r.store.cpu_capacity = 0
    r._init_history(12)  # three blocks
    r.history_retention_seconds = 10
    r.register_direct("a", "a", [0, 1, 2], [100, 101, 102])
    r.release_direct("a")
    await r._history_task
    assert list(r.history.blocks) == [102, 101, 100]
    assert r.reserve_history("lease", [100, 101, 102]) == [100, 101, 102]
    r.release_history("lease")
    r._history_probation[200] = None
    r.register_direct("b", "b", [3], [200])
    r.release_direct("b")
    await r._history_task
    assert r.reserve_history("read", [100, 101, 102]) == [100, 101]


@pytest.mark.asyncio
async def test_rejected_head_never_admits_repeated_tail(monkeypatch):
    r = history_runtime(monkeypatch)
    r._init_history(8)
    r.history_retention_seconds = 10
    r.register_direct("a", "a", [0, 1], [100, 101])
    r.release_direct("a")
    await r._history_task
    r._history_probation[202] = None
    r.register_direct("b", "b", [2, 3, 4], [200, 201, 202])
    r.release_direct("b")
    assert r.history.ready == {100, 101}
    assert r.metrics["history_admission_rejected_blocks"] == 3
    # A request longer than probation capacity retains its head, not its tail.
    r._history_probation_limit = 2
    r.register_direct("c", "c", [2, 3, 4], [300, 301, 302])
    r.release_direct("c")
    assert list(r._history_probation) == [301, 300]


@pytest.mark.asyncio
async def test_history_hit_counts_only_written_keys_once(monkeypatch):
    r = history_runtime(monkeypatch)
    r.register_direct("a", "a", [0, 1, 2], [100, 101, 102])
    r.release_direct("a")
    await r._history_task
    for lease in ("retry", "load"):
        assert r.reserve_history(lease, [100, 101, 102]) == [100, 101, 102]
        if lease == "retry":
            r.release_history(lease)
    assert r.history.metrics["gpu_hits"] == 0
    await r.load_history([{0: {"K": {101: 5}}}], ["load"])
    assert r.metrics["history_hit_blocks"] == 1
    assert r.metrics["history_gpu_hit_blocks"] == 1
    assert r.history.metrics["gpu_hits"] == 1


def test_prefill_retry_releases_previous_history_reservation(connector):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    connector._direct_handoff = connector._history_enabled = True
    actor = SimpleNamespace(
        reserve_direct_history_v1=AsyncMock(return_value=[123]),
        release_direct_history_v1=AsyncMock(),
    )
    connector._get_transfer_ref = AsyncMock(return_value=actor)
    request = SimpleNamespace(
        request_id="p", prompt_token_ids=list(range(33)), kv_transfer_params={}
    )
    assert connector.get_num_new_matched_tokens(request, 0) == (16, True)
    previous = connector._requests_need_load["p"].lease
    assert connector.get_num_new_matched_tokens(request, 0) == (16, True)
    actor.release_direct_history_v1.assert_awaited_once_with(previous)
    assert connector._requests_need_load["p"].lease != previous


@pytest.mark.asyncio
async def test_rejected_prefix_does_not_count_existing_tail_as_dropped(monkeypatch):
    r = history_runtime(monkeypatch)
    r._init_history(8)
    r.history_retention_seconds = 10
    r.register_direct("a", "a", [0, 1], [100, 101])
    r.release_direct("a")
    await r._history_task
    r.register_direct("b", "b", [2, 3], [200, 101])
    r.release_direct("b")
    assert r.metrics["history_admission_rejected_blocks"] == 1
    assert r.history.ready == {100, 101}
