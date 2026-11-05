# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

from types import SimpleNamespace

import pytest

from ..cache_lifecycle import (
    CacheSessionRegistry,
    order_free_session_blocks,
    validate_cache_session,
)
from ..transport import validate_gpu_cache_budget


def test_single_replica_resident_prefix_mode_needs_no_retained_history_budget():
    assert validate_gpu_cache_budget(0, True, 1) == 0
    with pytest.raises(ValueError):
        validate_gpu_cache_budget(1024, True, 1)
    with pytest.raises(ValueError):
        validate_gpu_cache_budget(0, False, 1)


class _FreeQueue(list):
    def prepend_n(self, blocks):
        self[:0] = blocks

    def append_n(self, blocks):
        self.extend(blocks)


def _pool(size=8, active=()):
    blocks = [
        SimpleNamespace(block_id=i, ref_cnt=int(i in active), is_null=False)
        for i in range(size)
    ]
    return SimpleNamespace(
        blocks=blocks, free_block_queue=_FreeQueue(b for b in blocks if not b.ref_cnt)
    )


def _offer(session, *blocks):
    return SimpleNamespace(session_id=session, block_ids=set(blocks))


def test_waiting_prefix_survives_other_work_without_reserving_capacity():
    pool = _pool()
    before = len(pool.free_block_queue)
    assert order_free_session_blocks(
        pool, [_offer("agent", 0, 1, 2)], {"agent": "waiting"}
    ) == (3, 0)
    assert len(pool.free_block_queue) == before
    assert all(block.ref_cnt == 0 for block in pool.blocks)
    # Five allocations can now run without overwriting the waiting prefix.
    allocated = [pool.free_block_queue.pop(0).block_id for _ in range(5)]
    assert allocated == [3, 4, 5, 6, 7]
    assert {b.block_id for b in pool.free_block_queue} == {0, 1, 2}


def test_finished_content_is_reclaimed_before_shared_waiting_prefixes():
    pool = _pool(active=(4,))
    offers = [_offer("done", 0, 1, 4), _offer("waiting", 1, 2), _offer(None, 0)]
    assert order_free_session_blocks(
        pool, offers, {"done": "finished", "waiting": "waiting"}
    ) == (2, 0)
    assert pool.blocks[4].ref_cnt == 1
    assert pool.blocks[4] not in pool.free_block_queue
    assert {b.block_id for b in pool.free_block_queue[-2:]} == {1, 2}
    order_free_session_blocks(pool, [_offer("done", 5, 6)], {"done": "finished"})
    assert {b.block_id for b in pool.free_block_queue[:2]} == {5, 6}
    assert len({b.block_id for b in pool.free_block_queue}) == 7


def test_missing_or_expired_hints_preserve_engine_eviction_order():
    pool = _pool()
    before = list(pool.free_block_queue)
    assert order_free_session_blocks(pool, [_offer("gone", 0, 1)], {}) == (0, 0)
    assert pool.free_block_queue == before


def test_late_waiting_cannot_override_resume_or_finish():
    registry = CacheSessionRegistry()
    assert registry.update({"id": "a", "revision": 2, "phase": "active"})
    assert not registry.update({"id": "a", "revision": 1, "phase": "waiting"})
    assert not registry.update({"id": "a", "revision": 2, "phase": "waiting"})
    assert registry.update({"id": "a", "revision": 3, "phase": "finished"})
    assert not registry.update({"id": "a", "revision": 2, "phase": "waiting"})
    assert registry.get(["a"]) == {"a": "finished"}


def test_crashed_clients_expire_and_registry_capacity_is_bounded():
    now = [0.0]
    registry = CacheSessionRegistry(capacity=2, ttl=5, clock=lambda: now[0])
    for key in ("a", "b", "c"):
        registry.update({"id": key, "revision": 1, "phase": "waiting"})
    assert set(registry.states) == {"b", "c"}
    now[0] = 5.0
    assert registry.get(["b", "c"]) == {}
    registry.update({"id": "d", "revision": 0, "phase": "active"})
    assert list(registry.states) == ["d"]


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"id": " ", "revision": 0},
        {"id": "a", "revision": True},
        {"id": "a", "revision": -1},
        {"id": "a", "revision": 2**63},
        {"id": "a", "revision": 1.0},
        {"id": "a" * 129, "revision": 0},
    ],
)
def test_invalid_cache_session_is_rejected(value):
    with pytest.raises(ValueError):
        validate_cache_session(value)


def test_chat_metadata_cannot_set_control_phase():
    assert (
        validate_cache_session({"id": "a", "revision": 0, "phase": "finished"})["phase"]
        == "active"
    )
    with pytest.raises(ValueError):
        validate_cache_session(
            {"id": "a", "revision": 0, "phase": "unknown"}, control=True
        )
