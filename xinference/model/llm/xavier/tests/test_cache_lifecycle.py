# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

from types import SimpleNamespace

import pytest

from ...vllm.xavier.cache_lifecycle import order_free_session_blocks
from ..cache_lifecycle import CacheSessionRegistry, validate_cache_session
from ..transport import validate_gpu_cache_budget


def test_single_replica_resident_prefix_mode_needs_no_retained_history_budget():
    assert validate_gpu_cache_budget(0, True, 1) == 0
    with pytest.raises(ValueError):
        validate_gpu_cache_budget(1024, True, 1)
    with pytest.raises(ValueError):
        validate_gpu_cache_budget(0, False, 1)


class _FreeQueue(list):
    """Track linked membership like vLLM's doubly linked free queue."""

    def __init__(self, blocks):
        super().__init__(blocks)
        self._link()

    def _link(self):
        for index, block in enumerate(self):
            block.prev_free_block = self[index - 1] if index else object()
            block.next_free_block = (
                self[index + 1] if index + 1 < len(self) else object()
            )

    def remove(self, block):
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError("remove() called on an invalid block")
        super().remove(block)
        block.prev_free_block = block.next_free_block = None
        self._link()

    def pop(self, index=0):
        block = self[index]
        self.remove(block)
        return block

    def append_n(self, blocks):
        assert all(
            b.prev_free_block is None and b.next_free_block is None for b in blocks
        )
        self.extend(blocks)
        self._link()


def _pool(size=8, active=()):
    blocks = [
        SimpleNamespace(
            block_id=i,
            ref_cnt=int(i in active),
            is_null=False,
            prev_free_block=None,
            next_free_block=None,
        )
        for i in range(size)
    ]
    return SimpleNamespace(
        blocks=blocks, free_block_queue=_FreeQueue(b for b in blocks if not b.ref_cnt)
    )


def _offer(session, *blocks):
    return SimpleNamespace(session_id=session, block_ids=list(blocks))


def test_waiting_prefix_survives_other_work_without_reserving_capacity():
    pool = _pool()
    before = len(pool.free_block_queue)
    assert (
        order_free_session_blocks(
            pool, [_offer("agent", 0, 1, 2)], {"agent": "waiting"}
        )
        == 3
    )
    assert len(pool.free_block_queue) == before
    assert all(block.ref_cnt == 0 for block in pool.blocks)
    # Five allocations can now run without overwriting the waiting prefix.
    allocated = [pool.free_block_queue.pop(0).block_id for _ in range(5)]
    assert allocated == [3, 4, 5, 6, 7]
    assert [b.block_id for b in pool.free_block_queue] == [2, 1, 0]


def test_waiting_shared_prefix_keeps_chain_order_and_active_ownership():
    pool = _pool(active=(4,))
    offers = [_offer("done", 0, 1, 4), _offer("waiting", 1, 2), _offer(None, 0)]
    assert (
        order_free_session_blocks(
            pool, offers, {"done": "finished", "waiting": "waiting"}
        )
        == 2
    )
    assert pool.blocks[4].ref_cnt == 1
    assert pool.blocks[4] not in pool.free_block_queue
    assert [b.block_id for b in pool.free_block_queue[-2:]] == [2, 1]
    before = list(pool.free_block_queue)
    order_free_session_blocks(pool, [_offer("done", 5, 6)], {"done": "finished"})
    assert pool.free_block_queue == before
    assert len({b.block_id for b in pool.free_block_queue}) == 7


def test_finished_prefix_survives_unrelated_work_with_free_capacity():
    pool = _pool()
    pool.free_block_queue = _FreeQueue(pool.blocks[i] for i in [4, 5, 6, 7, 3, 2, 1, 0])
    before = list(pool.free_block_queue)
    assert (
        order_free_session_blocks(
            pool, [_offer("done", 0, 1, 2, 3)], {"done": "finished"}
        )
        == 0
    )
    assert pool.free_block_queue == before
    assert [pool.free_block_queue.pop(0).block_id for _ in range(4)] == [4, 5, 6, 7]
    assert [b.block_id for b in pool.free_block_queue] == [3, 2, 1, 0]


def test_shared_waiting_branches_evict_tails_before_common_head():
    pool = _pool()
    order_free_session_blocks(
        pool,
        [_offer("a", 0, 1, 2), _offer("b", 0, 1, 3)],
        {"a": "waiting", "b": "waiting"},
    )
    assert [b.block_id for b in pool.free_block_queue[-4:]] == [2, 3, 1, 0]


def test_unlinked_and_null_blocks_are_not_requeued():
    pool = _pool()
    detached = pool.free_block_queue.pop(0)
    assert detached.ref_cnt == 0
    pool.blocks[1].is_null = True
    before = list(pool.free_block_queue)
    assert (
        order_free_session_blocks(
            pool, [_offer("waiting", 0, 1)], {"waiting": "waiting"}
        )
        == 0
    )
    assert pool.free_block_queue == before


def test_missing_or_expired_hints_preserve_engine_eviction_order():
    pool = _pool()
    before = list(pool.free_block_queue)
    assert order_free_session_blocks(pool, [_offer("gone", 0, 1)], {}) == 0
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
    assert len(registry.scopes) == len(registry.session_scopes) == 1


def test_equal_revision_refreshes_ttl_but_cannot_change_phase():
    now = [0.0]
    registry = CacheSessionRegistry(ttl=5, clock=lambda: now[0])
    session = {"id": "a", "revision": 2, "phase": "active"}
    assert registry.update(session)
    now[0] = 4
    assert registry.update(session)
    now[0] = 6
    assert registry.get(["a"]) == {"a": "active"}
    assert not registry.update(dict(session, phase="waiting"))
    now[0] = 9
    assert registry.get(["a"]) == {}


def test_one_scope_cannot_churn_another_scopes_hints():
    registry = CacheSessionRegistry(capacity=4, scope_capacity=2)
    registry.update(
        {"id": "other", "scope": "other", "revision": 0, "phase": "waiting"}
    )
    for index in range(20):
        registry.update(
            {
                "id": f"churn-{index}",
                "scope": "churn",
                "revision": 0,
                "phase": "waiting",
            }
        )
    assert registry.get(["other"]) == {"other": "waiting"}
    assert list(registry.states) == ["other", "churn-18", "churn-19"]
    assert set(registry.session_scopes) == set(registry.states)


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
