# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
import time
import weakref
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ..autoscaling import (
    POOL_ENV,
    AutoscalingPolicy,
    EmbeddingAutoscaler,
    EmbeddingOverloadedError,
    select_placements,
)
from ..exceptions import ModelNotReadyError
from ..model import ModelActor
from ..supervisor import SupervisorActor
from ..supervisor_upgrade import SupervisorMaintenance
from ..utils import build_replica_model_uid
from ..worker import WorkerActor


@pytest.mark.parametrize(
    "policy",
    [
        {"min_replicas": 0},
        {"max_replicas": True},
        {"min_replicas": 3, "max_replicas": 2},
    ],
)
def test_invalid_replica_bounds(policy):
    with pytest.raises(ValueError):
        AutoscalingPolicy(**policy)


def gpu_worker(address, total, occupied=()):
    return SimpleNamespace(
        address=address,
        get_gpu_allocation_status=AsyncMock(
            return_value={
                "total": total,
                "models": {device: ["occupied"] for device in occupied},
                "user_specified": {},
            }
        ),
    )


@pytest.mark.asyncio
async def test_pool_filters_workers_gpu_indexes_and_occupied_cards(monkeypatch):
    first = gpu_worker("a:1", [0, 1, 2], [0])
    second = gpu_worker("b:1", [0, 1])
    owner = SimpleNamespace(_worker_address_to_worker={"a:1": first, "b:1": second})
    monkeypatch.setenv(POOL_ENV, '{"test": {"a:1": [0, 1, 3]}}')
    targets = await select_placements(owner, "test", 2)
    assert [(t.devices[0].worker_ip, t.devices[0].gpu_idx) for t in targets] == [
        ("a:1", [1]),
    ]
    second.get_gpu_allocation_status.assert_not_awaited()
    assert len(await select_placements(owner, None, 5)) == 4
    with pytest.raises(ValueError, match="Unknown GPU pool"):
        await select_placements(owner, "missing", 1)


@pytest.mark.asyncio
async def test_exclusive_gpu_admission_is_checked_at_worker_and_released():
    worker = WorkerActor.__new__(WorkerActor)
    worker._total_gpu_devices = [0, 1]
    worker._allow_multi_replica_per_gpu = True
    worker._gpu_to_model_uids = defaultdict(set)
    worker._user_specified_gpu_to_model_uids = defaultdict(set)
    worker._gpu_to_model_uids[0].add("existing")
    with pytest.raises(RuntimeError, match="occupied"):
        await worker.allocate_devices_with_gpu_idx(
            "replica", "embedding", [0], exclusive=True
        )
    await worker.allocate_devices_with_gpu_idx(
        "replica", "embedding", [1], exclusive=True
    )
    assert worker.allocate_devices("ordinary", 1) == [0]
    with pytest.raises(RuntimeError, match="exclusively allocated"):
        await worker.allocate_devices_with_gpu_idx("ordinary", "embedding", [1])
    worker.release_devices("replica")
    assert worker.allocate_devices("ordinary", 1) == [1]


@pytest.mark.asyncio
async def test_bounded_queue_releases_cancelled_waiters():
    state = EmbeddingAutoscaler(
        AutoscalingPolicy(queue_capacity=1, queue_timeout_ms=80, scale_up_wait_ms=10)
    )
    replica = SimpleNamespace(reserve_embedding_request=AsyncMock(return_value=False))
    owner = SimpleNamespace(
        _model_uid_to_replica_info={"model": SupervisorActor._build_replica_info(1)},
        _get_replica_model=AsyncMock(return_value=replica),
    )
    pending = asyncio.create_task(state.admit(owner, "model", "first"))
    await asyncio.sleep(0)
    with pytest.raises(EmbeddingOverloadedError, match="full"):
        await state.admit(owner, "model", "second")
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert not state.waiters
    with pytest.raises(EmbeddingOverloadedError, match="Timed out"):
        await state.admit(owner, "model", "third")
    assert not state.waiters


@pytest.mark.asyncio
async def test_drain_checks_pending_reservation_and_active_work():
    actor = ModelActor.__new__(ModelActor)
    actor._model = SimpleNamespace(model_uid="embedding")
    actor._model_state = "ready"
    actor._serve_count = 0
    actor._request_limits = 1
    actor._chat_reservations = {}
    actor._chat_preflight_reservations = set()
    actor._realtime_runtime = None
    assert actor.reserve_embedding_request("request", 1)
    assert not actor.reserve_embedding_request("second", 1)
    assert not actor.begin_embedding_drain(0)
    assert actor.release_embedding_reservation("request")
    actor._serve_count = 1
    assert not actor.begin_embedding_drain(0)
    actor._serve_count = 0
    assert actor.begin_embedding_drain(0)
    assert not actor.reserve_embedding_request("third", 1)
    with pytest.raises(ModelNotReadyError, match="draining"):
        await actor.create_embedding("text")
    actor.cancel_embedding_drain()
    assert actor.reserve_embedding_request("third", 1)
    actor.release_embedding_reservation("third")


@pytest.mark.asyncio
async def test_controller_limits_scale_and_excludes_draining_replica(monkeypatch):
    monkeypatch.delenv(POOL_ENV, raising=False)
    owner = SupervisorActor.__new__(SupervisorActor)
    uid = "embedding"
    info = SupervisorActor._build_replica_info(1)
    state = EmbeddingAutoscaler(
        AutoscalingPolicy(max_replicas=2, scale_up_wait_ms=10, idle_timeout_seconds=1)
    )
    owner._embedding_autoscalers = {uid: state}
    owner._model_uid_to_replica_info = {uid: info}
    owner._model_replica_locks = weakref.WeakValueDictionary()
    owner._worker_address_to_worker = {"a:1": gpu_worker("a:1", [0, 1], [0])}
    owner._initial_model_launches = {}
    owner._model_reload_tasks = {}
    owner._upgrade_state = SimpleNamespace(maintenance=SupervisorMaintenance())
    second = build_replica_model_uid(uid, 1)
    owner._replica_model_uid_to_worker = {
        second: owner._worker_address_to_worker["a:1"]
    }

    async def add(_uid, placement):
        assert placement.devices[0].gpu_idx == [1]
        info.active_replica_ids.append(1)
        owner._refresh_replica_scheduler(info)

    async def terminate(_uid, index):
        assert index in state.draining
        info.active_replica_ids.remove(index)
        owner._refresh_replica_scheduler(info)

    replica = SimpleNamespace(begin_embedding_drain=AsyncMock(return_value=True))
    owner._get_replica_model = AsyncMock(return_value=replica)
    owner._add_model_replica = AsyncMock(side_effect=add)
    owner._terminate_model_replica = AsyncMock(side_effect=terminate)

    state.waiters.append((time.monotonic() - 1, asyncio.Event()))
    state.saturated = True
    await owner.reconcile_embedding_replicas(uid)
    assert len(info.active_replica_ids) == 2
    await owner.reconcile_embedding_replicas(uid)
    assert state.reason == "max_replicas_reached"
    owner._add_model_replica.assert_awaited_once()

    state.waiters.clear()
    state.saturated = False
    state.last_pressure = time.monotonic() - 2
    await owner.reconcile_embedding_replicas(uid)
    assert info.active_replica_ids == [0]
    assert not state.draining
    await owner.reconcile_embedding_replicas(uid)
    owner._terminate_model_replica.assert_awaited_once_with(uid, 1)
