# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .. import supervisor as supervisor_module
from ..supervisor import SupervisorActor, WorkerStatus


@pytest.fixture
def supervisor_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr("xinference.constants.XINFERENCE_HOME", str(tmp_path))
    monkeypatch.setattr(supervisor_module, "XINFERENCE_DISABLE_HEALTH_CHECK", True)
    monkeypatch.setattr(
        supervisor_module, "XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD", 1
    )
    monkeypatch.setattr(
        "xinference.constants.XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD", 1
    )
    worker = MagicMock(address="worker:1234")
    worker.worker_incarnation = AsyncMock(return_value="worker-incarnation")
    worker.list_models = AsyncMock(return_value={"model-rep0": {}})
    worker.get_extension_resources = AsyncMock(return_value={"extensions": {}})
    worker.get_node_metadata = AsyncMock(return_value={"software_version": "worker-v1"})
    worker.ping = AsyncMock()
    worker.get_model = AsyncMock(return_value=MagicMock(address="model:1234"))
    monkeypatch.setattr(
        supervisor_module.xo, "actor_ref", AsyncMock(return_value=worker)
    )
    monkeypatch.setattr("xinference.extensions.get_extensions", lambda: ())

    source = SupervisorActor()
    source._worker_address_to_worker[worker.address] = worker
    source._ownership_registry().incarnations[worker.address] = "worker-incarnation"
    source._rebuild_worker_replica_state(
        worker, [{"replica_model_uid": "model-rep0", "n_worker": 1, "shard": 0}]
    )
    source._worker_status[worker.address] = WorkerStatus(
        update_time=time.time() - 60, failure_remaining_count=1, status={}
    )
    source._block_tracker_mapping = {}
    source._collective_manager_mapping = {}
    source._schedule_autostart = MagicMock()
    source._upgrade_state.maintenance.pause()
    return source, worker


@pytest.fixture
async def restored_supervisor(supervisor_snapshot):
    source, worker = supervisor_snapshot
    snapshot = source.get_supervisor_snapshot()
    target = SupervisorActor(service_address="public:9999", standby=True)
    target._cache_tracker_ref = AsyncMock()
    target._schedule_autostart = MagicMock()
    target._relaunch_manager = MagicMock()
    await asyncio.wait_for(target.restore_supervisor(snapshot), 2)
    try:
        yield target, worker
    finally:
        await target.__pre_destroy__()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["activate", "rollback"])
async def test_first_health_sweep_retains_live_worker(
    restored_supervisor, supervisor_snapshot, outcome
):
    target, worker = (
        restored_supervisor if outcome == "activate" else supervisor_snapshot
    )
    assert time.time() - target._worker_status[worker.address].update_time > 10
    target._worker_metadata[worker.address] = {"software_version": "cached-v1"}
    if outcome == "activate":
        await asyncio.wait_for(target.activate_supervisor(), 2)
    else:
        target.resume_supervisor()
    # Run the first health sweep before any new Worker heartbeat can arrive.
    target._remove_dead_worker = AsyncMock(return_value=False)

    class StopSweep(Exception):
        pass

    with patch(
        "xinference.core.supervisor.asyncio.sleep", AsyncMock(side_effect=StopSweep)
    ):
        with pytest.raises(StopSweep):
            await target._check_dead_nodes()
    target._remove_dead_worker.assert_not_awaited()
    assert target._worker_status[worker.address].failure_remaining_count == 1
    assert target._replica_model_uid_to_worker["model-rep0"] is worker


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_metadata", [False, True])
async def test_restored_worker_metadata_refresh_is_target_local(
    restored_supervisor, monkeypatch, cached_metadata
):
    target, worker = restored_supervisor
    monkeypatch.setattr(supervisor_module, "_WORKER_METADATA_RETRY_DELAYS", (0,))
    if cached_metadata:
        target._worker_metadata[worker.address] = {"software_version": "cached-v1"}
    await target.activate_supervisor()
    if cached_metadata:
        assert not target._worker_metadata_refresh_tasks
        worker.get_node_metadata.assert_not_awaited()
    else:
        task = target._worker_metadata_refresh_tasks[worker.address]
        await asyncio.wait_for(task, 2)
        assert target._worker_metadata[worker.address] == {
            "software_version": "worker-v1"
        }
        assert not target._worker_metadata_refresh_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("coordinator", ["present", "missing", "unregistered-worker"])
async def test_restore_validates_coordinators_separately(
    supervisor_snapshot, coordinator
):
    source, worker = supervisor_snapshot
    source._replica_model_uid_to_worker["model-rank0"] = worker
    snapshot = source.get_supervisor_snapshot()
    assert set(snapshot["replicas"]) == {"model-rep0"}
    assert snapshot["coordinators"] == {"model-rank0": worker.address}
    target = SupervisorActor(service_address="public:9999", standby=True)
    if coordinator == "missing":
        worker.get_model.side_effect = KeyError("model-rank0")
        with pytest.raises(KeyError, match="model-rank0"):
            await target.restore_supervisor(snapshot)
    elif coordinator == "unregistered-worker":
        snapshot["coordinators"]["model-rank0"] = "unknown:1234"
        with pytest.raises(RuntimeError, match="coordinator inventory changed"):
            await target.restore_supervisor(snapshot)
    else:
        await target.restore_supervisor(snapshot)
        assert target._replica_model_uid_to_worker["model-rank0"] is worker
        assert target._replica_model_uid_to_worker["model-rep0"] is worker
        worker.get_model.assert_awaited_once_with("model-rank0")


@pytest.mark.asyncio
async def test_router_age_is_preserved_until_target_selection(
    supervisor_snapshot, monkeypatch
):
    source, _ = supervisor_snapshot
    registry = source._token_router_registry
    for instance_id, age in (("online", 5), ("offline", 100)):
        registry.register("router", instance_id, {"endpoint": "http://runtime"})
        registry._instances[instance_id]["last_heartbeat"] -= age
    snapshot = source.get_supervisor_snapshot()
    ages = {
        key: snapshot["captured_at"] - value["last_heartbeat"]
        for key, value in snapshot["router_instances"].items()
    }
    target = SupervisorActor(service_address="public:9999", standby=True)
    await target.restore_supervisor(snapshot)
    target._cache_tracker_ref = AsyncMock()
    monkeypatch.setattr(
        "xinference.core.supervisor_upgrade.time.time",
        lambda: snapshot["captured_at"] + 120,
    )
    await target.prepare_supervisor_activation()
    assert target._token_router_registry.get("online")["online"] is True
    assert target._token_router_registry.get("offline")["online"] is False
    for key, age in ages.items():
        assert target._token_router_registry.get(key)[
            "heartbeat_age_seconds"
        ] == pytest.approx(age)
