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
async def restored_supervisor(tmp_path, monkeypatch):
    monkeypatch.setattr("xinference.constants.XINFERENCE_HOME", str(tmp_path))
    monkeypatch.setattr(supervisor_module, "XINFERENCE_DISABLE_HEALTH_CHECK", True)
    monkeypatch.setattr(
        supervisor_module, "XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD", 1
    )
    worker = MagicMock(address="worker:1234")
    worker.worker_incarnation = AsyncMock(return_value="worker-incarnation")
    worker.list_models = AsyncMock(return_value={"model-rep0": {}})
    worker.get_extension_resources = AsyncMock(return_value={"extensions": {}})
    worker.get_node_metadata = AsyncMock(return_value={"software_version": "worker-v1"})
    worker.ping = AsyncMock()
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
    source._upgrade_state.maintenance.pause()
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
async def test_first_restored_health_sweep_retains_live_worker(restored_supervisor):
    target, worker = restored_supervisor
    assert time.time() - target._worker_status[worker.address].update_time > 10
    target._worker_metadata[worker.address] = {"software_version": "cached-v1"}
    await asyncio.wait_for(target.activate_supervisor(), 2)
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
