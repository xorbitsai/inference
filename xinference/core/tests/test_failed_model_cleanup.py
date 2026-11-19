# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ..status_guard import InstanceInfo, LaunchStatus, StatusGuardActor
from ..supervisor import SupervisorActor


def _instance(status: str = "ERROR", created: int = 1) -> InstanceInfo:
    return InstanceInfo(
        model_uid="demo",
        model_name="demo",
        model_version=None,
        model_ability=["chat"],
        replica=1,
        status=status,
        instance_created_ts=created,
    )


@pytest.fixture
def failed_model():
    guard = StatusGuardActor()
    guard.set_instance_info("demo", _instance())
    supervisor = SupervisorActor()
    supervisor.address = "supervisor:9999"
    supervisor._collective_manager_mapping = {}
    supervisor._block_tracker_mapping = {}
    supervisor._status_guard_ref = SimpleNamespace(
        remove_failed_instance_info=AsyncMock(
            side_effect=guard.remove_failed_instance_info
        )
    )
    return supervisor, guard


@pytest.mark.asyncio
async def test_delete_failed_model_without_runtime(failed_model):
    supervisor, guard = failed_model
    await supervisor.terminate_model("demo")
    assert guard.get_instance_info(model_uid="demo") == []
    assert guard.get_replica_statuses("demo") == []
    # A cleared UID can be reused normally.
    replacement = _instance("CREATING", created=2)
    guard.set_instance_info("demo", replacement)
    assert guard.get_instance_info(model_uid="demo") == [replacement]


@pytest.mark.asyncio
async def test_delete_unknown_model_still_rejects(failed_model):
    supervisor, guard = failed_model
    with pytest.raises(ValueError, match="Model not found"):
        await supervisor.terminate_model("missing")
    assert len(guard.get_instance_info(model_uid="demo")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", [s.name for s in LaunchStatus if s != LaunchStatus.ERROR]
)
async def test_delete_without_runtime_preserves_nonfailed_instance(
    failed_model, status
):
    supervisor, guard = failed_model
    info = _instance(status)
    guard.set_instance_info("demo", info)
    with pytest.raises(ValueError, match="Model not found"):
        await supervisor.terminate_model("demo")
    # Includes the retained TERMINATED entry that list APIs filter out.
    assert guard._model_uid_to_info["demo"] is info


@pytest.mark.asyncio
async def test_launch_rollback_preserves_failure_details(failed_model):
    supervisor, guard = failed_model
    await supervisor.terminate_model("demo", suppress_exception=True)
    assert guard.get_instance_info(model_uid="demo")[0].status == "ERROR"
    supervisor._status_guard_ref.remove_failed_instance_info.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_preserves_replacement_created_during_status_rpc(failed_model):
    supervisor, guard = failed_model
    replacement = _instance("CREATING", created=2)

    async def remove_after_relaunch(model_uid):
        supervisor._model_uid_to_replica_info[model_uid] = SimpleNamespace(
            active_replica_ids=[0]
        )
        guard.set_instance_info(model_uid, replacement)
        return guard.remove_failed_instance_info(model_uid)

    supervisor._status_guard_ref.remove_failed_instance_info.side_effect = (
        remove_after_relaunch
    )
    with pytest.raises(ValueError, match="Model not found"):
        await supervisor.terminate_model("demo")
    assert guard.get_instance_info(model_uid="demo") == [replacement]
    assert "demo" in supervisor._model_uid_to_replica_info


@pytest.mark.asyncio
async def test_delete_running_model_still_terminates_worker(failed_model):
    supervisor, guard = failed_model
    guard.set_instance_info("demo", _instance("READY"))
    worker = SimpleNamespace(terminate_model=AsyncMock())
    supervisor._model_uid_to_replica_info["demo"] = SimpleNamespace(
        active_replica_ids=[0]
    )
    supervisor._replica_model_uid_to_worker["demo-rep0"] = worker
    await supervisor.terminate_model("demo")
    worker.terminate_model.assert_awaited_once_with(model_uid="demo-rep0")
    supervisor._status_guard_ref.remove_failed_instance_info.assert_not_awaited()
    assert "demo" not in supervisor._model_uid_to_replica_info
    assert "demo-rep0" not in supervisor._replica_model_uid_to_worker
