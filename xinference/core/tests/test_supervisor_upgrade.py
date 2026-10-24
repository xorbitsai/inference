# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import xoscar as xo
from xoscar.utils import get_next_port

from .. import supervisor as supervisor_module
from ..router_config_store import RouterConfigStore
from ..router_orchestration import RouterOrchestrationController
from ..supervisor import SupervisorActor, WorkerStatus
from ..supervisor_upgrade import SupervisorGateway


@pytest.fixture
def supervisor_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr("xinference.constants.XINFERENCE_HOME", str(tmp_path))
    database_paths = {
        "XINFERENCE_DOWNLOAD_TASK_DB_PATH": str(tmp_path / "downloads.db"),
        "XINFERENCE_LAUNCH_HISTORY_DB_PATH": str(tmp_path / "history.db"),
        "XINFERENCE_TOKEN_ROUTER_DB_PATH": str(tmp_path / "routers.db"),
    }
    for name, path in database_paths.items():
        monkeypatch.setattr(supervisor_module, name, path)
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
    assert (
        source._download_task_store._db_path
        == database_paths["XINFERENCE_DOWNLOAD_TASK_DB_PATH"]
    )
    assert (
        source._launch_history_store._db_path
        == database_paths["XINFERENCE_LAUNCH_HISTORY_DB_PATH"]
    )
    assert (
        source._token_router_store._db_path
        == database_paths["XINFERENCE_TOKEN_ROUTER_DB_PATH"]
    )
    controller = source._token_router_orchestration
    for store in (
        controller.nodes,
        controller.assignments,
        controller.tokenizer_assets,
        controller.deployments,
    ):
        assert store._db_path == database_paths["XINFERENCE_TOKEN_ROUTER_DB_PATH"]
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


class AgentControl(xo.StatelessActor):
    def __init__(self, database_path):
        super().__init__()
        self._token_router_orchestration = RouterOrchestrationController(
            database_path, RouterConfigStore(database_path)
        )
        self.entered = asyncio.Event()

    heartbeat_token_router_node = SupervisorActor.heartbeat_token_router_node

    async def watch_token_router_assignments(
        self, node_id, after_cursor="", wait_seconds=30
    ):
        if wait_seconds:
            self.entered.set()
        return await SupervisorActor.watch_token_router_assignments(
            self, node_id, after_cursor, wait_seconds
        )

    async def wait_entered(self):
        await self.entered.wait()

    def finish_watch(self):
        self._token_router_orchestration._watch_epoch += "-changed"


@pytest.mark.asyncio
async def test_agent_heartbeat_preserves_ready_assets_during_long_poll_drain(
    supervisor_snapshot,
):
    source, _ = supervisor_snapshot
    store = source._token_router_store
    controller = source._token_router_orchestration
    fingerprint = "sha256:" + "a" * 64
    store.create(
        "router-a",
        {
            "logical_model": "logical",
            "tokenizer_asset_id": "asset-a",
            "tokenizer_asset_revision": "v1",
            "tokenizer_asset_fingerprint": fingerprint,
        },
    )
    controller.create_tokenizer_asset(
        {
            "asset_id": "asset-a",
            "origin": "shared_fs",
            "revision": "v1",
            "fingerprint": fingerprint,
            "source": {"type": "shared_fs", "path": "/assets/asset-a"},
        }
    )
    controller.register_node(
        {
            "node_id": "node-a",
            "advertise_host": "127.0.0.1",
            "port_range_start": 12080,
            "port_range_end": 12090,
            "max_instances": 5,
            "capabilities": {},
            "software_version": "test",
        }
    )
    controller.update_deployment(
        "router-a",
        {
            "management_mode": "managed",
            "desired_replicas": 1,
            "rollout": {"auto_failover": True},
        },
    )
    store.set_enabled("router-a", True)
    controller.router_enabled("router-a", True)
    binding = controller.list_tokenizer_asset_bindings(
        asset_id="asset-a", node_id="node-a"
    )[0]
    controller.report_tokenizer_asset_binding_status(
        "asset-a",
        "node-a",
        {
            "generation": binding["generation"],
            "observed_state": "ready",
            "observed_revision": "v1",
            "observed_fingerprint": fingerprint,
            "local_path": "/assets/asset-a",
        },
    )
    assignment = controller.list_assignments(router_uid="router-a")[0]

    pool = await xo.create_actor_pool(
        address=f"127.0.0.1:{get_next_port()}", n_process=0
    )
    watch = closing = heartbeat = None
    try:
        gateway = await xo.create_actor(
            SupervisorGateway, address=pool.external_address
        )
        agent = await xo.create_actor(
            AgentControl, store._db_path, address=pool.external_address
        )
        await gateway.select_supervisor(agent)
        await gateway.release_supervisor()
        initial = await gateway.watch_token_router_assignments("node-a", wait_seconds=0)
        watch = asyncio.create_task(
            gateway.watch_token_router_assignments("node-a", initial["cursor"], 60)
        )
        await asyncio.wait_for(agent.wait_entered(), 1)
        closing = asyncio.create_task(gateway.prepare_supervisor_switch(final=False))
        await asyncio.sleep(0.01)
        # Simulate a drain longer than the default 45-second offline window.
        with controller.nodes._connect() as connection:
            connection.execute(
                "UPDATE token_router_nodes SET last_seen_at = ? WHERE node_id = ?",
                (
                    (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat(),
                    "node-a",
                ),
            )
        heartbeat = asyncio.create_task(
            gateway.heartbeat_token_router_node("node-a", {})
        )
        try:
            await asyncio.wait_for(asyncio.shield(heartbeat), 1)
        except asyncio.TimeoutError:
            pass
        assert not controller.sweep_nodes()
        await asyncio.wait_for(heartbeat, 1)
        assert controller.get_node("node-a")["connectivity_status"] == "online"
        assert (
            controller.list_tokenizer_asset_bindings(
                asset_id="asset-a", node_id="node-a"
            )[0]["observed_state"]
            == "ready"
        )
        assert (
            controller.list_assignments(router_uid="router-a")[0][
                "assignment_generation"
            ]
            == assignment["assignment_generation"]
        )
        assert not closing.done()
    finally:
        if watch is not None:
            await agent.finish_watch()
            await asyncio.wait_for(watch, 2)
        if closing is not None:
            await asyncio.wait_for(closing, 2)
        if heartbeat is not None:
            await gateway.release_supervisor()
            await asyncio.wait_for(heartbeat, 1)
        await pool.stop()


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
