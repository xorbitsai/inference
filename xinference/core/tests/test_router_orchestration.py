# Copyright 2022-2026 Xinference Holdings Pte. Ltd

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta, timezone

import pytest

from xinference.core.router_config_store import RouterConfigStore
from xinference.core.router_orchestration import RouterOrchestrationController
from xinference.core.supervisor import SupervisorActor


def _config_store(tmp_path):
    store = RouterConfigStore(str(tmp_path / "routers.db"))
    store.create(
        "router-a",
        {
            "logical_model": "logical",
        },
    )
    return store


def _node(
    node_id: str,
    host: str,
    start: int,
    *,
    port_count: int = 10,
    max_instances: int = 5,
) -> dict:
    return {
        "node_id": node_id,
        "advertise_host": host,
        "port_range_start": start,
        "port_range_end": start + port_count - 1,
        "max_instances": max_instances,
        "reported_labels": {"zone": "a"},
        "capabilities": {},
        "software_version": "test",
    }


def _age_node(controller, node_id: str, seconds: float) -> None:
    last_seen = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    with controller.nodes._connect() as conn:
        conn.execute(
            "UPDATE token_router_nodes SET last_seen_at = ? WHERE node_id = ?",
            (last_seen, node_id),
        )


def test_existing_router_defaults_to_external(tmp_path):
    store = _config_store(tmp_path)
    controller = RouterOrchestrationController(str(tmp_path / "routers.db"), store)

    deployment = controller.get_deployment("router-a")

    assert deployment is not None
    assert deployment["management_mode"] == "external"
    assert controller.list_assignments(router_uid="router-a") == []


def test_managed_router_schedules_and_fences_runtime(tmp_path):
    store = _config_store(tmp_path)
    controller = RouterOrchestrationController(str(tmp_path / "routers.db"), store)
    controller.register_node(_node("node-a", "127.0.0.1", 12080))
    controller.update_deployment(
        "router-a",
        {
            "management_mode": "managed",
            "desired_replicas": 1,
            "placement": {"labels": {"zone": "a"}},
        },
    )
    store.set_enabled("router-a", True)
    controller.router_enabled("router-a", True)

    assignments = controller.list_assignments(router_uid="router-a")
    assert len(assignments) == 1
    assignment = assignments[0]
    assert assignment["node_id"] == "node-a"
    assert assignment["listen_port"] == 12080
    assert assignment["desired_state"] == "running"

    registration = {
        "assignment_id": assignment["assignment_id"],
        "assignment_generation": assignment["assignment_generation"],
        "node_id": "node-a",
        "endpoint": assignment["public_endpoint"],
    }
    assert controller.validate_runtime_registration("router-a", registration)
    with pytest.raises(ValueError, match="Stale"):
        controller.validate_runtime_registration(
            "router-a", {**registration, "assignment_generation": 99}
        )
    with pytest.raises(ValueError, match="endpoint"):
        controller.validate_runtime_registration(
            "router-a", {**registration, "endpoint": "http://127.0.0.1:1"}
        )


def test_assignment_status_preserves_observed_metadata_when_omitted(tmp_path):
    store = _config_store(tmp_path)
    controller = RouterOrchestrationController(str(tmp_path / "routers.db"), store)
    controller.register_node(_node("node-a", "127.0.0.1", 12080))
    controller.update_deployment("router-a", {"management_mode": "managed"})
    store.set_enabled("router-a", True)
    controller.router_enabled("router-a", True)
    assignment = controller.list_assignments(router_uid="router-a")[0]
    instance = {
        "assignment_id": assignment["assignment_id"],
        "assignment_generation": assignment["assignment_generation"],
        "instance_id": "instance-a",
    }

    controller.runtime_heartbeat(
        instance, {"status": "ready", "process": {"pid": 5001}}
    )
    controller.runtime_acked(instance)
    preserved = controller.report_assignment_status(
        assignment["assignment_id"],
        {
            "node_id": "node-a",
            "assignment_generation": assignment["assignment_generation"],
            "observed_state": "draining",
        },
    )

    assert preserved["observed"] == {"runtime_status": "ready"}

    cleared = controller.report_assignment_status(
        assignment["assignment_id"],
        {
            "node_id": "node-a",
            "assignment_generation": assignment["assignment_generation"],
            "observed_state": "draining",
            "observed": {},
        },
    )
    assert cleared["observed"] == {}


def test_replicas_spread_and_config_revision_does_not_bump_generation(tmp_path):
    store = _config_store(tmp_path)
    controller = RouterOrchestrationController(str(tmp_path / "routers.db"), store)
    controller.register_node(_node("node-a", "127.0.0.1", 12080))
    controller.register_node(_node("node-b", "127.0.0.2", 12080))
    controller.update_deployment(
        "router-a", {"management_mode": "managed", "desired_replicas": 2}
    )
    store.set_enabled("router-a", True)
    controller.router_enabled("router-a", True)
    before = controller.list_assignments(router_uid="router-a")

    assert {item["node_id"] for item in before} == {"node-a", "node-b"}
    store.update("router-a", {"logical_model": "logical-v2"})
    controller.router_config_updated("router-a")
    after = controller.list_assignments(router_uid="router-a")

    assert [item["assignment_generation"] for item in after] == [
        item["assignment_generation"] for item in before
    ]
    assert all(
        item["config_revision"] > old["config_revision"]
        for item, old in zip(after, before)
    )


def test_port_conflict_reassigns_port_and_rejects_stale_status(tmp_path):
    store = _config_store(tmp_path)
    controller = RouterOrchestrationController(str(tmp_path / "routers.db"), store)
    controller.register_node(_node("node-a", "127.0.0.1", 12080))
    controller.update_deployment("router-a", {"management_mode": "managed"})
    store.set_enabled("router-a", True)
    controller.router_enabled("router-a", True)
    assignment = controller.list_assignments(router_uid="router-a")[0]

    updated = controller.report_assignment_status(
        assignment["assignment_id"],
        {
            "node_id": "node-a",
            "assignment_generation": assignment["assignment_generation"],
            "observed_state": "port_conflict",
        },
    )
    assert updated["listen_port"] == 12081
    assert updated["assignment_generation"] == assignment["assignment_generation"] + 1
    with pytest.raises(ValueError, match="Stale"):
        controller.report_assignment_status(
            assignment["assignment_id"],
            {
                "node_id": "node-a",
                "assignment_generation": assignment["assignment_generation"],
                "observed_state": "starting",
            },
        )


def test_disable_waits_for_stopped_report_then_releases_assignment(tmp_path):
    store = _config_store(tmp_path)
    controller = RouterOrchestrationController(str(tmp_path / "routers.db"), store)
    controller.register_node(_node("node-a", "127.0.0.1", 12080))
    controller.update_deployment("router-a", {"management_mode": "managed"})
    store.set_enabled("router-a", True)
    controller.router_enabled("router-a", True)
    assignment = controller.list_assignments(router_uid="router-a")[0]

    store.set_enabled("router-a", False)
    controller.router_enabled("router-a", False)
    stopped_assignment = controller.list_assignments(router_uid="router-a")[0]
    assert stopped_assignment["desired_state"] == "stopped"
    result = controller.report_assignment_status(
        assignment["assignment_id"],
        {
            "node_id": "node-a",
            "assignment_generation": assignment["assignment_generation"],
            "observed_state": "stopped",
        },
    )
    assert result["released"] is True
    assert controller.list_assignments(router_uid="router-a") == []


def test_node_capacity_cannot_exceed_port_pool(tmp_path):
    store = _config_store(tmp_path)
    controller = RouterOrchestrationController(str(tmp_path / "routers.db"), store)

    with pytest.raises(ValueError, match="exceeds the Router node port range"):
        controller.register_node(
            _node(
                "node-a",
                "127.0.0.1",
                12080,
                port_count=1,
                max_instances=2,
            )
        )


def test_stopping_assignment_keeps_capacity_and_port_reserved(tmp_path):
    store = _config_store(tmp_path)
    store.create(
        "router-b",
        {"logical_model": "logical-b"},
    )
    controller = RouterOrchestrationController(str(tmp_path / "routers.db"), store)
    controller.register_node(
        _node(
            "node-a",
            "127.0.0.1",
            12080,
            port_count=1,
            max_instances=1,
        )
    )

    controller.update_deployment("router-a", {"management_mode": "managed"})
    store.set_enabled("router-a", True)
    controller.router_enabled("router-a", True)
    router_a_assignment = controller.list_assignments(router_uid="router-a")[0]

    store.set_enabled("router-a", False)
    controller.router_enabled("router-a", False)
    assert (
        controller.list_assignments(router_uid="router-a")[0]["desired_state"]
        == "stopped"
    )

    controller.update_deployment("router-b", {"management_mode": "managed"})
    store.set_enabled("router-b", True)
    controller.router_enabled("router-b", True)
    assert controller.list_assignments(router_uid="router-b") == []

    controller.report_assignment_status(
        router_a_assignment["assignment_id"],
        {
            "node_id": "node-a",
            "assignment_generation": router_a_assignment["assignment_generation"],
            "observed_state": "stopped",
        },
    )
    controller.scheduler.reconcile_router("router-b")
    router_b_assignment = controller.list_assignments(router_uid="router-b")[0]
    assert router_b_assignment["listen_port"] == 12080


def test_multiple_migrations_do_not_reuse_target_port(tmp_path):
    store = _config_store(tmp_path)
    controller = RouterOrchestrationController(str(tmp_path / "routers.db"), store)
    controller.register_node(_node("node-a", "127.0.0.1", 12080, max_instances=2))
    controller.update_deployment(
        "router-a", {"management_mode": "managed", "desired_replicas": 2}
    )
    store.set_enabled("router-a", True)
    controller.router_enabled("router-a", True)
    assert {
        item["node_id"] for item in controller.list_assignments(router_uid="router-a")
    } == {"node-a"}

    controller.register_node(_node("node-b", "127.0.0.2", 12080, max_instances=2))
    controller.set_node_state("node-a", "draining")

    migrated = controller.list_assignments(router_uid="router-a")
    assert {item["node_id"] for item in migrated} == {"node-b"}
    assert {item["listen_port"] for item in migrated} == {12080, 12081}


def test_managed_router_can_scale_to_zero_without_eligible_nodes(tmp_path):
    store = _config_store(tmp_path)
    controller = RouterOrchestrationController(str(tmp_path / "routers.db"), store)
    controller.update_deployment(
        "router-a", {"management_mode": "managed", "desired_replicas": 0}
    )

    assert controller.validate_managed_deployment("router-a") == []
    store.set_enabled("router-a", True)
    controller.router_enabled("router-a", True)
    assert controller.list_assignments(router_uid="router-a") == []


def test_runtime_current_fences_old_generation_and_external_registration(tmp_path):
    store = _config_store(tmp_path)
    controller = RouterOrchestrationController(str(tmp_path / "routers.db"), store)

    external = {
        "router_uid": "router-a",
        "instance_id": "external-instance",
        "endpoint": "http://127.0.0.1:10080",
    }
    assert controller.runtime_is_current(external) is True

    controller.register_node(_node("node-a", "127.0.0.1", 12080))
    controller.update_deployment("router-a", {"management_mode": "managed"})
    store.set_enabled("router-a", True)
    controller.router_enabled("router-a", True)
    assignment = controller.list_assignments(router_uid="router-a")[0]
    current = {
        "router_uid": "router-a",
        "instance_id": "managed-instance",
        "endpoint": assignment["public_endpoint"],
        "assignment_id": assignment["assignment_id"],
        "assignment_generation": assignment["assignment_generation"],
        "node_id": assignment["node_id"],
    }

    assert controller.runtime_is_current(external) is False
    assert controller.runtime_is_current(current) is True
    assert (
        controller.runtime_is_current(
            {
                **current,
                "assignment_generation": assignment["assignment_generation"] + 1,
            }
        )
        is False
    )


def test_placement_node_selection_filters_and_migrates_assignment(tmp_path):
    store = _config_store(tmp_path)
    controller = RouterOrchestrationController(str(tmp_path / "routers.db"), store)
    controller.register_node(_node("node-a", "127.0.0.1", 12080))
    controller.register_node(_node("node-b", "127.0.0.2", 12080))
    controller.update_deployment(
        "router-a",
        {
            "management_mode": "managed",
            "placement": {"node_ids": ["node-a"]},
        },
    )
    store.set_enabled("router-a", True)
    controller.router_enabled("router-a", True)
    before = controller.list_assignments(router_uid="router-a")[0]

    assert before["node_id"] == "node-a"

    controller.update_deployment("router-a", {"placement": {"node_ids": ["node-b"]}})
    after = controller.list_assignments(router_uid="router-a")[0]

    assert after["node_id"] == "node-b"
    assert after["assignment_generation"] == before["assignment_generation"] + 1


def test_unknown_placement_node_is_rejected_during_managed_validation(tmp_path):
    store = _config_store(tmp_path)
    controller = RouterOrchestrationController(str(tmp_path / "routers.db"), store)
    controller.register_node(_node("node-a", "127.0.0.1", 12080))
    controller.update_deployment(
        "router-a",
        {
            "management_mode": "managed",
            "placement": {"node_ids": ["missing-node"]},
        },
    )

    errors = controller.validate_managed_deployment("router-a")

    assert len(errors) == 1
    assert "No online active Router Agent node" in errors[0]


def test_scheduler_tolerates_null_node_capabilities(tmp_path):
    store = _config_store(tmp_path)
    controller = RouterOrchestrationController(str(tmp_path / "routers.db"), store)

    assert (
        controller.scheduler._node_has_asset(
            {"node_id": "node-a", "capabilities": None},
            {"tokenizer_asset_id": "asset-a"},
        )
        is False
    )


@pytest.mark.parametrize(
    "legacy_capability",
    [
        pytest.param(["asset-a"], id="string"),
        pytest.param(
            [
                {
                    "asset_id": "asset-a",
                    "revision": "v1",
                    "fingerprint": "sha256:" + "a" * 64,
                }
            ],
            id="structured",
        ),
    ],
)
def test_static_tokenizer_asset_capability_cannot_replace_binding(
    tmp_path, legacy_capability
):
    store = _config_store(tmp_path)
    store.update(
        "router-a",
        {
            "logical_model": "logical",
            "tokenizer_asset_id": "asset-a",
            "tokenizer_asset_revision": "v1",
            "tokenizer_asset_fingerprint": "sha256:" + "a" * 64,
        },
    )
    controller = RouterOrchestrationController(str(tmp_path / "routers.db"), store)
    controller.create_tokenizer_asset(
        {
            "asset_id": "asset-a",
            "origin": "shared_fs",
            "revision": "v1",
            "fingerprint": "sha256:" + "a" * 64,
            "source": {"type": "shared_fs", "path": "/assets/asset-a"},
        },
        "admin",
    )
    node = _node("node-a", "127.0.0.1", 12080)
    node["capabilities"] = {"tokenizer_assets": legacy_capability}

    registered = controller.register_node(node)

    assert controller.list_tokenizer_asset_bindings(node_id="node-a") == []
    assert (
        controller.scheduler._node_has_asset(registered, store.get("router-a")) is False
    )


def test_asset_binding_readiness_gates_assignment_and_stale_preserves_runtime(tmp_path):
    store = _config_store(tmp_path)
    store.update(
        "router-a",
        {
            "logical_model": "logical",
            "tokenizer_asset_id": "asset-a",
            "tokenizer_asset_revision": "v1",
            "tokenizer_asset_fingerprint": "sha256:" + "a" * 64,
        },
    )
    controller = RouterOrchestrationController(str(tmp_path / "routers.db"), store)
    controller.create_tokenizer_asset(
        {
            "asset_id": "asset-a",
            "origin": "shared_fs",
            "revision": "v1",
            "fingerprint": "sha256:" + "a" * 64,
            "source": {"type": "shared_fs", "path": "/assets/asset-a"},
        }
    )
    controller.register_node(_node("node-a", "127.0.0.1", 12080))
    controller.update_deployment(
        "router-a", {"management_mode": "managed", "desired_replicas": 1}
    )
    store.set_enabled("router-a", True)
    controller.router_enabled("router-a", True)

    binding = controller.list_tokenizer_asset_bindings(
        asset_id="asset-a", node_id="node-a"
    )[0]
    assert binding["binding_mode"] == "on_demand"
    assert binding["observed_state"] == "pending"
    assert controller.list_assignments(router_uid="router-a") == []

    controller.report_tokenizer_asset_binding_status(
        "asset-a",
        "node-a",
        {
            "generation": binding["generation"],
            "observed_state": "ready",
            "observed_revision": "v1",
            "observed_fingerprint": "sha256:" + "a" * 64,
            "local_path": "/assets/asset-a",
        },
    )
    assignment = controller.list_assignments(router_uid="router-a")[0]
    assert assignment["tokenizer_asset"]["local_path"] == "/assets/asset-a"

    controller.tokenizer_assets.mark_node_bindings_stale("node-a")
    controller.scheduler.reconcile_router("router-a")
    preserved = controller.list_assignments(router_uid="router-a")[0]
    controller.scheduler.reconcile_router("router-a")
    preserved_again = controller.list_assignments(router_uid="router-a")[0]

    assert preserved["desired_state"] == "running"
    assert preserved["assignment_generation"] == assignment["assignment_generation"]
    assert (
        preserved_again["assignment_generation"] == assignment["assignment_generation"]
    )


def test_managed_labels_survive_agent_reregistration(tmp_path):
    store = _config_store(tmp_path)
    controller = RouterOrchestrationController(str(tmp_path / "routers.db"), store)
    controller.register_node(_node("node-a", "127.0.0.1", 12080))
    controller.set_node_labels("node-a", {"environment": "test"})

    updated = _node("node-a", "127.0.0.1", 12080)
    updated["reported_labels"] = {"zone": "b"}
    node = controller.register_node(updated)

    assert node["reported_labels"] == {"zone": "b"}
    assert node["managed_labels"] == {"environment": "test"}
    assert node["labels"] == {"zone": "b", "environment": "test"}


def test_router_agent_lifecycle_hides_offline_node_and_preserves_runtime_contract(
    tmp_path,
):
    store = _config_store(tmp_path)
    controller = RouterOrchestrationController(
        str(tmp_path / "routers.db"),
        store,
        node_suspect_seconds=30,
        node_offline_seconds=45,
    )
    controller.register_node(_node("node-a", "127.0.0.1", 12080))
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
    before = controller.list_assignments(router_uid="router-a")[0]

    _age_node(controller, "node-a", 31)
    transitions = controller.sweep_nodes()
    suspected = controller.get_node("node-a")

    assert transitions[0]["connectivity_status"] == "suspected"
    assert suspected is not None
    assert suspected["connectivity_status"] == "suspected"
    assert suspected["can_schedule"] is False
    assert [
        node["node_id"] for node in controller.list_nodes(include_offline=False)
    ] == ["node-a"]

    _age_node(controller, "node-a", 46)
    transitions = controller.sweep_nodes()
    after = controller.list_assignments(router_uid="router-a")[0]

    assert transitions[0]["connectivity_status"] == "offline"
    assert controller.list_nodes(include_offline=False) == []
    assert (
        controller.list_nodes(include_offline=True)[0]["connectivity_status"]
        == "offline"
    )
    assert after["management_state"] == "node_lost"
    assert after["desired_state"] == "running"
    assert after["assignment_generation"] == before["assignment_generation"]


def test_router_agent_offline_auto_failover_moves_only_when_candidate_exists(tmp_path):
    store = _config_store(tmp_path)
    controller = RouterOrchestrationController(
        str(tmp_path / "routers.db"),
        store,
        node_suspect_seconds=30,
        node_offline_seconds=45,
    )
    controller.register_node(_node("node-a", "127.0.0.1", 12080))
    controller.register_node(_node("node-b", "127.0.0.2", 12080))
    controller.update_deployment(
        "router-a",
        {
            "management_mode": "managed",
            "desired_replicas": 1,
            "placement": {"node_ids": ["node-a", "node-b"]},
            "rollout": {"auto_failover": True},
        },
    )
    store.set_enabled("router-a", True)
    controller.router_enabled("router-a", True)
    before = controller.list_assignments(router_uid="router-a")[0]
    assert before["node_id"] == "node-a"

    _age_node(controller, "node-a", 46)
    controller.sweep_nodes()
    after = controller.list_assignments(router_uid="router-a")[0]

    assert after["node_id"] == "node-b"
    assert after["desired_state"] == "running"
    assert after["assignment_generation"] == before["assignment_generation"] + 1


@pytest.mark.asyncio
async def test_monitor_token_router_nodes_offloads_sweep(monkeypatch):
    main_thread_id = threading.get_ident()
    sweep_thread_ids = []

    class _Orchestration:
        def sweep_nodes(self):
            sweep_thread_ids.append(threading.get_ident())
            return []

    class _Supervisor:
        _monitor_token_router_nodes = SupervisorActor._monitor_token_router_nodes

        def __init__(self):
            self._token_router_orchestration = _Orchestration()

    class _StopMonitor(Exception):
        pass

    async def _stop_after_iteration(_delay):
        raise _StopMonitor

    monkeypatch.setattr(asyncio, "sleep", _stop_after_iteration)

    with pytest.raises(_StopMonitor):
        await _Supervisor()._monitor_token_router_nodes()

    assert len(sweep_thread_ids) == 1
    assert sweep_thread_ids[0] != main_thread_id
