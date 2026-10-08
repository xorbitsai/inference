from __future__ import annotations

from typing import Any

import pytest
from aioprometheus import Gauge

from xinference.core import metrics as metrics_module
from xinference.core.router_registry import RouterRuntimeRegistry
from xinference.core.supervisor import SupervisorActor


def test_sync_gauge_series_removes_stale_labelsets() -> None:
    gauge = Gauge("xinference:test_monitoring_v2_stale", "test gauge")

    metrics_module._sync_gauge_series(
        gauge,
        [({"router_uid": "router-a", "instance_id": "runtime-a"}, 1)],
    )
    assert gauge.get({"router_uid": "router-a", "instance_id": "runtime-a"}) == 1

    metrics_module._sync_gauge_series(gauge, [])

    with pytest.raises(KeyError):
        gauge.get({"router_uid": "router-a", "instance_id": "runtime-a"})


def test_monitoring_v2_exports_effective_and_controllable_separately() -> None:
    cluster_data = {
        "token_router_agents": [
            {
                "node_id": "agent-a",
                "advertise_host": "10.0.0.1",
                "software_version": "1.2.3",
                "software_revision": "abc123",
                "connectivity_status": "offline",
                "management_state": "active",
                "can_schedule": False,
                "heartbeat_age_seconds": 50,
                "max_instances": 20,
                "assignments": 1,
                "observed": {
                    "running_instances": 1,
                    "available_slots": 19,
                    "resources": {
                        "cpu": {"usage": 0.25, "total": 8},
                        "memory": {
                            "used": 1024,
                            "available": 3072,
                            "total": 4096,
                        },
                    },
                },
            }
        ],
        "token_router_assignments": [
            {
                "router_uid": "router-a",
                "assignment_id": "assignment-a",
                "replica_index": 0,
                "node_id": "agent-a",
                "desired_state": "running",
                "observed_state": "ready",
                "management_state": "node_lost",
                "assignment_generation": 2,
                "config_revision": 7,
            }
        ],
        "token_router_runtimes": [
            {
                "router_uid": "router-a",
                "assignment_id": "assignment-a",
                "replica_index": 0,
                "node_id": "agent-a",
                "instance_id": "runtime-a",
                "assignment_generation": 2,
                "software_version": "1.2.3",
                "software_revision": "abc123",
                "online": True,
                "status": "ready",
                "heartbeat_age_seconds": 10,
                "effective_ready": True,
                "controllable": False,
                "current": True,
                "expected_revision": 7,
                "acked_revision": 7,
                "config_synced": True,
            }
        ],
        "tokenizer_asset_bindings": [
            {
                "asset_id": "asset-a",
                "node_id": "agent-a",
                "desired_state": "present",
                "observed_state": "ready",
                "generation": 3,
                "synced": True,
                "ready": True,
            }
        ],
        "token_router_summaries": [
            {
                "router_uid": "router-a",
                "desired_replicas": 1,
                "effective_ready_replicas": 1,
                "controllable_ready_replicas": 0,
                "status": "degraded",
                "expected_revision": 7,
                "config_synced_replicas": 1,
            }
        ],
    }

    metrics_module._update_token_router_gauges(cluster_data)

    runtime_labels = {
        "router_uid": "router-a",
        "assignment_id": "assignment-a",
        "instance_id": "runtime-a",
    }
    assert metrics_module.token_router_runtime_effective_ready.get(runtime_labels) == 1
    assert metrics_module.token_router_runtime_controllable.get(runtime_labels) == 0
    assert metrics_module.token_router_runtime_config_synced.get(runtime_labels) == 1
    assert (
        metrics_module.token_router_agent_connectivity_status.get(
            {"node_id": "agent-a", "status": "offline"}
        )
        == 1
    )
    assert (
        metrics_module.token_router_assignment_observed_state.get(
            {
                "router_uid": "router-a",
                "assignment_id": "assignment-a",
                "replica_index": "0",
                "node_id": "agent-a",
                "state": "node_lost",
            }
        )
        == 1
    )
    assert (
        metrics_module.token_router_status.get(
            {"router_uid": "router-a", "status": "degraded"}
        )
        == 1
    )
    metrics_module._update_token_router_gauges({})
    with pytest.raises(KeyError):
        metrics_module.token_router_runtime_effective_ready.get(runtime_labels)


class _OrchestrationStub:
    def list_assignments(self) -> list[dict[str, Any]]:
        return [
            {
                "assignment_id": "assignment-a",
                "replica_index": 0,
                "assignment_generation": 2,
            }
        ]

    @staticmethod
    def runtime_is_current(instance: dict[str, Any]) -> bool:
        return int(instance.get("assignment_generation") or 0) == 2

    @staticmethod
    def runtime_is_controllable(instance: dict[str, Any]) -> bool:
        return False


@pytest.mark.asyncio
async def test_http_sd_retains_offline_runtime_until_registry_purge(
    monkeypatch,
) -> None:
    now = [100.0]
    monkeypatch.setattr("xinference.core.router_registry.time.time", lambda: now[0])

    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._token_router_registry = RouterRuntimeRegistry(
        heartbeat_timeout_seconds=90, stale_retention_seconds=300
    )
    supervisor._token_router_orchestration = _OrchestrationStub()
    supervisor._token_router_registry.register(
        "router-a",
        "runtime-a",
        {
            "assignment_id": "assignment-a",
            "assignment_generation": 2,
            "node_id": "agent-a",
            "endpoint": "http://10.0.0.1:10080",
            "status": "ready",
            "acked_revision": 7,
        },
    )

    _, effective, controllable = supervisor._token_router_runtime_health(
        {"router_uid": "router-a", "revision": 7}
    )
    assert [item["instance_id"] for item in effective] == ["runtime-a"]
    assert controllable == []

    targets = await supervisor.get_token_router_prometheus_http_sd("cluster-a")
    assert targets == [
        {
            "targets": ["10.0.0.1:10080"],
            "labels": {
                "job": "xinference-token-router-runtime",
                "cluster": "cluster-a",
                "router_uid": "router-a",
                "node_id": "agent-a",
                "assignment_id": "assignment-a",
                "replica_index": "0",
                "assignment_generation": "2",
                "instance_id": "runtime-a",
            },
        }
    ]

    now[0] = 191.0
    assert supervisor._token_router_registry.list()[0]["online"] is False
    assert await supervisor.get_token_router_prometheus_http_sd("cluster-a") == targets

    now[0] = 401.0
    assert await supervisor.get_token_router_prometheus_http_sd("cluster-a") == []
