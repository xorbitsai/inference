from typing import Dict, Union

import pytest

from xinference.core import supervisor as supervisor_module
from xinference.core.otel import ClusterMetricsCollector
from xinference.core.resource import GPUStatus, ResourceStatus
from xinference.core.supervisor import SupervisorActor, WorkerStatus


class DummySupervisor:
    get_cluster_device_info = SupervisorActor.get_cluster_device_info
    _list_token_router_cluster_info = SupervisorActor._list_token_router_cluster_info
    _with_token_router_status = SupervisorActor._with_token_router_status
    list_virtual_models = SupervisorActor.list_virtual_models
    resolve_token_router_runtime = SupervisorActor.resolve_token_router_runtime

    def __init__(
        self,
        address,
        worker_status,
        router_configs=None,
        router_instances=None,
        router_nodes=None,
    ):
        self.address = address
        self._worker_status = worker_status
        self._token_router_store = FakeRouterStore(router_configs or [])
        self._token_router_registry = FakeRouterRegistry(router_instances or {})
        self._token_router_orchestration = FakeRouterOrchestration(router_nodes or [])
        self._token_router_runtime_cursors = {}


class FakeRouterStore:
    def __init__(self, configs):
        self._configs = configs

    def list(self):
        return list(self._configs)

    def get_by_virtual_model_uid(self, virtual_model_uid):
        return next(
            (
                config
                for config in self._configs
                if config.get("virtual_model_uid") == virtual_model_uid
            ),
            None,
        )


class FakeRouterRegistry:
    def __init__(self, instances):
        self._instances = instances

    def list(self, router_uid=None):
        if router_uid is None:
            return [item for values in self._instances.values() for item in values]
        return list(self._instances.get(router_uid, []))


class FakeRouterOrchestration:
    def __init__(self, nodes):
        self._nodes = nodes

    def list_nodes(self, *, include_offline=True):
        if include_offline:
            return list(self._nodes)
        return [
            node for node in self._nodes if node.get("connectivity_status") != "offline"
        ]


def _build_worker_status(*, gpu_utils=None):
    if gpu_utils is None:
        gpu_utils = []

    status: Dict[str, Union[ResourceStatus, GPUStatus]] = {
        "cpu": ResourceStatus(
            usage=0.25,
            total=32,
            memory_used=128,
            memory_available=384,
            memory_total=512,
        )
    }
    for idx, gpu_util in enumerate(gpu_utils):
        status[f"gpu-{idx}"] = GPUStatus(
            name=f"GPU-{idx}",
            mem_total=1000,
            mem_free=400,
            mem_used=600,
            mem_usage=0.6,
            gpu_util=gpu_util,
        )
    return status


def _build_worker_status_without_cpu(*, gpu_utils=None):
    if gpu_utils is None:
        gpu_utils = []

    status: Dict[str, Union[ResourceStatus, GPUStatus]] = {}
    for idx, gpu_util in enumerate(gpu_utils):
        status[f"gpu-{idx}"] = GPUStatus(
            name=f"GPU-{idx}",
            mem_total=1000,
            mem_free=400,
            mem_used=600,
            mem_usage=0.6,
            gpu_util=gpu_util,
        )
    return status


def test_cluster_metrics_collector_normalizes_gpu_utilization():
    collector = ClusterMetricsCollector()
    collector.update("worker-1", _build_worker_status(gpu_utils=[75]))

    observations = list(collector._gpu_utilization_cb(None))

    assert len(observations) == 1
    assert observations[0].value == 75
    assert observations[0].attributes == {
        "worker_address": "worker-1",
        "gpu_index": "gpu-0",
        "gpu_name": "GPU-0",
    }


def test_cluster_metrics_collector_remove_worker_stops_observations():
    collector = ClusterMetricsCollector()
    collector.update("worker-1", _build_worker_status(gpu_utils=[50]))
    collector.remove_worker("worker-1")

    assert list(collector._gpu_utilization_cb(None)) == []


@pytest.mark.asyncio
async def test_supervisor_cluster_device_info_includes_gpu_utilization_average():
    supervisor = DummySupervisor(
        "127.0.0.1:9999",
        {
            "worker-1": WorkerStatus(
                update_time=0,
                failure_remaining_count=3,
                status=_build_worker_status(gpu_utils=[40, 60]),
            ),
            "worker-2": WorkerStatus(
                update_time=0,
                failure_remaining_count=3,
                status=_build_worker_status(),
            ),
        },
    )

    result = await supervisor.get_cluster_device_info(detailed=True)

    worker_with_gpu = next(item for item in result if item["ip_address"] == "worker-1")
    worker_without_gpu = next(
        item for item in result if item["ip_address"] == "worker-2"
    )
    supervisor_info = next(item for item in result if item["node_type"] == "Supervisor")

    assert worker_with_gpu["gpu_count"] == 2
    assert worker_with_gpu["gpu_utilization"] == 50.0
    assert worker_without_gpu["gpu_count"] == 0
    assert worker_without_gpu["gpu_utilization"] is None
    assert supervisor_info["gpu_utilization"] is None


@pytest.mark.asyncio
async def test_supervisor_cluster_device_info_missing_cpu_key_with_gpus():
    supervisor = DummySupervisor(
        "127.0.0.1:9999",
        {
            "worker-no-cpu": WorkerStatus(
                update_time=0,
                failure_remaining_count=3,
                status=_build_worker_status_without_cpu(gpu_utils=[40, 60]),
            ),
        },
    )

    result = await supervisor.get_cluster_device_info(detailed=True)

    w = next(item for item in result if item["ip_address"] == "worker-no-cpu")
    assert w["gpu_count"] == 2
    assert w["gpu_utilization"] == 50.0
    assert w["cpu_available"] is None
    assert w["cpu_count"] is None
    assert w["mem_used"] is None
    assert w["mem_available"] is None
    assert w["mem_total"] is None


@pytest.mark.asyncio
async def test_supervisor_cluster_device_info_missing_cpu_key_empty_status():
    supervisor = DummySupervisor(
        "127.0.0.1:9999",
        {
            "worker-empty": WorkerStatus(
                update_time=0,
                failure_remaining_count=3,
                status={},
            ),
        },
    )

    result = await supervisor.get_cluster_device_info(detailed=True)

    w = next(item for item in result if item["ip_address"] == "worker-empty")
    assert w["gpu_count"] == 0
    assert w["gpu_utilization"] is None
    assert w["cpu_available"] is None
    assert w["cpu_count"] is None
    assert w["mem_used"] is None
    assert w["mem_available"] is None
    assert w["mem_total"] is None


@pytest.mark.asyncio
async def test_supervisor_cluster_device_info_mixed_with_and_without_cpu():
    supervisor = DummySupervisor(
        "127.0.0.1:9999",
        {
            "worker-normal": WorkerStatus(
                update_time=0,
                failure_remaining_count=3,
                status=_build_worker_status(gpu_utils=[10]),
            ),
            "worker-no-cpu": WorkerStatus(
                update_time=0,
                failure_remaining_count=3,
                status=_build_worker_status_without_cpu(gpu_utils=[20]),
            ),
        },
    )

    result = await supervisor.get_cluster_device_info(detailed=True)

    normal = next(item for item in result if item["ip_address"] == "worker-normal")
    assert normal["cpu_count"] == 32
    assert normal["mem_total"] == 512

    no_cpu = next(item for item in result if item["ip_address"] == "worker-no-cpu")
    assert no_cpu["cpu_available"] is None
    assert no_cpu["gpu_count"] == 1


def _router_config(*, enabled=True, revision=2):
    return {
        "router_uid": "router-a",
        "virtual_model_uid": "virtual-a",
        "enabled": enabled,
        "revision": revision,
        "model_type": "LLM",
    }


def _router_node(*, node_id="router-node-1", connectivity_status="online"):
    return {
        "node_id": node_id,
        "advertise_host": node_id,
        "online": connectivity_status == "online",
        "connectivity_status": connectivity_status,
        "software_version": "3.1.0",
        "software_revision": "abcdef123456",
        "resources": {
            "cpu": {"usage": 0.25, "total": 4},
            "memory": {"used": 2048, "available": 6144, "total": 8192},
        },
    }


@pytest.mark.asyncio
async def test_cluster_device_info_keeps_default_response_compatible():
    supervisor = DummySupervisor(
        "127.0.0.1:9999", {}, router_configs=[_router_config()]
    )

    result = await supervisor.get_cluster_device_info(detailed=True)

    assert {item["node_type"] for item in result} == {"Supervisor"}


@pytest.mark.asyncio
async def test_cluster_device_info_includes_only_online_router_agent_hosts():
    instances = {
        "router-a": [
            {
                "router_uid": "router-a",
                "instance_id": "instance-ready",
                "endpoint": "http://router:10080",
                "online": True,
                "status": "ready",
                "acked_revision": 2,
                "heartbeat_age_seconds": 1.5,
                "last_heartbeat": 1.0,
                "config_error": "",
                "protocol_version": "1",
                "software_version": "3.1.0",
                "software_revision": "abcdef123456",
                "metadata": {"hostname": "router-host"},
                "metrics": {},
                "backend_health": {},
                "process": {
                    "active_requests": 1,
                    "resources": {
                        "cpu_percent": 12.5,
                        "cpu_cores": 0.125,
                        "cpu_count": 4.0,
                        "rss_bytes": 1024,
                        "memory_total_bytes": 8192,
                    },
                },
            },
        ]
    }
    supervisor = DummySupervisor(
        "127.0.0.1:9999",
        {},
        router_configs=[_router_config()],
        router_instances=instances,
        router_nodes=[
            _router_node(),
            _router_node(
                node_id="router-node-suspected", connectivity_status="suspected"
            ),
            _router_node(node_id="router-node-offline", connectivity_status="offline"),
        ],
    )

    result = await supervisor.get_cluster_device_info(
        detailed=True, include_routers=True
    )
    routers = [item for item in result if item["node_type"] == "Router"]

    assert len(routers) == 1
    assert routers[0]["node_id"] == "router-node-1"
    assert routers[0]["ip_address"] == "router-node-1"
    assert routers[0]["online"] is True
    assert routers[0]["connectivity_status"] == "online"
    assert routers[0]["cpu_count"] == 4.0
    assert routers[0]["cpu_available"] == 3.0
    assert routers[0]["mem_used"] == 2048
    assert routers[0]["mem_available"] == 6144
    assert routers[0]["mem_total"] == 8192
    assert routers[0]["software_version"] == "3.1.0"
    assert routers[0]["software_revision"] == "abcdef123456"
    assert "endpoint" not in routers[0]
    assert "instance_id" not in routers[0]
    assert "process" not in routers[0]


@pytest.mark.asyncio
async def test_cluster_device_info_does_not_emit_config_placeholder() -> None:
    supervisor = DummySupervisor(
        "127.0.0.1:9999", {}, router_configs=[_router_config()]
    )

    result = await supervisor.get_cluster_device_info(
        detailed=True, include_routers=True
    )

    assert all(item["node_type"] != "Router" for item in result)


@pytest.mark.asyncio
async def test_token_router_feature_disabled_hides_cluster_and_virtual_models(
    monkeypatch,
):
    monkeypatch.setattr(supervisor_module, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    supervisor = DummySupervisor(
        "127.0.0.1:9999", {}, router_configs=[_router_config()]
    )

    cluster = await supervisor.get_cluster_device_info(
        detailed=True, include_routers=True
    )
    virtual_models = await supervisor.list_virtual_models()
    resolution = await supervisor.resolve_token_router_runtime("virtual-a")

    assert all(item["node_type"] != "Router" for item in cluster)
    assert virtual_models == {}
    assert resolution["available"] is False
    assert resolution["error_code"] == "TOKEN_ROUTER_DISABLED"
