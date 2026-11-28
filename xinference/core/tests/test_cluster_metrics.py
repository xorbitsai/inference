# CI: linux

from types import SimpleNamespace
from typing import Dict, Union

import pynvml
import pytest

from xinference import device_utils
from xinference.core import supervisor as supervisor_module
from xinference.core.otel import ClusterMetricsCollector
from xinference.core.resource import GPUStatus, ResourceStatus, gather_node_info
from xinference.core.supervisor import SupervisorActor, WorkerStatus


class DummySupervisor:
    control_address = SupervisorActor.control_address
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
        worker_metadata=None,
    ):
        self.address = address
        self._worker_status = worker_status
        self._worker_metadata = worker_metadata or {}
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


@pytest.mark.parametrize("has_memory_v2", [True, False])
def test_unsupported_nvml_memory_preserves_gpu_status(
    monkeypatch, caplog, has_memory_v2
):
    mib = 1024**2
    memory = SimpleNamespace(total=8192 * mib, used=2048 * mib, free=6144 * mib)

    def get_memory(handle, version=None):
        if handle == 0:
            raise pynvml.NVMLError_NotSupported()
        return memory

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        device_utils,
        "_find_device",
        lambda: SimpleNamespace(get_gpu_info_fn=device_utils.get_nvidia_gpu_info),
    )
    monkeypatch.setattr(pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlShutdown", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetCount", lambda: 2)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByIndex", lambda index: index)
    monkeypatch.setattr(
        pynvml,
        "nvmlDeviceGetName",
        lambda handle: ("NVIDIA GB10", "Discrete GPU")[handle],
    )
    monkeypatch.setattr(pynvml, "nvmlDeviceGetMemoryInfo", get_memory)
    monkeypatch.setattr(
        pynvml, "nvmlDeviceGetUtilizationRates", lambda handle: SimpleNamespace(gpu=37)
    )
    if not has_memory_v2:
        monkeypatch.delattr(pynvml, "nvmlMemory_v2", raising=False)

    info = device_utils.get_gpu_info()
    assert info["gpu-0"] == {
        "name": "NVIDIA GB10",
        "total": 0,
        "used": 0,
        "free": 0,
        "util": 37,
    }
    assert info["gpu-1"]["free_memory_mib"] == 6144
    assert info["gpu-1"]["total_memory_mib"] == 8192
    assert gather_node_info()["gpu-0"] == GPUStatus("NVIDIA GB10", 0, 0, 0, 0, 37)
    assert "Fail to get GPU info" not in caplog.text


@pytest.mark.parametrize("visible", [None, "0,1,2"])
def test_lost_gpu_is_left_out_of_gpu_info(monkeypatch, caplog, visible):
    mib = 1024**2
    memory = SimpleNamespace(total=8192 * mib, used=2048 * mib, free=6144 * mib)

    def get_handle(index):
        if index == 1:
            raise pynvml.NVMLError(pynvml.NVML_ERROR_GPU_IS_LOST)
        return index

    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(
        device_utils,
        "_find_device",
        lambda: SimpleNamespace(get_gpu_info_fn=device_utils.get_nvidia_gpu_info),
    )
    monkeypatch.setattr(pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlShutdown", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetCount", lambda: 3)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByIndex", get_handle)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetName", lambda handle: "NVIDIA H100")
    monkeypatch.setattr(pynvml, "nvmlDeviceGetMemoryInfo", lambda *args: memory)
    monkeypatch.setattr(
        pynvml, "nvmlDeviceGetUtilizationRates", lambda handle: SimpleNamespace(gpu=37)
    )

    monkeypatch.setattr(device_utils, "_unqueryable_device_ids", set())

    info = device_utils.get_gpu_info()
    device_utils.get_gpu_info()

    # The other GPUs keep their info, under their own keys.
    assert sorted(info) == ["gpu-0", "gpu-2"]
    assert info["gpu-2"]["free_memory_mib"] == 6144
    # Warned once, not on every poll.
    skip_warnings = [
        r
        for r in caplog.records
        if r.levelname == "WARNING" and "Skipping GPU 1" in r.getMessage()
    ]
    assert len(skip_warnings) == 1


def test_nvml_memory_v2_used_when_v1_is_unsupported(monkeypatch):
    mib = 1024**2
    memory = SimpleNamespace(total=8192 * mib, used=2048 * mib, free=6144 * mib)
    memory_v2 = object()

    def get_memory(handle, version=None):
        if version is None:
            raise pynvml.NVMLError_NotSupported()
        assert version is memory_v2
        return memory

    monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByIndex", lambda index: index)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetName", lambda handle: "Discrete GPU")
    monkeypatch.setattr(pynvml, "nvmlMemory_v2", memory_v2, raising=False)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetMemoryInfo", get_memory)
    monkeypatch.setattr(
        pynvml, "nvmlDeviceGetUtilizationRates", lambda handle: SimpleNamespace(gpu=37)
    )

    info = device_utils._get_info_by_pynvml(0)
    assert info["free_memory_mib"] == 6144
    assert info["total_memory_mib"] == 8192


@pytest.mark.asyncio
async def test_supervisor_cluster_device_info_includes_worker_software_versions():
    supervisor = DummySupervisor(
        "127.0.0.1:9999",
        {
            "worker-1": WorkerStatus(
                update_time=0,
                failure_remaining_count=3,
                status=_build_worker_status(),
            ),
            "worker-2": WorkerStatus(
                update_time=0,
                failure_remaining_count=3,
                status=_build_worker_status(),
            ),
            "worker-3": WorkerStatus(
                update_time=0,
                failure_remaining_count=3,
                status=_build_worker_status(),
            ),
        },
        worker_metadata={
            "worker-1": {"software_version": "3.4.1.dev9+gabcdef"},
            "worker-2": {"software_version": "3.4.0"},
        },
    )

    detailed = await supervisor.get_cluster_device_info(detailed=True)
    compact = await supervisor.get_cluster_device_info(detailed=False)

    worker_1 = next(item for item in detailed if item["ip_address"] == "worker-1")
    worker_2 = next(item for item in detailed if item["ip_address"] == "worker-2")
    worker_3 = next(item for item in detailed if item["ip_address"] == "worker-3")
    assert worker_1["software_version"] == "3.4.1.dev9+gabcdef"
    assert worker_2["software_version"] == "3.4.0"
    assert worker_3["software_version"] is None
    assert all("software_version" not in item for item in compact)


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
@pytest.mark.parametrize("service_address", [None, "public-supervisor:9999"])
async def test_cluster_device_info_keeps_default_response_compatible(service_address):
    supervisor = DummySupervisor(
        "127.0.0.1:9999", {}, router_configs=[_router_config()]
    )
    supervisor._service_address = service_address

    result = await supervisor.get_cluster_device_info(detailed=True)

    assert {item["node_type"] for item in result} == {"Supervisor"}
    assert result[0]["ip_address"] == (service_address or supervisor.address)


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


def test_models_loaded_total_drops_a_type_with_no_loaded_models():
    from xinference.core import metrics

    labels = {"model_type": "embedding"}
    metrics.update_cluster_metrics(
        {}, {"emb": {"model_type": "embedding", "model_name": "bge"}}
    )
    assert metrics.models_loaded_total.get(labels) == 1

    metrics.update_cluster_metrics({}, {})
    with pytest.raises(KeyError):
        metrics.models_loaded_total.get(labels)


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [0, 1, 2, 4, 8, 16])
async def test_cluster_per_gpu_metrics(count):
    supervisor = DummySupervisor(
        "127.0.0.1:9999",
        {
            "worker": WorkerStatus(
                update_time=0,
                failure_remaining_count=3,
                status=_build_worker_status(gpu_utils=list(range(count))),
            )
        },
    )
    detailed = await supervisor.get_cluster_device_info(detailed=True)
    worker = next(item for item in detailed if item["node_type"] == "Worker")
    assert len(worker["gpus"]) == count
    for index, gpu in enumerate(worker["gpus"]):
        assert gpu == {
            "index": f"gpu-{index}",
            "name": f"GPU-{index}",
            "utilization": index,
            "memory_used": None,
            "memory_total": None,
            "memory_available": None,
        }
    compact = await supervisor.get_cluster_device_info(detailed=False)
    assert all("gpus" not in item for item in compact)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["ascend", "nvidia"])
async def test_collected_device_memory_is_projected_in_bytes(monkeypatch, backend):
    mib = 1024**2
    used_mib, total_mib = 8192, 65536
    free_mib = total_mib - used_mib
    if backend == "ascend":
        import subprocess

        def npu_smi(command, **kwargs):
            assert command == ["npu-smi", "info"]
            return SimpleNamespace(
                stdout=(
                    "| 0 Ascend 910B | OK | 0 |\n"
                    "| 0 | 0000:01:00.0 | 8192 / 65536 |\n"
                )
            )

        monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(subprocess, "run", npu_smi)
        collect = device_utils.get_npu_info
        index = "gpu-0-0"
        name = "Ascend 910B"
        legacy_scale = 1
    else:
        # Reserved memory means total - free is not necessarily used memory.
        free_mib -= 4096
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(pynvml, "nvmlInit", lambda: None)
        monkeypatch.setattr(pynvml, "nvmlShutdown", lambda: None)
        monkeypatch.setattr(pynvml, "nvmlDeviceGetCount", lambda: 1)
        monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByIndex", lambda index: index)
        monkeypatch.setattr(pynvml, "nvmlDeviceGetName", lambda handle: "NVIDIA H100")
        monkeypatch.setattr(
            pynvml,
            "nvmlDeviceGetMemoryInfo",
            lambda handle: SimpleNamespace(
                total=total_mib * mib, used=used_mib * mib, free=free_mib * mib
            ),
        )
        monkeypatch.setattr(
            pynvml,
            "nvmlDeviceGetUtilizationRates",
            lambda handle: SimpleNamespace(gpu=0),
        )
        collect = device_utils.get_nvidia_gpu_info
        index = "gpu-0"
        name = "NVIDIA H100"
        legacy_scale = mib
    monkeypatch.setattr(
        device_utils,
        "_find_device",
        lambda: SimpleNamespace(get_gpu_info_fn=collect),
    )
    status = gather_node_info()
    gpu = status[index]
    assert isinstance(gpu, GPUStatus)
    assert (gpu.mem_used, gpu.mem_total, gpu.mem_free) == (
        used_mib * legacy_scale,
        total_mib * legacy_scale,
        free_mib * legacy_scale,
    )
    supervisor = DummySupervisor(
        "127.0.0.1:9999",
        {
            "worker": WorkerStatus(
                update_time=0, failure_remaining_count=3, status=status
            )
        },
    )
    detailed = await supervisor.get_cluster_device_info(detailed=True)
    worker = next(item for item in detailed if item["node_type"] == "Worker")
    assert worker["gpus"] == [
        {
            "index": index,
            "name": name,
            "utilization": 0,
            "memory_used": used_mib * mib,
            "memory_total": total_mib * mib,
            "memory_available": free_mib * mib,
        }
    ]
    assert worker["gpu_vram_total"] == total_mib * legacy_scale
    assert worker["gpu_vram_available"] == free_mib * legacy_scale
    compact = await supervisor.get_cluster_device_info(detailed=False)
    assert next(item for item in compact if item["node_type"] == "Worker") == {
        "node_type": "Worker",
        "ip_address": "worker",
        "gpu_count": 1,
        "gpu_vram_total": f"{int(total_mib * legacy_scale / mib)}MiB",
    }
    assert all("gpus" not in item for item in compact)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["rocm", "torch", "gcu", "legacy-ascend"])
async def test_device_memory_without_normalized_measurements_is_unknown(
    monkeypatch, backend
):
    if backend == "rocm":
        sample = device_utils._get_rocm_gpu_mem_info(0)
    elif backend == "torch":
        sample = device_utils._get_info_by_torch(0)
    else:
        sample = {
            "name": backend,
            "total": 65536,
            "used": 8192,
            "free": 57344,
            "util": 0,
        }
    monkeypatch.setattr(
        "xinference.core.resource.get_gpu_info", lambda: {"gpu-0": sample}
    )
    status = gather_node_info()
    supervisor = DummySupervisor(
        "127.0.0.1:9999",
        {
            "worker": WorkerStatus(
                update_time=0, failure_remaining_count=3, status=status
            )
        },
    )
    detailed = await supervisor.get_cluster_device_info(detailed=True)
    worker = next(item for item in detailed if item["node_type"] == "Worker")
    assert worker["gpus"][0] == {
        "index": "gpu-0",
        "name": sample["name"],
        "utilization": sample["util"],
        "memory_used": None,
        "memory_total": None,
        "memory_available": None,
    }
    assert worker["gpu_vram_total"] == sample["total"]
    assert worker["gpu_vram_available"] == sample["free"]


@pytest.mark.asyncio
async def test_device_memory_uses_independent_normalized_values_and_preserves_zero():
    status = _build_worker_status(gpu_utils=[0])
    gpu = status["gpu-0"]
    assert isinstance(gpu, GPUStatus)
    gpu.mem_total_bytes = 1000
    gpu.mem_free_bytes = 0
    supervisor = DummySupervisor(
        "127.0.0.1:9999",
        {
            "worker": WorkerStatus(
                update_time=0, failure_remaining_count=3, status=status
            )
        },
    )
    worker = (await supervisor.get_cluster_device_info(detailed=True))[1]
    assert worker["gpus"][0]["memory_total"] == 1000
    assert worker["gpus"][0]["memory_available"] == 0
    assert worker["gpus"][0]["memory_used"] is None
