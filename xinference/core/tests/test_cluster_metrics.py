from typing import Dict, Union

import pytest

from xinference.core.otel import ClusterMetricsCollector
from xinference.core.resource import GPUStatus, ResourceStatus
from xinference.core.supervisor import SupervisorActor, WorkerStatus


class DummySupervisor:
    get_cluster_device_info = SupervisorActor.get_cluster_device_info

    def __init__(self, address, worker_status):
        self.address = address
        self._worker_status = worker_status


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
