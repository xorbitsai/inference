from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import psutil
import pytest

from xinference.router import process_metrics as process_metrics_module
from xinference.router.process_metrics import ProcessMetricsCollector


@pytest.fixture(autouse=True)
def fixed_process_capacity(monkeypatch):
    monkeypatch.setattr(process_metrics_module, "_cgroup_cpu_quota", lambda: 4.0)
    monkeypatch.setattr(process_metrics_module, "_cgroup_memory_limit", lambda: 1024)


class FakeProcess:
    def __init__(
        self,
        pid: int,
        *,
        cpu_values: list[float],
        rss: int,
        vms: int = 0,
        threads: int,
        cpu_user: float = 0.0,
        cpu_system: float = 0.0,
        created_at: float = 100.0,
        children: Optional[list["FakeProcess"]] = None,
        failure: Optional[Exception] = None,
    ) -> None:
        self.pid = pid
        self._cpu_values = iter(cpu_values)
        self._rss = rss
        self._vms = vms
        self._threads = threads
        self._cpu_user = cpu_user
        self._cpu_system = cpu_system
        self._created_at = created_at
        self._children = children or []
        self._failure = failure

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise self._failure

    def create_time(self) -> float:
        self._raise_if_failed()
        return self._created_at

    def children(self, recursive: bool = False) -> list["FakeProcess"]:
        self._raise_if_failed()
        if not recursive:
            return list(self._children)
        result: list[FakeProcess] = []
        pending = list(self._children)
        while pending:
            child = pending.pop(0)
            result.append(child)
            pending.extend(child._children)
        return result

    def cpu_percent(self, interval: Any = None) -> float:
        self._raise_if_failed()
        return next(self._cpu_values, 0.0)

    def memory_info(self) -> SimpleNamespace:
        self._raise_if_failed()
        return SimpleNamespace(rss=self._rss, vms=self._vms)

    def cpu_times(self) -> SimpleNamespace:
        self._raise_if_failed()
        return SimpleNamespace(user=self._cpu_user, system=self._cpu_system)

    def num_threads(self) -> int:
        self._raise_if_failed()
        return self._threads


def test_collects_recursive_process_tree_resources() -> None:
    grandchild = FakeProcess(
        3,
        cpu_values=[0.0, 20.0],
        rss=30,
        vms=300,
        threads=3,
        cpu_user=1.5,
        cpu_system=0.5,
    )
    child = FakeProcess(
        2,
        cpu_values=[0.0, 30.0],
        rss=50,
        vms=500,
        threads=4,
        cpu_user=2.0,
        cpu_system=1.0,
        children=[grandchild],
    )
    root = FakeProcess(
        1,
        cpu_values=[0.0, 50.0],
        rss=100,
        vms=1000,
        threads=5,
        cpu_user=4.0,
        cpu_system=1.0,
        created_at=100.0,
        children=[child],
    )

    resources = ProcessMetricsCollector(process=root, clock=lambda: 160.0).collect()

    assert resources == {
        "cpu_percent": 100.0,
        "cpu_cores": 1.0,
        "cpu_seconds_total": 10.0,
        "cpu_count": 4.0,
        "rss_bytes": 180,
        "virtual_memory_bytes": 1800,
        "memory_total_bytes": 1024,
        "main_process_rss_bytes": 100,
        "child_process_rss_bytes": 80,
        "child_process_count": 2,
        "thread_count": 12,
        "started_at": 100.0,
        "uptime_seconds": 60.0,
        "sampled_at": 160.0,
    }


def test_first_cpu_sample_is_non_blocking_zero() -> None:
    root = FakeProcess(1, cpu_values=[0.0, 0.0], rss=100, threads=1)

    resources = ProcessMetricsCollector(process=root, clock=lambda: 101.0).collect()

    assert resources["cpu_percent"] == 0.0
    assert resources["cpu_cores"] == 0.0


def test_process_affinity_precedes_cgroup_cpu_quota() -> None:
    root = FakeProcess(1, cpu_values=[0.0, 0.0], rss=100, threads=1)
    root.cpu_affinity = lambda: [0, 1]  # type: ignore[attr-defined]

    resources = ProcessMetricsCollector(process=root, clock=lambda: 101.0).collect()

    assert resources["cpu_count"] == 2.0


def test_exited_child_does_not_abort_snapshot() -> None:
    exited = FakeProcess(
        2,
        cpu_values=[],
        rss=0,
        threads=0,
        failure=psutil.NoSuchProcess(2),
    )
    root = FakeProcess(1, cpu_values=[0.0, 25.0], rss=100, threads=2, children=[exited])

    resources = ProcessMetricsCollector(process=root, clock=lambda: 120.0).collect()

    assert resources["cpu_percent"] == 25.0
    assert resources["rss_bytes"] == 100
    assert resources["child_process_count"] == 1
    assert resources["thread_count"] == 2


def test_access_denied_root_returns_partial_snapshot() -> None:
    root = FakeProcess(
        1,
        cpu_values=[],
        rss=0,
        threads=0,
        failure=psutil.AccessDenied(1),
    )

    resources = ProcessMetricsCollector(process=root, clock=lambda: 120.0).collect()

    assert resources["rss_bytes"] == 0
    assert resources["thread_count"] == 0
    assert "started_at" not in resources


def test_process_lookup_failure_returns_empty_snapshot(monkeypatch) -> None:
    def denied_process(pid: int) -> None:
        raise PermissionError(f"cannot inspect process {pid}")

    monkeypatch.setattr(process_metrics_module.psutil, "Process", denied_process)

    collector = ProcessMetricsCollector(pid=1, clock=lambda: 120.0)

    assert collector.started_at is None
    assert collector.collect() == {}
