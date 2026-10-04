# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    spec = importlib.util.spec_from_file_location(
        "benchmark_sglang_pd", Path(__file__).parents[1] / "benchmark_sglang_pd.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_isolation_rejects_unrelated_allocations_but_accepts_owned_children(
    benchmark, monkeypatch
):
    owner = Mock(pid=1)
    owner.children.return_value = [Mock(pid=2)]
    monkeypatch.setattr(benchmark.psutil, "Process", Mock(return_value=owner))
    monkeypatch.setattr(
        benchmark, "gpu_processes", lambda: [dict(pid=i) for i in (1, 2, 3, 4)]
    )
    assert benchmark.GPUIsolation([3], 10).snapshot()["external"] == [dict(pid=4)]


def test_gpu_monitor_error_cannot_produce_an_accepted_sample(benchmark, monkeypatch):
    guard = benchmark.GPUIsolation([], 10)

    def broken_monitor():
        guard.stop.set()
        raise RuntimeError("GPU telemetry unavailable")

    monkeypatch.setattr(guard, "snapshot", broken_monitor)
    guard.__enter__()
    guard.thread.join(timeout=2)
    assert not guard.thread.is_alive()
    assert "GPU telemetry unavailable" in guard.interference[0]["monitor_error"]


def test_client_readiness_and_failed_launch_always_clean_up_owned_server(
    benchmark, monkeypatch, tmp_path
):
    proc = Mock()
    proc.poll.return_value = None
    monkeypatch.setattr(benchmark, "get_next_port", lambda: 12345)
    monkeypatch.setattr(benchmark.subprocess, "Popen", Mock(return_value=proc))
    client = Mock()
    client.get_workers_info.return_value = [{"work-ip": "127.0.0.1:12345"}]
    client.launch_model.side_effect = RuntimeError("model launch failed")
    factory = Mock(side_effect=[benchmark.requests.ConnectionError(), client])
    monkeypatch.setattr(benchmark, "Client", factory)
    monkeypatch.setattr(benchmark.time, "sleep", lambda _: None)
    stop = Mock()
    monkeypatch.setattr(benchmark, "stop_server", stop)
    args = SimpleNamespace(
        model_path=Path("model"), memory_fraction=0.6, kv_tokens=524288
    )
    with pytest.raises(RuntimeError, match="model launch failed"):
        benchmark.run_backend(args, "nixl", 0, tmp_path, None)
    assert factory.call_count == 2
    stop.assert_called_once_with(proc)
