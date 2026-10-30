# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# CI: linux

import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ..weight_cache import (
    ModelReloadError,
    WeightCacheDaemon,
    WeightCachedModel,
    _cached_memory_fraction,
    _process_group_alive,
    _terminate_process_group,
    parse_weight_cache_option,
    validate_reload_patch,
)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
@pytest.mark.parametrize("living", [False, True])
def test_permission_error_distinguishes_live_group_from_zombies(monkeypatch, living):
    import psutil

    process = SimpleNamespace(
        pid=11,
        info={"status": psutil.STATUS_RUNNING if living else psutil.STATUS_ZOMBIE},
    )
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [process])
    monkeypatch.setattr(os, "getpgid", lambda pid: 10)
    monkeypatch.setattr(os, "killpg", MagicMock(side_effect=PermissionError("group")))
    assert _process_group_alive(10) is living
    if living:
        with pytest.raises(PermissionError, match="group"):
            _terminate_process_group(10)
    else:
        _terminate_process_group(10)


def test_cached_memory_counts_only_daemon_tree(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(hip=None))
    )
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            Process=lambda pid: SimpleNamespace(
                children=lambda **kw: [SimpleNamespace(pid=11)]
            )
        ),
    )
    nvml = SimpleNamespace(
        nvmlInit=MagicMock(),
        nvmlShutdown=MagicMock(),
        nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetComputeRunningProcesses=lambda index: [
            SimpleNamespace(pid=10, usedGpuMemory=100),
            SimpleNamespace(pid=11, usedGpuMemory=200 * (index + 1)),
            SimpleNamespace(pid=99, usedGpuMemory=800),
        ],
        nvmlDeviceGetMemoryInfo_v2=lambda handle: SimpleNamespace(total=1000, free=100),
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    assert _cached_memory_fraction(10) == 0.5
    nvml.nvmlShutdown.assert_called_once()


def test_cached_memory_rocm_records(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(hip="7"))
    )
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(Process=lambda pid: SimpleNamespace(children=lambda **kw: [])),
    )
    smi = SimpleNamespace(
        amdsmi_init=MagicMock(),
        amdsmi_shut_down=MagicMock(),
        amdsmi_get_processor_handles=lambda: [0],
        amdsmi_get_gpu_process_list=lambda handle: [
            {"pid": 10, "memory_usage": {"vram_mem": 300}},
            {"pid": 99, "memory_usage": {"vram_mem": 800}},
        ],
        amdsmi_get_gpu_memory_total=lambda *args: 1000,
        amdsmi_get_gpu_memory_usage=lambda *args: 900,
        AmdSmiMemoryType=SimpleNamespace(VRAM=0),
    )
    monkeypatch.setitem(sys.modules, "amdsmi", smi)
    assert _cached_memory_fraction(10) == 0.3
    smi.amdsmi_shut_down.assert_called_once()


@pytest.mark.parametrize("requested", [0.9, "0.9"])
def test_engine_budget_reserves_retained_memory_without_mutating_config(
    monkeypatch, requested
):
    daemon = WeightCacheDaemon("vllm", "/models/test", {})
    daemon.process = SimpleNamespace(pid=10)
    monkeypatch.setattr(
        "xinference.model.llm.weight_cache._cached_memory_fraction", lambda pid: 0.4
    )
    config = {"gpu_memory_utilization": requested}
    try:
        assert daemon.engine_config(config)["gpu_memory_utilization"] == 0.5
        assert daemon.engine_config(config)["gpu_memory_utilization"] == 0.5
        assert config == {"gpu_memory_utilization": requested}
        with pytest.raises(ValueError, match="retained weights"):
            daemon.engine_config({"gpu_memory_utilization": 0.3})
        explicit = {"gpu_memory_utilization": 0.3, "kv_cache_memory_bytes": 100}
        assert daemon.engine_config(explicit)["gpu_memory_utilization"] == 0.01
    finally:
        daemon.process = None
        daemon.stop()


@pytest.mark.parametrize(
    "value, expected",
    [(None, None), (True, True), (False, False), ("true", True), ("FALSE", False)],
)
def test_cache_option(value, expected):
    assert parse_weight_cache_option(value) is expected


@pytest.mark.parametrize("value", [0, 1, "yes", {}, "auto"])
def test_invalid_cache_option(value):
    with pytest.raises(ValueError):
        parse_weight_cache_option(value)


@pytest.mark.parametrize(
    "patch",
    [
        {},
        {"tensor_parallel_size": 2},
        {"dtype": "float16"},
        {"max_num_seqs": True},
        {"max_num_seqs": 1.5},
        {"gpu_memory_utilization": float("nan")},
        {"gpu_memory_utilization": 1.1},
        {"enforce_eager": "false"},
    ],
)
def test_reject_unsafe_or_invalid_reload(patch):
    with pytest.raises(ValueError):
        validate_reload_patch("vllm", patch)


def test_valid_reload_types():
    validate_reload_patch(
        "vllm",
        {"enforce_eager": False, "max_num_seqs": 32, "gpu_memory_utilization": 0.8},
    )
    validate_reload_patch(
        "sglang", {"chunked_prefill_size": -1, "cuda_graph_max_bs_decode": 16}
    )
    with pytest.raises(ValueError):
        validate_reload_patch("sglang", {"cuda_graph_max_bs": 16})


@pytest.mark.parametrize(
    "engine,key", [("vllm", "max_model_len"), ("sglang", "context_length")]
)
@pytest.mark.parametrize("launched", [None, 262144, 500000, 1000000])
def test_reload_rejects_context_increases_before_engine_validation(
    engine, key, launched
):
    from ..llm_family import match_llm

    model = WeightCachedModel()
    model.model_family = match_llm("qwen3.8", "pytorch", 27, "none")
    model._weight_cache_engine = engine
    model._reload_config = {key: launched} if launched else {}
    original = dict(model._reload_config)
    current = launched or 262144
    for requested in (current + 1, 2000000):
        with pytest.raises(ValueError, match="requires a new deployment"):
            model.validate_reload({key: requested})
    assert model._reload_config == original


@pytest.mark.parametrize(
    "engine,key,module,argument_class",
    [
        ("vllm", "max_model_len", "vllm.engine.arg_utils", "AsyncEngineArgs"),
        ("sglang", "context_length", "sglang.srt.server_args", "ServerArgs"),
    ],
)
@pytest.mark.parametrize("requested", [500000, 1000000])
def test_reload_keeps_launch_rope_settings_within_the_current_window(
    monkeypatch, engine, key, module, argument_class, requested
):
    from ..context_length import configure_auto_yarn, map_context_length
    from ..llm_family import match_llm

    model = WeightCachedModel()
    model.model_family = match_llm("qwen3.8", "pytorch", 27, "none")
    model._weight_cache_engine = engine
    model.model_path = "/models/qwen"
    model._reload_config, _ = configure_auto_yarn(
        model.model_family,
        engine,
        map_context_length(engine, {"context_length": 1000000}),
    )
    model._weight_cache = SimpleNamespace(
        client_config=lambda: {}, engine_config=lambda cfg: cfg
    )
    model._sanitize_model_config = lambda cfg: cfg
    validated = []

    class EngineArgs:
        def __init__(self, **kwargs):
            validated.append(kwargs)

        def create_engine_config(self):
            pass

    monkeypatch.setitem(
        sys.modules, module, SimpleNamespace(**{argument_class: EngineArgs})
    )
    result = model.validate_reload({key: requested})
    assert result[key] == requested
    assert validated[0][key] == requested
    assert model._reload_config[key] == 1000000
    rope_key = "hf_overrides" if engine == "vllm" else "json_model_override_args"
    assert result[rope_key] == model._reload_config[rope_key]


def test_reload_error_survives_actor_serialization():
    error = pickle.loads(pickle.dumps(ModelReloadError("oom", restored=True)))
    assert error.restored is True
    assert str(error) == "oom"


@pytest.mark.parametrize(
    "engine,tp", [("vllm", "tensor_parallel_size"), ("sglang", "tp_size")]
)
@pytest.mark.skipif(
    sys.platform == "win32", reason="Daemon process groups require POSIX"
)
def test_daemon_lifecycle(monkeypatch, engine, tp):
    daemon = WeightCacheDaemon(
        engine,
        "/models/test",
        {tp: 2, "launch_timeout": 300, "unused": None, "kv_cache_memory_bytes": 100},
    )
    calls = []
    process = MagicMock(pid=1234)
    process.poll.return_value = None

    def spawn(command, **kwargs):
        calls.append((command, kwargs))
        for rank in range(2):
            (daemon.directory / f"gpu-{rank}.sock").touch()
            import os

            (daemon.directory / f"gpu-{rank}.ready").write_text(f"pid={os.getpid()}\n")
        return process

    monkeypatch.setattr("xinference.model.llm.weight_cache.subprocess.Popen", spawn)
    kill = MagicMock()
    monkeypatch.setattr("xinference.model.llm.weight_cache.os.killpg", kill)
    # The mocked daemon PID must not collide with a real host process group.
    monkeypatch.setattr("psutil.process_iter", lambda attrs: [])
    monkeypatch.setattr("xinference.model.llm.weight_cache.socket.socket", MagicMock())
    monkeypatch.setenv("SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE", "previous")
    daemon.start(timeout=1)
    config = json.loads((daemon.directory / "config.yaml").read_text())
    assert "launch_timeout" not in config and "unused" not in config
    command, options = calls[0]
    assert options["start_new_session"]
    assert Path(command[1]).name == "weight_cache.py"
    assert Path(command[1]).is_absolute()
    assert command[2] == str(os.getpid())
    if engine == "vllm":
        client = daemon.client_config()
        assert client["load_format"] == "ipc_cache"
        assert client["model_loader_extra_config"]["fallback"] is False
        assert client["model_loader_extra_config"]["mode"] == "zero_copy"
    else:
        assert config["model-path"] == "/models/test"
        assert config["tp-size"] == 2
        assert config["weight-cache-mode"] == "off"
        assert daemon.client_config() == {"weight_cache_mode": "client"}
        assert "{device_uuid}" in options["env"]["SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE"]
    daemon.stop()
    daemon.stop()
    assert kill.call_count == 2
    assert not daemon.directory.exists()

    assert os.environ["SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE"] == "previous"


def test_container_driver_pids_and_unknown_memory(monkeypatch):
    monkeypatch.setattr(
        "xinference.model.llm.weight_cache._gpu_process_memory",
        lambda: {
            0: (1000, {900: 800, 901: 300, 902: None}),
            1: (1000, {900: 800, 903: 400}),
        },
    )
    # Namespace PIDs intentionally differ from all driver PIDs.
    assert _cached_memory_fraction(os.getpid(), {0: {900}, 1: {900}}) == 0.4
    with pytest.raises(RuntimeError, match="kv_cache_memory_bytes"):
        _cached_memory_fraction(os.getpid())


def test_reload_config_reads_committed_snapshot_during_engine_mutation():
    model = FakeCachedModel()

    class MutatingConfig(dict):
        def items(self):
            raise RuntimeError("dictionary changed size during iteration")

    model._model_config = MutatingConfig(max_num_seqs=32)
    assert model.get_reload_config()["model_config"] == {"max_num_seqs": 16}


@pytest.mark.skipif(sys.platform == "win32", reason="AF_UNIX daemon")
def test_stale_socket_cannot_pass_preflight():
    import socket

    daemon = WeightCacheDaemon("vllm", "/models/test", {})
    daemon.process = MagicMock()
    daemon.process.poll.return_value = None
    path = daemon.directory / "rank.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen()
    daemon._socket_paths = {path: path.stat().st_ino}
    try:
        daemon.client_config()
        connection, _ = server.accept()
        connection.close()
        server.close()  # leaves the same inode behind
        with pytest.raises(RuntimeError, match="unavailable"):
            daemon.client_config()
    finally:
        server.close()
        daemon.process = None
        daemon.stop()


def test_daemon_failure_restores_environment(monkeypatch):
    import os

    monkeypatch.setenv("SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE", "original")
    daemon = WeightCacheDaemon("sglang", "/models/test", {"not_serializable": object()})
    with pytest.raises(TypeError):
        daemon.start()
    assert os.environ["SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE"] == "original"
    assert not daemon.directory.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux AF_UNIX path limit")
def test_daemon_uses_short_socket_namespace_for_long_tmpdir(monkeypatch, tmp_path):
    import os
    import tempfile

    directory = tmp_path / ("long-temporary-directory-" * 4)
    directory.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(directory))
    daemon = WeightCacheDaemon("vllm", "/models/test", {})
    try:
        assert len(os.fsencode(daemon.directory)) <= 20
        assert daemon.directory.stat().st_mode & 0o777 == 0o700
        assert list(directory.iterdir()) == []
    finally:
        daemon.stop()


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.skipif(
    sys.platform == "win32", reason="Daemon process groups require POSIX"
)
def test_sglang_daemon_normalizes_legacy_graph_flag(monkeypatch, disabled):
    daemon = WeightCacheDaemon(
        "sglang", "/models/test", {"disable_cuda_graph": disabled}
    )
    process = MagicMock(pid=1234)
    process.poll.return_value = None

    def spawn(*args, **kwargs):
        import os

        (daemon.directory / "gpu.sock").touch()
        (daemon.directory / "gpu.ready").write_text(f"pid={os.getpid()}\n")
        return process

    monkeypatch.setattr("xinference.model.llm.weight_cache.subprocess.Popen", spawn)
    monkeypatch.setattr("xinference.model.llm.weight_cache.os.killpg", MagicMock())
    monkeypatch.setattr("psutil.process_iter", lambda attrs: [])
    try:
        daemon.start(timeout=1)
        config = json.loads((daemon.directory / "config.yaml").read_text())
        assert "disable-cuda-graph" not in config
        for field in ("cuda-graph-backend-decode", "cuda-graph-backend-prefill"):
            assert config.get(field) == ("disabled" if disabled else None)
    finally:
        daemon.stop()


class FakeCachedModel(WeightCachedModel):
    """Engine double: weights are external, each load creates a new engine."""

    _weight_cache_engine = "vllm"

    def __init__(self):
        self.model_uid = "test"
        self._model_config = {"max_num_seqs": 16}
        self._init_weight_cache({"enable_weight_cache": True, **self._model_config})
        self._weight_cache = SimpleNamespace(client_config=lambda: {}, weights=object())
        self.engine = object()
        self.loads = []
        self.stops = 0
        self.fail_restore = False

    def validate_reload(self, patch):
        validate_reload_patch("vllm", patch)
        return {**self._reload_config, **patch}

    def _sanitize_model_config(self, config):
        return config

    def _stop_engine(self):
        self.stops += 1
        self.engine = None

    def load(self):
        self.loads.append((dict(self._model_config), self._weight_cache.weights))
        if self._model_config["max_num_seqs"] > 32 or self.fail_restore:
            raise RuntimeError("out of memory")
        self.engine = object()

    def wait_for_load(self):
        pass


def test_reload_reuses_weights_and_commits_config():
    model = FakeCachedModel()
    weights, engine = model._weight_cache.weights, model.engine
    stages = []
    model.reload({"max_num_seqs": 32}, stages.append)
    assert model.loads == [({"max_num_seqs": 32}, weights)]
    assert model.engine is not engine
    assert model._reload_config["max_num_seqs"] == 32
    assert model.stops == 1


@pytest.mark.parametrize("fail_restore", [False, True])
def test_reload_failure_restores_previous_config(fail_restore):
    model = FakeCachedModel()
    weights = model._weight_cache.weights
    model.fail_restore = fail_restore
    stages = []
    with pytest.raises(ModelReloadError) as exc:
        model.reload({"max_num_seqs": 64}, stages.append)
    assert exc.value.restored is not fail_restore
    assert stages == ["loading", "restoring"]
    assert model._reload_config == {"max_num_seqs": 16}
    assert all(w is weights for _, w in model.loads)
    assert (model.engine is not None) is not fail_restore


def test_missing_socket_refuses_reload_even_if_launcher_is_alive(monkeypatch):
    daemon = WeightCacheDaemon("vllm", "/models/test", {})
    daemon.process = MagicMock()
    daemon.process.poll.return_value = None
    socket = daemon.directory / "rank.sock"
    socket.touch()
    daemon._socket_paths = {socket: socket.stat().st_ino}
    socket.unlink()
    try:
        with pytest.raises(RuntimeError, match="unavailable"):
            daemon.client_config()
    finally:
        daemon.process = None
        daemon.stop()


@pytest.fixture
def auto_cache_model(monkeypatch):
    from .. import weight_cache

    model = WeightCachedModel()
    model._weight_cache_engine = "vllm"
    model._model_config = {"max_num_seqs": 16}
    model._n_worker = 1
    model.model_path = "/models/test"
    model._init_weight_cache(dict(model._model_config))
    monkeypatch.setattr(weight_cache, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(weight_cache, "package_version", lambda engine: "1.0.0")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    daemon = MagicMock()
    daemon.client_config.return_value = {"load_format": "ipc_cache"}
    monkeypatch.setattr(
        weight_cache, "WeightCacheDaemon", MagicMock(return_value=daemon)
    )
    return model, daemon


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("option", [None, True, "true", False, "false"])
def test_cache_mode_resolves_in_model_environment(auto_cache_model, engine, option):
    model, daemon = auto_cache_model
    model._weight_cache_engine = engine
    model._init_weight_cache({"enable_weight_cache": option, **model._model_config})
    model._prepare_weight_cache()
    enabled = option not in (False, "false")
    assert model.get_reload_config()["enabled"] is enabled
    assert "enable_weight_cache" not in model._model_config
    assert "enable_weight_cache" not in model._reload_config
    assert daemon.start.call_count == int(enabled)


@pytest.mark.parametrize("option", [None, True])
@pytest.mark.parametrize(
    "case",
    [
        "old_engine",
        "platform",
        "cpu",
        "gguf",
        "executor",
        "worker",
        "pipeline",
        "data",
        "nodes",
        "xavier",
        "nixl",
        "pd",
        "sleep",
        "offload",
        "loader",
        "lora",
        "draft",
    ],
)
def test_unsupported_cache_auto_skips_but_explicit_enable_fails(
    auto_cache_model, monkeypatch, option, case
):
    from .. import weight_cache

    model, daemon = auto_cache_model
    if case == "old_engine":
        monkeypatch.setattr(weight_cache, "package_version", lambda engine: "0.30.0")
    elif case == "platform":
        monkeypatch.setattr(weight_cache, "sys", SimpleNamespace(platform="darwin"))
    elif case == "cpu":
        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        )
    elif case == "gguf":
        model.model_spec = SimpleNamespace(model_format="ggufv2")
    elif case == "executor":
        model._get_xinference_executor_backend = lambda: "xoscar"
    elif case == "worker":
        model._n_worker = 2
    else:
        config = {
            "pipeline": {"pipeline_parallel_size": 2},
            "data": {"data_parallel_size": 2},
            "nodes": {"nnodes": 2},
            "xavier": {"_xavier_cache_config": {}},
            "nixl": {"_nixl_config": {}},
            "pd": {"kv_transfer_config": {"kv_role": "kv_producer"}},
            "sleep": {"enable_sleep_mode": True},
            "offload": {"cpu_offload_gb": 1},
            "loader": {"load_format": "ipc_cache"},
            "lora": {"enable_lora": True},
            "draft": {"draft_model_path": "/models/draft"},
        }[case]
        model._model_config.update(config)
    model._init_weight_cache({"enable_weight_cache": option, **model._model_config})
    if option is True:
        with pytest.raises(ValueError):
            model._prepare_weight_cache()
    else:
        model._prepare_weight_cache()
        assert model.get_reload_config()["enabled"] is False
    daemon.start.assert_not_called()


def test_auto_cache_does_not_hide_daemon_startup_failure(auto_cache_model):
    model, daemon = auto_cache_model
    daemon.start.side_effect = RuntimeError("daemon failed")
    with pytest.raises(RuntimeError, match="daemon failed"):
        model._prepare_weight_cache()
    assert model._weight_cache is None


def test_disabled_cache_does_not_probe_environment(auto_cache_model, monkeypatch):
    model, daemon = auto_cache_model
    model._init_weight_cache({"enable_weight_cache": False})
    probe = MagicMock(side_effect=AssertionError("must not probe"))
    monkeypatch.setattr(model, "_weight_cache_unavailable_reason", probe)
    model._prepare_weight_cache()
    probe.assert_not_called()
    daemon.start.assert_not_called()
