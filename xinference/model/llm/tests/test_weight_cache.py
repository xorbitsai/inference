# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import json
import pickle
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ..weight_cache import (
    ModelReloadError,
    WeightCacheDaemon,
    WeightCachedModel,
    parse_weight_cache_option,
    validate_reload_patch,
)


@pytest.mark.parametrize(
    "value, expected", [(True, True), (False, False), ("true", True), ("FALSE", False)]
)
def test_cache_option(value, expected):
    assert parse_weight_cache_option(value) is expected


@pytest.mark.parametrize("value", [None, 0, 1, "yes", {}])
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


def test_reload_error_survives_actor_serialization():
    error = pickle.loads(pickle.dumps(ModelReloadError("oom", restored=True)))
    assert error.restored is True
    assert str(error) == "oom"


@pytest.mark.parametrize(
    "engine,tp", [("vllm", "tensor_parallel_size"), ("sglang", "tp_size")]
)
def test_daemon_lifecycle(monkeypatch, engine, tp):
    daemon = WeightCacheDaemon(
        engine, "/models/test", {tp: 2, "launch_timeout": 300, "unused": None}
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
    monkeypatch.setenv("SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE", "previous")
    daemon.start(timeout=1)
    config = json.loads((daemon.directory / "config.yaml").read_text())
    assert "launch_timeout" not in config and "unused" not in config
    command, options = calls[0]
    assert options["start_new_session"]
    assert "xinference.model.llm.weight_cache" in command
    if engine == "vllm":
        client = daemon.client_config()
        assert client["load_format"] == "ipc_cache"
        assert client["model_loader_extra_config"]["fallback"] is False
        assert client["model_loader_extra_config"]["mode"] == "zero_copy"
    else:
        assert daemon.client_config() == {"weight_cache_mode": "client"}
        assert "{device_uuid}" in options["env"]["SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE"]
    daemon.stop()
    daemon.stop()
    assert kill.call_count == 2
    assert not daemon.directory.exists()
    import os

    assert os.environ["SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE"] == "previous"


def test_daemon_failure_restores_environment(monkeypatch):
    import os

    monkeypatch.setenv("SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE", "original")
    daemon = WeightCacheDaemon("sglang", "/models/test", {"not_serializable": object()})
    with pytest.raises(TypeError):
        daemon.start()
    assert os.environ["SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE"] == "original"
    assert not daemon.directory.exists()


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
