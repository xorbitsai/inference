# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import os
import socket
import tempfile
from pathlib import Path

import pytest

from ..weight_cache import BorrowedWeightCache, WeightCacheDaemon


@pytest.mark.skipif(os.name != "posix", reason="Unix weight-cache sockets")
def test_borrowed_engine_cannot_stop_weight_owner():
    with (
        tempfile.TemporaryDirectory(prefix="xinfw-", dir="/tmp") as directory,
        socket.socket(socket.AF_UNIX) as owner,
    ):
        path = Path(directory) / "cache.sock"
        owner.bind(str(path))
        owner.listen(10)
        lease = BorrowedWeightCache(
            {
                "engine": "vllm",
                "config": {},
                "directory": directory,
                "pid": os.getpid(),
                "sockets": {str(path): path.stat().st_ino},
                "ready": [],
                "memory_fraction": 0.1,
            }
        )
        lease.stop()
        assert lease.client_config()["model_loader_extra_config"]["fallback"] is False
    with pytest.raises(RuntimeError, match="unavailable"):
        lease.client_config()


@pytest.mark.parametrize("explicit", [False, True])
def test_vllm_032_keeps_native_ipc_memory_accounting(monkeypatch, explicit):
    monkeypatch.setattr(
        "xinference.model.llm.weight_cache.package_version", lambda name: "0.32.0"
    )
    daemon = WeightCacheDaemon("vllm", "/models/test", {})
    config = {
        "gpu_memory_utilization": 0.4,
        **({"kv_cache_memory_bytes": 100} if explicit else {}),
    }
    assert daemon.engine_config(config) == config
