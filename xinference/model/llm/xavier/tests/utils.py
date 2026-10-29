# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Storage stand-in and L2 controls used by Xavier integration tests."""

import asyncio
import json
import os
from pathlib import Path

from ..backends import lmcache
from ..backends.torch.storage import XavierCacheActor


class MemoryStorage:
    def __init__(self, config, namespace, page_bytes):
        self.pages = {}
        self.namespace = namespace
        self.stopped = self.closed = False

    def store(self, key, page):
        self.pages[key] = page.clone()
        return True

    def load(self, keys):
        pages = {key: self.pages[key].clone() for key in keys if key in self.pages}
        return pages, set(keys) - pages.keys(), 0, len(pages)

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def mock_lmcache_backend(monkeypatch, namespace="deployment", cpu_bytes=8192):
    storages = []

    class Storage(MemoryStorage):
        def __init__(self, *args):
            super().__init__(*args)
            storages.append(self)

    monkeypatch.setattr(lmcache, "_LMCacheStorage", Storage)
    monkeypatch.setattr(lmcache, "check_lmcache_available", lambda: None)
    return {
        "name": "lmcache",
        "config": {
            "cpu_bytes": cpu_bytes,
            "disk_bytes": 32768,
            "disk_path": "/cache",
        },
        "namespace": namespace,
    }, storages


class DiskCacheActor(XavierCacheActor):
    async def force_disk_reads(self):
        """Drain filesystem writes and remove L1 objects before the consumer runs."""
        storage = self._store._storage
        deadline = asyncio.get_running_loop().time() + 30
        while True:
            controllers = storage.manager.report_status()["store_controllers"].values()
            if len(list(storage._disk_path.rglob("*.data"))) >= self._store.get_stats()[
                "pages"
            ] and all(
                state["pending_keys_count"] == 0 and state["in_flight_task_count"] == 0
                for state in controllers
            ):
                break
            assert (
                asyncio.get_running_loop().time() < deadline
            ), "Disk writes did not drain"
            await asyncio.sleep(0.05)
        storage.manager.clear()


def mooncake_backend_config(cpu_bytes=64 << 20):
    """Read explicit test-service configuration; ordinary CI needs no Mooncake."""
    import pytest

    location = os.environ.get("XINFERENCE_TEST_MOONCAKE_CONFIG")
    if not location:
        pytest.skip("requires XINFERENCE_TEST_MOONCAKE_CONFIG JSON file")
    return dict(
        l2_backend="mooncake",
        cpu_bytes=cpu_bytes,
        storage_bytes=128 << 20,
        read_timeout=10,
        mooncake_config=json.loads(Path(location).read_text()),
    )


class RemoteCacheActor(XavierCacheActor):
    async def force_remote_reads(self):
        storage = self._store._storage
        deadline = asyncio.get_running_loop().time() + 30
        while any(
            state["pending_keys_count"] or state["in_flight_task_count"]
            for state in storage.manager.report_status()["store_controllers"].values()
        ):
            assert (
                asyncio.get_running_loop().time() < deadline
            ), "Remote writes did not drain"
            await asyncio.sleep(0.05)
        storage.manager.clear()
