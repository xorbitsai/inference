# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import sys
from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch

from ..backends import lmcache
from ..backends.config import normalize_cache_backend


def mooncake_options(**overrides):
    config = dict(
        l2_backend="mooncake",
        cpu_bytes=4096,
        storage_bytes=4096,
        read_timeout=0.1,
        mooncake_config=dict(
            local_hostname="client",
            metadata_server="http://127.0.0.1:18080/metadata",
            master_server_addr="127.0.0.1:15051",
            protocol="tcp",
            global_segment_size=16 << 20,
        ),
    )
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    "overrides",
    [
        {"l2_backend": "unknown"},
        {"storage_bytes": True},
        {"storage_bytes": 0},
        {"disk_path": "/cache"},
        {"disk_bytes": 8192},
        {"mooncake_config": []},
        {"mooncake_config": {"protocol": "tcp"}},
        {
            "mooncake_config": {
                **mooncake_options()["mooncake_config"],
                "protocol": "unknown",
            }
        },
        {
            "mooncake_config": {
                **mooncake_options()["mooncake_config"],
                "global_segment_size": True,
            }
        },
        {"mooncake_config": {**mooncake_options()["mooncake_config"], "extra": {}}},
    ],
)
def test_invalid_mooncake_configuration(overrides):
    with pytest.raises(ValueError):
        normalize_cache_backend("lmcache", mooncake_options(**overrides))


def test_storage_selection_preserves_filesystem_defaults():
    _, config = normalize_cache_backend(
        "lmcache", dict(disk_path="/cache", disk_bytes=8192)
    )
    assert "l2_backend" not in config  # Keep legacy launch options unchanged.
    original = mooncake_options()
    _, config = normalize_cache_backend("lmcache", original)
    assert config["mooncake_config"]["global_segment_size"] == str(16 << 20)
    assert original["mooncake_config"]["global_segment_size"] == 16 << 20
    with pytest.raises(ValueError, match="require l2_backend"):
        normalize_cache_backend(
            "lmcache", dict(disk_path="/cache", disk_bytes=8192, mooncake_config={})
        )


@pytest.fixture
def storage_runtime(monkeypatch):
    """Exercise our real adapter constructor without native optional libraries."""
    pool = {}
    managers = []

    def config(*args, **kwargs):
        return SimpleNamespace(args=args, **kwargs)

    class Manager:
        def __init__(self, options):
            self.options = options
            self.pending = {}
            self.closed = False
            self.attached = True
            self.deleted = []
            managers.append(self)

        def reserve_write(self, keys, layout):
            count = layout.args[0][0][0]
            self.pending.update(
                {
                    key: SimpleNamespace(tensor=torch.empty(count, dtype=torch.uint8))
                    for key in keys
                }
            )
            return {key: self.pending[key] for key in keys}

        def finish_write(self, keys):
            for key in keys:
                pool[key] = self.pending.pop(key).tensor.clone()

        def report_status(self):
            return {
                "store_controllers": {
                    "l1": {
                        "pending_keys_count": len(self.pending),
                        "in_flight_task_count": 0,
                    }
                }
            }

        def l2_adapters(self):
            return [(SimpleNamespace(index=0), self)] if self.attached else []

        def delete_l2_adapter(self, index, timeout):
            assert index == 0
            self.attached = False

        def delete(self, keys):
            self.deleted.extend(keys)
            for key in keys:
                pool.pop(key, None)

        def close(self):
            self.closed = True

    modules = {
        "lmcache.v1.distributed.api": dict(
            ObjectKey=namedtuple("Key", "chunk_hash model_name kv_rank"),
            GroupedObjectKeys=config,
            PrefetchTaskSpec=config,
            MemoryLayoutDesc=config,
        ),
        "lmcache.v1.distributed.config": dict(
            EvictionConfig=config,
            L1ManagerConfig=config,
            L1MemoryManagerConfig=config,
            StorageManagerConfig=config,
        ),
        "lmcache.v1.distributed.l2_adapters.config": dict(L2AdaptersConfig=config),
        "lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter": dict(
            FSNativeL2AdapterConfig=config
        ),
        "lmcache.v1.distributed.l2_adapters.mooncake_store_l2_adapter": dict(
            MooncakeStoreL2AdapterConfig=config
        ),
        "lmcache.v1.distributed.storage_manager": dict(StorageManager=Manager),
        "lmcache.lmcache_mooncake": dict(LMCacheMooncakeClient=object),
        "lmcache.lmcache_fs": dict(LMCacheFSClient=object),
    }
    for name, attrs in modules.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**attrs))
    monkeypatch.setattr(lmcache, "check_lmcache_available", lambda: None)
    return pool, managers


def make_storage(namespace="first"):
    _, config = normalize_cache_backend("lmcache", mooncake_options())
    return lmcache._LMCacheStorage(config, namespace, 16)


def test_mooncake_uses_native_l2_without_requiring_filesystem_extension(
    storage_runtime, monkeypatch
):
    monkeypatch.setitem(sys.modules, "lmcache.lmcache_fs", None)
    pool, managers = storage_runtime
    first, second = make_storage(), make_storage("second")
    page = torch.arange(16, dtype=torch.uint8)
    try:
        assert first.store(1, page) and second.store(1, page + 1)
        assert len(pool) == 2
        adapter = managers[0].options.l2_adapter_config.args[0][0]
        assert adapter.setup_config["protocol"] == "tcp"
        assert managers[0].options.l1_manager_config.args[0].use_lazy is False
        first.close()
        assert list(pool) == [second._key(1)]
        assert torch.equal(pool[second._key(1)], page + 1)
        assert managers[0].closed
    finally:
        second.close()
    assert not pool


def test_remote_eviction_bounds_ownership_and_allows_later_admission(storage_runtime):
    pool, _ = storage_runtime
    storage = make_storage()
    page = torch.zeros(16, dtype=torch.uint8)
    try:
        assert storage.store(1, page) and storage.store(2, page)
        assert not storage.store(3, page)
        storage._delete_future.result(timeout=2)
        assert storage.store(3, page)
        assert set(storage._remote_keys) == {2, 3}
        assert storage._key(1) not in pool
        assert len(pool) == 2
    finally:
        storage.close()
    assert not pool


def test_missing_native_mooncake_extension_has_actionable_error(
    storage_runtime, monkeypatch
):
    monkeypatch.setitem(sys.modules, "lmcache.lmcache_mooncake", None)
    with pytest.raises(RuntimeError, match="BUILD_MOONCAKE=1"):
        make_storage()


def test_cleanup_failure_still_closes_client_and_preserves_other_namespaces(
    storage_runtime,
):
    pool, _ = storage_runtime
    storage = make_storage()
    storage.store(1, torch.zeros(16, dtype=torch.uint8))
    storage.manager.pending["blocked"] = None
    storage._read_timeout = 0
    storage.close()
    assert storage.manager.closed
    assert (
        len(pool) == 1
    )  # Cleanup is best effort when outstanding writes cannot drain.


def test_failed_repeated_write_keeps_existing_remote_ownership(storage_runtime):
    pool, _ = storage_runtime
    storage = make_storage()
    page = torch.zeros(16, dtype=torch.uint8)
    try:
        assert storage.store(1, page)
        storage.manager.reserve_write = lambda *args: {}
        assert not storage.store(1, page)
        assert list(storage._remote_keys) == [1]
    finally:
        storage.close()
    assert not pool
