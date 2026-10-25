# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from ..backends import lmcache as module
from ..backends.config import normalize_cache_backend
from ..backends.torch.backend import create_snapshot_backend
from ..backends.torch.request_transfer import LayerRead, pack_reads
from ..backends.torch.snapshot import KVSnapshotStore
from .utils import DiskCacheActor, MemoryStorage, mock_lmcache_backend


def options(**overrides):
    return dict(
        cpu_bytes=8192,
        disk_bytes=32768,
        disk_path="/cache",
        read_timeout=1,
        **overrides,
    )


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(module, "_LMCacheStorage", MemoryStorage)
    return create_snapshot_backend(
        8,
        "lmcache",
        options(),
        namespace="deployment-rank1",
        schema={"K": ((4,), torch.float16), "V": ((4,), torch.bfloat16)},
    )


def publish(backend, keys):
    k = torch.arange(len(keys) * 4).reshape(len(keys), 4).half()
    v = (k + 100).bfloat16()
    backend.stage("K", keys, k)
    backend.stage("V", keys, v.view(torch.float16), logical_dtype=torch.bfloat16)
    return backend.publish(keys, {"K", "V"}), k, v


@pytest.mark.parametrize("backend", [None, "native"])
def test_default_backend_is_existing_native_store(backend):
    store = create_snapshot_backend(2, backend or "native")
    assert type(store) is KVSnapshotStore
    store.stage("K", [1], torch.ones(1, 2))
    store.publish([1], {"K"})
    assert store.ready_keys([1, 2, 1]) == [1]
    assert store.has_layer("K", [1], torch.float32)
    assert not store.has_layer("K", [1], torch.float16)
    assert store.reserve("1:read", [1])
    assert not store.can_stage({2, 3})
    store.release("1:read")
    assert store.can_stage({2, 3})


@pytest.mark.parametrize(
    "name,config",
    [
        ("unknown", None),
        ("native", {"cpu_bytes": 1}),
        ("lmcache", []),
        ("lmcache", {}),
        ("lmcache", {"disk_path": "relative", "disk_bytes": 8192}),
        ("lmcache", {"disk_path": "/cache", "disk_bytes": True}),
        (
            "lmcache",
            {"disk_path": "/cache", "disk_bytes": 8192, "read_timeout": float("nan")},
        ),
        ("lmcache", {"disk_path": "/cache", "disk_bytes": 8192, "unknown": 1}),
    ],
)
def test_invalid_backend_config(name, config):
    with pytest.raises(ValueError):
        normalize_cache_backend(name, config)


def test_partial_pages_are_never_published(backend):
    backend.stage("K", [1], torch.ones(1, 4).half())
    assert backend.publish([1], {"K", "V"}) == []
    assert backend.ready_keys([1]) == []
    assert not backend._storage.pages


@pytest.mark.asyncio
async def test_lmcache_roundtrip_and_read_leases(backend):
    accepted, k, v = publish(backend, [11, 22])
    assert accepted == [11, 22]
    assert not backend.blocks  # No second retained CPU cache.
    assert backend.ready_keys([11, 22]) == [11, 22]
    assert backend.publish([11, 22], {"K", "V"}) == [11, 22]  # Directory refresh.
    assert await backend.prefetch([11, 22])
    assert backend.reserve("1:r", [11, 22])
    assert torch.equal(backend.read("K", [11, 22]), k)
    assert torch.equal(backend.read("V", [11, 22]).view(torch.bfloat16), v)
    reads = [LayerRead("V", [11, 22], [0, 1], (4,), torch.float16, torch.bfloat16)]
    assert torch.equal(pack_reads(backend, reads), v.view(torch.uint8).reshape(-1))
    backend.release("1:r")
    assert not backend.blocks and not backend.leases
    assert backend.get_stats()["l2_hits"] == 2
    await backend.close()
    assert backend._storage.closed


@pytest.mark.asyncio
async def test_overlapping_leases_keep_independent_read_copies(backend):
    publish(backend, [1, 2])
    assert await backend.prefetch([1, 2])
    assert backend.reserve("1:a", [1, 2])
    assert backend.reserve("2:b", [2])
    backend.release_consumer(1)
    assert backend.ready == {2}
    backend.release_consumer(2)
    assert not backend.ready


@pytest.mark.asyncio
async def test_lmcache_eviction_is_miss_and_invalidates_directory(backend):
    publish(backend, [1, 2])
    del backend._storage.pages[1]
    assert not await backend.prefetch([1, 2])
    assert backend.ready_keys([1, 2]) == [2]
    assert backend.take_evicted() == [1]
    assert not backend.reserve("r", [1])


@pytest.mark.asyncio
async def test_prefetch_pressure_does_not_evict_active_reader(backend):
    publish(backend, [1, 2])
    assert await backend.prefetch([1])
    assert backend.reserve("1:a", [1])
    publish(backend, [3])
    assert not await backend.prefetch([2, 3])  # Two-slot transient budget.
    assert backend.has_layer("K", [1])
    backend.release("1:a")
    assert await backend.prefetch([2, 3])


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_timed_out_or_cancelled_read_is_drained_and_bounded(backend, cancel):
    publish(backend, [1])
    release = threading.Event()
    original = backend._storage.load

    def slow(keys):
        release.wait(5)
        return original(keys)

    backend._storage.load = slow
    backend._config["read_timeout"] = 0.01 if not cancel else 5
    read = asyncio.create_task(backend.prefetch([1]))
    await asyncio.sleep(0.02)
    if cancel:
        read.cancel()
        with pytest.raises(asyncio.CancelledError):
            await read
    else:
        assert not await read
    assert not await backend.prefetch([1])  # No unbounded queue of I/O jobs.
    assert not backend.leases
    release.set()
    await backend._pending
    await backend.close()


def test_packed_export_owns_pages_independently(backend):
    packed = torch.arange(32, dtype=torch.uint8).reshape(2, 16)
    layers = [("K", (4,), torch.float16, 0, 8), ("V", (4,), torch.bfloat16, 8, 8)]
    backend.stage_packed_blocks([1, 2], packed, layers, retain_storage=True)
    assert backend.publish([1, 2], {"K", "V"}) == [1, 2]
    assert not backend.blocks and not backend._packed_slabs
    packed.zero_()
    assert backend._storage.pages[2].sum() > 0


@pytest.mark.asyncio
async def test_storage_adapter_releases_every_successful_read_lock():
    storage = module._LMCacheStorage.__new__(module._LMCacheStorage)
    storage._closed = threading.Event()
    storage._key = lambda key: key
    storage._layout = None
    storage._group_type = lambda *args: args
    storage._spec_type = lambda *args, **kwargs: args
    released = []
    bitmap = SimpleNamespace(get_indices_list=lambda: [0])
    result = SimpleNamespace(
        hit_cells=[bitmap], found_cells=[bitmap], l1_hit_count=0, l2_hit_count=1
    )

    @contextmanager
    def read(keys):
        yield [SimpleNamespace(tensor=torch.ones(16, dtype=torch.uint8))]

    storage.manager = SimpleNamespace(
        submit_prefetch_task=lambda spec: "task",
        wait_prefetch_status=lambda *args: True,
        query_prefetch_status=lambda task: result,
        read_prefetched_results=read,
        finish_read_prefetched=lambda keys: released.extend(keys),
    )
    pages, absent, _, _ = storage.load([1, 2])
    assert set(pages) == {1} and absent == {2}
    assert released == [1]


def _require_real_lmcache():
    import sys

    if sys.platform != "linux" or not torch.cuda.is_available():
        pytest.skip("LMCache disk integration requires Linux CUDA")
    pytest.importorskip("lmcache")


async def _wait_for_disk_writes(store, path, count):
    deadline = asyncio.get_running_loop().time() + 10
    while True:
        controllers = store._storage.manager.report_status()[
            "store_controllers"
        ].values()
        if len(list(path.rglob("*.data"))) >= count and all(
            status["pending_keys_count"] == 0 and status["in_flight_task_count"] == 0
            for status in controllers
        ):
            return
        assert asyncio.get_running_loop().time() < deadline, "Disk writes did not drain"
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
@pytest.mark.parametrize("read_mode", ["normal", "timeout", "cancel"])
async def test_lmcache_real_disk_roundtrip(tmp_path, read_mode):
    """Run automatically in a Linux CUDA environment with the optional extra."""
    _require_real_lmcache()
    config = dict(
        cpu_bytes=16 << 20,
        disk_bytes=32 << 20,
        disk_path=str(tmp_path),
        read_timeout=10,
    )
    store = create_snapshot_backend(
        8,
        "lmcache",
        config,
        namespace="disk-roundtrip",
        schema={"K": ((4,), torch.float16), "V": ((4,), torch.bfloat16)},
    )
    try:
        accepted, k, v = publish(store, [1, 2])
        assert accepted == [1, 2]
        await _wait_for_disk_writes(store, tmp_path, 2)
        store._storage.manager.clear()
        if read_mode == "timeout":
            store._config["read_timeout"] = 1e-9
            assert not await store.prefetch([1, 2])
            assert store.get_stats()["timeouts"] == 1
        elif read_mode == "cancel":
            reading = asyncio.create_task(store.prefetch([1, 2]))
            await asyncio.sleep(0)
            assert store._pending is not None
            reading.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reading
        if read_mode != "normal":
            pending = store._pending
            if pending is not None:
                await asyncio.shield(pending)
            assert not store.blocks and not store.leases
            # Drained reads must have released every LMCache lock. Otherwise
            # clear() retains those objects and the retry silently hits L1.
            store._storage.manager.clear()
            store._config["read_timeout"] = 10
        assert await store.prefetch([1, 2])
        assert store.reserve("1:disk", [1, 2])
        assert torch.equal(store.read("K", [1, 2]), k)
        assert torch.equal(store.read("V", [1, 2]).view(torch.bfloat16), v)
        assert store.get_stats()["l2_hits"] == 2
        store.release("1:disk")
    finally:
        await store.close()
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_lmcache_real_l1_pressure_keeps_disk_candidates(tmp_path):
    _require_real_lmcache()
    page_bytes = 2 << 20
    store = create_snapshot_backend(
        8,
        "lmcache",
        dict(
            cpu_bytes=16 << 20,
            disk_bytes=64 << 20,
            disk_path=str(tmp_path),
            read_timeout=10,
        ),
        namespace="disk-admission",
        schema={"KV": ((page_bytes,), torch.uint8)},
    )
    payload = torch.ones(2, page_bytes, dtype=torch.uint8)
    payload[0].fill_(7)
    payload[1].fill_(123)
    try:
        store.stage("KV", [1, 2], payload)
        assert store.publish([1, 2], {"KV"}) == [1, 2]
        await _wait_for_disk_writes(store, tmp_path, 2)
        manager = store._storage.manager
        manager.clear()
        held = {}
        # Keep the remaining DRAM write-locked, as during slow disk exports.
        for key in range(10, 30):
            reserved = manager.reserve_write(
                [store._storage._key(key)], store._storage._layout
            )
            if not reserved:
                break
            for obj in reserved.values():
                obj.tensor.zero_()
            held.update(reserved)
        assert held
        assert not await store.prefetch([1, 2])
        assert store.ready_keys([1, 2]) == [1, 2]
        assert store.take_evicted() == []
        manager.finish_write(list(held))
        await _wait_for_disk_writes(store, tmp_path, 2 + len(held))
        manager.clear()
        assert await store.prefetch([1, 2])
        assert store.reserve("1:retry", [1, 2])
        assert torch.equal(store.read("KV", [1, 2]), payload)
        store.release("1:retry")
    finally:
        await store.close()
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_prefetch_rejects_invalid_pages_before_modifying_residency(backend):
    publish(backend, [1, 2])
    backend._storage.pages[2] = torch.ones(16, dtype=torch.float16)
    with pytest.raises(ValueError, match="Invalid Xavier LMCache page"):
        await backend.prefetch([1, 2])
    assert not backend.blocks and not backend.leases
    assert backend._pending is None


@pytest.mark.asyncio
async def test_completed_read_job_does_not_retain_payload(backend):
    publish(backend, [1])
    assert await backend.prefetch([1])
    assert backend._pending is None
    assert backend.reserve("1:r", [1])
    backend.release("1:r")
    assert not backend.blocks


@pytest.mark.asyncio
async def test_shared_page_actor_bounds_reads_and_releases_copies(monkeypatch):
    from ..backends.torch.storage import XavierCacheActor
    from ..contract import KVCacheContract

    config, storages = mock_lmcache_backend(monkeypatch)
    actor = XavierCacheActor(8192, cache_backend=config)
    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 2, 2, 4, 4, "bfloat16"
    )
    namespace = actor.configure(contract.to_dict(), {"layout": "layer_first"})
    nbytes = contract.layer_nbytes * contract.num_layers
    page = torch.arange(nbytes, dtype=torch.int32).to(torch.uint8)
    assert actor.put(namespace, ["a", "b", "c"], [page] * 3) == [True] * 3
    # The disk candidate set can exceed L1; a single owned RPC result is bounded.
    assert actor.exists(namespace, ["a", "b", "c"]) == 3
    a, b, c = await actor.get(namespace, ["a", "b", "c"])
    assert torch.equal(a, page) and torch.equal(b, page) and c is None
    assert actor.get_stats()["used_bytes"] == 0
    assert not actor._store.leases
    a.zero_()
    assert torch.equal((await actor.get(namespace, ["a"]))[0], page)
    storages[0].pages.clear()
    assert await actor.get(namespace, ["a"]) == [None]
    assert actor.exists(namespace, ["a"]) == 0
    with pytest.raises(ValueError, match="P/D handoffs"):
        actor.prepare_handoff(namespace, ["b"])
    await actor.__pre_destroy__()
    assert storages[0].stopped and storages[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "engine,dtype", [("sglang", "float16"), ("trt-llm", "bfloat16")]
)
async def test_shared_actor_real_disk_roundtrip(tmp_path, engine, dtype):
    from ..contract import KVCacheContract

    _require_real_lmcache()
    config = dict(
        name="lmcache",
        config=dict(cpu_bytes=16 << 20, disk_bytes=32 << 20, disk_path=str(tmp_path)),
        namespace="shared-" + engine,
    )
    actor = DiskCacheActor(16 << 20, cache_backend=config)
    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 2, 4, 8, 16, dtype
    )
    try:
        namespace = actor.configure(
            contract.to_dict(), {"engine": engine, "layout": "layer_first"}
        )
        size = contract.layer_nbytes * contract.num_layers // 2
        pages = [
            (torch.arange(size, dtype=torch.float32) + offset)
            .to(getattr(torch, dtype))
            .view(torch.uint8)
            for offset in (0, 100)
        ]
        assert actor.put(namespace, ["first", "second"], pages) == [True, True]
        await actor.force_disk_reads()
        result = await actor.get(namespace, ["first", "second"])
        assert all(
            torch.equal(actual, expected) for actual, expected in zip(result, pages)
        )
        stats = actor.get_stats()
        assert stats["backend_stats"]["l2_hits"] == 2
        assert stats["used_bytes"] == 0
        assert not actor._store.leases
    finally:
        await actor.__pre_destroy__()
    assert not list(tmp_path.iterdir())
