# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
"""History storage is optional; live P/D ownership never belongs to LMCache."""

from dataclasses import replace

import pytest

from ..backends.bytes.history import XavierHistoryCacheActor
from ..prefix import prefix_keys
from .test_bytes_storage import contract
from .utils import mock_lmcache_backend


@pytest.mark.asyncio
async def test_real_history_actor_proxy_roundtrip_through_disk(tmp_path):
    import xoscar as xo

    from ..history import XavierHistoryProxy
    from .test_backends import _require_real_lmcache
    from .utils import DiskHistoryCacheActor

    _require_real_lmcache()

    c = contract()
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        store = await xo.create_actor(
            DiskHistoryCacheActor,
            dict(
                name="lmcache",
                namespace="pd-history",
                config=dict(
                    cpu_bytes=16 << 20,
                    disk_bytes=32 << 20,
                    disk_path=str(tmp_path),
                    read_timeout=10,
                ),
            ),
            address=pool.external_address,
        )
        refs = [store]
        try:
            proxy = await xo.create_actor(
                XavierHistoryProxy,
                dict(address=store.address, uid=store.uid),
                16 << 20,
                address=pool.external_address,
            )
            refs.append(proxy)
            await proxy.configure(c.to_dict())
            keys = prefix_keys(list(range(129)), cache_salt="a")
            pages = [bytes([i + 1]) * c.num_layers * c.layer_nbytes for i in range(2)]
            assert await proxy.put(c.fingerprint, keys, pages) == [True, True]
            await proxy.flush()
            await store.force_disk_reads()
            # A fresh P proxy discovers the same L2 entries without local metadata.
            fresh = await xo.create_actor(
                XavierHistoryProxy,
                dict(address=store.address, uid=store.uid),
                16 << 20,
                address=pool.external_address,
            )
            refs.append(fresh)
            await fresh.configure(c.to_dict())
            assert await fresh.exists(c.fingerprint, keys) == 2
            assert await fresh.get(c.fingerprint, keys) == pages
            assert (
                await fresh.get(
                    c.fingerprint, prefix_keys(list(range(129)), cache_salt="b")
                )
                == []
            )
            assert (await store.get_stats())["backend_stats"]["l2_hits"] == 2
        finally:
            for ref in reversed(refs):
                await xo.destroy_actor(ref)
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
async def test_history_owns_canonical_pages_after_l2_prefetch(monkeypatch, dtype):
    backend, storages = mock_lmcache_backend(monkeypatch, cpu_bytes=32768)
    actor = XavierHistoryCacheActor(backend)
    c = replace(contract(), logical_dtype=dtype)
    meta = actor.configure(c.to_dict())
    assert meta["namespace"] == c.fingerprint
    keys = prefix_keys(list(range(193)), cache_salt="customer-a")
    pages = [bytes([i + 1]) * (c.num_layers * c.layer_nbytes) for i in range(3)]
    assert actor.put(c.fingerprint, keys, pages) == [True] * 3
    assert not actor._store.ready  # Publication released the transient copies.
    assert await actor.get(c.fingerprint, keys) == pages
    assert not actor._store.ready and not actor._store.leases
    assert actor.get_stats()["backend_stats"]["l2_hits"] == 3
    assert (
        await actor.get(
            c.fingerprint, prefix_keys(list(range(193)), cache_salt="customer-b")
        )
        == []
    )
    # A later suffix can refresh old metadata without another cache payload read.
    assert actor.put(c.fingerprint, keys, pages[2:], 2) == [True]
    with pytest.raises(ValueError, match="namespace differs"):
        actor.configure(replace(c, weights_fingerprint="f" * 64).to_dict())
    await actor.__pre_destroy__()
    assert storages[0].stopped and storages[0].closed


@pytest.mark.asyncio
async def test_history_stops_at_missing_prefix_and_preserves_owned_result(monkeypatch):
    backend, _ = mock_lmcache_backend(monkeypatch, cpu_bytes=32768)
    actor = XavierHistoryCacheActor(backend)
    c = contract()
    actor.configure(c.to_dict())
    keys = prefix_keys(list(range(193)))
    page = bytes(c.num_layers * c.layer_nbytes)
    actor.put(c.fingerprint, keys[:1], [page])
    actor.put(c.fingerprint, keys[2:], [page])
    assert actor.exists(c.fingerprint, keys) == 1
    owned = await actor.get(c.fingerprint, keys)
    assert owned == [page]
    actor._store._storage.pages.clear()
    assert owned == [page]
    assert await actor.get(c.fingerprint, keys) == []
    with pytest.raises(ValueError, match="native Xavier backend"):
        actor.prepare_handoff(c.fingerprint, keys)
    await actor.__pre_destroy__()


@pytest.mark.asyncio
async def test_history_proxy_put_does_not_wait_for_l2_and_bounds_queued_copies(
    monkeypatch,
):
    import asyncio
    from types import SimpleNamespace

    from ..history import XavierHistoryProxy

    backend, _ = mock_lmcache_backend(monkeypatch, cpu_bytes=32768)
    store = XavierHistoryCacheActor(backend)
    gate = asyncio.Event()
    c = contract()

    async def configure(config):
        return store.configure(config)

    async def exists(*args):
        await gate.wait()
        return store.exists(*args)

    async def put(*args):
        return store.put(*args)

    proxy = XavierHistoryProxy({}, c.num_layers * c.layer_nbytes)
    proxy._ref = SimpleNamespace(
        configure=configure, exists=exists, put=put, get=store.get
    )
    await proxy.configure(c.to_dict())
    keys = prefix_keys(list(range(129)))
    page = bytes(c.num_layers * c.layer_nbytes)
    assert proxy.put(c.fingerprint, keys[:1], [page]) == [True]
    await asyncio.sleep(0)
    assert not gate.is_set() and proxy.get_stats()["pending_bytes"] == len(page)
    assert proxy.put(c.fingerprint, keys[1:], [page]) == [False]
    assert proxy.exists_local(c.fingerprint, keys) == 1
    assert await proxy.get(c.fingerprint, keys[:1]) == []
    assert proxy.exists_local(c.fingerprint, keys) == 0
    gate.set()
    await proxy.flush()
    assert proxy.get_stats()["pending_bytes"] == 0
    assert proxy.exists_local(c.fingerprint, keys) == 1
    assert await proxy.get(c.fingerprint, keys[:1]) == [page]
    proxy.record_import(c.fingerprint, 1)
    assert proxy.get_stats()["imported_pages"] == 1
    await proxy.__pre_destroy__()
    await store.__pre_destroy__()


@pytest.mark.asyncio
async def test_history_write_failure_drops_only_history_and_allows_prefill(monkeypatch):
    from types import SimpleNamespace

    from ..history import XavierHistoryProxy

    c = contract()

    async def configure(config):
        return {"namespace": c.fingerprint, "capacity_pages": 4, "max_keys": 4096}

    async def unavailable(*args):
        raise ConnectionError("L2 is unavailable")

    proxy = XavierHistoryProxy({}, 4 * c.num_layers * c.layer_nbytes)
    proxy._ref = SimpleNamespace(configure=configure, exists=unavailable)
    await proxy.configure(c.to_dict())
    keys = prefix_keys(list(range(129)))
    page = bytes(c.num_layers * c.layer_nbytes)
    assert proxy.put(c.fingerprint, keys, [page, page]) == [True, True]
    await proxy.flush()
    assert proxy.get_stats()["dropped_pages"] == 2
    assert proxy.get_stats()["pending_bytes"] == 0
    assert await proxy.exists(c.fingerprint, keys) == 0
    assert await proxy.get(c.fingerprint, keys) == []
    await proxy.__pre_destroy__()
