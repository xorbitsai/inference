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
    assert proxy.pending_prefix(c.fingerprint, keys) == 1
    # Queued publication is not proof that the store can serve these pages.
    assert await proxy.get(c.fingerprint, keys[:1]) == []
    assert proxy.pending_prefix(c.fingerprint, keys) == 1
    gate.set()
    await proxy.flush()
    assert proxy.get_stats()["pending_bytes"] == 0
    assert proxy.pending_prefix(c.fingerprint, keys) == 0
    assert await proxy.exists(c.fingerprint, keys) == 1
    assert await proxy.get(c.fingerprint, keys[:1]) == [page]
    proxy.record_import(c.fingerprint, 1)
    assert proxy.get_stats()["imported_pages"] == 1
    await proxy.__pre_destroy__()
    await store.__pre_destroy__()


@pytest.fixture
async def history_proxy(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from ..history import XavierHistoryProxy

    backend, _ = mock_lmcache_backend(monkeypatch, cpu_bytes=32768)
    store = XavierHistoryCacheActor(backend)
    c = contract()
    proxy = XavierHistoryProxy({}, 4 * c.num_layers * c.layer_nbytes)
    proxy._ref = SimpleNamespace(
        **{
            method: AsyncMock(side_effect=getattr(store, method))
            for method in ("configure", "exists", "get", "put")
        }
    )
    await proxy.configure(c.to_dict())
    try:
        yield proxy, store, c
    finally:
        await proxy.__pre_destroy__()
        await store.__pre_destroy__()


@pytest.mark.asyncio
async def test_history_proxy_revalidates_evicted_pages_and_republishes(history_proxy):
    proxy, store, c = history_proxy
    keys = prefix_keys(list(range(129)))
    pages = [bytes(c.num_layers * c.layer_nbytes)] * 2
    store._store._directory_capacity = 2
    proxy.put(c.fingerprint, keys, pages)
    await proxy.flush()
    assert await proxy.exists(c.fingerprint, keys) == 2
    store.put(c.fingerprint, prefix_keys([9] * 129), pages)
    assert await proxy.exists(c.fingerprint, keys) == 0
    assert await proxy.get(c.fingerprint, keys) == []
    assert proxy.pending_prefix(c.fingerprint, keys) == 0
    proxy.put(c.fingerprint, keys, pages)
    await proxy.flush()
    assert await proxy.get(c.fingerprint, keys) == pages


@pytest.mark.asyncio
async def test_history_proxy_drops_suffix_without_history_head(history_proxy):
    proxy, store, c = history_proxy
    keys = prefix_keys(list(range(129)))
    page = bytes(c.num_layers * c.layer_nbytes)
    proxy.put(c.fingerprint, keys, [page], start=1)
    assert proxy.pending_prefix(c.fingerprint, keys) == 0
    await proxy.flush()
    proxy._ref.put.assert_not_awaited()
    assert proxy.get_stats()["dropped_pages"] == 1
    assert not proxy._pending
    assert await store.get(c.fingerprint, keys[1:]) == []
    proxy.put(c.fingerprint, keys, [page, page])
    await proxy.flush()
    assert await proxy.get(c.fingerprint, keys) == [page, page]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["exists", "get"])
@pytest.mark.parametrize("failure", ["error", "timeout"])
async def test_history_proxy_read_backoff_does_not_drop_writes(
    history_proxy, method, failure
):
    import asyncio

    proxy, store, c = history_proxy
    proxy.read_timeout = 0.01
    keys = prefix_keys(list(range(129)))
    page = bytes(c.num_layers * c.layer_nbytes)
    read = getattr(proxy._ref, method)

    async def unavailable(*args):
        if failure == "error":
            raise ConnectionError("slow history read")
        await asyncio.Event().wait()

    read.side_effect = unavailable
    if method == "get" and failure == "error":
        # Hold an already queued write while the separate read fails.
        gate = asyncio.Event()

        async def exists(*args):
            await gate.wait()
            return store.exists(*args)

        proxy._ref.exists.side_effect = exists
        proxy.put(c.fingerprint, keys[:1], [page])
    assert await getattr(proxy, method)(c.fingerprint, keys) == (
        0 if method == "exists" else []
    )
    if method == "get" and failure == "error":
        gate.set()
    elif method == "exists":
        read.side_effect = store.exists
    # A subsequent publication also remains available during read backoff.
    assert proxy.put(c.fingerprint, keys, [page, page]) == [True, True]
    await proxy.flush()
    assert await store.get(c.fingerprint, keys) == [page, page]
    assert proxy.get_stats()["dropped_pages"] == 0
    calls = read.await_count
    assert await proxy.get(c.fingerprint, keys) == []
    assert await proxy.exists(c.fingerprint, keys) == 0
    assert read.await_count == calls


@pytest.mark.asyncio
async def test_history_proxy_write_backoff_does_not_disable_reads(history_proxy):
    proxy, store, c = history_proxy
    keys = prefix_keys(list(range(129)))
    pages = [bytes(c.num_layers * c.layer_nbytes)] * 2
    store.put(c.fingerprint, keys, pages)
    proxy._ref.put.side_effect = ConnectionError("write unavailable")
    other = prefix_keys([9] * 129)
    proxy.put(c.fingerprint, other, pages)
    await proxy.flush()
    assert proxy.put(c.fingerprint, other, pages) == [False, False]
    assert await proxy.exists(c.fingerprint, keys) == 2
    assert await proxy.get(c.fingerprint, keys) == pages


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_history_proxy_shutdown_releases_stalled_writer(history_proxy, cancel):
    import asyncio

    proxy, _, c = history_proxy
    proxy.read_timeout = 0.01
    writer = proxy._writer = asyncio.create_task(asyncio.Event().wait())
    keys = prefix_keys(list(range(129)))
    page = bytes(c.num_layers * c.layer_nbytes)
    proxy.put(c.fingerprint, keys, [page, page])
    closing = asyncio.create_task(proxy.__pre_destroy__())
    if cancel:
        await asyncio.sleep(0)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
    else:
        await asyncio.wait_for(closing, 2)
    assert writer.cancelled()
    assert not proxy._queue and not proxy._pending
    assert proxy.get_stats()["pending_bytes"] == 0
    assert proxy.put(c.fingerprint, keys, [page, page]) == [False, False]


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
