# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
"""Pressure, expiry and actual restore semantics at the shared storage boundary."""

import pytest
import torch

from ..backends.bytes.history import XavierHistoryCacheActor
from ..backends.bytes.storage import XavierBytesCacheActor
from ..backends.torch.storage import XavierCacheActor
from ..cache_lifecycle import SessionPrefixPolicy
from .test_bytes_storage import contract
from .utils import mock_lmcache_backend


def session(phase="active", revision=0, id="agent"):
    return dict(id=id, revision=revision, phase=phase, scope="credential")


@pytest.mark.asyncio
async def test_concurrent_opaque_reads_share_one_staging_slot(monkeypatch):
    import asyncio

    backend, _ = mock_lmcache_backend(monkeypatch, cpu_bytes=4096)
    actor = XavierHistoryCacheActor(backend)
    namespace = "a" * 64
    actor.configure_pages(namespace, 4096)
    keys = [f"{i:064x}" for i in [1, 2]]
    pages = [bytes([i]) * 4096 for i in [1, 2]]
    assert actor.put(namespace, keys, pages) == [True, True]
    try:
        for _ in range(10):
            results = await asyncio.gather(
                *(actor.get(namespace, [key]) for key in keys)
            )
            assert results == [[page] for page in pages]
        assert not actor._store.ready and not actor._store.leases
        assert actor.get_stats()["missed_pages"] == 0
    finally:
        await actor.__pre_destroy__()


@pytest.mark.parametrize("backend", ["bytes", "torch"])
@pytest.mark.parametrize("end", ["waiting", "finished", "expired"])
def test_waiting_prefix_survives_pressure_but_does_not_pin_capacity(backend, end):
    c = contract()
    page_bytes = c.num_layers * c.layer_nbytes
    keys = [f"{i:064x}" for i in range(8)]
    if backend == "bytes":
        actor = XavierBytesCacheActor(3 * page_bytes)
        namespace = actor.configure(c.to_dict())["namespace"]
        page = bytes(page_bytes)
    else:
        actor = XavierCacheActor(3 * page_bytes)
        namespace = actor.configure(c.to_dict(), {"layout": "layer_first"})
        page = torch.zeros(page_bytes, dtype=torch.uint8)
    actor.put(namespace, keys[:2], [page, page])
    actor.bind_cache_session(session(), keys[:2])
    actor.update_cache_session(session("waiting", 1))
    if end == "finished":
        actor.update_cache_session(session(end, 2))
    elif end == "expired":
        actor._sessions.registry.clock = lambda: float("inf")
    for key in keys[2:5]:
        actor.put(namespace, [key], [page])
    assert actor.exists(namespace, keys[:2]) == (2 if end == "waiting" else 0)
    assert actor.exists(namespace, keys[4:5]) == 1
    # Even a cache entirely occupied by waiting prefixes remains allocatable.
    actor.bind_cache_session(session(id="other"), keys[4:5])
    actor.update_cache_session(session("waiting", 1, id="other"))
    assert actor.put(namespace, keys[5:6], [page]) == [True]


def _bind_with_stale_sessions(policy: SessionPrefixPolicy, end: str) -> list[str]:
    now = [0.0]
    policy.registry.clock = lambda: now[0]
    keys = [f"{i:064x}" for i in range(4200)]
    assert policy.bind(session("waiting", id="live"), keys[:1000])
    now[0] = 10.0
    for i in range(3):
        assert policy.bind(
            session(id=f"stale-{i}"), keys[1000 * (i + 1) : 1000 * (i + 2)]
        )
    assert policy.key_count == 4000
    now[0] = 240.0
    assert policy.registry.update(session("waiting", id="live"))
    if end == "finished":
        for i in range(3):
            assert policy.registry.update(session("finished", 1, id=f"stale-{i}"))
    now[0] = 320.0
    # The live hint lasts until 540; older hints expired at 310 or finished.
    # Bind before any storage operation gets a chance to prune the old hints.
    assert policy.bind(session(id="new"), keys[4000:])
    return keys


@pytest.mark.parametrize("end", ["expired", "finished"])
def test_stale_bindings_are_pruned_before_session_key_limit(end):
    policy = SessionPrefixPolicy()
    keys = _bind_with_stale_sessions(policy, end)
    assert policy.key_count == 1200
    assert set(policy.prefixes) == {"live", "new"}
    assert policy.waiting_keys() == list(reversed(keys[:1000]))


@pytest.mark.parametrize("backend", ["bytes", "torch"])
@pytest.mark.parametrize("end", ["expired", "finished"])
def test_stale_binding_page_is_evicted_before_live_waiting_page(backend, end):
    c = contract()
    page_bytes = c.num_layers * c.layer_nbytes
    live, stale, new = [f"{i:064x}" for i in (0, 1000, 4000)]
    if backend == "bytes":
        actor = XavierBytesCacheActor(2 * page_bytes)
        namespace = actor.configure(c.to_dict())["namespace"]
        page = bytes(page_bytes)
    else:
        actor = XavierCacheActor(2 * page_bytes)
        namespace = actor.configure(c.to_dict(), {"layout": "layer_first"})
        page = torch.zeros(page_bytes, dtype=torch.uint8)
    # Make the live page the oldest, so only its waiting hint saves it.
    assert actor.put(namespace, [live], [page]) == [True]
    assert actor.put(namespace, [stale], [page]) == [True]
    _bind_with_stale_sessions(actor._sessions, end)
    assert actor.put(namespace, [new], [page]) == [True]
    assert actor.exists(namespace, [live]) == 1
    assert actor.exists(namespace, [stale]) == 0
    assert actor.exists(namespace, [new]) == 1
    assert actor.get_stats()["evicted_pages"] == 1


def test_session_bindings_are_bounded_and_ignore_reordered_hints():
    policy = SessionPrefixPolicy()
    keys = [f"{i:064x}" for i in range(4096)]
    assert policy.bind(session(), keys)
    assert policy.registry.update(session("finished", 2))
    assert not policy.bind(session(revision=1), keys)
    assert policy.waiting_keys() == []
    assert policy.key_count == 0
    for i in range(8):
        policy.bind(session(id=str(i)), keys[:1024])
    assert policy.key_count <= 4096
    assert len(policy.registry.states) <= 4096
    with pytest.raises(ValueError, match="prefix keys"):
        policy.bind(session(), ["unvalidated"])


@pytest.mark.asyncio
async def test_opaque_lmcache_restore_exceeds_ram_and_reconciles_eviction(monkeypatch):
    backend, storages = mock_lmcache_backend(monkeypatch, cpu_bytes=4096)
    actor = XavierHistoryCacheActor(backend)
    namespace = "a" * 64
    meta = actor.configure_pages(namespace, 4096)
    assert meta["capacity_pages"] > 1
    keys = [f"{i:064x}" for i in range(3)]
    pages = [bytes([i + 1]) * 4096 for i in range(3)]
    assert actor.put(namespace, keys, pages) == [True] * 3
    actor.bind_cache_session(session(), keys)
    actor.update_cache_session(session("waiting", 1))
    if actor._session_warmup is not None:
        await actor._session_warmup
    assert await actor.get(namespace, keys) == pages
    assert actor.get_stats()["backend_stats"]["l2_hits"] >= 3
    assert actor._store.leases == {}
    # Independent L2 eviction leaves optimistic metadata until an actual read.
    storage_keys = actor._keys(namespace, keys)
    storages[0].pages.pop(storage_keys[1])
    assert actor.contains(namespace, keys) == [True] * 3
    assert await actor.get(namespace, keys) == pages[:1]
    assert actor.contains(namespace, keys) == [True, False, True]
    assert actor.put(namespace, keys[1:2], pages[1:2]) == [True]
    assert await actor.get(namespace, keys) == pages
    with pytest.raises(ValueError, match="namespace"):
        actor.configure_pages("b" * 64, 4096)
    with pytest.raises(ValueError, match="native"):
        actor.prepare_handoff(namespace, keys)
    await actor.__pre_destroy__()


@pytest.mark.asyncio
async def test_waiting_disk_warmup_does_not_delay_control_or_escape_shutdown(
    monkeypatch,
):
    backend, storages = mock_lmcache_backend(monkeypatch, cpu_bytes=4096)
    actor = XavierHistoryCacheActor(backend)
    actor.configure_pages("a" * 64, 4096)
    actor.bind_cache_session(session(), ["1" * 64])
    assert actor.update_cache_session(session("waiting", 1))
    await actor.__pre_destroy__()
    assert actor._session_warmup.done()
    assert storages[0].closed


@pytest.mark.asyncio
async def test_real_disk_restore_does_not_fill_l1_with_earlier_read_batches(tmp_path):
    import xoscar as xo

    from .test_backends import _require_real_lmcache
    from .utils import DiskHistoryCacheActor

    _require_real_lmcache()
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        actor = await xo.create_actor(
            DiskHistoryCacheActor,
            dict(
                name="lmcache",
                namespace="opaque-pressure",
                config=dict(
                    cpu_bytes=16 << 20,
                    disk_bytes=64 << 20,
                    disk_path=str(tmp_path),
                    read_timeout=10,
                ),
            ),
            address=pool.external_address,
        )
        try:
            namespace = "a" * 64
            await actor.configure_pages(namespace, 2 << 20)
            keys = [f"{i:064x}" for i in range(6)]
            pages = [bytes([i + 1]) * (2 << 20) for i in range(6)]
            assert await actor.put(namespace, keys, pages) == [True] * 6
            await actor.force_disk_reads()
            await actor.fill_l1_below_watermark(6)
            assert await actor.get(namespace, keys) == pages
            assert (await actor.get_stats())["backend_stats"]["l2_hits"] == 6
        finally:
            await xo.destroy_actor(actor)


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_head_metadata", [False, True])
async def test_rewrite_reconciles_entire_stale_suffix_beyond_ram(
    monkeypatch, missing_head_metadata
):
    backend, storages = mock_lmcache_backend(monkeypatch, cpu_bytes=4096)
    actor = XavierHistoryCacheActor(backend)
    namespace = "a" * 64
    actor.configure_pages(namespace, 4096)
    keys = [f"{i:064x}" for i in range(3)]
    pages = [bytes([i + 1]) * 4096 for i in range(3)]
    assert actor.put(namespace, keys, pages) == [True] * 3
    storages[0].pages.clear()
    if missing_head_metadata:
        actor._store._known.pop(actor._keys(namespace, keys[:1])[0])
    assert await actor.verified_prefix(namespace, keys) == 0
    assert actor.put(namespace, keys, pages) == [True] * 3
    assert await actor.get(namespace, keys) == pages
    await actor.__pre_destroy__()


def test_opaque_registration_cannot_reinterpret_canonical_pages(monkeypatch):
    backend, _ = mock_lmcache_backend(monkeypatch, cpu_bytes=32768)
    actor = XavierHistoryCacheActor(backend)
    c = contract()
    actor.configure(c.to_dict())
    with pytest.raises(ValueError, match="namespace"):
        actor.configure_pages(c.fingerprint, c.layer_nbytes * c.num_layers)
