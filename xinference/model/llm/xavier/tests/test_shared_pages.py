# CI: linux
# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import os

import pytest
import torch
import xoscar as xo

from ..backends.torch import shared_pages
from ..backends.torch.shared_pages import SharedPageArena, map_pages
from ..backends.torch.storage import XavierCacheActor
from ..contract import KVCacheContract


@pytest.fixture
def arena(tmp_path):
    store = SharedPageArena(str(tmp_path), "pages", 8, 2)
    yield store
    store.pages = None


def test_writers_publish_only_committed_slots(arena):
    first, second, third = arena.allocate([1, 2, 3])
    assert third is None  # Two slots, both being written.
    assert arena.ready_keys([1, 2]) == []
    arena.pages[first].fill_(7)
    assert arena.commit([1], [first]) == 1
    arena.abort([2], [second])
    assert arena.allocate([1]) == [-1]  # Stored keys are never copied again.
    # Another process maps the same bytes.
    info = arena.info()
    peer = map_pages(
        info["path"], bytes.fromhex(info["nonce"]), info["page_bytes"], info["slots"]
    )
    assert peer[first].tolist() == [7] * 8
    with pytest.raises(ValueError, match="another cache"):
        map_pages(info["path"], os.urandom(16), 8, 2)


def test_eviction_skips_pins_and_writers(arena, monkeypatch):
    slots = arena.allocate([1, 2])
    arena.commit([1, 2], slots)
    assert arena.reserve("reader", [1])
    (slot,) = arena.allocate([3])
    assert slot == slots[1] and arena.ready_keys([1, 2]) == [1]
    assert arena.allocate([4]) == [None]  # Key 1 is pinned, key 3 being written.
    arena.release("reader")
    # A writer that never commits gives its slot back after the timeout.
    later = shared_pages.time.monotonic() + 1000
    monkeypatch.setattr(shared_pages.time, "monotonic", lambda: later)
    assert arena.allocate([5]) == [slot]
    assert arena.allocate([6]) == [slots[0]]
    assert sorted(arena.take_evicted()) == [1, 2]


@pytest.mark.asyncio
async def test_cache_actor_shares_page_bytes_with_rpc_clients(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_pages, "shared_directory", lambda: str(tmp_path))
    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 1, 1, 2, 2, "bfloat16"
    )
    page_bytes = contract.num_layers * contract.layer_nbytes
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        actor = await xo.create_actor(
            XavierCacheActor, 2 * page_bytes, address=pool.external_address
        )
        fmt = {"layout": "layer_first", "shared_memory": True}
        namespace = await actor.configure(contract.to_dict(), fmt)
        info = await actor.shared_pages(namespace)
        pages = map_pages(
            info["path"], bytes.fromhex(info["nonce"]), page_bytes, info["slots"]
        )
        (slot,) = await actor.allocate_shared(namespace, ["a"])
        pages[slot].copy_(torch.arange(page_bytes, dtype=torch.uint8))
        assert await actor.commit_shared(namespace, ["a"], [slot]) == 1
        # Remote engines read and write the same slots through RPC.
        (page,) = await actor.get(namespace, ["a"])
        assert torch.equal(page, pages[slot])
        await actor.put(namespace, ["b"], [torch.ones(page_bytes, dtype=torch.uint8)])
        pinned = await actor.reserve_shared(namespace, ["a", "b"])
        assert pages[pinned["slots"][1]].tolist() == [1] * page_bytes
        assert await actor.reserve_shared(namespace, ["missing"]) is None
        stats = await actor.get_stats()
        assert stats["stored_pages"] == 2 and stats["active_handoffs"] == 1
        await actor.release_handoff(pinned["ticket"])
        await xo.destroy_actor(actor)
    assert not os.path.exists(info["path"])  # The actor removes its file.


def test_unavailable_shared_memory_falls_back_to_actor_storage(tmp_path, monkeypatch):
    def full(*args):
        raise OSError("no space")

    monkeypatch.setattr(shared_pages.shutil, "disk_usage", full)
    actor = XavierCacheActor.__new__(XavierCacheActor)
    XavierCacheActor.__init__(actor, 1024)
    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 1, 1, 2, 2, "bfloat16"
    )
    namespace = actor.configure(
        contract.to_dict(), {"layout": "layer_first", "shared_memory": True}
    )
    assert actor.shared_pages(namespace) is None
