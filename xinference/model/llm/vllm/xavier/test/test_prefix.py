# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ....xavier.backends.bytes.storage import XavierBytesCacheActor
from ....xavier.prefix import prefix_keys
from ....xavier.tests.test_bytes_storage import contract


@pytest.fixture
def prefix_module(connector_module, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "xinference.model.llm.vllm.xavier.v1_connector", connector_module
    )
    name = "xinference.model.llm.vllm.xavier._test_prefix"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "prefix.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_prefix_admission_owns_pages_across_eviction_and_tracks_salt(
    prefix_module, connector_config
):
    c = contract()
    config = dict(
        contract=c.to_dict(),
        shared_prefix=True,
        capacity_bytes=2 * c.num_layers * c.layer_nbytes,
    )
    connector_config.cache_config.block_size = 64
    connector_config.kv_transfer_config.get_from_extra_config = lambda *a: config
    caches = SimpleNamespace(
        num_blocks=4,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=[
                    "model.layers.0.self_attn.attn",
                    "model.layers.1.self_attn.attn",
                ],
                kv_cache_spec=SimpleNamespace(),
            )
        ],
    )
    client = prefix_module.PrefixConnector(connector_config, None, caches)
    store = XavierBytesCacheActor(config["capacity_bytes"])
    ns = store.configure(c.to_dict())["namespace"]
    client._rpc = lambda method, *args: getattr(store, method)(*args)
    native = {
        f"model.layers.{i}.self_attn.attn": torch.zeros(
            4, 2, 64, 2, 8, dtype=torch.float16
        )
        for i in range(2)
    }
    client.register_kv_caches(native)
    keys = prefix_keys(list(range(129))[:-1], cache_salt="a")
    page = bytes([1] * c.num_layers * c.layer_nbytes)
    store.put(ns, keys, [page, page])
    request = SimpleNamespace(
        request_id="r", prompt_token_ids=list(range(129)), cache_salt="b"
    )
    assert client.get_num_new_matched_tokens(request, 0) == (0, False)
    request.cache_salt = "a"
    assert client.get_num_new_matched_tokens(request, 0) == (128, False)
    store.put(ns, prefix_keys([5] * 128), [page, page])
    assert store.get(ns, keys) == []
    client.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda: ([2, 0, 3],)), 128
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="r",
                prompt_token_ids=request.prompt_token_ids,
                block_ids=([2, 0, 3],),
                num_computed_tokens=128,
            )
        ],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={"r": 1},
    )
    metadata = client.build_connector_meta(output)
    client._get_connector_metadata = lambda: metadata
    client.start_load_kv(None)
    assert store.get_stats()["imported_pages"] == 2
    client.wait_for_save()
    assert store.get(ns, keys) == [page, page]
    assert client.request_finished(request, [2, 0, 3]) == (False, None)
    assert not client._prefix_pending and not client._prefix_allocated
    assert not client._request_cache_salts and not client._chunked_prefill
    client.shutdown()


def test_unavailable_storage_is_a_miss_and_request_state_is_released(
    prefix_module, connector_config
):
    c = contract()
    config = dict(
        contract=c.to_dict(),
        shared_prefix=True,
        capacity_bytes=c.num_layers * c.layer_nbytes,
    )
    connector_config.cache_config.block_size = 64
    connector_config.kv_transfer_config.get_from_extra_config = lambda *a: config
    caches = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[
            SimpleNamespace(layer_names=["layer"], kv_cache_spec=SimpleNamespace())
        ],
    )
    client = prefix_module.PrefixConnector(connector_config, None, caches)

    def unavailable(*args):
        raise ConnectionError("storage unavailable")

    client._rpc = unavailable
    request = SimpleNamespace(
        request_id="failed", prompt_token_ids=list(range(65)), cache_salt="a"
    )
    assert client.get_num_new_matched_tokens(request, 0) == (0, False)
    assert client.get_num_new_matched_tokens(request, 1) == (0, False)
    client.request_finished(request, [])
    assert not client._prefix_pending and not client._request_cache_salts
    client.shutdown()


@pytest.fixture
def shared_connector(prefix_module, connector_config):
    clients = []

    def create(capacity=6, blocks=7):
        c = contract()
        config = dict(
            address="unused",
            uid="prefix",
            contract=c.to_dict(),
            shared_prefix=True,
            capacity_bytes=capacity * c.num_layers * c.layer_nbytes,
        )
        connector_config.cache_config.block_size = c.block_size
        connector_config.kv_transfer_config.get_from_extra_config = lambda *a: config
        caches = SimpleNamespace(
            num_blocks=blocks,
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=[
                        f"model.layers.{i}.self_attn.attn" for i in range(c.num_layers)
                    ],
                    kv_cache_spec=SimpleNamespace(),
                )
            ],
        )
        client = prefix_module.PrefixConnector(connector_config, None, caches)
        clients.append(client)
        store = XavierBytesCacheActor(config["capacity_bytes"])
        store.configure(c.to_dict())
        client._rpc = lambda method, *args: getattr(store, method)(*args)
        native = {
            f"model.layers.{i}.self_attn.attn": torch.full(
                (blocks, 2, c.block_size, c.num_kv_heads, c.head_dim),
                i + 1,
                dtype=torch.float16,
            )
            for i in range(c.num_layers)
        }
        client.register_kv_caches(native)
        return client, store, native

    yield create
    for client in clients:
        client.shutdown()


def store_meta(module, client, tokens, slots, computed=None):
    meta = module.PrefixMetadata()
    client._add_store_request(meta, "r", tokens, [slots], computed)
    client._get_connector_metadata = lambda: meta


@pytest.mark.parametrize("cached", [0, 1, 3])
def test_publication_only_copies_missing_suffix_and_full_hit_is_copy_free(
    prefix_module, shared_connector, monkeypatch, cached
):
    client, store, _ = shared_connector()
    c = client.contract
    tokens = list(range(193))
    keys = prefix_keys(tokens[:-1])
    slots = [3, 0, 2]
    pages = prefix_module.pack_pages(client._prefix_views, slots, c)
    store.put(c.fingerprint, keys[:cached], pages[:cached])
    calls = []
    pack = prefix_module.pack_pages

    def capture(views, selected, contract):
        calls.append(list(selected))
        return pack(views, selected, contract)

    monkeypatch.setattr(prefix_module, "pack_pages", capture)
    store_meta(prefix_module, client, tokens, slots)
    client.wait_for_save()
    assert calls == ([slots[cached:]] if cached < len(slots) else [])
    assert store.get(c.fingerprint, keys) == pages
    calls.clear()
    client.wait_for_save()
    assert not calls


def test_chunked_publication_copies_each_complete_page_only_once(
    prefix_module, shared_connector, monkeypatch
):
    client, store, _ = shared_connector()
    tokens = list(range(193))
    slots = [3, 0, 2, 1]
    calls = []
    pack = prefix_module.pack_pages

    def capture(views, selected, contract):
        calls.extend(selected)
        return pack(views, selected, contract)

    monkeypatch.setattr(prefix_module, "pack_pages", capture)
    for computed in (64, 96, 128, 193):
        store_meta(prefix_module, client, tokens, slots, computed)
        client.wait_for_save()
    assert calls == slots[:3]
    assert len(store.get(client.contract.fingerprint, prefix_keys(tokens[:-1]))) == 3


def test_batched_publication_refreshes_whole_chain_and_bounds_capacity(
    prefix_module, shared_connector
):
    client, store, _ = shared_connector(capacity=140, blocks=130)
    c = client.contract
    tokens = list(range(129 * 64 + 1))
    keys = prefix_keys(tokens[:-1])
    store_meta(prefix_module, client, tokens, list(range(129)))
    client.wait_for_save()
    assert len(store.get(c.fingerprint, keys)) == 129
    other = prefix_keys([7] * (20 * 64))
    page = bytes(c.num_layers * c.layer_nbytes)
    store.put(c.fingerprint, other, [page] * len(other))
    assert len(store.get(c.fingerprint, keys)) == 120

    # A prompt longer than storage capacity can still publish its usable head.
    limited, small_store, _ = shared_connector(capacity=2)
    store_meta(prefix_module, limited, list(range(5 * 64 + 1)), list(range(5)))
    limited.wait_for_save()
    assert len(small_store.get(c.fingerprint, prefix_keys(list(range(5 * 64))))) == 2


@pytest.mark.parametrize("local_pages", [1, 2])
def test_waiting_prefix_retains_snapshot_and_maps_nonzero_local_offset(
    prefix_module, shared_connector, local_pages
):
    client, store, native = shared_connector(capacity=3)
    c = client.contract
    tokens = list(range(257))
    keys = prefix_keys(tokens[:-1], cache_salt="a")
    pages = [bytes([i + 1]) * (c.num_layers * c.layer_nbytes) for i in range(3)]
    store.put(c.fingerprint, keys[:3], pages)
    reads = []
    rpc = client._rpc

    def capture(method, *args):
        if method == "get":
            reads.append(args[1])
        return rpc(method, *args)

    client._rpc = capture
    request = SimpleNamespace(request_id="r", prompt_token_ids=tokens, cache_salt="a")
    expected = (3 - local_pages) * 64
    assert client.get_num_new_matched_tokens(request, local_pages * 64) == (
        expected,
        False,
    )
    store.put(c.fingerprint, prefix_keys([7] * 192), pages)
    assert client.get_num_new_matched_tokens(request, local_pages * 64) == (
        expected,
        False,
    )
    assert len(reads) == 1
    slots = [4, 2, 0, 3, 1]
    client.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda: (slots,)), expected
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="r",
                prompt_token_ids=tokens,
                block_ids=(slots,),
                num_computed_tokens=192,
            )
        ],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={"r": 65},
    )
    metadata = client.build_connector_meta(output)
    selected = slots[local_pages:3]
    assert metadata.prefix_loads == [(selected, pages[local_pages:])]
    before = {name: cache.clone() for name, cache in native.items()}
    client._get_connector_metadata = lambda: metadata
    client.start_load_kv(None)
    assert (
        prefix_module.pack_pages(client._prefix_views, selected, c)
        == pages[local_pages:]
    )
    for name, cache in native.items():
        untouched = sorted(set(range(len(cache))) - set(selected))
        assert torch.equal(cache[untouched], before[name][untouched])
    client.request_finished(request, slots)
    assert not client._prefix_pending and not client._prefix_allocated


def test_prefix_rpc_times_out_once_then_backs_off_and_recovers(
    prefix_module, shared_connector, monkeypatch
):
    import asyncio
    from unittest.mock import AsyncMock

    client, _, _ = shared_connector()
    # Exercise the actual synchronous engine-thread RPC path.
    client._rpc = prefix_module.PrefixConnector._rpc.__get__(client)
    now = [10.0]
    monkeypatch.setattr(
        prefix_module, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    monkeypatch.setattr(prefix_module, "PREFIX_RPC_TIMEOUT", 0.01)
    hung = [True]
    calls = []

    async def get(*args):
        calls.append(args)
        if hung[0]:
            await asyncio.Event().wait()
        return []

    ref = SimpleNamespace(configure=AsyncMock(), get=get)
    resolve = AsyncMock(return_value=ref)
    monkeypatch.setattr(prefix_module.xo, "actor_ref", resolve)
    with pytest.raises(asyncio.TimeoutError):
        client._rpc("get", client.contract.fingerprint, [])
    with pytest.raises(ConnectionError, match="backoff"):
        client._rpc("get", client.contract.fingerprint, [])
    assert len(calls) == 1 and resolve.await_count == 1
    hung[0] = False
    now[0] += 5.0
    assert client._rpc("get", client.contract.fingerprint, []) == []
    assert len(calls) == 2 and resolve.await_count == 2


@pytest.mark.parametrize("change", ["offset", "salt", "unaligned"])
def test_waiting_snapshot_invalidates_when_admission_scope_changes(
    shared_connector, change
):
    client, store, _ = shared_connector(capacity=3)
    c = client.contract
    tokens = list(range(257))
    page = bytes(c.num_layers * c.layer_nbytes)
    store.put(c.fingerprint, prefix_keys(tokens[:-1], cache_salt="a")[:3], [page] * 3)
    request = SimpleNamespace(request_id="r", prompt_token_ids=tokens, cache_salt="a")
    assert client.get_num_new_matched_tokens(request, 64) == (128, False)
    computed = 128 if change == "offset" else 65 if change == "unaligned" else 64
    if change == "salt":
        request.cache_salt = "b"
    assert client.get_num_new_matched_tokens(request, computed) == (
        64 if change == "offset" else 0,
        False,
    )
    if change == "offset":
        assert client._prefix_pending["r"] == (2, [page])
    else:
        assert not client._prefix_pending
    client.request_finished(request, [])
    assert not client._prefix_pending and not client._request_cache_salts
