# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import xoscar as xo

from ...xavier.backends.torch.storage import XavierCacheActor
from ...xavier.contract import KVCacheContract
from ...xavier.tests.utils import mock_lmcache_backend
from ..xavier import CONFIG_ENV, configure_xavier


@pytest.fixture
def connector(monkeypatch):
    class Base:
        def __init__(self, args):
            self._metadata = None

        def get_connector_meta(self):
            return self._metadata

        def bind_connector_meta(self, metadata):
            self._metadata = metadata

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.pyexecutor.kv_cache_connector",
        SimpleNamespace(KvCacheConnectorScheduler=Base, KvCacheConnectorWorker=Base),
    )
    name = "xinference.model.llm.trt_llm._test_connector"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "connector.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    yield module
    module.close_clients()


@pytest.mark.asyncio
@pytest.mark.parametrize("logical_dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("backend", ["native", "lmcache"])
async def test_two_instances_restore_owned_prefix_after_eviction(
    monkeypatch, connector, logical_dtype, backend
):
    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 2, 2, 4, 4, logical_dtype
    )
    page_bytes = contract.num_layers * contract.layer_nbytes
    options = {}
    storages = []
    if backend == "lmcache":
        config, storages = mock_lmcache_backend(monkeypatch)
        options["cache_backend"] = config
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        actor = await xo.create_actor(
            XavierCacheActor,
            2 * page_bytes,
            **options,
            address=pool.external_address,
            uid="trt-cache",
        )
        monkeypatch.setenv(
            CONFIG_ENV,
            json.dumps(
                {
                    "address": actor.address,
                    "uid": "trt-cache",
                    "contract": contract.to_dict(),
                }
            ),
        )
        a, b = [
            await asyncio.to_thread(connector.XavierConnectorScheduler, None)
            for _ in range(2)
        ]
        source, target = [
            await asyncio.to_thread(connector.XavierConnectorWorker, None)
            for _ in range(2)
        ]
        source_pool = torch.arange(384, dtype=getattr(torch, logical_dtype)).reshape(
            3, 2, 2, 32
        )
        target_pool = torch.zeros_like(source_pool)
        source.register_kv_caches(source_pool)
        target.register_kv_caches(target_pool)
        request = SimpleNamespace(request_id=1, get_tokens=lambda beam: list(range(10)))
        assert await asyncio.to_thread(a.get_num_new_matched_tokens, request, 0) == (
            0,
            False,
        )
        a.update_state_after_alloc(request, [0, 1, 2])
        metadata = a.build_connector_meta(
            SimpleNamespace(
                new_requests=[
                    SimpleNamespace(
                        request_id=1,
                        new_block_ids=[0, 1, 2],
                        computed_position=0,
                        num_scheduled_tokens=10,
                    )
                ]
            )
        )
        source.bind_connector_meta(metadata)
        await asyncio.to_thread(
            source.wait_for_save, SimpleNamespace(synchronize=lambda: None)
        )
        stats = await actor.get_stats()
        assert stats["stored_pages"] == 2
        assert stats["backend"] == backend
        request.request_id = 2
        request.get_tokens = lambda beam: [*range(10), 100]
        assert await asyncio.to_thread(b.get_num_new_matched_tokens, request, 0) == (
            8,
            False,
        )
        if storages:
            assert (await actor.get_stats())["backend_stats"]["l2_hits"] == 2
            storages[0].pages.clear()
        b.update_state_after_alloc(request, [2, 0, 1])
        # Remove source data and evict the shared pages after the admission query.
        expected = source_pool[:2].clone()
        source_pool.zero_()
        namespace = stats["namespace"]
        await actor.put(
            namespace,
            ["evict-a", "evict-b"],
            [torch.zeros(page_bytes, dtype=torch.uint8)] * 2,
        )
        metadata = b.build_connector_meta(
            SimpleNamespace(
                new_requests=[
                    SimpleNamespace(
                        request_id=2,
                        new_block_ids=[2, 0, 1],
                        computed_position=0,
                        num_scheduled_tokens=3,
                    )
                ]
            )
        )
        target.bind_connector_meta(metadata)
        await asyncio.to_thread(target.start_load_kv, None)
        assert torch.equal(target_pool[2], expected[0])
        assert torch.equal(target_pool[0], expected[1])
        assert not target_pool[1].any()
        assert (await actor.get_stats())["imported_pages"] == 2
        # Pending owned copies also share a budget: concurrent queries cannot
        # multiply the shared-cache capacity without a bound.
        b._pending[9] = (0, [expected[0].view(torch.uint8).flatten()] * 2)
        assert await asyncio.to_thread(b.get_num_new_matched_tokens, request, 0) == (
            0,
            False,
        )
        b._pending.clear()
        # Equal block tokens behind different causal prefixes must never match.
        assert a._keys(list(range(8)))[1] != a._keys([9, 8, 7, 6, 4, 5, 6, 7])[1]
        assert await asyncio.to_thread(b.get_num_new_matched_tokens, request, 1) == (
            0,
            False,
        )
        await xo.destroy_actor(actor)
        if storages:
            assert storages[0].closed and storages[0].stopped
        assert await asyncio.to_thread(b.get_num_new_matched_tokens, request, 0) == (
            0,
            False,
        )
        await asyncio.to_thread(connector.close_clients)
        for client in (a.client, b.client, source.client, target.client):
            assert not client._thread.is_alive() and client._loop.is_closed()


def test_bad_page_slots_cannot_address_another_requests_last_page(connector):
    worker = connector.XavierConnectorWorker.__new__(connector.XavierConnectorWorker)
    worker.pool = torch.zeros(2, 1, 2, 8, dtype=torch.float16)
    worker.bind_connector_meta(
        connector.XavierMetadata(loads=[(-1, torch.zeros(32, dtype=torch.uint8))])
    )
    with pytest.raises(ValueError, match="page slots"):
        worker.start_load_kv(None)
    assert not worker.pool.any()


def test_prefix_lookup_leaves_last_token_and_respects_local_hits(connector):
    scheduler = connector.XavierConnectorScheduler.__new__(
        connector.XavierConnectorScheduler
    )
    scheduler.contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 1, 1, 1, 4, "float16"
    )
    scheduler._pending = {}
    scheduler._allocations = {}
    scheduler._pd_requests = {}
    seen = []

    def read(keys):
        seen.extend(keys)
        return [torch.zeros(16, dtype=torch.uint8) for _ in keys]

    scheduler._pd_sources = {}
    scheduler._pd_destinations = {}
    scheduler.client = SimpleNamespace(
        pd_gpu=False, capacity_bytes=4096, read_prefix=read, config={}
    )
    request = SimpleNamespace(request_id=1, get_tokens=lambda beam: list(range(12)))
    assert scheduler.get_num_new_matched_tokens(request, 0) == (8, False)
    assert len(seen) == 2  # The third complete page includes the final token.
    seen.clear()
    assert scheduler.get_num_new_matched_tokens(request, 4) == (4, False)
    assert seen == [scheduler._keys(list(range(12)))[1]]
    scheduler.request_finished(request, [])
    assert not scheduler._pending


@pytest.fixture
def checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "torch_dtype": "bfloat16",
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "hidden_size": 16,
                "max_position_embeddings": 1024,
            }
        )
    )
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "tokenizer.json").write_text("{}")
    return str(tmp_path)


def test_config_sets_versioned_native_pool_and_fingerprints(monkeypatch, checkpoint):
    monkeypatch.setenv(CONFIG_ENV, "")
    config = {"max_seq_len": 512}
    configure_xavier(checkpoint, config, {"address": "worker:123", "uid": "cache"})
    shared = json.loads(__import__("os").environ[CONFIG_ENV])
    assert shared["contract"]["block_size"] == 64
    assert shared["contract"]["logical_dtype"] == "bfloat16"
    assert config["disable_overlap_scheduler"] and not config["enable_chunked_prefill"]
    assert (
        config["kv_connector_config"]["connector_worker_class"]
        == "XavierConnectorWorker"
    )


@pytest.mark.parametrize(
    "options",
    [
        {"tensor_parallel_size": 2},
        {"dtype": "float16"},
        {"enable_chunked_prefill": True},
        {"disable_overlap_scheduler": False},
        {"speculative_config": {}},
        {"kv_cache_config": {"host_cache_size": 100}},
        {"kv_cache_config": {"enable_block_reuse": False}},
        {"kv_cache_config": {"tokens_per_block": 32}},
    ],
)
def test_incompatible_execution_options_are_rejected(checkpoint, options):
    with pytest.raises(ValueError):
        configure_xavier(checkpoint, options, {})


@pytest.mark.asyncio
async def test_trt_imports_foreign_shared_prefix_after_storage_eviction(
    connector, monkeypatch
):
    from ...xavier.backends.bytes.storage import XavierBytesCacheActor
    from ...xavier.backends.torch.prefix import pack_pages
    from ...xavier.prefix import prefix_keys
    from ..cross_engine import canonical_trt_views

    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 2, 2, 4, 4, "bfloat16"
    )
    page_bytes = contract.num_layers * contract.layer_nbytes
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        actor = await xo.create_actor(
            XavierBytesCacheActor,
            2 * page_bytes,
            address=pool.external_address,
            uid="cross-prefix",
        )
        await actor.configure(contract.to_dict())
        monkeypatch.setenv(
            CONFIG_ENV,
            json.dumps(
                dict(
                    address=actor.address,
                    uid="cross-prefix",
                    contract=contract.to_dict(),
                    shared_prefix=True,
                )
            ),
        )
        scheduler = await asyncio.to_thread(connector.XavierConnectorScheduler, None)
        worker = await asyncio.to_thread(connector.XavierConnectorWorker, None)
        native = (
            torch.arange(3 * 2 * 2 * 32, dtype=torch.int32)
            .to(torch.bfloat16)
            .reshape(3, 2, 2, 32)
        )
        pages = pack_pages(canonical_trt_views(native, contract), [0, 1], contract)
        keys = prefix_keys(list(range(9)), 4)
        await actor.put(contract.fingerprint, keys, pages)
        request = SimpleNamespace(request_id=7, get_tokens=lambda beam: list(range(9)))
        assert await asyncio.to_thread(
            scheduler.get_num_new_matched_tokens, request, 0
        ) == (8, False)
        await actor.put(contract.fingerprint, prefix_keys([5] * 8, 4), pages)
        assert await actor.get(contract.fingerprint, keys) == []
        scheduler.update_state_after_alloc(request, [2, 0, 1])
        metadata = scheduler.build_connector_meta(
            SimpleNamespace(
                new_requests=[
                    SimpleNamespace(
                        request_id=7,
                        new_block_ids=[2, 0, 1],
                        computed_position=0,
                        num_scheduled_tokens=1,
                    )
                ]
            )
        )
        worker.register_kv_caches(torch.zeros_like(native))
        worker.bind_connector_meta(metadata)
        await asyncio.to_thread(worker.start_load_kv, None)
        assert (
            pack_pages(canonical_trt_views(worker.pool, contract), [2, 0], contract)
            == pages
        )
        await asyncio.to_thread(
            worker.wait_for_save, SimpleNamespace(synchronize=lambda: None)
        )
        assert await actor.get(contract.fingerprint, keys) == pages
        scheduler.client.close()
        worker.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("capacity", [128, 200])
@pytest.mark.parametrize("local_pages", [0, 32])
async def test_shared_prefix_gap_stops_admission_before_later_read_batch(
    connector, monkeypatch, capacity, local_pages
):
    from ...xavier.backends.bytes.storage import XavierBytesCacheActor
    from ...xavier.backends.torch.prefix import pack_pages
    from ...xavier.prefix import prefix_keys
    from ..cross_engine import canonical_trt_views

    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 1, 1, 1, 64, "bfloat16"
    )
    tokens = list(range(256 * contract.block_size + 1))
    keys = prefix_keys(tokens, contract.block_size)
    source = (
        torch.arange(257, dtype=torch.bfloat16)[:, None, None, None]
        .expand(257, 1, 2, 64)
        .contiguous()
    )
    pages = pack_pages(
        canonical_trt_views(source, contract), list(range(256)), contract
    )
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        actor = await xo.create_actor(
            XavierBytesCacheActor,
            capacity * contract.num_layers * contract.layer_nbytes,
            address=pool.external_address,
            uid="gap-prefix",
        )
        await actor.configure(contract.to_dict())
        # Normal 128-page publication batches leave a hole after a retained head.
        for start in (0, 128):
            await actor.put(
                contract.fingerprint,
                keys[start : start + 128],
                pages[start : start + 128],
            )
        retained_head = capacity - 128
        assert len(await actor.get(contract.fingerprint, keys[:128])) == retained_head
        assert len(await actor.get(contract.fingerprint, keys[128:])) == 128
        monkeypatch.setenv(
            CONFIG_ENV,
            json.dumps(
                dict(
                    address=actor.address,
                    uid="gap-prefix",
                    contract=contract.to_dict(),
                    shared_prefix=True,
                )
            ),
        )
        scheduler = await asyncio.to_thread(connector.XavierConnectorScheduler, None)
        worker = await asyncio.to_thread(connector.XavierConnectorWorker, None)
        reads = []
        original_rpc = scheduler.client.rpc

        def rpc(method, *args):
            if method == "get":
                reads.append(args[1])
            return original_rpc(method, *args)

        scheduler.client.rpc = rpc
        assert (
            await asyncio.to_thread(scheduler.client.read_prefix, keys)
            == pages[:retained_head]
        )
        assert reads == [keys[:128]]
        reads.clear()
        request = SimpleNamespace(request_id=1, get_tokens=lambda beam: tokens)
        expected_pages = max(retained_head - local_pages, 0)
        computed = local_pages * contract.block_size
        assert await asyncio.to_thread(
            scheduler.get_num_new_matched_tokens, request, computed
        ) == (expected_pages * contract.block_size, False)
        assert reads == [keys[local_pages : local_pages + 128]]
        # Imported pages retain their original logical positions despite shuffled slots.
        slots = list(reversed(range(257)))
        scheduler.update_state_after_alloc(request, slots)
        metadata = scheduler.build_connector_meta(
            SimpleNamespace(
                new_requests=[
                    SimpleNamespace(
                        request_id=1,
                        new_block_ids=slots,
                        computed_position=computed,
                        num_scheduled_tokens=len(tokens)
                        - computed
                        - expected_pages * contract.block_size,
                    )
                ]
            )
        )
        imported_slots = slots[local_pages : local_pages + expected_pages]
        assert [slot for slot, _ in metadata.loads] == imported_slots
        assert [page for _, page in metadata.loads] == pages[
            local_pages : local_pages + expected_pages
        ]
        worker.register_kv_caches(torch.zeros_like(source))
        worker.bind_connector_meta(metadata)
        await asyncio.to_thread(worker.start_load_kv, None)
        if imported_slots:
            assert (
                pack_pages(
                    canonical_trt_views(worker.pool, contract), imported_slots, contract
                )
                == pages[local_pages:retained_head]
            )
        untouched = sorted(set(slots) - set(imported_slots))
        assert torch.count_nonzero(worker.pool[untouched]) == 0
        scheduler.client.close()
        worker.client.close()


@pytest.mark.parametrize("shared_prefix", [False, True])
def test_shared_prefix_save_avoids_discarded_native_cpu_copies(
    connector, monkeypatch, shared_prefix
):
    from ...xavier.backends.torch.prefix import pack_pages
    from ...xavier.prefix import prefix_keys
    from ..cross_engine import canonical_trt_views

    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 2, 2, 4, 4, "bfloat16"
    )
    native = (
        torch.arange(3 * 2 * 2 * 32, dtype=torch.int32)
        .to(torch.bfloat16)
        .reshape(3, 2, 2, 32)
    )
    slots = [2, 0]
    expected = pack_pages(canonical_trt_views(native, contract), slots, contract)
    keys = prefix_keys(list(range(8)), 4)
    published = []
    worker = connector.XavierConnectorWorker.__new__(connector.XavierConnectorWorker)
    worker.contract = contract
    worker.pool = native
    worker._metadata = connector.XavierMetadata(saves=list(zip(keys, slots)))

    def rpc(method, namespace, out_keys, *args):
        if method == "exists":
            return 0
        pages, *start = args
        published.append((out_keys, pages))

    worker.client = SimpleNamespace(
        config={"shared_prefix": shared_prefix},
        namespace=contract.fingerprint,
        capacity_bytes=3 * contract.num_layers * contract.layer_nbytes,
        rpc=rpc,
    )
    copies = []
    original_cpu = torch.Tensor.cpu

    def cpu(tensor, *args, **kwargs):
        copies.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    worker.wait_for_save(SimpleNamespace(synchronize=lambda: None))
    assert len(copies) == (1 if shared_prefix else len(slots))
    assert published[0][0] == keys
    if shared_prefix:
        assert published[0][1] == expected
    else:
        assert all(
            torch.equal(page, native[slot].view(torch.uint8).reshape(-1))
            for page, slot in zip(published[0][1], slots)
        )


@pytest.mark.parametrize("oom", [False, True])
def test_shared_prefix_publication_bounds_bytes_and_contains_pack_failure(
    connector, monkeypatch, oom
):
    from ...xavier.backends.torch import prefix
    from .. import cross_engine

    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 20, 32, 128, 64, "bfloat16"
    )
    worker = connector.XavierConnectorWorker.__new__(connector.XavierConnectorWorker)
    worker.contract = contract
    worker.pool = torch.zeros(3)
    keys = [f"{i:064x}" for i in range(3)]
    worker._metadata = connector.XavierMetadata(saves=list(zip(keys, range(3))))
    published, batches = [], []
    worker.client = SimpleNamespace(
        config={"shared_prefix": True},
        namespace=contract.fingerprint,
        capacity_bytes=3 * contract.num_layers * contract.layer_nbytes,
        rpc=lambda method, *args: 0 if method == "exists" else published.append(args),
    )
    monkeypatch.setattr(cross_engine, "canonical_trt_views", lambda *args: {})

    def pack(views, slots, c):
        batches.append(slots)
        assert len(slots) * c.num_layers * c.layer_nbytes <= 32 * 1024**2
        if oom:
            raise torch.cuda.OutOfMemoryError("snapshot allocation failed")
        return [b"page"] * len(slots)

    monkeypatch.setattr(prefix, "pack_pages", pack)
    worker.wait_for_save(SimpleNamespace(synchronize=lambda: None))
    assert batches == ([[0]] if oom else [[0], [1], [2]])
    assert len(published) == (0 if oom else 3)


@pytest.fixture
def shared_worker(connector, monkeypatch):
    from ...xavier.backends.bytes.storage import XavierBytesCacheActor

    def create(capacity=6, blocks=7):
        c = KVCacheContract(
            "a" * 64, "b" * 64, "c" * 64, "d" * 64, 1, 1, 1, 64, "bfloat16"
        )
        store = XavierBytesCacheActor(capacity * c.num_layers * c.layer_nbytes)
        store.configure(c.to_dict())
        client = SimpleNamespace(
            contract=c,
            config={"shared_prefix": True},
            pd_gpu=False,
            namespace=c.fingerprint,
            capacity_bytes=store.capacity_bytes,
            rpc=lambda method, *args: getattr(store, method)(*args),
        )
        monkeypatch.setattr(connector, "XavierClient", lambda: client)
        worker = connector.XavierConnectorWorker(None)
        native = (
            torch.arange(blocks, dtype=torch.bfloat16)[:, None, None, None]
            .expand(blocks, 1, 2, 64)
            .contiguous()
        )
        worker.register_kv_caches(native)
        return worker, store

    return create


@pytest.mark.parametrize("cached", [0, 1, 3])
def test_burst_shared_prefix_publication_deduplicates_and_skips_cached_pages(
    connector, shared_worker, monkeypatch, cached
):
    from ...xavier.backends.torch import prefix
    from ..cross_engine import canonical_trt_views

    worker, store = shared_worker()
    scheduler = connector.XavierConnectorScheduler(None)
    requests = [
        SimpleNamespace(request_id=i, get_tokens=lambda beam, tokens=t: tokens)
        for i, t in enumerate((list(range(129)), list(range(64)) + [7] * 65))
    ]
    for request, slots in zip(requests, ([0, 1], [2, 3])):
        scheduler.update_state_after_alloc(request, slots)
    meta = scheduler.build_connector_meta(
        SimpleNamespace(
            new_requests=[
                SimpleNamespace(
                    request_id=i,
                    new_block_ids=slots,
                    computed_position=0,
                    num_scheduled_tokens=129,
                )
                for i, slots in enumerate(([0, 1], [2, 3]))
            ]
        )
    )
    assert len(meta.saves) == 4 and meta.saves[0][0] == meta.saves[2][0]
    keys = [meta.saves[i][0] for i in (0, 1, 3)]
    expected = prefix.pack_pages(
        canonical_trt_views(worker.pool, worker.contract), [0, 1, 3], worker.contract
    )
    store.put(worker.client.namespace, keys[:cached], expected[:cached])
    copied = []
    pack = prefix.pack_pages

    def capture(views, slots, c):
        copied.extend(slots)
        return pack(views, slots, c)

    monkeypatch.setattr(prefix, "pack_pages", capture)
    worker.bind_connector_meta(meta)
    worker.wait_for_save(SimpleNamespace(synchronize=lambda: None))
    assert copied == [0, 1, 3][cached:]
    assert store.get(worker.client.namespace, keys) == expected
    # The duplicate key must use the first request's slot, not slot 2.
    assert store.get(
        worker.client.namespace, scheduler._keys(list(requests[1].get_tokens(0)))
    ) == [expected[0], expected[2]]
    copied.clear()
    worker.wait_for_save(SimpleNamespace(synchronize=lambda: None))
    assert not copied


def test_trt_batched_prefix_publication_preserves_heads_and_capacity(
    connector, shared_worker
):
    from ...xavier.prefix import prefix_keys

    worker, store = shared_worker(capacity=140, blocks=130)
    c = worker.contract
    keys = prefix_keys(list(range(129 * 64)))
    worker.bind_connector_meta(
        connector.XavierMetadata(saves=list(zip(keys, range(129))))
    )
    worker.wait_for_save(SimpleNamespace(synchronize=lambda: None))
    assert len(store.get(c.fingerprint, keys)) == 129
    others = prefix_keys([7] * (20 * 64))
    store.put(c.fingerprint, others, [bytes(c.num_layers * c.layer_nbytes)] * 20)
    assert len(store.get(c.fingerprint, keys)) == 120
    limited, small = shared_worker(capacity=2)
    limited.bind_connector_meta(
        connector.XavierMetadata(saves=list(zip(keys[:5], range(5))))
    )
    limited.wait_for_save(SimpleNamespace(synchronize=lambda: None))
    assert len(small.get(c.fingerprint, keys[:5])) == 2


@pytest.mark.parametrize("cached", [6, 65])
def test_trt_large_shared_reads_are_byte_bounded_and_stop_at_a_gap(connector, cached):
    c = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 28, 8, 128, 64, "bfloat16"
    )
    client = connector.XavierClient.__new__(connector.XavierClient)
    client.contract = c
    client.config = {"shared_prefix": True}
    client.namespace = c.fingerprint
    keys = [f"{i:064x}" for i in range(65)]
    page_bytes = c.num_layers * c.layer_nbytes
    page = bytes(page_bytes)
    reads = []

    def rpc(method, namespace, batch):
        assert method == "get" and len(batch) * page_bytes <= 32 * 1024**2
        start = keys.index(batch[0])
        reads.append(batch)
        return [page] * max(0, min(len(batch), cached - start))

    client.rpc = rpc
    assert client.read_prefix(keys) == [page] * cached
    assert len(reads) == (2 if cached == 6 else 17)
