# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import xoscar as xo

from ...xavier.backends.torch.storage import XavierCacheActor
from ...xavier.contract import KVCacheContract
from ..compat import prepare_runtime, supports_tensor_parallel, validate_runtime_version
from ..xavier import CONFIG_ENV, WORKER_CONFIG_ENV, configure_xavier

NEW = "1.4.0rc3"


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
    name = "xinference.model.llm.trt_llm._test_tp_connector"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "connector.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    yield module
    module.close_clients()


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


@pytest.mark.parametrize("version", [NEW, "1.4.0", "1.4.2"])
def test_parallel_runtime_versions_are_accepted(version):
    assert supports_tensor_parallel(version)
    validate_runtime_version({}, version)
    validate_runtime_version({"_xavier_cache_config": {}}, version)


@pytest.mark.parametrize(
    "config",
    [
        {"_xavier_cache_config": {"role": "prefill"}},
        {"_xavier_cache_config": {"heterogeneous": True}},
        {"_nixl_config": {"role": "decode"}},
        {
            "kv_connector_config": {
                "connector_module": "xinference.model.llm.trt_llm.connector"
            }
        },
    ],
)
def test_pd_keeps_its_validated_versions_on_the_parallel_runtime(config):
    with pytest.raises(ValueError, match="Xavier shared caching and P/D require"):
        validate_runtime_version(config, NEW)


@pytest.mark.parametrize("version", ["1.4.0rc2", "1.5.0", "1.3.0"])
def test_tensor_parallelism_requires_the_parallel_runtime(monkeypatch, version):
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    if version == "1.5.0":
        with pytest.raises(ValueError, match="requires"):
            validate_runtime_version({}, version)
    with pytest.raises(ValueError, match="tensor parallelism requires"):
        prepare_runtime({"tensor_parallel_size": 2}, engine_version=version)


def test_tensor_parallel_runtime_spawns_rank_workers(monkeypatch):
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    prepare_runtime({"tensor_parallel_size": 2}, engine_version=NEW)
    assert "TLLM_WORKER_USE_SINGLE_PROCESS" not in os.environ
    prepare_runtime({}, engine_version=NEW)
    assert os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] == "1"
    for options in ({"tensor_parallel_size": 0}, {"pipeline_parallel_size": 2}):
        with pytest.raises(ValueError):
            prepare_runtime(options, engine_version=NEW)


def test_ordinary_xavier_shards_the_contract_per_rank(monkeypatch, checkpoint):
    monkeypatch.delenv(WORKER_CONFIG_ENV, raising=False)
    config = {"tensor_parallel_size": 2, "max_seq_len": 512}
    cache = {"address": "worker:123", "uid": "cache"}
    configure_xavier(checkpoint, config, cache, engine_version=NEW)
    shared = json.loads(os.environ[WORKER_CONFIG_ENV])
    assert shared == json.loads(os.environ[CONFIG_ENV])
    assert shared["tensor_parallel_size"] == 2
    assert shared["contract"]["num_kv_heads"] == 1
    assert cache["contract"]["num_kv_heads"] == 1
    assert config["kv_cache_config"]["host_cache_size"] == 0
    # A TP1 relaunch in the same process must not inherit the rank config.
    configure_xavier(checkpoint, {}, {"address": "a", "uid": "b"}, engine_version=NEW)
    assert WORKER_CONFIG_ENV not in os.environ
    assert json.loads(os.environ[CONFIG_ENV])["contract"]["num_kv_heads"] == 2


@pytest.mark.parametrize(
    "options, cache, version, match",
    [
        ({"tensor_parallel_size": 2}, {}, "1.2.1", "TP>1"),
        ({"tensor_parallel_size": 2}, {"role": "prefill", "rank": 0}, NEW, "TP>1"),
        ({"tensor_parallel_size": 3}, {}, NEW, "divisible"),
        ({"tensor_parallel_size": 2}, {"read_timeout": 5}, NEW, "native"),
        ({"pipeline_parallel_size": 2}, {}, NEW, "PP=CP=1"),
        ({"kv_cache_config": {"host_cache_size": 1}}, {}, NEW, "offload"),
    ],
)
def test_unsupported_parallel_xavier_options(
    monkeypatch, checkpoint, options, cache, version, match
):
    monkeypatch.setattr(
        "xinference.model.llm.xavier.transport.gpu_pool_options", lambda *_: None
    )
    with pytest.raises(ValueError, match=match):
        configure_xavier(checkpoint, options, cache, engine_version=version)


def _meta(request_id, blocks, computed, scheduled):
    return SimpleNamespace(
        new_requests=[
            SimpleNamespace(
                request_id=request_id,
                new_block_ids=blocks,
                computed_position=computed,
                num_scheduled_tokens=scheduled,
            )
        ]
    )


def _flushed(worker):
    return worker._publisher is None or worker._publisher.flush(10)


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["1.2.1", NEW])
async def test_tensor_parallel_replicas_exchange_rank_shards(
    monkeypatch, tmp_path, connector, version
):
    # TRT 1.4 maps the cache actor's shared-memory pages; older versions and
    # remote actors exchange the same pages through actor RPC.
    from ...xavier.backends.torch import shared_pages

    monkeypatch.setattr(shared_pages, "shared_directory", lambda: str(tmp_path))
    # One rank holds one of two KV heads: 2 layers x K/V x 4 tokens x 1 head x 4.
    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 2, 1, 4, 4, "bfloat16"
    )
    page_bytes = contract.num_layers * contract.layer_nbytes
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        actor = await xo.create_actor(
            XavierCacheActor,
            8 * page_bytes,
            address=pool.external_address,
            uid="trt-tp-cache",
        )
        monkeypatch.delenv(CONFIG_ENV, raising=False)
        monkeypatch.setenv(
            WORKER_CONFIG_ENV,
            json.dumps(
                {
                    "address": actor.address,
                    "uid": "trt-tp-cache",
                    "contract": contract.to_dict(),
                    "tensor_parallel_size": 2,
                    "engine_version": version,
                }
            ),
        )

        async def replica():
            scheduler = await asyncio.to_thread(
                connector.XavierConnectorScheduler, None
            )
            workers = []
            for rank in range(2):
                monkeypatch.setattr(connector, "_local_rank", lambda rank=rank: rank)
                worker = await asyncio.to_thread(connector.XavierConnectorWorker, None)
                worker.register_kv_caches(
                    torch.zeros(3, 2, 2, 16, dtype=torch.bfloat16)
                )
                workers.append(worker)
            return scheduler, workers

        (a, source), (b, target) = [await replica() for _ in range(2)]
        for rank, worker in enumerate(source):
            worker.pool.copy_(
                torch.arange(192, dtype=torch.bfloat16).reshape(3, 2, 2, 16)
                + 1000 * rank
            )
        request = SimpleNamespace(request_id=1, get_tokens=lambda beam: list(range(10)))
        assert await asyncio.to_thread(a.get_num_new_matched_tokens, request, 0) == (
            0,
            False,
        )
        a.update_state_after_alloc(request, [0, 1, 2])
        metadata = a.build_connector_meta(_meta(1, [0, 1, 2], 0, 10))
        # Only rank 0 publishes first: half a page set is never a hit.
        source[0].bind_connector_meta(metadata)
        await asyncio.to_thread(
            source[0].wait_for_save, SimpleNamespace(synchronize=lambda: None)
        )
        assert await asyncio.to_thread(_flushed, source[0])
        request.request_id = 2
        request.get_tokens = lambda beam: [*range(10), 100]
        assert await asyncio.to_thread(b.get_num_new_matched_tokens, request, 0) == (
            0,
            False,
        )
        source[1].bind_connector_meta(metadata)
        await asyncio.to_thread(
            source[1].wait_for_save, SimpleNamespace(synchronize=lambda: None)
        )
        assert await asyncio.to_thread(_flushed, source[1])
        assert (await actor.get_stats())["stored_pages"] == 4
        # Republishing an already stored prefix copies nothing to the host.
        for worker in source:
            worker.bind_connector_meta(metadata)
            await asyncio.to_thread(
                worker.wait_for_save, SimpleNamespace(synchronize=lambda: None)
            )
            assert worker._publisher is None or worker._publisher.in_flight == 0
        assert (await actor.get_stats())["stored_pages"] == 4

        request.request_id = 3
        assert await asyncio.to_thread(b.get_num_new_matched_tokens, request, 0) == (
            8,
            False,
        )
        store = actor  # Pinned pages survive eviction pressure before the copy.
        namespace = (await store.get_stats())["namespace"]
        await store.put(
            namespace,
            [f"evict-{i}" for i in range(8)],
            [torch.zeros(page_bytes, dtype=torch.uint8)] * 8,
        )
        b.update_state_after_alloc(request, [2, 0, 1])
        metadata = b.build_connector_meta(_meta(3, [2, 0, 1], 0, 3))
        assert [slot for slot, _, _ in metadata.shard_loads] == [2, 0]
        shared = version == NEW
        assert all((slots is not None) == shared for *_, slots in metadata.shard_loads)
        assert (await actor.get_stats())["backend_stats"].get("shared_memory") == (
            shared or None
        )
        assert not metadata.loads
        for worker in target:
            worker.bind_connector_meta(metadata)
            await asyncio.to_thread(worker.start_load_kv, None)
        for rank in range(2):
            expected = source[rank].pool
            assert torch.equal(target[rank].pool[2], expected[0])
            assert torch.equal(target[rank].pool[0], expected[1])
            assert not target[rank].pool[1].any()
        # Imported pages are known to be stored: no host copy, no store RPC.
        stored = (await actor.get_stats())["stored_pages"]
        for worker in target:
            await asyncio.to_thread(
                worker.wait_for_save, SimpleNamespace(synchronize=lambda: None)
            )
            assert worker._publisher is None
        stats = await actor.get_stats()
        assert stats["stored_pages"] == stored
        assert stats["imported_pages"] == 2  # Counted once, not per rank.
        assert stats["active_handoffs"] == 1
        # The next scheduler step follows every rank's copy and drops the pin.
        b.build_connector_meta(SimpleNamespace(new_requests=[]))
        assert (await actor.get_stats())["active_handoffs"] == 0

        # A lookup whose request ends before allocation releases its pin too.
        request.request_id = 4
        assert (await asyncio.to_thread(b.get_num_new_matched_tokens, request, 0))[
            0
        ] in (0, 8)
        b.request_finished(request, [])
        assert (await actor.get_stats())["active_handoffs"] == 0
        await asyncio.to_thread(connector.close_clients)


def test_publication_is_bounded_and_survives_store_failures(connector):
    calls = []

    def rpc(method, namespace, keys=None, pages=None):
        if method == "eviction_count":
            return 0
        if method == "exists":
            # "e" is already stored by another replica.
            return int(keys[0] == "e")
        calls.append((keys, [bytes(page) for page in pages]))
        if len(calls) == 1:
            raise RuntimeError("store unavailable")
        return [True] * len(keys)

    pool = torch.arange(24, dtype=torch.bfloat16).reshape(3, 1, 2, 4)
    page_bytes = pool[0].numel() * pool.element_size()
    publisher = connector._Publisher(
        SimpleNamespace(rpc=rpc, namespace="ns"), budget=2 * page_bytes
    )
    try:
        # Only two pages fit; the third is skipped instead of blocking TRT.
        assert publisher.submit(pool, [("a", 2), ("b", 0), ("c", 1)]) == 2
        assert publisher.flush(10)
        assert publisher.submit(pool, [("d", 1)]) == 1
        assert publisher.flush(10)
        assert publisher.submit(pool, [("e", 0), ("f", 2)]) == 2
        assert publisher.flush(10)
    finally:
        publisher.close()
    assert [keys for keys, _ in calls] == [["a", "b"], ["d"], ["f"]]
    assert calls[1][1] == [bytes(pool[1].view(torch.uint8).reshape(-1))]


@pytest.mark.parametrize(
    "version, devices, expected", [(NEW, 4, 4), ("1.2.1", 4, None), (NEW, 1, None)]
)
def test_tensor_parallel_defaults_to_allocated_gpus(
    monkeypatch, version, devices, expected
):
    from .. import core

    seen = {}

    class Stop(Exception):
        pass

    def configure(config, **kwargs):
        seen.update(config)
        raise Stop

    monkeypatch.setattr(core.importlib.metadata, "version", lambda name: version)
    monkeypatch.setattr("xinference.utils.cuda_count", lambda: devices)
    monkeypatch.setattr(
        "xinference.model.llm.trt_llm.compat.prepare_runtime", configure
    )
    monkeypatch.setitem(
        sys.modules, "tensorrt_llm", SimpleNamespace(LLM=SimpleNamespace)
    )
    model = core.TRTLLMModel.__new__(core.TRTLLMModel)
    model._model_config = {"max_seq_len": 128}
    model._engine = model._pd = None
    model._requests = {}
    model._cache_enabled = False
    model.model_path = "/model"
    model.model_family = SimpleNamespace(context_length=128)
    model.prepare_parse_reasoning_content = lambda *args: None
    with pytest.raises(Stop):
        model.load()
    assert seen.get("tensor_parallel_size") == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_staged_publication_copies_pages_after_the_engine_stream(connector, tmp_path):
    from ...xavier.backends.torch.shared_pages import SharedPageArena

    pool = torch.arange(4 * 2048, dtype=torch.float32, device="cuda").reshape(4, 2048)
    page_bytes = pool[0].numel() * pool.element_size()
    arena = SharedPageArena(str(tmp_path), "pages", page_bytes, 4)
    allocated = {}

    def rpc(method, namespace, keys, *args):
        if method == "allocate_shared":
            slots = arena.allocate([hash(key) for key in keys], *args)
            allocated.update(zip(keys, slots))
            return slots
        assert method == "commit_shared"
        slots, ok = args
        assert ok
        return arena.commit([hash(key) for key in keys], slots)

    client = SimpleNamespace(
        namespace="ns",
        rpc=rpc,
        writer="w",
        pages=lambda: arena.pages,
        evictions=lambda: arena.evictions,
        writing=lambda delta: None,
    )
    writer = connector._StagedWriter(client, pool, budget=2 * page_bytes)
    try:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(50_000_000)  # A long forward pass still writing.
            pool[3].fill_(7)
        # Two rows: the third page is left to the synchronous fallback.
        assert writer.submit(pool, stream, [("a", 3), ("b", 1), ("c", 2)]) == 2
        # The engine reuses slot 3 right after; the staged copy keeps old bytes.
        with torch.cuda.stream(stream):
            pool[3].fill_(-1)
        assert writer.flush(30)
        expected = torch.full((2048,), 7, dtype=torch.float32)
        page = arena.pages[allocated["a"]].view(torch.float32)
        assert torch.equal(page, expected)
        assert torch.equal(
            arena.pages[allocated["b"]].view(torch.float32), pool[1].cpu()
        )
        assert len(arena.ready) == 2
    finally:
        writer.close()
        arena.pages = None


@pytest.mark.asyncio
async def test_a_page_evicted_by_another_writer_is_published_again(
    monkeypatch, tmp_path, connector
):
    from ...xavier.backends.torch import shared_pages

    monkeypatch.setattr(shared_pages, "shared_directory", lambda: str(tmp_path))
    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 1, 1, 2, 4, "bfloat16"
    )
    page_bytes = contract.num_layers * contract.layer_nbytes
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        # The store holds a single page.
        actor = await xo.create_actor(
            XavierCacheActor, page_bytes, address=pool.external_address, uid="one"
        )
        monkeypatch.delenv(WORKER_CONFIG_ENV, raising=False)
        monkeypatch.setenv(
            CONFIG_ENV,
            json.dumps(
                {
                    "address": actor.address,
                    "uid": "one",
                    "contract": contract.to_dict(),
                    "engine_version": NEW,
                }
            ),
        )
        worker = await asyncio.to_thread(connector.XavierConnectorWorker, None)
        worker.register_kv_caches(torch.ones(2, 1, 2, 8, dtype=torch.bfloat16))
        assert worker.client.pages() is not None
        namespace = (await actor.get_stats())["namespace"]

        async def save():
            worker.bind_connector_meta(connector.XavierMetadata(saves=[("A", 1)]))
            await asyncio.to_thread(
                worker.wait_for_save, SimpleNamespace(synchronize=lambda: None)
            )
            return (await actor.get_stats())["stored_pages"]

        assert await save() == 1
        assert await save() == 1  # A known page is not copied again.
        # Another writer evicts A from the one-page store.
        await actor.put(namespace, ["B"], [torch.zeros(page_bytes, dtype=torch.uint8)])
        assert await actor.exists(namespace, ["A"]) == 0
        assert await save() == 3  # The stale hint is dropped; A is stored again.
        assert await actor.exists(namespace, ["A"]) == 1
        await asyncio.to_thread(connector.close_clients)


def test_failed_shared_writes_return_slots_only_after_draining(connector, tmp_path):
    from ...xavier.backends.torch.shared_pages import SharedPageArena

    arena = SharedPageArena(str(tmp_path), "pages", 4, 2)
    calls = []

    def rpc(method, namespace, *args):
        calls.append(method)
        if method == "allocate_shared":
            keys, writer = args
            return arena.allocate(keys, writer)
        if method == "commit_shared":
            keys, slots, ok = args
            if ok:
                return arena.commit(keys, slots)
            return arena.abort(keys, slots)
        assert method == "release_writer"
        return arena.release_writer(args[0])

    client = SimpleNamespace(
        namespace="ns",
        rpc=rpc,
        writer="w",
        pages=lambda: arena.pages,
        evictions=lambda: arena.evictions,
        writing=lambda delta: None,
    )

    def broken(index, target):
        raise RuntimeError("copy failed")

    # Drained copies: the writer gives the slot back at once.
    with pytest.raises(RuntimeError):
        connector._store_shared(client, ["a"], broken, lambda: None)
    assert calls[-1] == "commit_shared" and len(arena.free) == 2

    # A copy that may still be running keeps its slot reserved.
    def stuck():
        raise RuntimeError("device lost")

    with pytest.raises(RuntimeError):
        connector._store_shared(client, ["b"], broken, stuck)
    (spare,) = arena.free
    assert arena.allocate(["c"], "x") == [spare]
    assert arena.allocate(["d"], "x") == [None]
    # Only the writer's own release returns it.
    assert arena.release_writer("w") == 1
    arena.pages = None


def test_stored_hints_expire_on_eviction_and_age(connector, monkeypatch):
    hints = connector._StoredHints(limit=2, ttl=10)
    now = [100.0]
    monkeypatch.setattr(connector.time, "monotonic", lambda: now[0])
    hints.add(["a", "b"], 5)
    assert hints.unknown(["a", "b", "c"], 5) == ["c"]
    assert hints.unknown(["a"], 6) == ["a"]  # Any eviction invalidates it.
    hints.add(["a"], 6)
    now[0] += 11
    assert hints.unknown(["a"], 6) == ["a"]  # Too old.
    hints.add(["x", "y", "z"], 6)
    assert hints.unknown(["x", "y", "z"], 6) == ["x"]  # Bounded size.


@pytest.mark.asyncio
async def test_undrained_writes_keep_their_slots_after_client_shutdown(
    monkeypatch, tmp_path, connector
):
    from ...xavier.backends.torch import shared_pages

    monkeypatch.setattr(shared_pages, "shared_directory", lambda: str(tmp_path))
    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 1, 1, 2, 4, "bfloat16"
    )
    page_bytes = contract.num_layers * contract.layer_nbytes
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        # A single slot: whoever reuses it would overwrite the stalled copy.
        actor = await xo.create_actor(
            XavierCacheActor, page_bytes, address=pool.external_address, uid="w"
        )
        config = {
            "address": actor.address,
            "uid": "w",
            "contract": contract.to_dict(),
            "engine_version": NEW,
        }
        first, second = [
            await asyncio.to_thread(connector.XavierClient, config) for _ in range(2)
        ]
        pending = []

        def deferred(index, target):
            pending.append(target)  # The copy has not run yet.

        def lost():
            raise RuntimeError("device lost")

        with pytest.raises(RuntimeError):
            await asyncio.to_thread(
                connector._store_shared, first, ["A"], deferred, lost
            )
        await asyncio.to_thread(first.close)
        # The first writer never confirmed its copy: its slot stays reserved.
        assert await asyncio.to_thread(
            second.rpc, "allocate_shared", second.namespace, ["B"], second.writer
        ) == [None]
        pending[0].fill_(7)  # The stalled copy lands harmlessly.
        await asyncio.to_thread(second.close)


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["1.2.1", NEW])
async def test_admitted_pages_stay_pinned_for_a_slow_rank(
    monkeypatch, tmp_path, connector, version
):
    from ...xavier.backends.torch import shared_pages, storage

    monkeypatch.setattr(shared_pages, "shared_directory", lambda: str(tmp_path))
    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 1, 1, 2, 4, "bfloat16"
    )
    page_bytes = contract.num_layers * contract.layer_nbytes
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        # Room for exactly the two shards of one page.
        actor = await xo.create_actor(
            XavierCacheActor, 2 * page_bytes, address=pool.external_address, uid="p"
        )
        monkeypatch.delenv(CONFIG_ENV, raising=False)
        monkeypatch.setenv(
            WORKER_CONFIG_ENV,
            json.dumps(
                {
                    "address": actor.address,
                    "uid": "p",
                    "contract": contract.to_dict(),
                    "tensor_parallel_size": 2,
                    "engine_version": version,
                }
            ),
        )
        scheduler = await asyncio.to_thread(connector.XavierConnectorScheduler, None)
        ranks = []
        for rank in range(2):
            monkeypatch.setattr(connector, "_local_rank", lambda rank=rank: rank)
            worker = await asyncio.to_thread(connector.XavierConnectorWorker, None)
            worker.register_kv_caches(torch.zeros(2, 1, 2, 8, dtype=torch.bfloat16))
            ranks.append(worker)
        namespace = (await actor.get_stats())["namespace"]
        key = scheduler._keys(list(range(4)))[0]
        shards = [f"{key}@tp{rank}" for rank in range(2)]
        await actor.put(
            namespace, shards, [torch.full((page_bytes,), 7, dtype=torch.uint8)] * 2
        )
        request = SimpleNamespace(request_id=1, get_tokens=lambda beam: list(range(5)))
        assert await asyncio.to_thread(
            scheduler.get_num_new_matched_tokens, request, 0
        ) == (4, False)
        scheduler.update_state_after_alloc(request, [1, 0])
        metadata = scheduler.build_connector_meta(_meta(1, [1, 0], 0, 1))
        ranks[0].bind_connector_meta(metadata)
        await asyncio.to_thread(ranks[0].start_load_kv, None)
        # Rank 1 stalls past every handoff deadline while another replica
        # publishes; the admitted pages must not be evicted under it.
        later = time.monotonic() + 3600
        monkeypatch.setattr(storage.time, "monotonic", lambda: later)
        others = [f"other@tp{rank}" for rank in range(2)]
        await actor.put(
            namespace, others, [torch.full((page_bytes,), 99, dtype=torch.uint8)] * 2
        )
        ranks[1].bind_connector_meta(metadata)
        await asyncio.to_thread(ranks[1].start_load_kv, None)
        for worker in ranks:
            assert worker.pool[1].view(torch.uint8).unique().tolist() == [7]
        # The next scheduler step follows every rank's copy and releases.
        scheduler.build_connector_meta(SimpleNamespace(new_requests=[]))
        assert (await actor.get_stats())["active_handoffs"] == 0
        await asyncio.to_thread(connector.close_clients)


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["1.2.1", NEW])
async def test_pins_survive_lost_rpcs_until_released(
    monkeypatch, tmp_path, connector, version
):
    from ...xavier.backends.torch import shared_pages

    monkeypatch.setattr(shared_pages, "shared_directory", lambda: str(tmp_path))
    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 1, 1, 2, 4, "bfloat16"
    )
    page_bytes = contract.num_layers * contract.layer_nbytes
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        # Room for the two shards of one page.
        actor = await xo.create_actor(
            XavierCacheActor, 2 * page_bytes, address=pool.external_address, uid="r"
        )
        monkeypatch.delenv(CONFIG_ENV, raising=False)
        monkeypatch.setenv(
            WORKER_CONFIG_ENV,
            json.dumps(
                {
                    "address": actor.address,
                    "uid": "r",
                    "contract": contract.to_dict(),
                    "tensor_parallel_size": 2,
                    "engine_version": version,
                }
            ),
        )
        scheduler = await asyncio.to_thread(connector.XavierConnectorScheduler, None)
        client = scheduler.client
        namespace = (await actor.get_stats())["namespace"]
        key = scheduler._keys(list(range(4)))[0]
        page = torch.full((page_bytes,), 7, dtype=torch.uint8)
        await actor.put(namespace, [f"{key}@tp{r}" for r in range(2)], [page] * 2)
        real = client.rpc
        faults = {}

        def flaky(method, *args):
            queued = faults.get(method) or [None]
            fault = queued.pop(0)
            if fault == "before":
                raise TimeoutError(f"{method} did not reach the actor")
            result = real(method, *args)
            if fault == "after":
                raise TimeoutError(f"{method} reply was lost")
            return result

        monkeypatch.setattr(client, "rpc", flaky)

        async def pins():
            return (await actor.get_stats())["active_handoffs"]

        serial = iter(range(100))

        stored = []

        async def replace():
            others = [f"x{next(serial)}-{r}" for r in range(2)]
            stored[:] = others
            return await actor.put(namespace, others, [page] * 2)

        request = SimpleNamespace(request_id=1, get_tokens=lambda beam: list(range(5)))

        async def admit():
            return await asyncio.to_thread(
                scheduler.get_num_new_matched_tokens, request, 0
            )

        # 1. The reserve reply is lost: a miss, and the pin is released.
        faults["reserve_shared" if version == NEW else "reserve_handoff"] = ["after"]
        assert await admit() == (0, False)
        assert await pins() == 0
        # 2. A release that fails keeps the admitted pages pinned until a
        #    later step retries it.
        assert await admit() == (4, False)
        scheduler.update_state_after_alloc(request, [1, 0])
        scheduler.build_connector_meta(_meta(1, [1, 0], 0, 1))
        # The release and its immediate retry both fail.
        faults["release_handoff"] = ["before", "before"]
        scheduler.build_connector_meta(SimpleNamespace(new_requests=[]))
        assert await pins() == 1
        assert await replace() == [False, False]  # Pinned pages cannot go.
        scheduler.build_connector_meta(SimpleNamespace(new_requests=[]))
        assert await pins() == 0
        # 3. A release lost at shutdown: closing releases by owner.
        request.request_id = 2
        assert await admit() == (4, False)
        faults["release_handoff"] = ["before"]
        scheduler.request_finished(request, [])
        assert await pins() == 1
        await asyncio.to_thread(client.close)
        assert await pins() == 0
        assert await replace() == [True, True]
        # A reserve still in flight from the closed owner is refused.
        method = "reserve_shared" if version == NEW else "reserve_handoff"
        args = (False,) if version != NEW else ()
        late = await getattr(actor, method)(
            namespace, stored, *args, "late", client.writer
        )
        assert late is None and await pins() == 0
        assert await actor.exists(namespace, stored) == 2  # Refused, not missing.
        await asyncio.to_thread(connector.close_clients)


@pytest.mark.asyncio
async def test_a_reserve_arriving_after_its_release_is_refused(tmp_path, monkeypatch):
    from ...xavier.backends.torch import shared_pages

    monkeypatch.setattr(shared_pages, "shared_directory", lambda: str(tmp_path))
    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 1, 1, 2, 4, "bfloat16"
    )
    page_bytes = contract.num_layers * contract.layer_nbytes
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        actor = await xo.create_actor(
            XavierCacheActor, page_bytes, address=pool.external_address
        )
        namespace = await actor.configure(
            contract.to_dict(), {"layout": "layer_first", "shared_memory": True}
        )
        await actor.put(namespace, ["A"], [torch.zeros(page_bytes, dtype=torch.uint8)])
        await actor.release_handoff("t1")  # The release overtakes the reserve.
        assert await actor.reserve_shared(namespace, ["A"], "t1", "o") is None
        assert await actor.reserve_handoff(namespace, ["A"], False, "t1", "o") is None
        assert (await actor.get_stats())["active_handoffs"] == 0
        pinned = await actor.reserve_shared(namespace, ["A"], "t2", "o")
        assert pinned is not None and (await actor.get_stats())["active_handoffs"] == 1
