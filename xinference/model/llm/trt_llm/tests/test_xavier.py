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
        pd_gpu=False, capacity_bytes=4096, read_prefix=read
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
