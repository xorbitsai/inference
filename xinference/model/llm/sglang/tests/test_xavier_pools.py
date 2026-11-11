# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Multi-pool SGLang Xavier: tensor-parallel, quantized and DeepSeek-V4 models."""

import asyncio
import enum
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import pytest
import torch
import xoscar as xo

from ...xavier.backends.bytes.pools import PoolPageStore
from ...xavier.backends.torch.storage import XavierCacheActor
from ...xavier.contract import KVCacheContract
from ..xavier import config as xavier_config
from ..xavier.config import configure_xavier


class PoolName(str, enum.Enum):
    KV = "kv"
    SWA = "swa"
    C4 = "deepseek_v4_c4"

    def __str__(self):
        return self.value


class PoolHitPolicy(str, enum.Enum):
    ALL_PAGES = "all_pages"
    TRAILING_PAGES = "trailing_pages"
    BLOCK_GROUPS = "block_groups"


@dataclass
class PoolTransferResult:
    kv_hit_pages: int
    extra_pool_hit_pages: dict
    restorable_prefix_pages: Optional[List[int]] = None


@dataclass
class PoolTransfer:
    name: Any
    host_indices: Optional[torch.Tensor] = None
    keys: Optional[List[str]] = None
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES
    extra: dict = field(default_factory=dict)


@pytest.fixture
def storage_class(monkeypatch):
    class Base:
        def register_mem_pool_host(self, pool):
            self.mem_pool_host = pool

        def register_mem_host_pool_v2(self, pool, name):
            if not hasattr(self, "registered_pools"):
                self.registered_pools = {}
            self.registered_pools[name] = pool

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.hicache_storage",
        SimpleNamespace(
            HiCacheStorage=Base,
            PoolHitPolicy=PoolHitPolicy,
            PoolTransferResult=PoolTransferResult,
        ),
    )
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.5.23")
    name = "xinference.model.llm.sglang.xavier._test_pool_storage"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "xavier/storage.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module.XavierHiCacheStorage


class LogicalPool:
    """DeepSeek-V4's KV anchor: allocation indices only, no bytes."""

    page_size = 4
    layout = "layer_first"

    def get_dummy_flat_data_page(self):
        return torch.empty(0, dtype=torch.uint8)


class BytePool:
    layout = "layer_first"

    def __init__(self, page_size=4, item_bytes=6, pages=4, dtype=torch.uint8):
        self.page_size = page_size
        self.layer_num = 2
        self.dtype = dtype
        self.data = torch.zeros(pages, self.layer_num, item_bytes, dtype=torch.uint8)

    def fill(self, seed):
        self.data.copy_(
            torch.arange(self.data.numel(), dtype=torch.int64)
            .add(seed)
            .remainder(251)
            .to(torch.uint8)
            .reshape(self.data.shape)
        )

    def get_dummy_flat_data_page(self):
        return torch.zeros(self.data[0].numel(), dtype=torch.uint8).view(self.dtype)

    def get_data_page(self, index, flat=True):
        return self.data[index // self.page_size].flatten().view(self.dtype)

    def set_from_flat_data_page(self, index, page):
        self.data[index // self.page_size].copy_(
            page.view(torch.uint8).reshape(self.data[0].shape)
        )


def make_storage(storage_class, actor, replica, rank=0, size=2, mla=True):
    config = SimpleNamespace(
        tp_rank=rank,
        tp_size=size,
        pp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        is_mla_model=mla,
        extra_config={
            "address": actor.address,
            "uid": actor.uid,
            "pools": True,
            "replica": replica,
            "identity": {"weights": "w", "config": "c"},
        },
    )
    storage = storage_class(config)
    pools = {
        PoolName.KV: LogicalPool(),
        PoolName.C4: BytePool(),
        PoolName.SWA: BytePool(item_bytes=4, dtype=torch.bfloat16),
    }
    storage.register_mem_pool_host(pools[PoolName.KV])
    for name, pool in pools.items():
        storage.register_mem_host_pool_v2(pool, name)
    storage.on_host_pools_registered()
    return storage, pools


def transfers(keys, swa_keys=None):
    indices = torch.arange(len(keys) * 4)
    swa_keys = keys if swa_keys is None else swa_keys
    return [
        PoolTransfer(PoolName.C4, indices, list(keys), PoolHitPolicy.ALL_PAGES),
        PoolTransfer(
            PoolName.SWA,
            torch.arange(len(swa_keys) * 4),
            list(swa_keys),
            PoolHitPolicy.TRAILING_PAGES,
        ),
    ]


@pytest.mark.asyncio
async def test_replicas_share_mla_pools_across_tensor_parallel_ranks(
    storage_class, caplog
):
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        actor = await xo.create_actor(
            XavierCacheActor, 4096, address=pool.external_address, uid="cache"
        )
        writer, source = await asyncio.to_thread(
            make_storage, storage_class, actor, "replica-a"
        )
        reader, target = await asyncio.to_thread(
            make_storage, storage_class, actor, "replica-b", rank=1
        )
        source[PoolName.C4].fill(3)
        source[PoolName.SWA].fill(7)
        keys = ["p0", "p1"]
        indices = torch.arange(8)
        # Side pools first, then the anchor marker (SGLang's backup order).
        assert await asyncio.to_thread(writer.batch_set_v2, transfers(keys)) == {
            PoolName.C4: [True, True],
            PoolName.SWA: [True, True],
        }
        assert await asyncio.to_thread(writer.batch_set_v1, keys, indices) == [
            True,
            True,
        ]
        hit = await asyncio.to_thread(
            reader.batch_exists_v2, keys + ["missing"], transfers(keys, ["p1"])
        )
        assert hit.kv_hit_pages == 2 and hit.restorable_prefix_pages == [1, 2]
        assert await asyncio.to_thread(reader.batch_exists, keys) == 2
        caplog.set_level("DEBUG")
        assert await asyncio.to_thread(reader.batch_get_v2, transfers(keys)) == {
            PoolName.C4: [True, True],
            PoolName.SWA: [True, True],
        }
        assert torch.equal(target[PoolName.C4].data[:2], source[PoolName.C4].data[:2])
        assert torch.equal(target[PoolName.SWA].data[:2], source[PoolName.SWA].data[:2])
        assert "written by replica-a:4" in caplog.text
        # The logical anchor transfers nothing and reports success.
        assert await asyncio.to_thread(reader.batch_get_v1, keys, indices) == [
            True,
            True,
        ]
        stats = await actor.get_stats()
        assert stats["read_pages"] == 6 and stats["pools"] == {
            "kv": 0,
            "deepseek_v4_c4": 12,
            "swa": 8,
        }


@pytest.mark.asyncio
async def test_missing_side_pools_and_sharded_ranks_are_misses(storage_class):
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        actor = await xo.create_actor(
            XavierCacheActor, 4096, address=pool.external_address, uid="cache"
        )
        writer, _ = await asyncio.to_thread(
            make_storage, storage_class, actor, "a", mla=False
        )
        same_rank, _ = await asyncio.to_thread(
            make_storage, storage_class, actor, "b", mla=False
        )
        other_rank, _ = await asyncio.to_thread(
            make_storage, storage_class, actor, "b", rank=1, mla=False
        )
        keys = ["p0", "p1", "p2"]
        # Only the last page carries sliding-window state.
        await asyncio.to_thread(
            writer.batch_set_v2,
            [
                PoolTransfer(PoolName.C4, torch.arange(12), keys),
                PoolTransfer(PoolName.SWA, torch.arange(4), ["p2"]),
            ],
        )
        await asyncio.to_thread(writer.batch_set_v1, keys, torch.arange(12))
        hit = await asyncio.to_thread(
            same_rank.batch_exists_v2,
            keys,
            [
                PoolTransfer(PoolName.C4, None, keys),
                PoolTransfer(PoolName.SWA, None, ["p2"], PoolHitPolicy.TRAILING_PAGES),
            ],
        )
        # A window can restore only the prefix that ends on its stored page.
        assert hit.kv_hit_pages == 3 and hit.restorable_prefix_pages == [3]
        # TP-sharded pages are per rank: another rank's shard never matches.
        assert await asyncio.to_thread(other_rank.batch_exists, keys) == 0
        with pytest.raises(ValueError, match="hit policy"):
            await asyncio.to_thread(
                same_rank.batch_exists_v2,
                keys,
                [PoolTransfer(PoolName.C4, None, keys, PoolHitPolicy.BLOCK_GROUPS)],
            )
        # A mismatched host index count fails only that pool.
        assert await asyncio.to_thread(
            same_rank.batch_get_v2, [PoolTransfer(PoolName.C4, torch.arange(3), keys)]
        ) == {PoolName.C4: [False, False, False]}


@pytest.mark.asyncio
async def test_replicas_with_different_pool_layouts_cannot_share(storage_class):
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        actor = await xo.create_actor(
            XavierCacheActor, 4096, address=pool.external_address, uid="cache"
        )
        await asyncio.to_thread(make_storage, storage_class, actor, "a")
        config = SimpleNamespace(
            tp_rank=0,
            tp_size=2,
            pp_size=1,
            attn_cp_size=1,
            is_mla_model=True,
            extra_config={
                "address": actor.address,
                "uid": actor.uid,
                "pools": True,
                "identity": {"weights": "w", "config": "c"},
            },
        )
        storage = storage_class(config)
        storage.register_mem_pool_host(LogicalPool())
        storage.register_mem_host_pool_v2(BytePool(item_bytes=8), PoolName.C4)
        with pytest.raises(ValueError, match="differs between replicas"):
            await asyncio.to_thread(storage.on_host_pools_registered)
        # A single-pool contract cannot share the multi-pool namespace.
        contract = KVCacheContract(
            "a" * 64, "b" * 64, "c" * 64, "d" * 64, 3, 2, 4, 4, "float16"
        )
        with pytest.raises(ValueError, match="multi-pool"):
            await actor.configure(contract.to_dict(), {"layout": "layer_first"})


def test_pipeline_or_context_parallel_storage_is_rejected(storage_class):
    config = SimpleNamespace(
        tp_rank=0,
        tp_size=4,
        pp_size=2,
        attn_cp_size=1,
        is_mla_model=True,
        extra_config={"pools": True, "identity": {}},
    )
    with pytest.raises(ValueError, match="without PP"):
        storage_class(config)


def test_pool_store_budget_keeps_chain_heads():
    store = PoolPageStore(10)
    namespace = store.configure({"model": 1}, {"a": {"page_bytes": 4}})
    entries = [["a", "h"], ["a", "m"], ["a", "t"]]
    # Only two pages fit: the head of the chain is kept, not the tail.
    assert store.put(namespace, entries, [b"0000", b"1111", b"2222"], "r") == [
        True,
        True,
        False,
    ]
    assert store.get(namespace, [["a", "h"]]) == {
        "pages": [b"0000"],
        "writers": {"r": 1},
    }
    # New pages evict the least recently used one first.
    assert store.put(namespace, [["a", "n"]], [b"3333"]) == [True]
    assert store.contains(namespace, entries) == [True, False, False]
    assert store.get_stats()["used_bytes"] == 8
    with pytest.raises(ValueError, match="geometry"):
        store.put(namespace, [["a", "x"]], [b"0"])
    with pytest.raises(ValueError, match="pool page key"):
        store.contains(namespace, [["b", "x"]])
    with pytest.raises(ValueError, match="cannot hold one complete page"):
        PoolPageStore(2).configure({}, {"a": {"page_bytes": 4}})


@pytest.mark.asyncio
async def test_lmcache_backend_rejects_multi_pool_layouts(monkeypatch):
    from ...xavier.tests.utils import mock_lmcache_backend

    config, _ = mock_lmcache_backend(monkeypatch)
    actor = XavierCacheActor(4096, cache_backend=config)
    with pytest.raises(ValueError, match="native CPU backend"):
        actor.configure_pools({}, {"a": {"page_bytes": 4}})


@pytest.fixture
def deepseek_path(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV41ForCausalLM"],
                "model_type": "deepseek_v41",
                "quantization_config": {"quant_method": "fp8"},
                "vision_config": {"model_type": "deepseek_v41_vision"},
            }
        )
    )
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"one")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"two")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors"}})
    )
    (tmp_path / "tokenizer.json").write_text("{}")
    return tmp_path


def _storage_config(model_config):
    return json.loads(model_config["hicache_storage_backend_extra_config"])


def test_deepseek_v41_shares_quantized_multimodal_pools_at_tp4(
    deepseek_path, monkeypatch
):
    hashed = []

    def fingerprint(files, **kwargs):
        hashed.extend(files)
        return "f" * 64

    monkeypatch.setattr(xavier_config, "fingerprint_files", fingerprint)
    model_config = {"tp_size": 4}
    cache = {"address": "worker:1234", "uid": "cache", "replica": "rep0"}
    configure_xavier(str(deepseek_path), model_config, cache)
    storage = _storage_config(model_config)
    assert storage["pools"] is True and storage["replica"] == "rep0"
    assert storage["prefetch_threshold"] == 256 and "contract" not in storage
    # SGLang chooses DeepSeek-V4's page size, dtype and attention backend.
    assert "page_size" not in model_config and "dtype" not in model_config
    assert model_config["enable_hierarchical_cache"] is True
    # DeepSeek-V4 host pools cannot be auto-sized; Xavier sets a ratio.
    assert model_config["hicache_ratio"] == 0.5
    assert "hicache_host_memory_fraction" not in model_config
    assert cache["pools"] is True and cache["identity"] == storage["identity"]
    # Weights are identified by their manifest; only the tokenizer is hashed.
    assert hashed == ["tokenizer.json"]
    identity = storage["identity"]
    (deepseek_path / "model-00002-of-00002.safetensors").write_bytes(b"three")
    changed = {"tp_size": 4}
    configure_xavier(str(deepseek_path), changed, {**cache, "replicas": 2})
    assert _storage_config(changed)["identity"]["weights"] != identity["weights"]
    assert changed["hicache_ratio"] == 0.25


def test_tensor_parallel_text_models_use_pools_with_64_token_pages(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "tokenizer.json").write_text("{}")
    model_config = {"tp_size": 2, "quantization": "fp8"}
    cache = {"address": "a", "uid": "c", "replicas": 2}
    configure_xavier(str(tmp_path), model_config, cache)
    assert model_config["page_size"] == 64
    assert _storage_config(model_config)["prefetch_threshold"] == 64
    # Replicas on one host split SGLang's per-engine host-memory share.
    assert model_config["hicache_host_memory_fraction"] == 0.25
    explicit = {"tp_size": 2, "hicache_ratio": 3.0}
    configure_xavier(str(tmp_path), explicit, dict(cache))
    assert explicit["hicache_ratio"] == 3.0
    assert "hicache_host_memory_fraction" not in explicit
    single = {"tp_size": 2}
    configure_xavier(str(tmp_path), single, {"address": "a", "uid": "c"})
    assert "hicache_host_memory_fraction" not in single


@pytest.mark.parametrize(
    "config,options,message",
    [
        ({"model_type": "gemma3"}, {"tp_size": 2}, "Qwen2, Qwen3, Llama and DeepSeek"),
        (
            {"model_type": "qwen3", "use_sliding_window": True},
            {"tp_size": 2},
            "full-attention",
        ),
        ({"model_type": "deepseek_v41"}, {"pp_size": 2}, "without PP"),
        (
            {"model_type": "deepseek_v41"},
            {"speculative_algorithm": "EAGLE"},
            "speculation",
        ),
        (
            {"model_type": "deepseek_v41"},
            {"hicache_storage_backend": "file"},
            "another HiCache",
        ),
    ],
)
def test_pool_mode_rejects_unvalidated_models_and_options(
    tmp_path, config, options, message
):
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "tokenizer.json").write_text("{}")
    with pytest.raises(ValueError, match=message):
        configure_xavier(str(tmp_path), dict(options), {"address": "a", "uid": "c"})
