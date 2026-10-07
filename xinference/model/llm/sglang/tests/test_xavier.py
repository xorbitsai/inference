# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import xoscar as xo

from ...xavier.backends.torch.storage import XavierCacheActor
from ...xavier.contract import KVCacheContract
from ..xavier.config import configure_xavier


@pytest.fixture
def contract():
    return KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 3, 2, 4, 4, "float16"
    )


def test_cpu_cache_get_keeps_hot_pages_in_lru(contract):
    page_bytes = contract.layer_nbytes * contract.num_layers
    actor = XavierCacheActor(2 * page_bytes)
    ns = actor.configure(contract.to_dict(), {"layout": "layer_first"})
    page = torch.zeros(page_bytes, dtype=torch.uint8)
    assert actor.put(ns, ["hot", "cold"], [page, page]) == [True, True]
    assert actor.get(ns, ["missing", "hot"])[0] is None
    assert actor.put(ns, ["new"], [page]) == [True]
    hot, cold, new = actor.get(ns, ["hot", "cold", "new"])
    assert hot is not None and cold is None and new is not None
    assert actor.get_stats()["evicted_pages"] == 1


@pytest.fixture
def storage_class(monkeypatch):
    class Base:
        def register_mem_pool_host(self, pool):
            self.mem_pool_host = pool

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.hicache_storage",
        SimpleNamespace(HiCacheStorage=Base),
    )
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.5.21")
    name = "xinference.model.llm.sglang.xavier._test_storage"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "xavier/storage.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module.XavierHiCacheStorage


class HostPool:
    dtype = torch.float16
    page_size = 4
    layer_num = 3
    head_num = 2
    head_dim = 4
    layout = "layer_first"

    def __init__(self):
        self.data = torch.arange(576, dtype=self.dtype).reshape(2, 3, 12, 2, 4)

    def get_data_page(self, index, flat=True):
        return self.data[:, :, index : index + self.page_size].flatten()

    def set_from_flat_data_page(self, index, page):
        self.data[:, :, index : index + self.page_size].copy_(
            page.reshape(2, 3, 4, 2, 4)
        )


@pytest.mark.asyncio
async def test_two_adapters_restore_remote_pages_and_fail_closed(
    contract, storage_class
):
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        actor = await xo.create_actor(
            XavierCacheActor, 768, address=pool.external_address, uid="cache"
        )
        config = SimpleNamespace(
            tp_size=1,
            pp_size=1,
            attn_cp_size=1,
            is_mla_model=False,
            extra_config={
                "address": actor.address,
                "uid": actor.uid,
                "contract": contract.to_dict(),
            },
        )
        source, target = storage_class(config), storage_class(config)
        source_pool, target_pool = HostPool(), HostPool()
        target_pool.data.zero_()
        await asyncio.to_thread(source.register_mem_pool_host, source_pool)
        await asyncio.to_thread(target.register_mem_pool_host, target_pool)
        indices = torch.arange(8)
        expected = source_pool.data[:, :, :8].clone()
        assert await asyncio.to_thread(source.batch_set_v1, ["a", "b"], indices) == [
            True,
            True,
        ]
        source_pool.data.zero_()
        assert await asyncio.to_thread(target.batch_exists, ["a", "missing", "b"]) == 1
        assert await asyncio.to_thread(target.batch_get_v1, ["a", "b"], indices) == [
            True,
            True,
        ]
        assert torch.equal(target_pool.data[:, :, :8], expected)
        stats = await actor.get_stats()
        assert stats["read_pages"] == 2 and stats["used_bytes"] == 768
        assert await asyncio.to_thread(
            target.batch_get_v1, ["missing"], torch.arange(4)
        ) == [False]
        # Snapshot eviction and remote loss both become misses, preserving local computation.
        assert await asyncio.to_thread(
            source.batch_set_v1, ["c"], torch.arange(8, 12)
        ) == [True]
        assert (await actor.get_stats())["evicted_pages"] == 1
        await xo.destroy_actor(actor)
        assert await asyncio.to_thread(target.batch_get_v1, ["b"], torch.arange(4)) == [
            False
        ]
        source.close()
        target.close()
        for storage in (source, target):
            assert not storage._rpc_thread.is_alive()
            assert storage._rpc_loop.is_closed()
            storage.close()
        assert await asyncio.to_thread(target.batch_exists, ["b"]) == 0


def test_namespace_mismatch_and_bad_pages_do_not_mutate_cache(contract):
    actor = XavierCacheActor(768)
    fmt = {"engine": "sglang", "version": "0.5.21", "layout": "layer_first"}
    namespace = actor.configure(contract.to_dict(), fmt)
    with pytest.raises(ValueError, match="namespace"):
        actor.configure(
            replace(contract, tokenizer_fingerprint="e" * 64).to_dict(), fmt
        )
    with pytest.raises(ValueError, match="geometry"):
        actor.put(namespace, ["a"], [torch.zeros(192, dtype=torch.bfloat16)])
    with pytest.raises(ValueError, match="geometry"):
        actor.put(namespace, ["a"], [torch.zeros(191, dtype=torch.float16)])
    assert actor.get_stats()["pages"] == 0


def test_pd_lease_pins_complete_prefix_until_release(contract, monkeypatch):
    actor = XavierCacheActor(768)
    namespace = actor.configure(contract.to_dict(), {"layout": "layer_first"})
    page = torch.zeros(384, dtype=torch.uint8)
    assert actor.reserve_handoff(namespace, ["a", "b"]) is None
    actor.put(namespace, ["a", "b"], [page, page])
    ticket = actor.reserve_handoff(namespace, ["a", "b"])
    assert actor.validate_handoff(namespace, ["a", "b"], ticket)
    assert not actor.validate_handoff(namespace, ["b", "a"], ticket)
    assert actor.put(namespace, ["c"], [page]) == [False]
    actor.release_handoff(ticket)
    actor.release_handoff(ticket)
    assert actor.put(namespace, ["c"], [page]) == [True]
    assert not actor.validate_handoff(namespace, ["a", "b"], ticket)
    ticket = actor.reserve_handoff(namespace, ["c"])
    monkeypatch.setattr(
        "xinference.model.llm.xavier.backends.torch.storage.time.monotonic",
        lambda: float("inf"),
    )
    assert actor.get_stats()["active_handoffs"] == 0
    assert not actor.validate_handoff(namespace, ["c"], ticket)
    with pytest.raises(ValueError, match="budget"):
        actor.reserve_handoff(namespace, ["a", "b", "c"])


def test_pd_capacity_is_reserved_before_publication(contract, monkeypatch):
    actor = XavierCacheActor(768)
    namespace = actor.configure(contract.to_dict(), {"layout": "layer_first"})
    page = torch.zeros(384, dtype=torch.uint8)
    ticket = actor.prepare_handoff(namespace, ["a", "b"])
    assert not actor.validate_handoff(namespace, ["a", "b"], ticket)
    with pytest.raises(RuntimeError, match="active PD"):
        actor.prepare_handoff(namespace, ["other"])
    with pytest.raises(ValueError, match="budget"):
        actor.prepare_handoff(namespace, ["a", "b", "c"])
    # Backup may include an extra decode/tail page. It must not evict the
    # leased prompt prefix even if the budget holds exactly that prefix.
    assert actor.put(namespace, ["a", "b", "tail"], [page] * 3) == [True, True, False]
    assert actor.validate_handoff(namespace, ["a", "b"], ticket)
    actor.release_handoff(ticket)
    assert actor.get_stats()["pages"] == 0
    assert actor.put(namespace, ["other"], [page]) == [True]
    ticket = actor.prepare_handoff(namespace, ["pending"])
    assert actor.put(namespace, ["pending"], [page]) == [True]
    monkeypatch.setattr(
        "xinference.model.llm.xavier.backends.torch.storage.time.monotonic",
        lambda: float("inf"),
    )
    assert actor.get_stats()["active_handoffs"] == 0
    assert actor.exists(namespace, ["pending"]) == 0
    assert actor.exists(namespace, ["other"]) == 1


@pytest.fixture
def model_path(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "num_hidden_layers": 3,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "hidden_size": 16,
            }
        )
    )
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "tokenizer.json").write_text("{}")
    return str(tmp_path)


@pytest.mark.parametrize("field", ["layer_types", "head_dim", "num_key_value_heads"])
def test_null_optional_model_geometry_uses_defaults(model_path, field):
    path = Path(model_path) / "config.json"
    config = json.loads(path.read_text())
    config[field] = None
    path.write_text(json.dumps(config))
    cache = dict(role="decode")
    configure_xavier(model_path, {}, cache)
    contract = KVCacheContract.from_dict(cache["contract"])
    assert contract.head_dim == 4
    assert contract.num_kv_heads == (4 if field == "num_key_value_heads" else 2)


@pytest.mark.parametrize("field", ["head_dim", "num_key_value_heads"])
def test_zero_model_geometry_is_rejected(model_path, field):
    path = Path(model_path) / "config.json"
    config = json.loads(path.read_text())
    config[field] = 0
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Invalid"):
        configure_xavier(model_path, {}, dict(role="decode"))


@pytest.mark.asyncio
async def test_storage_reuses_one_thread_affine_loop_across_calling_threads(
    contract, storage_class, monkeypatch
):
    import concurrent.futures
    import threading

    config = SimpleNamespace(
        tp_size=1,
        pp_size=1,
        attn_cp_size=1,
        is_mla_model=False,
        extra_config=dict(address="unused", uid="unused", contract=contract.to_dict()),
    )
    from unittest.mock import AsyncMock

    calls = []

    async def stats():
        calls.append((threading.get_ident(), asyncio.get_running_loop()))
        return {"pages": 0}

    ref = SimpleNamespace(get_stats=stats)
    lookup = AsyncMock(return_value=ref)
    monkeypatch.setattr(xo, "actor_ref", lookup)
    storage = storage_class(config)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            assert executor.submit(storage.get_stats).result() == {"pages": 0}
            assert executor.submit(storage.get_stats).result() == {"pages": 0}
        assert calls[0] == calls[1]
        assert calls[0][0] == storage._rpc_thread.ident
        lookup.assert_awaited_once()
    finally:
        storage.close()
    assert storage._rpc_loop.is_closed()


def test_cross_engine_cannot_select_unfingerprinted_pt_weights(model_path):
    with pytest.raises(ValueError, match="automatic or safetensors"):
        configure_xavier(model_path, {"load_format": "pt"}, {"heterogeneous": True})


def test_launch_configuration_consumes_adapter_options(model_path):
    config = {"tp_size": 1, "page_size": 4}
    configure_xavier(model_path, config, {"address": "worker:1234", "uid": "cache"})
    extra = json.loads(config["hicache_storage_backend_extra_config"])
    assert (
        extra["interface_v1"] is True and extra["class_name"] == "XavierHiCacheStorage"
    )
    assert config["hicache_storage_backend"] == "dynamic"
    assert config["dtype"] == "float16" and config["enable_hierarchical_cache"] is True
    assert KVCacheContract.from_dict(extra["contract"]).block_size == 4


def test_pd_uses_gpu_slots_and_rejects_cpu_hicache(model_path):
    cache = {"address": "worker:1234", "uid": "cache", "role": "decode"}
    config = {}
    configure_xavier(model_path, config, cache)
    assert config["disaggregation_mode"] == "decode"
    assert config["disaggregation_transfer_backend"] == "xavier"
    assert "enable_hierarchical_cache" not in config
    assert "hicache_storage_backend" not in config
    assert config["disaggregation_decode_enable_radix_cache"] is False
    with pytest.raises(ValueError, match="CPU HiCache"):
        configure_xavier(model_path, {"hicache_host_memory_mode": "cache"}, cache)


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_pd_rejects_decode_radix_cache_at_launch(model_path, role):
    with pytest.raises(ValueError, match="decode radix cache"):
        configure_xavier(
            model_path,
            {"disaggregation_decode_enable_radix_cache": True},
            {"address": "worker:1234", "uid": "cache", "role": role},
        )


@pytest.mark.parametrize(
    "option,value",
    [
        ("tp_size", 2),
        ("pp_size", 2),
        ("dp_size", 2),
        ("nnodes", 2),
        ("attn_cp_size", 2),
        ("dcp_size", 2),
        ("load_format", "dummy"),
        ("model_impl", "transformers"),
        ("tokenizer_mode", "slow"),
        ("tokenizer_backend", "fastokens"),
        ("model_loader_extra_config", '{"custom": true}'),
        ("dtype", "bfloat16"),
        ("kv_cache_dtype", "fp8"),
        ("quantization", "awq"),
        ("enable_lora", True),
        ("speculative_algorithm", "EAGLE"),
        ("hicache_mem_layout", "page_first"),
        ("hicache_storage_backend", "file"),
    ],
)
def test_unsupported_engine_options_are_rejected(model_path, option, value):
    with pytest.raises(ValueError):
        configure_xavier(
            model_path, {option: value}, {"address": "worker:1234", "uid": "cache"}
        )


@pytest.mark.asyncio
async def test_large_hicache_batches_preserve_remote_pages_and_prefix_misses(
    contract, storage_class
):
    count = 259
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        actor = await xo.create_actor(
            XavierCacheActor,
            count * (contract.num_layers * contract.layer_nbytes),
            address=pool.external_address,
            uid="large-cache",
        )
        cfg = SimpleNamespace(
            tp_size=1,
            pp_size=1,
            attn_cp_size=1,
            is_mla_model=False,
            extra_config=dict(
                address=actor.address, uid=actor.uid, contract=contract.to_dict()
            ),
        )
        source, target = storage_class(cfg), storage_class(cfg)
        try:
            for adapter in (source, target):
                await asyncio.to_thread(adapter.register_mem_pool_host, HostPool())
            keys = [f"{i:064x}" for i in range(count)]
            pages = [
                torch.full(
                    ((contract.num_layers * contract.layer_nbytes) // 2,),
                    i,
                    dtype=torch.float16,
                )
                for i in range(count)
            ]
            assert await asyncio.to_thread(source.batch_set, keys, pages)
            assert await asyncio.to_thread(target.batch_exists, keys) == count
            restored = await asyncio.to_thread(target.batch_get, keys)
            assert all(torch.equal(a, b) for a, b in zip(restored, pages))
            missing = list(keys)
            missing[130] = "f" * 64
            assert await asyncio.to_thread(target.batch_exists, missing) == 130
            restored = await asyncio.to_thread(target.batch_get, missing)
            assert restored[130] is None and torch.equal(restored[258], pages[258])
        finally:
            source.close()
            target.close()


def test_bf16_checkpoint_cast_is_visible(model_path, caplog):
    import logging

    cfg_path = Path(model_path) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["torch_dtype"] = "bfloat16"
    cfg_path.write_text(json.dumps(cfg))
    output = {"dtype": "auto"}
    with caplog.at_level(logging.WARNING):
        configure_xavier(str(model_path), output, {"address": "a", "uid": "u"})
    assert output["dtype"] == "float16"
    assert "BF16 checkpoint to FP16" in caplog.text


def test_inactive_plugin_does_not_import_or_patch_optional_sglang(monkeypatch):
    import builtins

    from ..xavier.plugin import register
    from ..xavier.settings import GPU_CONFIG_ENV

    monkeypatch.delenv(GPU_CONFIG_ENV, raising=False)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("sglang"):
            raise ImportError("old or absent SGLang")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    register()
