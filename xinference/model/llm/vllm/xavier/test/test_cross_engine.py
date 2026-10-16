# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def cross_module(connector_module, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "xinference.model.llm.vllm.xavier.v1_connector", connector_module
    )
    name = "xinference.model.llm.vllm.xavier._test_cross_engine"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "cross_engine.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def decode(cross_module, tokens):
    instance = cross_module.CrossEngineConnector.__new__(
        cross_module.CrossEngineConnector
    )
    instance._is_consumer = True
    instance._is_producer = False
    instance._cross_requests = {}
    instance._cross_allocated = {}
    instance._direct_sends = set()
    instance._direct_handoff = True
    instance._cross_prepare_errors = {}
    instance._pending_aborts = {}
    instance._call = lambda coro: coro
    instance._prepare = lambda request: 123
    request = SimpleNamespace(
        request_id="r",
        prompt_token_ids=tokens,
        kv_transfer_params=dict(do_remote_prefill=True),
    )
    return instance, request


@pytest.mark.parametrize("allocated_pages", [3, 4])
def test_vllm_decode_reuses_all_but_final_prompt_token(cross_module, allocated_pages):
    instance, request = decode(cross_module, list(range(193)))
    assert instance.get_num_new_matched_tokens(request, 0) == (192, True)
    destinations = [9, 5, 7, 8][:allocated_pages]
    blocks = SimpleNamespace(get_block_ids=lambda: (destinations,))
    instance.update_state_after_alloc(request, blocks, 192)
    meta = instance.build_connector_meta(SimpleNamespace(num_scheduled_tokens={}))
    assert meta.loads == [("r", 123, destinations)]
    assert request.kv_transfer_params["do_remote_prefill"] is False
    assert instance.get_num_new_matched_tokens(request, 0) == (0, False)
    # Preemption subsequently recomputes locally; the consumed room is single-use.
    assert instance._cross_allocated == {}


def test_vllm_decode_reports_zero_reuse_prompt_as_load_failure(cross_module):
    instance, request = decode(cross_module, [1])
    assert instance.get_num_new_matched_tokens(request, 0) == (1, True)
    assert "two prompt tokens" in instance._cross_prepare_errors["r"]


@pytest.mark.asyncio
async def test_cross_prepare_checks_actual_token_ids_once(cross_module):
    instance = cross_module.CrossEngineConnector.__new__(
        cross_module.CrossEngineConnector
    )
    instance.contract = SimpleNamespace(fingerprint="a" * 64)
    instance._xavier_config = dict(role="decode", rank=1)
    directory = AsyncMock()
    instance.directory = AsyncMock(return_value=directory)
    request = SimpleNamespace(
        request_id="r",
        prompt_token_ids=[1, 2, 3],
        kv_transfer_params={
            "sglang_xavier": dict(mode="gpu", room=9),
            "xavier_prompt_digest": cross_module.prompt_digest([1, 2, 3]),
        },
    )
    assert await instance._prepare(request) == 9
    assert await instance._prepare(request) == 9
    directory.prepare.assert_not_awaited()
    request.prompt_token_ids.append(4)
    with pytest.raises(ValueError, match="these token IDs"):
        await instance._prepare(request)


@pytest.mark.parametrize(
    "option,value",
    [
        ("quantization", "awq"),
        ("dtype", "bfloat16"),
        ("block_size", 16),
        ("hf_overrides", {"rope_theta": 500000}),
        ("tokenizer", "another"),
        ("load_format", "pt"),
    ],
)
def test_cross_engine_rejects_incompatible_runtime_options(cross_module, option, value):
    with pytest.raises(ValueError):
        cross_module.configure_cross_engine(
            "unused", {option: value}, {"role": "decode"}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [RuntimeError("room cancelled"), TimeoutError("transfer")]
)
async def test_failed_load_reports_all_blocks_without_crashing_vllm(
    cross_module, error
):
    instance, _ = decode(cross_module, list(range(193)))
    instance._loop = object()
    instance._invalid_block_ids = set()
    actor = AsyncMock()
    actor.receive.side_effect = error
    instance._ensure_gpu_cache_mapping = AsyncMock(return_value=actor)
    instance._gpu_load_jobs = {}

    async def load(request_id, room, destinations):
        meta = cross_module.CrossEngineMetadata(
            loads=[(request_id, room, destinations)]
        )
        instance._get_connector_metadata = lambda: meta
        coroutines = []
        instance._call = coroutines.append
        instance.start_load_kv(None)
        for coroutine in coroutines:
            await coroutine
        await asyncio.gather(*instance._gpu_load_jobs)
        instance._call = lambda coro: coro.close()

    await load("r", 123, [9, 5, 7, 8])
    assert instance.get_finished(set()) == (set(), {"r"})
    assert instance.get_block_ids_with_load_errors() == {9, 5, 7, 8}
    assert not instance._gpu_load_jobs
    actor.receive.side_effect = None
    actor.receive.return_value = set()
    await load("next", 124, [1, 2])
    assert instance.get_finished(set()) == (set(), {"next"})
    assert instance.get_block_ids_with_load_errors() == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_stage", ["prepare", "mapping"])
async def test_hook_failure_finishes_only_its_request(cross_module, failed_stage):
    instance, request = decode(cross_module, list(range(193)))
    instance._loop = object()
    instance._gpu_load_jobs = {}
    instance._invalid_block_ids = set()
    if failed_stage == "prepare":

        def fail(request):
            raise ValueError("prompt mismatch")

        instance._prepare = fail
    else:
        instance._ensure_gpu_cache_mapping = AsyncMock(
            side_effect=OSError("actor unavailable")
        )
    matched, pending = instance.get_num_new_matched_tokens(request, 0)
    assert pending
    destinations = [9, 5, 7, 8][: (matched + 63) // 64]
    instance.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda: (destinations,)), matched
    )
    meta = instance.build_connector_meta(SimpleNamespace(num_scheduled_tokens={}))
    instance._get_connector_metadata = lambda: meta
    coroutines = []
    instance._call = coroutines.append
    instance.start_load_kv(None)
    for coroutine in coroutines:
        await coroutine
    await asyncio.gather(*instance._gpu_load_jobs)
    instance._call = lambda coro: coro.close()
    assert instance.get_finished(set()) == (set(), {"r"})
    assert instance.get_block_ids_with_load_errors() == set(destinations)


def test_producer_registration_failure_is_drained_without_engine_exception(
    cross_module, monkeypatch
):
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.request",
        SimpleNamespace(RequestStatus=SimpleNamespace(FINISHED_ABORTED="aborted")),
    )
    instance, request = decode(cross_module, [1, 2])
    instance._is_producer, instance._is_consumer = True, False
    instance._block_size = 64
    request.status, request.output_token_ids = "finished", [42]
    request.kv_transfer_params["do_remote_decode"] = True
    actor = AsyncMock()
    actor.register_prefill.side_effect = OSError("reply lost")
    actor.abort.side_effect = OSError("temporary disconnect")
    instance._get_transfer_ref = AsyncMock(return_value=actor)
    instance._prepare = AsyncMock(return_value=123)
    instance._call = asyncio.run
    hold, metadata = instance.request_finished(request, [9])
    assert hold and metadata["xavier_error"] == "reply lost"
    assert instance._pending_aborts == {"r": 123}
    worker, _ = decode(cross_module, [1, 2])
    worker._is_producer, worker._is_consumer = True, False
    worker._gpu_load_jobs = {}
    worker._transfer_ref = actor
    worker._get_transfer_ref = AsyncMock(return_value=actor)
    worker._call = asyncio.run
    actor.poll_direct_gpu_v1.return_value = set()
    meta = pickle.loads(
        pickle.dumps(
            instance.build_connector_meta(SimpleNamespace(num_scheduled_tokens={}))
        )
    )
    assert not instance._pending_aborts and not instance._direct_sends
    assert not worker._pending_aborts
    worker._get_connector_metadata = lambda: meta
    worker.start_load_kv(None)
    assert worker._pending_aborts == {"r": 123}
    assert worker.get_finished(set()) == (set(), set())
    assert worker._pending_aborts == {"r": 123} and worker._direct_sends == {"r"}
    meta = instance.build_connector_meta(SimpleNamespace(num_scheduled_tokens={}))
    worker.start_load_kv(None)
    assert worker._pending_aborts == {"r": 123}
    actor.abort.side_effect = None
    assert worker.get_finished(set()) == ({"r"}, set())
    assert not worker._pending_aborts and not worker._direct_sends
    assert worker.get_finished(set()) == (set(), set())


def test_completed_send_clears_pending_abort_without_duplicate_finish(cross_module):
    worker, _ = decode(cross_module, [1, 2])
    worker._is_producer, worker._is_consumer = True, False
    worker._direct_sends = {"r"}
    worker._pending_aborts = {"r": 123}
    worker._gpu_load_jobs = {}
    actor = AsyncMock()
    actor.abort.side_effect = OSError("RPC unavailable")
    actor.poll_direct_gpu_v1.return_value = {"r"}
    worker._transfer_ref = actor
    worker._get_transfer_ref = AsyncMock(return_value=actor)
    worker._call = asyncio.run
    assert worker.get_finished(set()) == ({"r"}, set())
    assert not worker._pending_aborts and not worker._direct_sends
    assert worker.get_finished(set()) == (set(), set())


def test_completion_rpc_failure_retains_pinned_request_for_retry(cross_module):
    instance, _ = decode(cross_module, [1, 2])
    instance._is_producer, instance._is_consumer = True, False
    instance._direct_sends = {"r"}
    instance._gpu_load_jobs = {}
    instance._transfer_ref = AsyncMock()
    instance._transfer_ref.poll_direct_gpu_v1.side_effect = OSError("RPC failed")
    instance._call = asyncio.run
    assert instance.get_finished(set()) == (set(), set())
    assert instance._direct_sends == {"r"}
    instance._transfer_ref.poll_direct_gpu_v1.side_effect = None
    instance._transfer_ref.poll_direct_gpu_v1.return_value = {"r"}
    assert instance.get_finished(set()) == ({"r"}, set())


@pytest.mark.parametrize("context", [None, 4096, 8192])
def test_engine_adapters_share_contract_and_cache_weight_hashes(
    cross_module, tmp_path, monkeypatch, context
):
    from xinference.model.llm.sglang.xavier import config as sglang_config
    from xinference.model.llm.xavier import pd_contract

    config = dict(
        model_type="qwen2",
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=8,
        head_dim=None,
        num_key_value_heads=None,
        layer_types=None,
        max_position_embeddings=4096,
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "tokenizer.json").write_bytes(b"tokenizer")
    monkeypatch.setattr(pd_contract, "XINFERENCE_CACHE_DIR", str(tmp_path / "cache"))
    from unittest.mock import Mock

    old_open = Path.open
    reads = []

    def opened(path, *args, **kwargs):
        if path.name == "model.safetensors":
            reads.append(path)
        return old_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    duplicate = Mock(side_effect=AssertionError("unused homogeneous contract"))
    monkeypatch.setattr(sglang_config, "fingerprint_files", duplicate)
    vcache, scache = dict(role="decode"), dict(role="decode", heterogeneous=True)
    cross_module.configure_cross_engine(
        str(tmp_path), dict(max_model_len=context), vcache
    )
    sglang_config.configure_xavier(str(tmp_path), dict(context_length=context), scache)
    assert vcache["contract"] == scache["contract"]
    assert len(reads) == 1
    duplicate.assert_not_called()
