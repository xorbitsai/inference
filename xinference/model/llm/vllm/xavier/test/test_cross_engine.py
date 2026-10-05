# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
import importlib.util
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
    instance._prepared = set()
    instance._call = lambda coro: coro
    instance._prepare = lambda request: 123
    request = SimpleNamespace(
        request_id="r",
        prompt_token_ids=tokens,
        kv_transfer_params=dict(do_remote_prefill=True),
    )
    return instance, request


def test_vllm_decode_reuses_all_but_final_prompt_token(cross_module):
    instance, request = decode(cross_module, list(range(193)))
    assert instance.get_num_new_matched_tokens(request, 0) == (192, True)
    blocks = SimpleNamespace(get_block_ids=lambda: ([9, 5, 7, 8],))
    instance.update_state_after_alloc(request, blocks, 192)
    meta = instance.build_connector_meta(SimpleNamespace(num_scheduled_tokens={}))
    assert meta.loads == [("r", 123, [9, 5, 7, 8])]
    assert request.kv_transfer_params["do_remote_prefill"] is False
    assert instance.get_num_new_matched_tokens(request, 0) == (0, False)
    # Preemption subsequently recomputes locally; the consumed room is single-use.
    assert instance._cross_allocated == {}


def test_vllm_decode_rejects_zero_reuse_prompt(cross_module):
    instance, request = decode(cross_module, [1])
    with pytest.raises(ValueError, match="two prompt tokens"):
        instance.get_num_new_matched_tokens(request, 0)


@pytest.mark.asyncio
async def test_cross_prepare_checks_actual_token_ids_once(cross_module):
    instance = cross_module.CrossEngineConnector.__new__(
        cross_module.CrossEngineConnector
    )
    instance._prepared = set()
    instance.contract = SimpleNamespace(fingerprint="a" * 64)
    instance._xavier_config = dict(role="decode")
    directory = AsyncMock()
    instance.directory = AsyncMock(return_value=directory)
    request = SimpleNamespace(
        request_id="r",
        prompt_token_ids=[1, 2, 3],
        kv_transfer_params={"sglang_xavier": dict(mode="gpu", room=9)},
    )
    assert await instance._prepare(request) == 9
    assert await instance._prepare(request) == 9
    directory.prepare.assert_awaited_once()
    args = directory.prepare.call_args.args
    assert args == (9, "a" * 64, cross_module.prompt_digest([1, 2, 3]), "decode", 3)


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
    instance._call = lambda coro: coro.close()
    actor = AsyncMock()
    actor.receive.side_effect = error
    task = asyncio.create_task(instance._receive_pages(actor, 123, [9, 5, 7, 8]))
    instance._gpu_load_jobs = {task: ["r"]}
    await task
    assert instance.get_finished(set()) == (set(), {"r"})
    assert instance.get_block_ids_with_load_errors() == {9, 5, 7, 8}
    assert not instance._gpu_load_jobs
    actor.receive.side_effect = None
    actor.receive.return_value = set()
    task = asyncio.create_task(instance._receive_pages(actor, 124, [1, 2]))
    instance._gpu_load_jobs = {task: ["next"]}
    await task
    assert instance.get_finished(set()) == (set(), {"next"})
    assert instance.get_block_ids_with_load_errors() == set()
