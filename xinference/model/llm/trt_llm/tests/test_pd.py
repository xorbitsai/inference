# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import copy
import json
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import xoscar as xo

from ...xavier.backends.torch.storage import XavierCacheActor
from ...xavier.contract import KVCacheContract
from ..pd import (
    TRTLLMPD,
    configure_native,
    forget_request,
    request_state,
    submit_xavier,
)
from ..xavier import CONFIG_ENV
from .test_core import model as model
from .test_xavier import checkpoint as checkpoint
from .test_xavier import connector as connector


@pytest.mark.parametrize(
    "options",
    [
        {"kv_connector_config": {}},
        {"enable_chunked_prefill": True},
        {"disable_overlap_scheduler": False},
        {"cache_transceiver_config": {"backend": "UCX"}},
    ],
)
def test_native_pd_rejects_conflicting_execution(checkpoint, options):
    with pytest.raises(ValueError):
        configure_native(checkpoint, options, {"role": "prefill"})


def test_native_pd_config(checkpoint):
    config = {"max_seq_len": 512}
    contract = configure_native(checkpoint, config, {"role": "decode"})
    assert contract.logical_dtype == "bfloat16"
    assert config["cache_transceiver_config"] == dict(
        backend="NIXL", kv_transfer_timeout_ms=30000
    )
    assert config["disable_overlap_scheduler"]
    assert not config["enable_chunked_prefill"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["async_generate", "async_chat"])
@pytest.mark.parametrize("stream", [False, True])
async def test_native_context_round_trip_keeps_tokens_and_cleans_source(
    model, monkeypatch, method, stream
):
    from .test_core import Request

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.disaggregated_params",
        SimpleNamespace(DisaggregatedParams=SimpleNamespace),
    )
    contract = SimpleNamespace(fingerprint="checkpoint")
    p, d = TRTLLMPD("prefill", "nixl", contract), TRTLLMPD("decode", "nixl", contract)
    submitted = []

    def generate(tokens, params, streaming, disaggregated_params):
        request = Request()
        request.disaggregated_params = SimpleNamespace(
            first_gen_tokens=[9],
            ctx_request_id=321,
            opaque_state=b"\x00\xffnative",
            draft_tokens=None,
        )
        submitted.append((request, disaggregated_params, params.max_tokens))
        return request

    monkeypatch.setattr(model._engine, "generate_async", generate)
    inputs = "hi" if method == "async_generate" else [{"role": "user", "content": "hi"}]
    config = {"max_tokens": 4, "_pd_kv_transfer_params": {"do_remote_decode": True}}
    original = copy.deepcopy(config)
    model._pd = p
    prefill = await getattr(model, method)(inputs, config, "public-id")
    assert config == original
    transfer = json.loads(json.dumps(prefill["_pd_kv_transfer_params"]))
    assert submitted[0][1].request_type == "context_only"
    assert submitted[0][2] == 1
    assert "public-id" in p.active and not submitted[0][0].aborted
    model._pd = d
    result = await getattr(model, method)(
        inputs,
        {"max_tokens": 4, "stream": stream, "_pd_kv_transfer_params": transfer},
        "public-id",
    )
    if stream:
        result = [chunk async for chunk in result]
        assert result
    else:
        assert result["usage"]["completion_tokens"] == 3
    native = submitted[1][1]
    assert native.request_type == "generation_only"
    assert native.first_gen_tokens == [9] and native.ctx_request_id == 321
    assert native.opaque_state == b"\x00\xffnative"
    assert submitted[1][2] == 4
    assert not model._requests
    await p.release("public-id")
    assert submitted[0][0].aborted and not p.active


@pytest.mark.asyncio
async def test_failed_native_prefill_releases_request(model, monkeypatch):
    from .test_core import Request

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.disaggregated_params",
        SimpleNamespace(DisaggregatedParams=SimpleNamespace),
    )
    request = Request()
    request.disaggregated_params = None
    monkeypatch.setattr(model._engine, "generate_async", lambda *a, **kw: request)
    model._pd = TRTLLMPD("prefill", "nixl", SimpleNamespace(fingerprint="model"))
    with pytest.raises(RuntimeError, match="context metadata"):
        await model.async_generate(
            "hi", {"_pd_kv_transfer_params": {"do_remote_decode": True}}, "failed"
        )
    assert request.aborted and not model._pd.active and not model._requests


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["prefill", "decode"])
async def test_pd_replicas_require_router_metadata(model, role):
    model._pd = TRTLLMPD(role, "nixl", SimpleNamespace(fingerprint="model"))
    with pytest.raises(ValueError, match="require P/D metadata"):
        await model.async_generate("hi")
    assert not model._engine.requests


@pytest.mark.asyncio
async def test_actor_release_accepts_rpc_context(model, monkeypatch):
    from xinference.core.model import ModelActor
    from xinference.core.rpc_context import actor_call

    monkeypatch.setattr(model.model_family, "to_description", lambda: {}, raising=False)
    actor = ModelActor("test:1", "test:2", model, "tiny-rep0")
    model._pd = TRTLLMPD("prefill", "nixl")
    request = SimpleNamespace(abort=Mock())
    model._pd.active["r"] = dict(request=request)
    await actor_call(
        actor, "release_trt_pd_request", "r", _rpc_operation_request_id="r"
    )
    request.abort.assert_called_once()
    assert not model._pd.active


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_task", [False, True])
async def test_cancellation_during_xavier_capacity_reservation_recovers_ticket(
    cancel_task,
):
    started, finish = threading.Event(), threading.Event()
    released = []

    def rpc(method, *args):
        if method == "prepare_handoff":
            started.set()
            assert finish.wait(timeout=3)
            return "reserved-ticket"
        assert method == "release_handoff"
        released.append(args[0])

    pd = TRTLLMPD("prefill", "nixl")
    pd.client = SimpleNamespace(rpc=rpc, namespace="namespace")
    task = asyncio.create_task(
        pd.prepare("r", list(range(70)), {"do_remote_decode": True})
    )
    assert await asyncio.to_thread(started.wait, 3)
    if cancel_task:
        task.cancel()
    else:
        await pd.release("r")
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert released == ["reserved-ticket"]
    assert not pd.active


def test_xavier_enqueue_failure_cleans_callback_state(monkeypatch):
    from ..pd import XavierRequest

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.inputs",
        SimpleNamespace(get_cache_salt_id=lambda salt: 123),
    )
    state = XavierRequest("decode", (1, 2), [])

    def enqueue(*args, **kwargs):
        assert request_state(SimpleNamespace(cache_salt_id=123)) is state
        raise RuntimeError("enqueue failed")

    engine = SimpleNamespace(generate_async=enqueue)
    with pytest.raises(RuntimeError, match="enqueue failed"):
        submit_xavier(engine, [1, 2], None, state)
    assert request_state(SimpleNamespace(cache_salt_id=123)) is None


@pytest.mark.asyncio
async def test_xavier_owned_partial_page_handoff_survives_source_release(
    monkeypatch, connector
):
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.inputs",
        SimpleNamespace(get_cache_salt_id=lambda salt: int(salt[:16], 16)),
    )
    monkeypatch.setitem(
        sys.modules, "xinference.model.llm.trt_llm.connector", connector
    )
    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 1, 1, 1, 64, "bfloat16"
    )
    page_bytes = contract.num_layers * contract.layer_nbytes
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        cache = await xo.create_actor(
            XavierCacheActor,
            4 * page_bytes,
            address=pool.external_address,
            uid="trt-pd-cpu",
        )
        monkeypatch.setenv(
            CONFIG_ENV,
            json.dumps(
                dict(
                    address=cache.address, uid="trt-pd-cpu", contract=contract.to_dict()
                )
            ),
        )
        p, d = [
            await asyncio.to_thread(TRTLLMPD, role, "xavier")
            for role in ("prefill", "decode")
        ]
        ps, ds = [
            await asyncio.to_thread(connector.XavierConnectorScheduler, None)
            for _ in range(2)
        ]
        pw, dw = [
            await asyncio.to_thread(connector.XavierConnectorWorker, None)
            for _ in range(2)
        ]
        source = torch.arange(384, dtype=torch.bfloat16).reshape(3, 1, 2, 64)
        target = torch.zeros_like(source)
        pw.register_kv_caches(source)
        dw.register_kv_caches(target)
        tokens = list(range(70))
        state = await p.prepare("r", tokens, {"do_remote_decode": True})
        handle = SimpleNamespace(request_id=11, abort=Mock())
        engine = SimpleNamespace(generate_async=Mock(return_value=handle))
        p.submit(engine, tokens, None, state)
        request = SimpleNamespace(
            request_id=101,
            cache_salt_id=int(
                engine.generate_async.call_args.kwargs["cache_salt"][:16], 16
            ),
            get_tokens=lambda beam: tokens,
        )
        assert request_state(request) is state["context"]
        assert ps.get_num_new_matched_tokens(request, 0) == (0, False)
        ps.update_state_after_alloc(request, [0, 1])

        def output(rid, scheduled):
            return SimpleNamespace(
                new_requests=[
                    SimpleNamespace(
                        request_id=rid,
                        new_block_ids=[],
                        computed_position=0,
                        num_scheduled_tokens=scheduled,
                    )
                ]
            )

        pw.bind_connector_meta(ps.build_connector_meta(output(101, 70)))
        await asyncio.to_thread(
            pw.wait_for_save, SimpleNamespace(synchronize=lambda: None)
        )
        transfer = await p.publish("r", tokens, None)
        assert (await cache.get_stats())["active_handoffs"] == 1
        concurrent = await p.prepare("same-prompt", tokens, {"do_remote_decode": True})
        assert set(concurrent["context"].keys).isdisjoint(state["context"].keys)
        assert concurrent["ticket"] != state["ticket"]
        with pytest.raises(RuntimeError, match="full of active"):
            await p.prepare("excess", tokens, {"do_remote_decode": True})
        assert "excess" not in p.active
        decoded = await d.prepare("r", tokens, transfer)
        expected = source[:2].clone()
        await p.release("r")
        assert (await cache.get_stats())["active_handoffs"] == 1
        await p.release("same-prompt")
        source.zero_()
        assert (await cache.get_stats())["active_handoffs"] == 0
        with pytest.raises(ValueError, match="incomplete"):
            await d.prepare("again", tokens, transfer)
        handle_d = SimpleNamespace(request_id=12, abort=Mock())
        engine.generate_async.return_value = handle_d
        d.submit(engine, tokens, None, decoded)
        request.request_id = 102
        request.cache_salt_id = int(
            engine.generate_async.call_args.kwargs["cache_salt"][:16], 16
        )
        assert ds.get_num_new_matched_tokens(request, 0) == (69, False)
        ds.update_state_after_alloc(request, [2, 0])
        dw.bind_connector_meta(ds.build_connector_meta(output(102, 1)))
        with pytest.raises(RuntimeError, match="did not import"):
            d.check_import(decoded)
        await asyncio.to_thread(dw.start_load_kv, None)
        d.check_import(decoded)
        assert torch.equal(target[2], expected[0])
        assert torch.equal(target[0], expected[1])
        assert not target[1].any()
        assert (await cache.get_stats())["imported_pages"] == 2
        forget_request(handle)
        forget_request(handle_d)
        assert request_state(request) is None
        await asyncio.to_thread(connector.close_clients)
