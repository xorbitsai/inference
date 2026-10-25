# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import copy
import json
import sys
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ...xavier.contract import KVCacheContract
from ..pd import (
    TRTLLMPD,
    configure_native,
    forget_request,
    request_state,
    submit_xavier,
)
from .test_core import model as model
from .test_xavier import checkpoint as checkpoint
from .test_xavier import connector as connector


@pytest.mark.parametrize(
    "options",
    [
        {"kv_connector_config": {}},
        {"dtype": "float16"},
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


@pytest.mark.parametrize("sender_timeout", [None, 250])
def test_native_sender_wait_is_bounded(checkpoint, sender_timeout):
    config = {
        "cache_transceiver_config": {
            "kv_transfer_timeout_ms": None,
            "kv_transfer_sender_future_timeout_ms": sender_timeout,
        }
    }
    configure_native(checkpoint, config, {"role": "prefill"})
    assert config["cache_transceiver_config"] == {
        "backend": "NIXL",
        "kv_transfer_timeout_ms": None,
        "kv_transfer_sender_future_timeout_ms": (
            1000 if sender_timeout is None else sender_timeout
        ),
    }


@pytest.fixture
def xavier_decode(model, monkeypatch):
    from ...xavier.pd_contract import prompt_digest

    client = SimpleNamespace(
        namespace="namespace",
        config={"rank": 1},
        rpc=Mock(),
        close=Mock(),
    )
    monkeypatch.setitem(
        sys.modules,
        "xinference.model.llm.trt_llm.connector",
        SimpleNamespace(XavierClient=lambda: client),
    )
    model._pd = TRTLLMPD("decode", "xavier")
    transfer = {
        "do_remote_prefill": True,
        "trt_llm": {
            "backend": "xavier",
            "namespace": client.namespace,
            "prompt_digest": prompt_digest([0, 1]),
            "room": 123,
        },
    }
    return model, transfer


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["async_generate", "async_chat"])
@pytest.mark.parametrize("close", ["aclose", "destroy_wrapper"])
async def test_unstarted_xavier_stream_releases_prepared_state(
    xavier_decode, method, close
):
    import gc

    from xinference.core.rpc_context import build_rpc_metadata, wrap_async_iterator

    model, transfer = xavier_decode
    inputs = "hi" if method == "async_generate" else [{"role": "user", "content": "hi"}]
    config = {"stream": True, "_pd_kv_transfer_params": transfer}
    stream = await getattr(model, method)(inputs, config, "unstarted")
    assert "unstarted" in model._pd.active
    assert not model._engine.requests
    if close == "aclose":
        await stream.aclose()
    else:
        wrapped = wrap_async_iterator(
            stream, build_rpc_metadata(operation_request_id="unstarted")
        )
        del stream, wrapped
        gc.collect()
        for _ in range(100):
            if not model._pd.active:
                break
            await asyncio.sleep(0.001)
    assert not model._pd.active and not model._requests
    assert not model._engine.requests
    retry = await getattr(model, method)(inputs, config, "unstarted")
    await retry.aclose()
    assert not model._pd.active


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["async_generate", "async_chat"])
async def test_xavier_stream_enqueue_error_releases_prepared_state(
    xavier_decode, monkeypatch, method
):
    from .. import pd

    model, transfer = xavier_decode
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.inputs",
        SimpleNamespace(get_cache_salt_id=lambda salt: 42),
    )
    monkeypatch.setattr(
        model._engine,
        "generate_async",
        Mock(side_effect=RuntimeError("enqueue failed")),
    )
    inputs = "hi" if method == "async_generate" else [{"role": "user", "content": "hi"}]
    stream = await getattr(model, method)(
        inputs,
        {"stream": True, "_pd_kv_transfer_params": transfer},
        "enqueue-error",
    )
    with pytest.raises(RuntimeError, match="enqueue failed"):
        await anext(stream)
    assert not model._pd.active and not model._requests
    assert 42 not in pd._requests


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
    release_native = Mock()
    p._native = SimpleNamespace(release=release_native)

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
    release_native.assert_called_once_with(321)
    assert not submitted[0][0].aborted and not p.active


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
@pytest.mark.parametrize("cancellations", [0, 1, 2])
async def test_gpu_registration_cancellation_drains_before_release(
    monkeypatch, cancellations
):
    from .. import gpu

    started, finish = threading.Event(), threading.Event()
    released = []
    abort = Mock()
    monkeypatch.setattr(gpu, "service", lambda: SimpleNamespace(rpc=abort))

    def rpc(method, *args, **kwargs):
        if method == "prepare":
            started.set()
            assert finish.wait(timeout=3)
        else:
            assert method == "release"
            released.append(args)

    pd = TRTLLMPD("prefill", "nixl")
    pd.client = SimpleNamespace(rpc=rpc, namespace="namespace", config={"rank": 1})
    task = asyncio.create_task(
        pd.prepare("r", list(range(70)), {"do_remote_decode": True})
    )
    assert await asyncio.to_thread(started.wait, 3)
    room = pd.active["r"]["context"].room
    cleanup = None
    if cancellations:
        for _ in range(cancellations):
            task.cancel()
            await asyncio.sleep(0)
    else:
        cleanup = asyncio.create_task(pd.release("r"))
        await asyncio.sleep(0)
        assert not cleanup.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    if cleanup is not None:
        await cleanup
    assert released == [(room, None)]
    abort.assert_called_once_with("abort", room)
    assert not pd.active


def test_xavier_enqueue_failure_cleans_callback_state(monkeypatch):
    from ..pd import XavierRequest

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.inputs",
        SimpleNamespace(get_cache_salt_id=lambda salt: 123),
    )
    state = XavierRequest("decode", (1, 2), 1)

    def enqueue(*args, **kwargs):
        assert request_state(SimpleNamespace(cache_salt_id=123)) is state
        raise RuntimeError("enqueue failed")

    engine = SimpleNamespace(generate_async=enqueue)
    with pytest.raises(RuntimeError, match="enqueue failed"):
        submit_xavier(engine, [1, 2], None, state)
    assert request_state(SimpleNamespace(cache_salt_id=123)) is None


def test_xavier_callback_uses_native_salt_and_cleans_registration(monkeypatch):
    from ..pd import XavierRequest

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.inputs",
        SimpleNamespace(get_cache_salt_id=lambda salt: 456),
    )
    state = XavierRequest("prefill", (1, 2), 1)
    handle = SimpleNamespace(request_id=11)
    engine = SimpleNamespace(generate_async=Mock(return_value=handle))
    submit_xavier(engine, [1, 2], None, state)
    callback = SimpleNamespace(request_id=101, cache_salt_id=456)
    assert request_state(callback) is state
    forget_request(handle)
    assert request_state(callback) is None


@pytest.mark.parametrize("prompt_tokens", [1, 64, 65, 70])
def test_gpu_connector_pins_source_until_worker_completion(
    monkeypatch, connector, prompt_tokens
):
    from .. import gpu
    from ..pd import XavierRequest

    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 1, 1, 1, 64, "bfloat16"
    )
    client = SimpleNamespace(
        pd_gpu=True,
        contract=contract,
        read_prefix=Mock(side_effect=AssertionError("P/D must not read CPU pages")),
    )
    monkeypatch.setattr(connector, "XavierClient", lambda: client)
    finished = set()
    calls = []

    def rpc(method, *args):
        calls.append((method, args))
        if method == "poll_direct_gpu_v1":
            result = set(finished)
            finished.clear()
            return result

    def submit(method, *args):
        future = Future()
        future.set_result(rpc(method, *args))
        return future

    monkeypatch.setattr(gpu, "service", lambda: SimpleNamespace(rpc=rpc, submit=submit))
    ps, ds = connector.XavierConnectorScheduler(
        None
    ), connector.XavierConnectorScheduler(None)
    pw, dw = connector.XavierConnectorWorker(None), connector.XavierConnectorWorker(
        None
    )
    pw.pool = dw.pool = torch.zeros((3, 1, 2, 64), dtype=torch.bfloat16)
    tokens = tuple(range(prompt_tokens))
    source = XavierRequest("prefill", tokens, 1)
    monkeypatch.setattr(connector, "request_state", lambda request: source)
    request = SimpleNamespace(request_id=101, get_tokens=lambda beam: tokens)
    assert ps.get_num_new_matched_tokens(request, 0) == (0, False)
    ps.update_state_after_alloc(request, [0, 1])

    def output(rid):
        return SimpleNamespace(
            new_requests=[
                SimpleNamespace(
                    request_id=rid,
                    new_block_ids=[],
                    computed_position=0,
                    num_scheduled_tokens=prompt_tokens,
                )
            ]
        )

    pw.bind_connector_meta(ps.build_connector_meta(output(101)))
    pw.wait_for_save(SimpleNamespace(synchronize=lambda: None))
    assert calls == [("publish", (1, 101, [0, 1][: source.num_pages]))]
    # The Python result may already be consumed and its callback registration
    # removed. The scheduler must still retain the C++ request's slots.
    monkeypatch.setattr(connector, "request_state", lambda request: None)
    assert ps.request_finished(request, [0, 1]) is bool(source.num_pages)
    assert pw.get_finished([], []) == ([], [])
    if source.num_pages:
        assert pw.get_finished([101], []) == ([], [])
        finished.update({101, 102})
        assert pw.get_finished([], []) == ([101], [])
        # Completion arriving before this ID is admitted stays pending locally.
        assert pw.get_finished([102], []) == ([102], [])

    target = XavierRequest("decode", tokens, 1)
    monkeypatch.setattr(connector, "request_state", lambda request: target)
    request.request_id = 201
    assert ds.get_num_new_matched_tokens(request, 0) == (
        max(prompt_tokens - 1, 0),
        prompt_tokens > 1,
    )
    ds.update_state_after_alloc(request, [2, 0])
    dw.bind_connector_meta(ds.build_connector_meta(output(201)))
    pd = TRTLLMPD("decode", "nixl")
    if prompt_tokens > 1:
        with pytest.raises(RuntimeError, match="did not import"):
            pd.check_import({"context": target})
    dw.start_load_kv(None)
    if prompt_tokens > 1:
        assert dw.get_finished([], []) == ([], [])
        assert target.imported_tokens == 0
        assert dw.get_finished([], [201]) == ([], [201])
    pd.check_import({"context": target})
    assert calls[-1] == ("load", (1, [2, 0][: target.num_pages]))
    client.read_prefix.assert_not_called()


def test_failed_gpu_load_does_not_report_an_import(monkeypatch, connector):
    from .. import gpu
    from ..pd import XavierRequest

    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 1, 1, 1, 64, "bfloat16"
    )
    monkeypatch.setattr(
        connector,
        "XavierClient",
        lambda: SimpleNamespace(pd_gpu=True, contract=contract),
    )
    future = Future()
    future.set_exception(RuntimeError("NIXL transfer failed"))
    monkeypatch.setattr(
        gpu, "service", lambda: SimpleNamespace(submit=lambda *args: future)
    )
    worker = connector.XavierConnectorWorker(None)
    worker.pool = torch.zeros((2, 1, 2, 64), dtype=torch.bfloat16)
    context = XavierRequest("decode", tuple(range(70)), 1, engine_request_id=7)
    worker.bind_connector_meta(connector.XavierMetadata(pd_loads=[(context, [0, 1])]))
    worker.start_load_kv(None)
    with pytest.raises(RuntimeError, match="NIXL transfer failed"):
        worker.get_finished([], [7])
    assert context.imported_tokens == 0


@pytest.mark.asyncio
async def test_source_abort_drains_reads_and_rejects_late_publication():
    from unittest.mock import AsyncMock

    from ..gpu import TRTGPUActor

    directory = SimpleNamespace(
        request_info=AsyncMock(return_value={"prompt_tokens": 70})
    )
    actor = TRTGPUActor(None, 0, None, directory, 1)
    state = SimpleNamespace(reading=True, released=False)
    active = {"1": state}

    def release(ticket):
        state.released = True
        if not state.reading:
            active.pop(ticket, None)

    actor._gpu_transfer = SimpleNamespace(
        direct_requests=active, release_direct=release
    )
    actor.sources[1] = "1"
    abort = asyncio.create_task(actor.abort(1))
    await asyncio.sleep(0.01)
    assert state.released and not abort.done()
    state.reading = False
    release("1")
    await abort
    assert not actor.sources
    with pytest.raises(ValueError, match="Invalid TRT Xavier source"):
        await actor.publish(1, 101, [0, 1])


@pytest.mark.asyncio
async def test_concurrent_gpu_drain_keeps_ownership_after_repeated_cancellation():
    from unittest.mock import AsyncMock

    from ..gpu import TRTGPUActor

    actor = TRTGPUActor(None, 0, None, None, 1)
    started, complete = asyncio.Event(), asyncio.Event()

    async def close():
        started.set()
        await complete.wait()

    actor._gpu_transfer = SimpleNamespace(
        direct_requests={},
        tasks=set(),
        poll_direct=lambda: set(),
        close=AsyncMock(side_effect=close),
    )
    first = asyncio.create_task(actor.drain())
    await started.wait()
    second = asyncio.create_task(actor.drain())
    for _ in range(2):
        first.cancel()
        await asyncio.sleep(0)
        assert not first.done() and not second.done()
    complete.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    actor._gpu_transfer.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_prefill_cancel_before_decode_drops_directory_room(monkeypatch):
    from ...sglang.xavier.directory import XavierPDDirectory
    from .. import gpu

    directory = XavierPDDirectory()
    directory.configure("namespace")
    monkeypatch.setattr(gpu, "service", lambda: SimpleNamespace(rpc=Mock()))
    pd = TRTLLMPD("prefill", "nixl")
    pd.client = SimpleNamespace(
        namespace="namespace",
        config={"rank": 1},
        rpc=lambda method, *args, **kwargs: getattr(directory, method)(*args, **kwargs),
    )
    await pd.prepare("cancelled", [1, 2, 3], {"do_remote_decode": True})
    assert directory.get_stats()["active_handoffs"] == 1
    await pd.release("cancelled")
    assert directory.get_stats()["active_handoffs"] == 0


def test_shutdown_retains_cpp_owner_if_importer_cannot_stop(model, monkeypatch):
    from .. import gpu

    owner = model._engine
    model._cache_enabled = True
    with monkeypatch.context() as patch:
        patch.setattr(gpu, "close_service", Mock(side_effect=RuntimeError("live DMA")))
        with pytest.raises(RuntimeError, match="live DMA"):
            model.stop()
        assert model._engine is owner
        assert not getattr(owner, "stopped", False)
    model.stop()
    assert owner.stopped


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.asyncio
async def test_async_decode_cancel_drains_destination_before_return(
    monkeypatch, connector, fail
):
    from .. import gpu
    from ..pd import XavierRequest

    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 1, 1, 1, 64, "bfloat16"
    )
    monkeypatch.setattr(
        connector,
        "XavierClient",
        lambda: SimpleNamespace(pd_gpu=True, contract=contract),
    )
    future = Future()
    monkeypatch.setattr(
        gpu, "service", lambda: SimpleNamespace(submit=lambda *args: future)
    )
    scheduler = connector.XavierConnectorScheduler(None)
    worker = connector.XavierConnectorWorker(None)
    worker.pool = torch.zeros((2, 1, 2, 64), dtype=torch.bfloat16)
    state = XavierRequest("decode", tuple(range(70)), 1)
    monkeypatch.setattr(connector, "request_state", lambda request: state)
    request = SimpleNamespace(request_id=7, get_tokens=lambda beam: state.tokens)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (69, True)
    scheduler.update_state_after_alloc(request, [1, 0])
    # Suspended requests are omitted by TRT's scheduler output.
    metadata = scheduler.build_connector_meta(SimpleNamespace(new_requests=[]))
    assert metadata.pd_loads == [(state, [1, 0])]
    worker.bind_connector_meta(metadata)
    worker.start_load_kv(None)
    assert worker.get_finished([], [7]) == ([], [])
    assert not future.done() and state.imported_tokens == 0
    cancelled = asyncio.create_task(
        asyncio.to_thread(scheduler.request_finished, request, [1, 0])
    )
    await asyncio.sleep(0.02)
    assert state.cancelled and not cancelled.done() and not future.cancelled()
    if fail:
        future.set_exception(RuntimeError("transfer failed after partial writes"))
    else:
        future.set_result(None)
    assert await cancelled is False
    # The manager retires the suspended load; late completion never resumes it.
    assert worker.get_finished([], []) == ([], [])
    assert state.imported_tokens == 0
    assert not worker._loading and not worker._waiting_loads
    assert not scheduler._pd_destinations and not scheduler._pd_requests


def test_async_decode_completions_are_admitted_and_can_finish_out_of_order(
    monkeypatch, connector
):
    from .. import gpu
    from ..pd import XavierRequest

    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 1, 1, 1, 64, "bfloat16"
    )
    monkeypatch.setattr(
        connector,
        "XavierClient",
        lambda: SimpleNamespace(pd_gpu=True, contract=contract),
    )
    futures = {rid: Future() for rid in (1, 2, 3)}
    monkeypatch.setattr(
        gpu,
        "service",
        lambda: SimpleNamespace(submit=lambda method, room, blocks: futures[room]),
    )
    worker = connector.XavierConnectorWorker(None)
    worker.pool = torch.zeros((3, 1, 2, 64), dtype=torch.bfloat16)
    states = [
        XavierRequest("decode", (1, 2), rid, engine_request_id=rid) for rid in futures
    ]
    worker.bind_connector_meta(
        connector.XavierMetadata(
            pd_loads=[(state, [idx]) for idx, state in enumerate(states)]
        )
    )
    worker.start_load_kv(None)
    futures[2].set_result(None)
    assert worker.get_finished([], []) == ([], [])
    assert worker.get_finished([], [1, 2, 3]) == ([], [2])
    failure = Mock()
    worker.load_failed = failure
    error = RuntimeError("source lost")
    futures[3].set_exception(error)
    assert worker.get_finished([], []) == ([], [])
    failure.assert_called_once_with(3, error)
    assert states[2].imported_tokens == 0
    futures[1].set_result(None)
    assert worker.get_finished([], []) == ([], [1])
    assert not worker._loading and not worker._waiting_loads


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_gpu_service_shutdown_drains_load_before_closing_ipc(monkeypatch, fail):
    from unittest.mock import AsyncMock

    from .. import gpu

    future = Future()
    directory = SimpleNamespace(unregister_peer=AsyncMock())
    transport = gpu.GPUService.__new__(gpu.GPUService)
    transport.client = SimpleNamespace(
        call=asyncio.run,
        directory=AsyncMock(return_value=directory),
        config={"rank": 1},
    )
    transport.actor = SimpleNamespace(address="nixl://127.0.0.1:1")
    transport.pool = SimpleNamespace(stop=AsyncMock())
    transport._loads = {future}
    transport._closing = False
    owner = transport.cache = object()
    destroy = AsyncMock()
    monkeypatch.setattr(gpu.xo, "destroy_actor", destroy)
    close = asyncio.create_task(asyncio.to_thread(transport.close))
    await asyncio.sleep(0.02)
    assert transport._closing and not close.done()
    assert transport.cache is owner
    destroy.assert_not_called()
    with pytest.raises(RuntimeError, match="shutting down"):
        transport.submit("load", 1, [0])
    if fail:
        future.set_exception(RuntimeError("writes drained after error"))
    else:
        future.set_result(None)
    await close
    assert transport.cache is None
    destroy.assert_awaited_once_with(transport.actor)
    transport.pool.stop.assert_awaited_once()


@pytest.mark.parametrize("cancel_during_rpc", [False, True])
def test_cancelled_source_publication_does_not_fail_other_sources(
    monkeypatch, connector, cancel_during_rpc
):
    from .. import gpu
    from ..pd import XavierRequest

    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 1, 1, 1, 64, "bfloat16"
    )
    monkeypatch.setattr(
        connector,
        "XavierClient",
        lambda: SimpleNamespace(pd_gpu=True, contract=contract),
    )
    cancelled = XavierRequest("prefill", (1, 2), 1, engine_request_id=11)
    other = XavierRequest("prefill", (3, 4), 2, engine_request_id=12)
    cancelled.cancelled = not cancel_during_rpc
    published = []

    def rpc(method, room, request_id, blocks):
        assert method == "publish"
        if room == 1:
            cancelled.cancelled = True
            raise ValueError("P/D room already released")
        published.append(room)

    monkeypatch.setattr(gpu, "service", lambda: SimpleNamespace(rpc=rpc))
    worker = connector.XavierConnectorWorker(None)
    worker.pool = torch.zeros((2, 1, 2, 64), dtype=torch.bfloat16)
    worker.bind_connector_meta(
        connector.XavierMetadata(pd_saves=[(cancelled, [0]), (other, [1])])
    )
    worker.wait_for_save(SimpleNamespace(synchronize=lambda: None))
    assert published == [2]
    assert not cancelled.registered and other.registered
    # A genuine publication failure on a live request must still surface.
    other.cancelled = False
    monkeypatch.setattr(
        gpu,
        "service",
        lambda: SimpleNamespace(rpc=Mock(side_effect=RuntimeError("NIXL failed"))),
    )
    worker.bind_connector_meta(connector.XavierMetadata(pd_saves=[(other, [1])]))
    with pytest.raises(RuntimeError, match="NIXL failed"):
        worker.wait_for_save(SimpleNamespace(synchronize=lambda: None))


@pytest.mark.asyncio
async def test_zero_page_cleanup_expires_tombstones_without_poll(monkeypatch):
    from unittest.mock import AsyncMock

    from .. import gpu

    now = [0.0]
    monkeypatch.setattr(gpu.time, "monotonic", lambda: now[0])
    actor = gpu.TRTGPUActor(
        None,
        0,
        None,
        SimpleNamespace(request_info=AsyncMock(return_value={"prompt_tokens": 1})),
        1,
    )
    for room in range(100):
        now[0] = room * 301.0
        await actor.abort(room)
        assert actor.aborted == {room: now[0] + 300}
    # The current, unexpired cancellation still rejects a late zero-page publish.
    with pytest.raises(ValueError, match="Invalid TRT Xavier source"):
        await actor.publish(99, 1, [])


@pytest.mark.asyncio
async def test_native_completed_source_releases_live_transmission():
    release = Mock()
    request = SimpleNamespace(
        request_id=7,
        abort=Mock(),
        disaggregated_params=SimpleNamespace(ctx_request_id=123),
    )
    pd = TRTLLMPD("prefill", "nixl")
    pd._native = SimpleNamespace(release=release)
    pd.active["r"] = dict(request=request, ticket=123)
    await pd.release("r")
    release.assert_called_once_with(123)
    request.abort.assert_not_called()
