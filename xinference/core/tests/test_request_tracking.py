# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
import gc
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from .test_request_priority import _actor, _close, _wait_for


def _streaming_actor(method, native, sync=False):
    actor, model = _actor(capacity=2)
    model.supports_native_request_priority = native
    actor._lock = None  # Async engines, like vLLM, permit concurrent streams.
    closed = asyncio.Event()
    entered = asyncio.Event()

    async def chunks():
        try:
            yield {"choices": []}
            entered.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    def sync_chunks():
        try:
            yield {"choices": []}
            yield {"choices": []}
        finally:
            closed.set()

    async def infer(*args, **kwargs):
        return chunks()

    setattr(
        model,
        method if sync else f"async_{method}",
        (lambda *args, **kwargs: sync_chunks()) if sync else infer,
    )
    if native:
        model.abort_request = AsyncMock(return_value="DONE")
    return actor, closed, entered


async def _start(actor, method):
    # The entry task finishes, exactly as an actor RPC does before next().
    stream = await asyncio.create_task(
        getattr(actor, method)("hello", {"stream": True}, request_id="r")
    )
    stream._actor_ref = actor
    stream._gc_destroy = False
    await asyncio.sleep(0)
    gc.collect()
    return stream


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "chat"])
@pytest.mark.parametrize("native,sync", [(False, False), (False, True), (True, False)])
async def test_stream_rejects_duplicate_until_closed(method, native, sync):
    actor, _, _ = _streaming_actor(method, native, sync)
    stream = await _start(actor, method)
    duplicate = None
    try:
        with pytest.raises(Exception, match="Duplicate request id"):
            duplicate = await _start(actor, method)
    finally:
        await _close(stream)
        if duplicate is not None:
            await _close(duplicate)
    assert "r" not in actor._running_tasks
    replacement = await _start(actor, method)
    await _close(replacement)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "chat"])
@pytest.mark.parametrize("native,sync", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("reading", [False, True])
@pytest.mark.parametrize("block_duration", [0, 30])
async def test_abort_tracks_stream_after_entry_task_finishes(
    method, native, sync, reading, block_duration
):
    actor, closed, entered = _streaming_actor(method, native, sync)
    stream = await _start(actor, method)
    pending = None
    block = None
    try:
        await stream.__anext__()
        if reading and not sync:
            pending = asyncio.create_task(stream.__anext__())
            await entered.wait()
        assert await actor.abort_request("r", block_duration=block_duration) == "DONE"
        block = actor._running_tasks.get("r")
        await _wait_for(closed.is_set)
        if pending is not None:
            with pytest.raises(StopAsyncIteration):
                await pending
        else:
            with pytest.raises((asyncio.CancelledError, StopAsyncIteration)):
                await stream.__anext__()
        if block_duration:
            with pytest.raises(Exception, match="request has been aborted"):
                await _start(actor, method)
        else:
            replacement = await _start(actor, method)
            await _close(replacement)
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await _close(stream)
        if block is not None:
            block.cancel()
            await asyncio.gather(block, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_aborts", [False, True])
async def test_abort_reserves_id_until_native_cancellation_and_cleanup_finish(
    monkeypatch, cleanup_aborts
):
    from ...model.llm.sglang.core import SGLANGModel

    actor, model = _actor(capacity=2)
    model.supports_native_request_priority = True
    actor._lock = None
    engine = object.__new__(SGLANGModel)
    engine._engine = SimpleNamespace(url="http://unused")
    engine._model_config = {}
    engine._active_request_ids = set()
    native_entered, native_release, reading, cleanup = [
        asyncio.Event() for _ in range(4)
    ]
    native_finished, cleanup_release = asyncio.Event(), asyncio.Event()
    calls = []

    @asynccontextmanager
    async def post(url, **kwargs):
        calls.append((url, kwargs["json"]))
        native_entered.set()
        if len(calls) == 1:
            await native_release.wait()
        yield SimpleNamespace(raise_for_status=lambda: None)
        native_finished.set()

    @asynccontextmanager
    async def session(**kwargs):
        yield SimpleNamespace(post=post)

    monkeypatch.setattr("aiohttp.ClientSession", session)
    model.abort_request = engine.abort_request

    async def chunks():
        try:
            yield {"choices": []}
            reading.set()
            await asyncio.Event().wait()
        finally:
            cleanup.set()
            if cleanup_aborts:
                # SGLang pd_stream aborts interrupted output before dropping its ID.
                await engine.abort_request("r")
                await cleanup_release.wait()
            engine._active_request_ids.discard("r")
            engine.__dict__.get("_abort_request_futures", {}).pop("r", None)

    async def infer(*args, **kwargs):
        engine._active_request_ids.add(kwargs["request_id"])
        return chunks()

    model.async_generate = infer
    stream = await _start(actor, "generate")
    pending = aborting = duplicate = replacement = None
    try:
        await anext(stream)
        pending = asyncio.create_task(anext(stream))
        await reading.wait()
        aborting = asyncio.create_task(actor.abort_request("r", block_duration=0))
        await native_entered.wait()
        await cleanup.wait()
        if not cleanup_aborts:
            await asyncio.gather(pending, return_exceptions=True)
        assert calls == [("http://unused/abort_request", {"rid": "r"})]
        assert not aborting.done()
        with pytest.raises(Exception, match="Duplicate request id"):
            duplicate = await _start(actor, "generate")
        native_release.set()
        if cleanup_aborts:
            await native_finished.wait()
            assert not aborting.done()
            with pytest.raises(Exception, match="Duplicate request id"):
                duplicate = await _start(actor, "generate")
            cleanup_release.set()
        assert await aborting == "DONE"
        await asyncio.gather(pending, return_exceptions=True)
        assert not engine.__dict__.get("_abort_request_futures")
        replacement = await _start(actor, "generate")
    finally:
        native_release.set()
        cleanup_release.set()
        await asyncio.gather(
            *(task for task in (pending, aborting) if task is not None),
            return_exceptions=True,
        )
        for owned in (stream, duplicate, replacement):
            if owned is not None:
                await _close(owned)


@pytest.mark.asyncio
async def test_sglang_joined_abort_survives_first_caller_cancellation(monkeypatch):
    from ...model.llm.sglang.core import SGLANGModel

    engine = object.__new__(SGLANGModel)
    engine._engine = SimpleNamespace(url="http://unused")
    engine._model_config = {}
    engine._active_request_ids = {"r"}
    entered = asyncio.Event()

    @asynccontextmanager
    async def post(url, **kwargs):
        entered.set()
        await asyncio.Event().wait()
        yield

    @asynccontextmanager
    async def session(**kwargs):
        yield SimpleNamespace(post=post)

    monkeypatch.setattr("aiohttp.ClientSession", session)
    first = asyncio.create_task(engine.abort_request("r"))
    await entered.wait()
    joined = asyncio.create_task(engine.abort_request("r"))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    # Stream cleanup joined the abort; it must not see the first caller's cancel.
    with pytest.raises(RuntimeError, match="interrupted"):
        await asyncio.wait_for(joined, 5)
    assert not engine._abort_request_futures


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", [False, True])
async def test_stream_uid_abort_only_cancels_its_original_request(replacement):
    actor, _, _ = _streaming_actor("generate", native=True)
    old = await _start(actor, "generate")
    current = None
    try:
        await anext(old)
        assert (
            await actor.abort_request("r", block_duration=0, stream_uid=old._uid)
            == "DONE"
        )
        if replacement:
            current = await _start(actor, "generate")
            await anext(current)
        assert await actor.abort_request("r", stream_uid=old._uid) == "NO_OP"
        actor._model.abort_request.assert_awaited_once_with("r")
        assert "r" not in actor._running_tasks or (
            actor._running_tasks["r"].get_name() != actor._CANCEL_TASK_NAME
        )
    finally:
        await _close(old)
        if current is not None:
            await _close(current)
        blocks = [
            t
            for t in actor._running_tasks.values()
            if t.get_name() == actor._CANCEL_TASK_NAME
        ]
        for task in blocks:
            task.cancel()
        await asyncio.gather(*blocks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "chat"])
@pytest.mark.parametrize(
    "ending", ["live_close", "complete", "close", "abort", "expired_block"]
)
async def test_vllm_stream_reuse_preserves_replacement_abort(
    method, ending, monkeypatch
):
    from ...model.llm.vllm.core import VLLMModel

    actor, model = _actor(capacity=2)
    model.supports_native_request_priority = True
    actor._lock = None
    tracker = object.__new__(VLLMModel)
    tracker._active_request_ids = set()
    finish, cleanup, cleanup_release, native, native_release = [
        asyncio.Event() for _ in range(5)
    ]

    async def abort(request_id):
        native.set()
        await native_release.wait()

    tracker._engine = SimpleNamespace(abort=AsyncMock(side_effect=abort))
    model.abort_request = tracker.abort_request
    expire_block = asyncio.Event()
    if ending == "expired_block":
        sleep = asyncio.sleep

        async def sleep_after_release(delay, result=None):
            task = asyncio.current_task()
            if (
                delay == 0.05
                and task is not None
                and task.get_name() == actor._CANCEL_TASK_NAME
            ):
                await expire_block.wait()
            return await sleep(delay, result)

        # Keep the block live until its rejection has been checked.
        monkeypatch.setattr(asyncio, "sleep", sleep_after_release)

    async def results():
        try:
            yield {"choices": []}
            await finish.wait()
        finally:
            cleanup.set()
            await cleanup_release.wait()

    async def infer(*args, **kwargs):
        return tracker._track_engine_request(kwargs["request_id"], results())

    setattr(model, f"async_{method}", infer)
    stream = await _start(actor, method)
    entry_task = actor._running_tasks["r"]
    assert entry_task.done()
    settling = duplicate = replacement = block = None
    try:
        await anext(stream)
        if ending == "live_close":
            with pytest.raises(Exception, match="Duplicate request id"):
                duplicate = await _start(actor, method)
        if ending == "complete":
            finish.set()
            settling = asyncio.create_task(anext(stream, None))
        elif ending in ("live_close", "close"):
            settling = asyncio.create_task(_close(stream))
        else:
            settling = asyncio.create_task(
                actor.abort_request(
                    "r", block_duration=0.05 if ending == "expired_block" else 0
                )
            )
        if ending != "complete":
            await asyncio.wait_for(native.wait(), 3)
            if ending == "expired_block":
                block = actor._running_tasks["r"]
                with pytest.raises(Exception, match="The request has been aborted"):
                    duplicate = await _start(actor, method)
                expire_block.set()
                await asyncio.wait_for(block, 3)
                assert actor._running_tasks["r"] is block and block.done()
            with pytest.raises(
                Exception, match="Duplicate request id|The request has been aborted"
            ):
                duplicate = await _start(actor, method)
            native_release.set()
        await asyncio.wait_for(cleanup.wait(), 3)
        assert not settling.done()
        with pytest.raises(
            Exception, match="Duplicate request id|The request has been aborted"
        ):
            duplicate = await _start(actor, method)
        cleanup_release.set()
        result = await settling
        if ending in ("abort", "expired_block"):
            assert result == "DONE"
        assert not tracker._active_request_ids and "r" not in actor._request_streams
        native_release.set()
        replacement = await _start(actor, method)
        await anext(replacement)
        assert tracker._active_request_ids == {"r"}
        # Retaining and closing the old stream must not remove the new native ID.
        await _close(stream)
        before = tracker._engine.abort.await_count
        assert await actor.abort_request("r", block_duration=0) == "DONE"
        assert tracker._engine.abort.await_count == before + 1
        tracker._engine.abort.assert_awaited_with("r")
        with pytest.raises((asyncio.CancelledError, StopAsyncIteration)):
            await anext(replacement)
        assert not tracker._active_request_ids and "r" not in actor._request_streams
    finally:
        expire_block.set()
        finish.set()
        native_release.set()
        cleanup_release.set()
        if settling is not None:
            await asyncio.gather(settling, return_exceptions=True)
        for owned in (stream, duplicate, replacement):
            if owned is not None:
                await _close(owned)
        if block is not None:
            block.cancel()
            await asyncio.gather(block, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [False, True])
async def test_realtime_file_stream_reserves_id_until_lease_drains(sync):
    actor, _ = _actor()
    actor._realtime_runtime = SimpleNamespace(active_count=0)
    entered, native_release, cleanup, cleanup_release = [
        threading.Event() for _ in range(4)
    ]

    def sync_chunks(**kwargs):
        try:
            yield {"choices": []}
            entered.set()
            assert native_release.wait(5)
        finally:
            cleanup.set()
            assert cleanup_release.wait(5)

    async def chunks(**kwargs):
        try:
            yield {"choices": []}
            entered.set()
            assert await asyncio.to_thread(native_release.wait, 5)
        finally:
            cleanup.set()
            assert await asyncio.to_thread(cleanup_release.wait, 5)

    async def infer(**kwargs):
        return chunks(**kwargs)

    async def request():
        return await actor._call_wrapper_json(
            sync_chunks if sync else infer, request_id="r"
        )

    handler = asyncio.create_task(actor._handle_pending_requests())
    setup = asyncio.create_task(request())
    stream = await setup
    replacement = None
    try:
        await _wait_for(entered.is_set)
        assert setup.done() and not actor._request_streams
        lease = next(iter(actor._realtime_file_streams))
        assert actor._running_tasks["r"] is setup
        with pytest.raises(Exception, match="Duplicate request id"):
            await asyncio.wait_for(request(), 1)
        assert await actor.abort_request("r", block_duration=0) == "DONE"
        assert "r" not in actor._running_tasks
        with pytest.raises(Exception, match="Duplicate request id"):
            await asyncio.wait_for(request(), 1)
        native_release.set()
        await _wait_for(cleanup.is_set)
        assert not lease.done.done()
        with pytest.raises(Exception, match="Duplicate request id"):
            await asyncio.wait_for(request(), 1)
        cleanup_release.set()
        await asyncio.wait_for(lease.done, 3)
        assert not actor._realtime_file_streams
        replacement = await asyncio.create_task(request())
        await _wait_for(lambda: not actor._realtime_file_streams)
    finally:
        native_release.set()
        cleanup_release.set()
        await _wait_for(lambda: not actor._realtime_file_streams)
        for owned in (stream, replacement):
            if owned is not None:
                if sync:
                    await asyncio.to_thread(owned.close)
                else:
                    await owned.aclose()
        handler.cancel()
        await asyncio.gather(handler, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "chat"])
@pytest.mark.parametrize("native", [False, True])
async def test_completed_non_stream_call_reuses_id(method, native):
    actor, model = _actor()
    model.supports_native_request_priority = native

    async def infer(*args, **kwargs):
        return {"choices": []}

    setattr(model, f"async_{method}", infer)
    first = asyncio.create_task(getattr(actor, method)("hello", {}, request_id="r"))
    assert await first
    assert first.done() and actor._running_tasks["r"] is first
    assert not actor._request_streams
    replacement = asyncio.create_task(
        getattr(actor, method)("hello", {}, request_id="r")
    )
    assert await replacement
    assert actor._running_tasks["r"] is replacement
