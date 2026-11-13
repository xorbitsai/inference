# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
import gc
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
