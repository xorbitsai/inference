# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
import gc
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
async def test_abort_tracks_stream_after_entry_task_finishes(
    method, native, sync, reading
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
        assert await actor.abort_request("r", block_duration=30) == "DONE"
        block = actor._running_tasks.get("r")
        await _wait_for(closed.is_set)
        if pending is not None:
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            with pytest.raises((asyncio.CancelledError, StopAsyncIteration)):
                await stream.__anext__()
        with pytest.raises(Exception, match="request has been aborted"):
            await _start(actor, method)
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await _close(stream)
        if block is not None:
            block.cancel()
            await asyncio.gather(block, return_exceptions=True)
