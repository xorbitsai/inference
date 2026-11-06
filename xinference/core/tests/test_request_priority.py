# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ...model.llm.core import LLM
from ...request_priority import PriorityScheduler
from ..model import ModelActor


class _Model(LLM):
    def load(self):
        pass

    @classmethod
    def check_lib(cls):
        return True

    @classmethod
    def match_json(cls, *args):
        return True

    def get_request_priority_capacity(self):
        return self.capacity


def _actor(capacity=1, request_limits=None):
    model = object.__new__(_Model)
    model.capacity = capacity
    model.model_uid = "test-model"
    model.model_family = SimpleNamespace(to_description=lambda: {"model_type": "LLM"})
    actor = ModelActor("test:123", "test:456", model, "test-model-0", request_limits)
    actor._model_state = "ready"
    actor.record_metrics = AsyncMock()
    return actor, model


async def _wait_for(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), 3)


async def _close(stream):
    stream._gc_destroy = False
    await stream.destroy()


@pytest.mark.asyncio
async def test_cancel_after_slot_reservation_does_not_leak_capacity():
    scheduler = PriorityScheduler(1)
    await scheduler.acquire(0)
    waiter = asyncio.create_task(scheduler.acquire(-1))
    await asyncio.sleep(0)
    scheduler.release()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await asyncio.wait_for(scheduler.acquire(0), 1)
    scheduler.release()


@pytest.mark.asyncio
async def test_stopping_fails_queued_requests():
    scheduler = PriorityScheduler(1)
    await scheduler.acquire(0)
    waiters = [asyncio.create_task(scheduler.acquire(p)) for p in (-1, 0, 1)]
    await asyncio.sleep(0)
    scheduler.close()
    for waiter in waiters:
        with pytest.raises(RuntimeError, match="stopping"):
            await waiter
    scheduler.release()
    with pytest.raises(RuntimeError, match="stopping"):
        await scheduler.acquire(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "chat"])
@pytest.mark.parametrize("capacity", [1, 2, 4])
async def test_waiting_requests_run_by_priority_then_arrival(method, capacity):
    actor, model = _actor(capacity)
    started = []
    unblock = asyncio.Event()

    async def infer(prompt, config):
        assert "priority" not in config
        started.append(config["request_id"])
        if config["request_id"].startswith("busy"):
            await unblock.wait()
        return {"choices": []}

    setattr(model, method, infer)
    handler = getattr(actor, method)
    active = [
        asyncio.create_task(handler("prompt", {"request_id": f"busy-{i}"}))
        for i in range(capacity)
    ]
    await _wait_for(lambda: len(started) == capacity)
    waiting = [
        asyncio.create_task(
            handler("prompt", {"request_id": name, "priority": priority})
        )
        for name, priority in [("low", 5), ("high", -5), ("tie", -5), ("default", 0)]
    ]
    await _wait_for(lambda: len(actor._request_priority_scheduler._waiting) == 4)
    unblock.set()
    await asyncio.gather(*active, *waiting)
    assert started[capacity:] == ["high", "tie", "default", "low"]
    assert actor._serve_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_result", ["DONE", "NO_OP", "NOT_FOUND"])
async def test_queued_abort_never_reaches_engine(backend_result):
    actor, model = _actor()
    started = []
    unblock = asyncio.Event()
    model.abort_request = AsyncMock(return_value=backend_result)

    async def infer(prompt, config):
        started.append(config["request_id"])
        if config["request_id"] == "busy":
            await unblock.wait()
        return {"choices": []}

    model.generate = infer
    busy = asyncio.create_task(actor.generate("hello", {"request_id": "busy"}))
    await _wait_for(lambda: started == ["busy"])
    cancelled = asyncio.create_task(
        actor.generate("hello", {"request_id": "cancel", "priority": -5})
    )
    next_request = asyncio.create_task(actor.generate("hello", {"request_id": "next"}))
    await _wait_for(lambda: len(actor._request_priority_scheduler._waiting) == 2)
    assert await actor.abort_request("cancel", block_duration=0) == "DONE"
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    unblock.set()
    await asyncio.gather(busy, next_request)
    assert started == ["busy", "next"]
    assert actor._serve_count == 0


@pytest.mark.asyncio
async def test_abort_acknowledges_actor_task_after_engine_cleanup():
    actor, model = _actor()
    started, cleaned = asyncio.Event(), asyncio.Event()

    async def infer(prompt, config):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    async def abort(request_id):
        await asyncio.wait_for(cleaned.wait(), 3)
        return "NOT_FOUND"

    model.generate = infer
    model.abort_request = AsyncMock(side_effect=abort)
    task = asyncio.create_task(actor.generate("hello", {"request_id": "active"}))
    try:
        await asyncio.wait_for(started.wait(), 3)
        assert await actor.abort_request("active", block_duration=0) == "DONE"
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await actor.abort_request("unknown", block_duration=0) == "NOT_FOUND"
        assert actor._serve_count == 0
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("consume_first", [False, True])
@pytest.mark.parametrize("sync", [False, True])
async def test_stream_owns_slot_until_close_and_releases_unread_streams(
    consume_first, sync
):
    actor, model = _actor()
    closed = threading.Event()
    started = []

    async def async_chunks():
        try:
            yield {"choices": []}
            await asyncio.Event().wait()
        finally:
            closed.set()

    def sync_chunks():
        try:
            yield {"choices": []}
            yield {"choices": []}
        finally:
            closed.set()

    async def infer(prompt, config):
        started.append(prompt)
        return (
            (sync_chunks() if sync else async_chunks())
            if prompt == "stream"
            else {"choices": []}
        )

    model.generate = infer
    model.abort_request = AsyncMock(return_value="DONE")
    stream = await actor.generate("stream", {"stream": True})
    stream._actor_ref = actor
    if consume_first:
        await stream.__anext__()
    waiting = asyncio.create_task(actor.generate("next", {"priority": -5}))
    await _wait_for(lambda: bool(actor._request_priority_scheduler._waiting))
    assert started == ["stream"]
    await _close(stream)
    await waiting
    assert started == ["stream", "next"]
    if consume_first:
        assert closed.is_set()
    model.abort_request.assert_awaited_once()
    await actor.decrease_serve_count()
    assert actor._serve_count == 0


@pytest.mark.asyncio
async def test_cancelled_sync_call_retains_slot_until_native_thread_finishes():
    actor, model = _actor()
    entered = threading.Event()
    release = threading.Event()
    started = []

    def infer(prompt, config):
        started.append(prompt)
        if prompt == "busy":
            entered.set()
            assert release.wait(3)
        return {"choices": []}

    model.generate = infer
    busy = asyncio.create_task(actor.generate("busy", {}))
    await _wait_for(entered.is_set)
    waiting = asyncio.create_task(actor.generate("next", {"priority": -1}))
    await _wait_for(lambda: bool(actor._request_priority_scheduler._waiting))
    busy.cancel()
    await asyncio.sleep(0)
    assert not busy.done()
    assert started == ["busy"]
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await busy
    await waiting
    assert started == ["busy", "next"]


@pytest.mark.asyncio
async def test_destroy_cancels_pending_stream_read_before_releasing_slot():
    actor, model = _actor()
    reading = asyncio.Event()
    closed = asyncio.Event()

    async def chunks():
        try:
            reading.set()
            await asyncio.Event().wait()
            yield {"choices": []}
        finally:
            await asyncio.sleep(0)
            closed.set()

    async def infer(prompt, config):
        if prompt == "stream":
            return chunks()
        assert closed.is_set()
        return {"choices": []}

    model.generate = infer
    stream = await actor.generate("stream", {"stream": True})
    stream._actor_ref = actor
    read = asyncio.create_task(stream.__anext__())
    await reading.wait()
    waiting = asyncio.create_task(actor.generate("next", {"priority": -1}))
    await _wait_for(lambda: bool(actor._request_priority_scheduler._waiting))
    await asyncio.wait_for(_close(stream), 3)
    with pytest.raises(asyncio.CancelledError):
        await read
    await waiting


@pytest.mark.asyncio
async def test_priority_queue_does_not_bypass_request_limits():
    actor, model = _actor(request_limits=1)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def infer(prompt, config):
        entered.set()
        await release.wait()
        return {"choices": []}

    model.generate = infer
    task = asyncio.create_task(actor.generate("busy", {}))
    await entered.wait()
    with pytest.raises(RuntimeError, match="Rate limit reached"):
        await actor.generate("next", {"priority": -10})
    release.set()
    await task


@pytest.mark.asyncio
async def test_cancelled_metrics_do_not_leak_registered_stream_slot():
    actor, model = _actor()
    blocked = asyncio.Event()

    async def chunks():
        yield {"choices": []}

    async def infer(prompt, config):
        return chunks() if prompt == "stream" else {"choices": []}

    async def metrics(name, *args):
        if name == "model_request_total":
            blocked.set()
            await asyncio.Event().wait()

    model.generate = infer
    actor.record_metrics = metrics
    task = asyncio.create_task(actor.generate("stream", {"stream": True}))
    await blocked.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not actor._generators
    assert actor._serve_count == 0
    actor.record_metrics = AsyncMock()
    await asyncio.wait_for(actor.generate("next", {}), 3)


@pytest.mark.asyncio
async def test_async_engine_and_queue_share_request_id_without_mutating_input():
    actor, model = _actor()
    observed = []

    async def infer(prompt, generate_config, request_id):
        observed.append((generate_config, request_id))
        return {"choices": []}

    model.async_generate = infer
    config = {"request_id": "public-id", "priority": -5}
    await actor.generate("hello", config)
    assert observed == [({"request_id": "public-id"}, "public-id")]
    assert "public-id" in actor._running_tasks
    assert config == {"request_id": "public-id", "priority": -5}


@pytest.mark.asyncio
@pytest.mark.parametrize("native,fallback", [(True, True), (False, False)])
async def test_engines_without_a_fallback_queue_keep_engine_metadata(native, fallback):
    actor, model = _actor()
    model.supports_native_request_priority = native
    model.supports_request_priority_fallback = fallback

    async def infer(prompt, config):
        assert config["priority"] == -5
        return {"choices": []}

    model.generate = infer
    await actor.generate("hello", {"priority": -5})
    assert "_request_priority_scheduler" not in actor.__dict__
