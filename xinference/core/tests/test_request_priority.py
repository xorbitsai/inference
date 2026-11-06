# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ...model.llm.core import LLM
from ...request_priority import PriorityScheduler
from ..model import ModelActor
from ..rpc_context import RPC_METADATA_KEY, RpcMetadata


class _Model(LLM):
    def load(self):
        pass

    @classmethod
    def check_lib(cls):
        return True

    @classmethod
    def match_json(cls, *args):
        return True

    def get_request_priority_capacity(self, generate_config=None):
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
async def test_cancelled_waiters_reclaim_storage_while_capacity_is_occupied():
    scheduler = PriorityScheduler(1)
    await scheduler.acquire(0)
    for _ in range(3):
        waiters = [asyncio.create_task(scheduler.acquire(i)) for i in range(100)]
        await _wait_for(lambda: len(scheduler._waiting) == 100)
        for waiter in waiters:
            waiter.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)
        assert scheduler._waiting == []
        assert scheduler._available == 0
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
@pytest.mark.parametrize("method", ["generate", "chat"])
@pytest.mark.parametrize("correlated", [False, True])
async def test_stream_owns_slot_until_close_and_releases_unread_streams(
    consume_first, sync, method, correlated
):
    actor, model = _actor()
    closed = threading.Event()
    started = []
    owned_streams = []
    priority_stream = actor._priority_stream

    def retain_stream(*args):
        result = priority_stream(*args)
        # Hold a reference so GC cannot make a broken wrapper appear to close
        # the owned stream successfully.
        owned_streams.append(result)
        return result

    actor._priority_stream = retain_stream

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

    setattr(model, method, infer)
    model.abort_request = AsyncMock(return_value="DONE")
    handler = getattr(actor, method)
    metadata = (
        {RPC_METADATA_KEY: RpcMetadata(correlation_id="rest-correlation").to_payload()}
        if correlated
        else {}
    )
    stream = await handler("stream", {"stream": True}, **metadata)
    stream._actor_ref = actor
    if consume_first:
        await stream.__anext__()
    waiting = asyncio.create_task(handler("next", {"priority": -5}))
    try:
        await _wait_for(lambda: bool(actor._request_priority_scheduler._waiting))
        assert started == ["stream"]
        await _close(stream)
        model.abort_request.assert_awaited_once()
        await waiting
        assert started == ["stream", "next"]
        if consume_first:
            assert closed.is_set()
        await actor.decrease_serve_count()
        assert actor._serve_count == 0
    finally:
        for owned in owned_streams:
            await owned.aclose()
        await waiting


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
async def test_cancelled_native_stream_creation_drains_off_loop_before_admission():
    from ...model.llm.llama_cpp.core import _owned_native_stream

    actor, model = _actor()
    creating = threading.Event()
    return_stream = threading.Event()
    closing = threading.Event()
    release_native = threading.Event()
    native = Future()
    started = []
    close_threads = []

    def finish_native():
        # Also unblock a regressed event-loop-thread close so the test fails
        # rather than deadlocking the entire test process.
        release_native.wait(3)
        native.set_result(None)

    worker = threading.Thread(target=finish_native)
    worker.start()

    def chunks():
        try:
            yield None
        finally:
            close_threads.append(threading.get_ident())
            closing.set()

    def infer(prompt, config):
        started.append(prompt)
        if prompt != "busy":
            return {"choices": []}
        creating.set()
        assert return_stream.wait(3)
        iterator = chunks()
        next(iterator)
        return _owned_native_stream(iterator, native)

    model.generate = infer
    busy = asyncio.create_task(actor.generate("busy", {"stream": True}))
    waiting = None
    try:
        await _wait_for(creating.is_set)
        busy.cancel()
        return_stream.set()
        # This await also proves generator cleanup did not block the event loop.
        await _wait_for(closing.is_set)
        waiting = asyncio.create_task(actor.generate("next", {}))
        await _wait_for(lambda: bool(actor._request_priority_scheduler._waiting))
        assert not busy.done()
        assert started == ["busy"]
        assert len(close_threads) == 1
        assert close_threads[0] != threading.get_ident()
        release_native.set()
        with pytest.raises(asyncio.CancelledError):
            await busy
        await waiting
        assert started == ["busy", "next"]
        assert actor._serve_count == 0
    finally:
        return_stream.set()
        release_native.set()
        await asyncio.gather(
            busy, *([waiting] if waiting else []), return_exceptions=True
        )
        await asyncio.to_thread(worker.join)


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
async def test_waiting_requests_count_toward_limits_until_cancelled():
    actor, model = _actor(request_limits=2)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def infer(prompt, config):
        if prompt == "busy":
            entered.set()
            await release.wait()
        return {"choices": []}

    model.generate = infer
    busy = asyncio.create_task(actor.generate("busy", {}))
    await entered.wait()
    low = asyncio.create_task(actor.generate("low", {"priority": 100}))
    await _wait_for(lambda: bool(actor._request_priority_scheduler._waiting))
    with pytest.raises(RuntimeError, match="Rate limit reached"):
        await actor.generate("high", {"priority": -100})
    low.cancel()
    with pytest.raises(asyncio.CancelledError):
        await low
    high = asyncio.create_task(actor.generate("high", {"priority": -100}))
    await _wait_for(lambda: bool(actor._request_priority_scheduler._waiting))
    release.set()
    await asyncio.gather(busy, high)
    assert actor._serve_count == 0


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


@pytest.mark.asyncio
async def test_serial_and_batch_waiters_share_capacity_and_cancel_safely():
    scheduler = PriorityScheduler(4)
    await scheduler.acquire(0, 1)
    low = asyncio.create_task(scheduler.acquire(10, 1))
    high = asyncio.create_task(scheduler.acquire(-10, 1))
    await _wait_for(lambda: len(scheduler._waiting) == 2)
    # A blocked serial path must not prevent otherwise eligible batch work.
    for _ in range(3):
        await asyncio.wait_for(scheduler.acquire(0, 4), 1)
    extra_batch = asyncio.create_task(scheduler.acquire(0, 4))
    await _wait_for(lambda: len(scheduler._waiting) == 3)
    assert scheduler._available == 0
    scheduler.release(1)
    high.cancel()  # Return both path and total capacity after reservation.
    with pytest.raises(asyncio.CancelledError):
        await high
    await asyncio.wait_for(extra_batch, 1)
    assert not low.done()
    scheduler.release(4)
    await asyncio.wait_for(low, 1)
    scheduler.release(1)
    for _ in range(3):
        scheduler.release(4)
    assert scheduler._available == 4


@pytest.mark.asyncio
async def test_close_rejects_waiters_in_each_execution_path():
    scheduler = PriorityScheduler(4)
    await scheduler.acquire(0, 1)
    for _ in range(3):
        await scheduler.acquire(0, 4)
    waiters = [asyncio.create_task(scheduler.acquire(0, path)) for path in (1, 4)]
    await _wait_for(lambda: len(scheduler._waiting) == 2)
    scheduler.close()
    for waiter in waiters:
        with pytest.raises(RuntimeError, match="stopping"):
            await waiter
    scheduler.release(1)
    for _ in range(3):
        scheduler.release(4)


@pytest.mark.asyncio
async def test_admitted_sync_call_keeps_rpc_context_and_stops_its_executor():
    from unittest.mock import Mock

    from ..rpc_context import get_current_rpc_metadata

    actor, model = _actor()
    observed = []

    def infer(prompt, config):
        metadata = get_current_rpc_metadata()
        observed.append(metadata.correlation_id if metadata else None)
        return {"choices": []}

    model.generate = infer
    model.stop = Mock()
    metadata = {
        RPC_METADATA_KEY: RpcMetadata(correlation_id="sync-request").to_payload()
    }
    await actor.generate("hello", {}, **metadata)
    executor = actor._request_priority_executor
    assert observed == ["sync-request"]
    await actor.stop()
    model.stop.assert_called_once()
    with pytest.raises(RuntimeError, match="shutdown"):
        executor.submit(lambda: None)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "chat"])
@pytest.mark.parametrize("native", [False, True])
async def test_rpc_metadata_is_bound_during_stream_iteration(method, native):
    from ..rpc_context import get_current_rpc_metadata

    actor, model = _actor()
    model.supports_native_request_priority = native
    if native:
        actor._lock = None  # Native schedulers use the concurrent actor path.
    observed = []

    async def chunks():
        try:
            for _ in range(2):
                metadata = get_current_rpc_metadata()
                observed.append(metadata.correlation_id if metadata else None)
                yield {"choices": []}
        finally:
            metadata = get_current_rpc_metadata()
            observed.append(metadata.correlation_id if metadata else None)

    async def infer(prompt, config, **kwargs):
        return chunks()

    setattr(model, method, infer)
    metadata = {
        RPC_METADATA_KEY: RpcMetadata(correlation_id="stream-request").to_payload()
    }
    stream = await getattr(actor, method)("hello", {"stream": True}, **metadata)
    stream._actor_ref = actor
    assert get_current_rpc_metadata() is None
    assert len([chunk async for chunk in stream]) == 2
    assert observed == ["stream-request"] * 3
    assert get_current_rpc_metadata() is None


@pytest.mark.asyncio
async def test_non_stream_batch_cancellation_retains_admission_until_native_removal():
    actor, model = _actor()
    model.allow_batch = True
    active, release, removed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    entered = []

    async def infer(prompt, config):
        entered.append(prompt)
        if prompt == "busy":
            active.set()
            try:
                await release.wait()
            finally:
                removed.set()
        return {"choices": []}

    model.generate = infer
    busy = asyncio.create_task(actor.generate("busy", {}))
    await active.wait()
    waiting = asyncio.create_task(actor.generate("next", {}))
    await _wait_for(lambda: bool(actor._request_priority_scheduler._waiting))
    busy.cancel()
    try:
        await asyncio.sleep(0)
        assert not removed.is_set()
        assert not busy.done()
        assert entered == ["busy"]
    finally:
        release.set()
        await asyncio.gather(busy, waiting, return_exceptions=True)
    assert removed.is_set()
    assert entered == ["busy", "next"]
    assert actor._serve_count == 0


@pytest.mark.asyncio
async def test_native_stream_close_does_not_occupy_shared_executor(monkeypatch):
    import contextvars
    import functools
    from concurrent.futures import ThreadPoolExecutor
    from unittest.mock import Mock

    from ...model.llm.llama_cpp.core import _owned_native_stream

    shared = ThreadPoolExecutor(max_workers=1)

    async def shared_thread(fn, *args, **kwargs):
        call = functools.partial(contextvars.copy_context().run, fn, *args, **kwargs)
        return await asyncio.get_running_loop().run_in_executor(shared, call)

    monkeypatch.setattr(asyncio, "to_thread", shared_thread)
    actor, model = _actor()
    native = Future()
    closed = threading.Event()

    def source():
        try:
            yield None
            yield {"choices": []}
        finally:
            closed.set()

    def infer(prompt, config):
        iterator = source()
        next(iterator)
        return _owned_native_stream(iterator, native)

    model.generate = infer
    model.stop = Mock()
    stream = await actor.generate("stream", {"stream": True})
    stream._actor_ref = actor
    await stream.__anext__()
    closing = asyncio.create_task(_close(stream))
    try:
        await _wait_for(closed.is_set)
        assert (
            await asyncio.wait_for(asyncio.to_thread(lambda: "responsive"), 1)
            == "responsive"
        )
        assert not closing.done()
        # Stop fences admission but must leave the executor available to drain.
        await actor.stop()
        assert not closing.done()
    finally:
        native.set_result(None)
        await closing
        shared.shutdown(wait=True)
    assert "_request_priority_executor" not in actor.__dict__


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
async def test_explicit_ids_are_unique_and_aborted_ids_remain_blocked(native):
    actor, model = _actor()
    model.supports_native_request_priority = native
    entered = asyncio.Event()
    release = asyncio.Event()

    async def infer(prompt, config, **kwargs):
        entered.set()
        await release.wait()
        return {"choices": []}

    model.async_generate = infer
    model.abort_request = AsyncMock(return_value="DONE")
    task = asyncio.create_task(actor.generate("hello", {}, request_id="public-id"))
    block = None
    try:
        await entered.wait()
        with pytest.raises(Exception, match="Duplicate request id"):
            await actor.generate("hello", {}, request_id="public-id")
        assert await actor.abort_request("public-id") == "DONE"
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(Exception, match="request has been aborted"):
            await actor.generate("hello", {}, request_id="public-id")
        block = actor._running_tasks.get("public-id")
    finally:
        release.set()
        task.cancel()
        if block is not None:
            block.cancel()
        await asyncio.gather(task, *([block] if block else []), return_exceptions=True)
