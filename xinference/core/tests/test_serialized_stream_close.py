# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
import inspect
import threading

import pytest

from .test_request_priority import _actor, _Model


async def _next(stream):
    if inspect.isasyncgen(stream):
        return await stream.__anext__()
    return await asyncio.to_thread(next, stream)


async def _close(stream):
    if inspect.isasyncgen(stream):
        await stream.aclose()
    else:
        await asyncio.to_thread(stream.close)


async def _drain(stream):
    if inspect.isasyncgen(stream):
        return [item async for item in stream]
    return await asyncio.to_thread(list, stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("output_type", ["json", "binary"])
@pytest.mark.parametrize("source", ["async", "sync"])
@pytest.mark.parametrize("request_id", [None, "closed-request"], ids=["no-id", "id"])
@pytest.mark.parametrize("path", ["serialized", "concurrent", "realtime-lease"])
async def test_consumer_close_stops_production(
    monkeypatch, path, request_id, source, output_type
):
    # Set the flag before construction so ModelActor selects its own lock.
    monkeypatch.setattr(_Model, "allow_batch", path == "concurrent")
    actor, model = _actor()
    if path == "realtime-lease":
        # Select the actual file-stream lease code without starting a gateway.
        actor._realtime_runtime = object()
    assert (actor._lock is None) == (path == "concurrent")

    loop = asyncio.get_running_loop()
    paused, async_release = asyncio.Event(), asyncio.Event()
    sync_release = threading.Event()
    produced, raw_streams = [], []

    def payload(number):
        return {"item": number} if output_type == "json" else str(number).encode()

    def sync_chunks(name):
        if name == "barrier":
            return
        for number in range(5):
            produced.append(number)
            if number == 1:
                # Count this in-flight item before close, then hold its yield.
                loop.call_soon_threadsafe(paused.set)
                sync_release.wait()
            yield payload(number)

    async def async_chunks(name):
        if name == "barrier":
            return
        for number in range(5):
            produced.append(number)
            if number == 1:
                paused.set()
                await async_release.wait()
            yield payload(number)

    def infer(name, **kwargs):
        raw = async_chunks(name) if source == "async" else sync_chunks(name)
        raw_streams.append(raw)  # Keep sources alive independently of consumers.
        return raw

    async def async_infer(name, **kwargs):
        return infer(name, **kwargs)

    model.generate = async_infer if source == "async" else infer

    async def open_stream(name, identifier):
        # Exercise the shared actor boundary without LLM priority admission,
        # which can select a different path from the serialized queue.
        return await actor._call_wrapper(
            output_type, model.generate, name, request_id=identifier
        )

    handler = asyncio.create_task(actor._handle_pending_requests())
    consumer = barrier = None
    try:
        consumer = await open_stream("first", request_id)
        assert (request_id in actor._request_streams) == (
            request_id is not None and path != "realtime-lease"
        )
        assert await asyncio.wait_for(_next(consumer), 10) is not None
        if path != "concurrent":
            await asyncio.wait_for(paused.wait(), 10)
        at_close = list(produced)
        assert at_close == ([0] if path == "concurrent" else [0, 1])
        await asyncio.wait_for(_close(consumer), 10)
        assert not actor._request_streams
        if request_id is not None and path != "realtime-lease":
            assert request_id not in actor._running_tasks

        sync_release.set()
        async_release.set()
        barrier = await asyncio.wait_for(open_stream("barrier", None), 10)
        # On queued paths, exhaustion of this second stream proves that the
        # single producer has finished processing the first queued stream.
        assert await asyncio.wait_for(_drain(barrier), 10) == []
        if path == "realtime-lease":
            assert not actor._realtime_file_streams
            assert not actor._lock.locked()
        assert produced == at_close, "source advanced after consumer close"
    finally:
        sync_release.set()
        async_release.set()
        handler.cancel()
        await asyncio.gather(handler, return_exceptions=True)
        for stream in (consumer, barrier, *raw_streams):
            if stream is not None:
                await _close(stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("output_type", ["json", "binary"])
@pytest.mark.parametrize("source", ["async", "sync"])
@pytest.mark.parametrize("request_id", [None, "closed-request"], ids=["no-id", "id"])
async def test_consumer_close_before_first_read_stops_production(
    monkeypatch, request_id, source, output_type
):
    """Closing an unstarted consumer must still set cancelled and stop the serialized producer."""
    actor, model = _actor()
    loop = asyncio.get_running_loop()
    paused, async_release = asyncio.Event(), asyncio.Event()
    sync_release = threading.Event()
    produced, raw_streams = [], []

    def payload(number):
        return {"item": number} if output_type == "json" else str(number).encode()

    def sync_chunks(name):
        if name == "barrier":
            return
        for number in range(5):
            produced.append(number)
            if number == 0:
                loop.call_soon_threadsafe(paused.set)
                sync_release.wait()
            yield payload(number)

    async def async_chunks(name):
        if name == "barrier":
            return
        for number in range(5):
            produced.append(number)
            if number == 0:
                paused.set()
                await async_release.wait()
            yield payload(number)

    def infer(name, **kwargs):
        raw = async_chunks(name) if source == "async" else sync_chunks(name)
        raw_streams.append(raw)
        return raw

    async def async_infer(name, **kwargs):
        return infer(name, **kwargs)

    model.generate = async_infer if source == "async" else infer

    async def open_stream(name, identifier):
        return await actor._call_wrapper(
            output_type, model.generate, name, request_id=identifier
        )

    handler = asyncio.create_task(actor._handle_pending_requests())
    consumer = barrier = None
    try:
        consumer = await open_stream("first", request_id)
        # Wait for the serialized handler to pull the first item into flight.
        await asyncio.wait_for(paused.wait(), 10)
        at_close = list(produced)
        assert at_close == [0]
        # Close without ever reading from the consumer
        await asyncio.wait_for(_close(consumer), 10)
        assert not actor._request_streams
        if request_id is not None:
            assert request_id not in actor._running_tasks

        sync_release.set()
        async_release.set()
        barrier = await asyncio.wait_for(open_stream("barrier", None), 10)
        assert await asyncio.wait_for(_drain(barrier), 10) == []
        assert produced == [0], f"source advanced after early close: {produced}"
    finally:
        sync_release.set()
        async_release.set()
        handler.cancel()
        await asyncio.gather(handler, return_exceptions=True)
        for stream in (consumer, barrier, *raw_streams):
            if stream is not None:
                await _close(stream)
