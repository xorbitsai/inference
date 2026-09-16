# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import asyncio

import pytest

from ..streaming_outcome import (
    FailureOrigin,
    StreamingOutcomeReporter,
    StreamState,
    observe_stream,
)


@pytest.mark.asyncio
async def test_observe_stream_preserves_chunks_and_reports_failure():
    reporter = StreamingOutcomeReporter()

    async def source():
        yield b"one"
        yield b"two"
        raise RuntimeError("failed")

    chunks = []
    with pytest.raises(RuntimeError, match="failed"):
        async for chunk in observe_stream(
            source(), reporter, failure_origin=FailureOrigin.UPSTREAM
        ):
            chunks.append(chunk)

    assert chunks == [b"one", b"two"]
    assert reporter.outcome.state is StreamState.FAILED
    assert reporter.outcome.failure_origin is FailureOrigin.UPSTREAM
    assert reporter.outcome.error_type == "RuntimeError"


def test_first_terminal_state_wins():
    reporter = StreamingOutcomeReporter()
    reporter.failed(ValueError("first"), FailureOrigin.PROTOCOL)
    reporter.completed()
    reporter.failed(RuntimeError("second"), FailureOrigin.SERVER)

    assert reporter.outcome.state is StreamState.FAILED
    assert reporter.outcome.failure_origin is FailureOrigin.PROTOCOL
    assert reporter.outcome.error_message == "first"


@pytest.mark.asyncio
async def test_nested_observers_keep_precise_failure_origin():
    upstream_reporter = StreamingOutcomeReporter()

    async def upstream_failure():
        yield b"one"
        raise OSError("upstream disconnected")

    async def pass_through(source):
        async for item in source:
            yield item

    stream = observe_stream(
        pass_through(
            observe_stream(
                upstream_failure(),
                upstream_reporter,
                failure_origin=FailureOrigin.UPSTREAM,
            )
        ),
        upstream_reporter,
        failure_origin=FailureOrigin.PROTOCOL,
    )
    with pytest.raises(OSError, match="upstream disconnected"):
        async for _ in stream:
            pass
    assert upstream_reporter.outcome.failure_origin is FailureOrigin.UPSTREAM

    protocol_reporter = StreamingOutcomeReporter()

    async def valid_upstream():
        yield b"one"

    async def protocol_failure(source):
        async for _ in source:
            raise ValueError("invalid protocol frame")
            yield  # pragma: no cover

    stream = observe_stream(
        protocol_failure(
            observe_stream(
                valid_upstream(),
                protocol_reporter,
                failure_origin=FailureOrigin.UPSTREAM,
            )
        ),
        protocol_reporter,
        failure_origin=FailureOrigin.PROTOCOL,
    )
    with pytest.raises(ValueError, match="invalid protocol frame"):
        async for _ in stream:
            pass
    assert protocol_reporter.outcome.failure_origin is FailureOrigin.PROTOCOL


@pytest.mark.asyncio
async def test_observe_stream_closes_source_without_changing_chunks():
    reporter = StreamingOutcomeReporter()
    closed = False

    async def source():
        nonlocal closed
        try:
            yield b"one"
            yield b"two"
        finally:
            closed = True

    stream = observe_stream(
        source(), reporter, failure_origin=FailureOrigin.MODEL_GENERATOR
    )
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
        break
    await stream.aclose()

    assert chunks == [b"one"]
    assert closed is True
    assert reporter.outcome.state is StreamState.PENDING


@pytest.mark.asyncio
async def test_observe_stream_leaves_cancellation_for_outer_response_layer():
    reporter = StreamingOutcomeReporter()

    async def source():
        raise asyncio.CancelledError()
        yield  # pragma: no cover

    with pytest.raises(asyncio.CancelledError):
        async for _ in observe_stream(
            source(), reporter, failure_origin=FailureOrigin.MODEL_GENERATOR
        ):
            pass

    assert reporter.outcome.state is StreamState.PENDING
