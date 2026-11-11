# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
import threading
from unittest.mock import AsyncMock

import pytest

from ..streaming import generate_stream
from .test_gemma4 import _direct_model


@pytest.mark.asyncio
async def test_hf_streamer_close_retains_serial_admission(monkeypatch):
    import transformers

    from .....core.model import ModelActor

    real_streamer = transformers.TextIteratorStreamer
    model = _direct_model(monkeypatch)
    monkeypatch.setattr(transformers, "TextIteratorStreamer", real_streamer)
    model._batch_scheduler = None
    model._pytorch_model_config = {}
    model.allow_batch = False
    cancelled, release = threading.Event(), threading.Event()
    entered = []
    owned = []
    original = model.build_streaming_iter

    def retain(*args):
        result = original(*args)
        owned.append(result[0])
        return result

    model.build_streaming_iter = retain

    def generate(**kwargs):
        entered.append(True)
        streamer = kwargs["streamer"]
        if len(entered) == 1:
            streamer.on_finalized_text("first ", stream_end=False)
            criteria = kwargs.get("stopping_criteria") or []
            while not any(check(None, None) for check in criteria):
                if release.wait(0.001):
                    break
            cancelled.set()
            assert release.wait(5)
        streamer.on_finalized_text("done", stream_end=True)

    model._model.generate = generate
    actor = ModelActor("test:123", "test:456", model, "gemma-4-0", None)
    actor._model_state = "ready"
    actor.record_metrics = AsyncMock()
    messages = [{"role": "user", "content": "你好"}]
    stream = await actor.chat(messages, {"stream": True})
    stream._actor_ref = actor
    await stream.__anext__()
    waiting = asyncio.create_task(actor.chat(messages, {"max_tokens": 8}))
    stream._gc_destroy = False
    closing = asyncio.create_task(stream.destroy())
    try:

        async def until_cancelled():
            while not cancelled.is_set():
                await asyncio.sleep(0)

        await asyncio.wait_for(until_cancelled(), 2)
        assert not closing.done()
        assert not waiting.done()
        assert len(entered) == 1
    finally:
        release.set()
        await asyncio.gather(closing, waiting, return_exceptions=True)
        for source in owned:
            await asyncio.to_thread(source.close)
        actor._shutdown_request_priority_executor()


@pytest.mark.asyncio
@pytest.mark.parametrize("read", [False, True])
async def test_owned_thread_stream_closes_unread_and_keeps_existing_criteria(read):
    from transformers import TextIteratorStreamer

    cancelled, release = threading.Event(), threading.Event()
    captured = []
    original = lambda *args, **kwargs: False
    criteria = [original]
    streamer = TextIteratorStreamer(object())

    def generate(**kwargs):
        captured.extend(kwargs["stopping_criteria"])
        streamer.on_finalized_text("chunk", stream_end=False)
        while not captured[-1](None, None):
            assert not release.wait(0.001)
        cancelled.set()
        assert release.wait(3)
        streamer.end()

    stream = generate_stream(
        generate, {"streamer": streamer, "stopping_criteria": criteria}
    )
    if read:
        assert next(stream) == "chunk"
    closing = asyncio.create_task(asyncio.to_thread(stream.close))
    try:

        async def until_cancelled():
            while not cancelled.is_set():
                await asyncio.sleep(0)

        await asyncio.wait_for(until_cancelled(), 2)
        assert criteria == [original]
        assert captured[0] is original
        assert not closing.done()
    finally:
        release.set()
        await closing


def test_owned_thread_stream_propagates_generation_errors():
    from transformers import TextIteratorStreamer

    def fail(**kwargs):
        raise ValueError("native generation failed")

    stream = generate_stream(fail, {"streamer": TextIteratorStreamer(object())})
    with pytest.raises(ValueError, match="native generation failed"):
        list(stream)
