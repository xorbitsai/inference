# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import AsyncMock

import pytest

from ....scheduler.core import AbortRequestMessage
from ..core import VLLMModel


def _new_model(engine):
    model = object.__new__(VLLMModel)
    model._engine = engine
    model._active_request_ids = set()
    return model


async def _engine_results(*values):
    for value in values:
        yield value


class _ClosableAsyncIterator:
    def __init__(self, *values):
        self._values = iter(values)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._values)
        except StopIteration:
            raise StopAsyncIteration

    async def aclose(self):
        self.closed = True


class _AsyncIterableOnly:
    def __init__(self, *values):
        self.iterator = _ClosableAsyncIterator(*values)
        self.aiter_calls = 0

    def __aiter__(self):
        self.aiter_calls += 1
        return self.iterator


@pytest.mark.asyncio
async def test_abort_request_statuses():
    model = _new_model(None)
    assert await model.abort_request("missing") == AbortRequestMessage.NOT_FOUND.name

    class EngineWithoutAbort:
        pass

    engine = EngineWithoutAbort()
    model = _new_model(engine)
    model._active_request_ids.add("unsupported")
    assert await model.abort_request("unsupported") == AbortRequestMessage.NO_OP.name
    assert "unsupported" in model._active_request_ids

    assert await model.abort_request("missing") == AbortRequestMessage.NOT_FOUND.name


@pytest.mark.asyncio
async def test_abort_request_calls_engine_once_for_duplicate_aborts():
    class Engine:
        abort = AsyncMock()

    engine = Engine()
    model = _new_model(engine)
    model._active_request_ids.add("request-1")

    assert await model.abort_request("request-1") == AbortRequestMessage.DONE.name
    assert await model.abort_request("request-1") == AbortRequestMessage.NOT_FOUND.name
    engine.abort.assert_awaited_once_with("request-1")


@pytest.mark.asyncio
async def test_early_stream_close_aborts_engine_request():
    class Engine:
        abort = AsyncMock()

    engine = Engine()
    model = _new_model(engine)
    stream = model._track_engine_request(
        "request-1", _engine_results("first", "second")
    )

    assert "request-1" in model._active_request_ids
    assert await anext(stream) == "first"
    await stream.aclose()

    engine.abort.assert_awaited_once_with("request-1")
    assert "request-1" not in model._active_request_ids


@pytest.mark.asyncio
async def test_early_close_closes_same_concrete_async_iterator():
    class Engine:
        abort = AsyncMock()

    engine = Engine()
    model = _new_model(engine)
    results = _AsyncIterableOnly("first", "second")
    stream = model._track_engine_request("request-1", results)

    assert results.aiter_calls == 1
    assert await anext(stream) == "first"
    await stream.aclose()

    engine.abort.assert_awaited_once_with("request-1")
    assert results.aiter_calls == 1
    assert results.iterator.closed


@pytest.mark.asyncio
async def test_normal_stream_completion_does_not_abort():
    class Engine:
        abort = AsyncMock()

    engine = Engine()
    model = _new_model(engine)
    stream = model._track_engine_request(
        "request-1", _engine_results("first", "second")
    )

    assert [item async for item in stream] == ["first", "second"]
    engine.abort.assert_not_awaited()
    assert "request-1" not in model._active_request_ids


@pytest.mark.asyncio
async def test_concurrent_streams_are_tracked_independently():
    class Engine:
        abort = AsyncMock()

    engine = Engine()
    model = _new_model(engine)
    first = model._track_engine_request("request-1", _engine_results(1, 2))
    second = model._track_engine_request("request-2", _engine_results(3, 4))

    assert model._active_request_ids == {"request-1", "request-2"}
    assert await anext(first) == 1
    assert [item async for item in second] == [3, 4]
    assert model._active_request_ids == {"request-1"}

    await first.aclose()
    engine.abort.assert_awaited_once_with("request-1")
    assert not model._active_request_ids
