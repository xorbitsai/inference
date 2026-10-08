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
from unittest.mock import MagicMock

import pytest

from ...utils import ChatModelMixin
from .. import utils
from ..utils import vllm_check


class _AsyncIterableOnly:
    def __init__(self, model, fail: bool = False):
        self._model = model
        self._fail = fail

    def __aiter__(self):
        async def iterator():
            try:
                yield "first"
                if self._fail:
                    raise utils.VLLM_ENGINE_DEAD_ERRORS[0]("EngineCore died")
                yield "second"
            finally:
                self._model.iterator_closed()

        return iterator()


class _Model:
    def __init__(self):
        self.stop = MagicMock()
        self.iterator_closed = MagicMock()

    @vllm_check
    async def stream(self, fail: bool):
        async def iterator():
            yield "first"
            if fail:
                raise utils.VLLM_ENGINE_DEAD_ERRORS[0]("EngineCore died")
            yield "second"

        return iterator()

    @vllm_check
    async def fail_before_returning(self):
        raise utils.VLLM_ENGINE_DEAD_ERRORS[0]("EngineCore died")

    @vllm_check
    async def closable_stream(self):
        async def iterator():
            try:
                yield "first"
                yield "second"
            finally:
                self.iterator_closed()

        return iterator()

    @vllm_check
    async def async_iterable_stream(self, fail: bool = False):
        return _AsyncIterableOnly(self, fail)

    @vllm_check
    async def async_generate(self):
        async def engine_stream():
            try:
                yield {
                    "id": "completion-id",
                    "model": "test-model",
                    "created": 0,
                    "object": "text_completion",
                    "choices": [
                        {
                            "index": 0,
                            "text": "first",
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
                yield {
                    "id": "completion-id",
                    "model": "test-model",
                    "created": 0,
                    "object": "text_completion",
                    "choices": [
                        {
                            "index": 0,
                            "text": "second",
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
            finally:
                self.iterator_closed()

        return engine_stream()

    @vllm_check
    async def async_chat(self):
        chunks = await self.async_generate()
        return ChatModelMixin._async_to_chat_completion_chunks(chunks)


@pytest.mark.asyncio
async def test_vllm_check_guards_async_generator(monkeypatch):
    model = _Model()
    exit_process = MagicMock()
    monkeypatch.setattr("xinference.model.llm.vllm.utils.os._exit", exit_process)

    iterator = await model.stream(fail=True)
    assert [item async for item in iterator] == ["first"]
    model.stop.assert_called_once_with()
    exit_process.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_vllm_check_keeps_successful_async_generator(monkeypatch):
    model = _Model()
    exit_process = MagicMock()
    monkeypatch.setattr("xinference.model.llm.vllm.utils.os._exit", exit_process)

    iterator = await model.stream(fail=False)
    assert [item async for item in iterator] == ["first", "second"]
    model.stop.assert_not_called()
    exit_process.assert_not_called()


@pytest.mark.asyncio
async def test_vllm_check_handles_failure_before_generator_return(monkeypatch):
    model = _Model()
    exit_process = MagicMock()
    monkeypatch.setattr("xinference.model.llm.vllm.utils.os._exit", exit_process)

    assert await model.fail_before_returning() is None
    model.stop.assert_called_once_with()
    exit_process.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_vllm_check_closes_wrapped_async_generator(monkeypatch):
    model = _Model()
    exit_process = MagicMock()
    monkeypatch.setattr("xinference.model.llm.vllm.utils.os._exit", exit_process)

    iterator = await model.closable_stream()
    assert await anext(iterator) == "first"

    await iterator.aclose()

    model.iterator_closed.assert_called_once_with()
    model.stop.assert_not_called()
    exit_process.assert_not_called()


@pytest.mark.asyncio
async def test_vllm_check_guards_async_iterable(monkeypatch):
    model = _Model()
    exit_process = MagicMock()
    monkeypatch.setattr("xinference.model.llm.vllm.utils.os._exit", exit_process)

    iterable = await model.async_iterable_stream(fail=True)
    assert [item async for item in iterable] == ["first"]
    model.stop.assert_called_once_with()
    exit_process.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_vllm_check_closes_async_iterable_iterator(monkeypatch):
    model = _Model()
    exit_process = MagicMock()
    monkeypatch.setattr("xinference.model.llm.vllm.utils.os._exit", exit_process)

    iterable = await model.async_iterable_stream()
    assert await anext(iterable) == "first"

    await iterable.aclose()

    model.iterator_closed.assert_called_once_with()
    model.stop.assert_not_called()
    exit_process.assert_not_called()


@pytest.mark.asyncio
async def test_vllm_check_closes_nested_generate_stream(monkeypatch):
    model = _Model()
    exit_process = MagicMock()
    monkeypatch.setattr("xinference.model.llm.vllm.utils.os._exit", exit_process)

    stream = await model.async_chat()
    first_chunk = await anext(stream)
    assert first_chunk["choices"][0]["delta"]["content"] == "first"

    await stream.aclose()

    model.iterator_closed.assert_called_once_with()
    model.stop.assert_not_called()
    exit_process.assert_not_called()
