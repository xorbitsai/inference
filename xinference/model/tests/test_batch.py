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

import asyncio
from contextlib import suppress

import pytest
from xoscar import extensible

from ..batch import BatchMixin


class _BatchProbe(BatchMixin):
    def __init__(self, batch_size):
        self.calls = []
        BatchMixin.__init__(self, self.run, batch_size=batch_size, batch_interval=0.001)

    @extensible
    def run(self, values):
        return values

    @run.batch  # type: ignore
    async def run(self, args_list, kwargs_list):
        assert not any(kwargs_list)
        values = [args[0] for args in args_list]
        self.calls.append(values)
        return values

    def _get_batch_size(self, values):
        return len(values)


async def _shutdown_batch_processor(probe):
    task = probe._process_batch_task
    if task is not None:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_sizes", "expected_batch_sizes"),
    [
        ([3, 2], [[3], [2]]),
        ([2, 2, 1], [[2, 2], [1]]),
        ([3, 2, 2], [[3], [2, 2]]),
        ([5, 1], [[5], [1]]),
    ],
)
async def test_batch_mixin_respects_batch_size_and_order(
    request_sizes, expected_batch_sizes
):
    probe = _BatchProbe(batch_size=4)
    requests = [
        [f"request-{request_index}-item-{item_index}" for item_index in range(size)]
        for request_index, size in enumerate(request_sizes)
    ]

    try:
        results = await asyncio.gather(*(probe.run(request) for request in requests))

        assert results == requests
        assert [
            [len(request) for request in batch] for batch in probe.calls
        ] == expected_batch_sizes
        assert [
            request[0].split("-item-")[0] for batch in probe.calls for request in batch
        ] == [f"request-{i}" for i in range(len(requests))]
    finally:
        await _shutdown_batch_processor(probe)


class _BatchModel(BatchMixin):
    def __init__(self):
        self.batch_started = asyncio.Event()
        self.batch_completed = asyncio.Event()
        self.release_batch = asyncio.Event()
        self.batch_error = None
        self.batch_values = []
        BatchMixin.__init__(self, self.infer, batch_interval=0.05)

    def _get_batch_size(self, *args, **kwargs) -> int:
        return 1

    @extensible
    def infer(self, value: int) -> int:
        return value * 2

    @infer.batch  # type: ignore[no-redef]
    async def infer(self, args_list, kwargs_list):
        self.batch_values = [args[0] for args in args_list]
        self.batch_started.set()
        try:
            await self.release_batch.wait()
            if self.batch_error is not None:
                raise self.batch_error
            return [args[0] * 2 for args in args_list]
        finally:
            self.batch_completed.set()


class _QueuedBatchModel(_BatchModel):
    def __init__(self):
        self.release_processor = asyncio.Event()
        super().__init__()

    async def _process_batch(self):
        await self.release_processor.wait()
        await super()._process_batch()


class _FailOnceProcessorModel(_BatchModel):
    def __init__(self):
        super().__init__()
        self.processor_failed = asyncio.Event()

    async def _process_batch(self):
        if not self.processor_failed.is_set():
            # Remove the request from the queue before failing, exercising the
            # in-flight cleanup path rather than only draining queued requests.
            await self.queue.get()
            self.processor_failed.set()
            raise RuntimeError("processor failed")
        await super()._process_batch()


async def _stop_batch_processor(model):
    task = model._process_batch_task
    if task is None:
        return
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_batch_processor_uses_distinct_exceptions_for_pending_callers():
    probe = _BatchProbe(batch_size=4)
    loop = asyncio.get_running_loop()
    first = loop.create_future()
    second = loop.create_future()
    probe._pending_batch_futures.update((first, second))

    exception = RuntimeError("processor failed")
    probe._fail_pending_requests(exception)

    first_exception = first.exception()
    second_exception = second.exception()
    assert isinstance(first_exception, RuntimeError)
    assert isinstance(second_exception, RuntimeError)
    assert first_exception.args == exception.args
    assert second_exception.args == exception.args
    assert first_exception is not exception
    assert second_exception is not exception
    assert first_exception is not second_exception


@pytest.mark.asyncio
async def test_batch_processor_survives_cancelled_caller():
    model = _BatchModel()

    first = asyncio.create_task(model.infer(1))
    second = asyncio.create_task(model.infer(2))
    try:
        await model.batch_started.wait()
        assert model.batch_values == [1, 2]

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        model.release_batch.set()
        assert await asyncio.wait_for(second, timeout=1) == 4
        assert model._process_batch_task is not None
        assert not model._process_batch_task.done()
    finally:
        await _stop_batch_processor(model)


@pytest.mark.asyncio
async def test_batch_processor_skips_caller_cancelled_while_queued():
    model = _QueuedBatchModel()

    first = asyncio.create_task(model.infer(1))
    second = asyncio.create_task(model.infer(2))
    try:
        while model.queue.qsize() < 2:
            await asyncio.sleep(0)

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        model.release_processor.set()
        await model.batch_started.wait()
        assert model.batch_values == [2]

        model.release_batch.set()
        assert await asyncio.wait_for(second, timeout=1) == 4
    finally:
        model.release_processor.set()
        await _stop_batch_processor(model)


@pytest.mark.asyncio
async def test_batch_processor_survives_cancelled_caller_when_batch_fails():
    model = _BatchModel()

    first = asyncio.create_task(model.infer(1))
    try:
        await model.batch_started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        model.batch_error = RuntimeError("batch failed")
        model.release_batch.set()
        await model.batch_completed.wait()
        model.batch_error = None

        assert await asyncio.wait_for(model.infer(2), timeout=1) == 4
        assert model._process_batch_task is not None
        assert not model._process_batch_task.done()
    finally:
        await _stop_batch_processor(model)


@pytest.mark.asyncio
async def test_batch_processor_fails_inflight_request_after_unexpected_exit():
    model = _FailOnceProcessorModel()

    first = asyncio.create_task(model.infer(1))
    try:
        await model.processor_failed.wait()
        with pytest.raises(RuntimeError, match="processor failed"):
            await asyncio.wait_for(first, timeout=1)

        model.release_batch.set()
        assert await asyncio.wait_for(model.infer(2), timeout=1) == 4
        assert model._process_batch_task is not None
        assert not model._process_batch_task.done()
    finally:
        await _stop_batch_processor(model)
