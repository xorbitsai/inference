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

import pytest

from ..batch import BatchScheduler
from ..core import XINFERENCE_STREAMING_ERROR_FLAG


@pytest.fixture
def no_empty_cache(monkeypatch):
    monkeypatch.setattr(BatchScheduler, "_empty_cache", staticmethod(lambda: None))


@pytest.mark.asyncio
async def test_batch_inference_failure_completes_non_streaming_request(no_empty_cache):
    class FailingModel:
        def get_max_num_seqs(self):
            return 2

        def batch_inference(self, req_list):
            raise RuntimeError("cache failure")

    scheduler = BatchScheduler(FailingModel())
    future = asyncio.get_running_loop().create_future()
    await scheduler.add_request(
        "prompt", future, "generate", {"request_id": "cache-failure"}
    )

    await scheduler.step()

    with pytest.raises(RuntimeError, match="Batch inference failed: cache failure"):
        await future
    assert not scheduler._running_queue
    assert not scheduler._waiting_queue
    assert not scheduler._id_to_req


@pytest.mark.asyncio
async def test_batch_inference_failure_emits_streaming_error(no_empty_cache):
    class FailingModel:
        def get_max_num_seqs(self):
            return 2

        def batch_inference(self, req_list):
            raise RuntimeError("model failure")

    scheduler = BatchScheduler(FailingModel())
    queue = asyncio.Queue()
    await scheduler.add_request(
        "prompt",
        queue,
        "generate",
        {"request_id": "stream-failure", "stream": True},
    )

    await scheduler.step()

    assert await queue.get() == (
        XINFERENCE_STREAMING_ERROR_FLAG + "Batch inference failed: model failure"
    )
    assert not scheduler._id_to_req


@pytest.mark.asyncio
async def test_scheduler_survives_batch_failure(no_empty_cache):
    class FailOnceModel:
        def __init__(self):
            self.failed = False

        def get_max_num_seqs(self):
            return 2

        def batch_inference(self, req_list):
            if not self.failed:
                self.failed = True
                raise RuntimeError("temporary failure")
            for req in req_list:
                req.stopped = True
                req.completion = [{"text": req.prompt}]

    model = FailOnceModel()
    scheduler = BatchScheduler(model)
    scheduler._running = True
    task = asyncio.create_task(scheduler._run())
    try:
        first = asyncio.get_running_loop().create_future()
        await scheduler.add_request("first", first, "generate", {"request_id": "first"})
        with pytest.raises(RuntimeError, match="temporary failure"):
            await asyncio.wait_for(first, timeout=1)

        second = asyncio.get_running_loop().create_future()
        await scheduler.add_request(
            "second", second, "generate", {"request_id": "second"}
        )
        assert await asyncio.wait_for(second, timeout=1) == {"text": "second"}
        assert not task.done()
    finally:
        scheduler._running = False
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_cache_reduction_failure_fails_active_requests(no_empty_cache):
    class ReductionFailingModel:
        def get_max_num_seqs(self):
            return 2

        def batch_inference(self, req_list):
            completed, active = req_list
            completed.stopped = True
            completed.new_tokens.append(1)
            completed.completion = [{"text": "done"}]
            active.kv_cache = object()

        def build_reduced_kv_cache(self, cache, skipped_indexes):
            raise RuntimeError("reduction failure")

    scheduler = BatchScheduler(ReductionFailingModel())
    completed_future = asyncio.get_running_loop().create_future()
    active_future = asyncio.get_running_loop().create_future()
    await scheduler.add_request(
        "completed", completed_future, "generate", {"request_id": "completed"}
    )
    await scheduler.add_request(
        "active", active_future, "generate", {"request_id": "active"}
    )

    await scheduler.step()

    assert await completed_future == {"text": "done"}
    with pytest.raises(RuntimeError, match="Batch inference failed: reduction failure"):
        await active_future
    assert not scheduler._running_queue
    assert not scheduler._id_to_req
