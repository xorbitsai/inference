# Copyright 2022-2023 XProbe Inc.
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
import pytest_asyncio
import xoscar as xo
from xoscar import create_actor_pool

from ..model import ModelActor

TEST_EVENT = None
TEST_VALUE = None


class MockModel:
    async def generate(self, prompt, **kwargs):
        global TEST_VALUE
        TEST_VALUE = True
        assert isinstance(TEST_EVENT, asyncio.Event)
        await TEST_EVENT.wait()
        yield {"test1": prompt}
        yield {"test2": prompt}


class MockModelActor(ModelActor):
    def __init__(
        self,
        supervisor_address: str,
        worker_address: str,
    ):
        super().__init__(supervisor_address, worker_address, MockModel())  # type: ignore
        self._lock = asyncio.locks.Lock()

    async def __pre_destroy__(self):
        pass

    async def record_metrics(self, name, op, kwargs):
        pass


@pytest_asyncio.fixture
async def setup_pool():
    pool = await create_actor_pool(
        f"test://127.0.0.1:{xo.utils.get_next_port()}", n_process=0
    )
    async with pool:
        yield pool


@pytest.mark.asyncio
async def test_concurrent_call(setup_pool):
    pool = setup_pool
    addr = pool.external_address

    global TEST_EVENT
    TEST_EVENT = asyncio.Event()

    worker: xo.ActorRefType[MockModelActor] = await xo.create_actor(  # type: ignore
        MockModelActor,
        address=addr,
        uid=MockModelActor.default_uid(),
        supervisor_address="test:123",
        worker_address="test:345",
    )

    await worker.generate("test_prompt1")
    assert TEST_VALUE is not None
    # This request is waiting for the TEST_EVENT, so the queue is empty.
    pending_count = await worker.get_pending_requests_count()
    assert pending_count == 0
    await worker.generate("test_prompt3")
    # This request is waiting in the queue because the previous request is waiting for TEST_EVENT.
    pending_count = await worker.get_pending_requests_count()
    assert pending_count == 1

    async def _check():
        gen = await worker.generate("test_prompt2")
        result = []
        async for g in gen:
            result.append(g)
        assert result == [
            b'data: {"test1": "test_prompt2"}\r\n\r\n',
            b'data: {"test2": "test_prompt2"}\r\n\r\n',
        ]

    check_task = asyncio.create_task(_check())
    await asyncio.sleep(2)
    assert not check_task.done()
    # Pending 2 requests: test_prompt3 and test_prompt2
    pending_count = await worker.get_pending_requests_count()
    assert pending_count == 2
    TEST_EVENT.set()
    await check_task
    pending_count = await worker.get_pending_requests_count()
    assert pending_count == 0
