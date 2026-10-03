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
from types import SimpleNamespace

import pytest
import xoscar as xo

from ..model import ModelActor, request_limit


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "streaming,metric_call",
    [(False, call) for call in range(1, 5)] + [(True, call) for call in range(1, 4)],
)
async def test_cancelled_metrics_release_request_limit(metric_call, streaming):
    entered = asyncio.Event()
    calls = 0

    async def record_metrics(*args):
        nonlocal calls
        calls += 1
        if calls == metric_call:
            entered.set()
            await asyncio.Event().wait()

    actor = SimpleNamespace(
        _request_limits=1,
        _serve_count=0,
        _metrics_labels={},
        record_metrics=record_metrics,
        model_uid=lambda: "test",
    )

    @request_limit
    async def generate(self):
        if streaming:

            async def stream():
                yield "chunk"

            return stream()
        return "result"

    task = asyncio.create_task(generate(actor))
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert actor._serve_count == 0
    result = await generate(actor)
    if streaming:
        assert actor._serve_count == 1
        assert [chunk async for chunk in result] == ["chunk"]
    else:
        assert result == "result"
        assert actor._serve_count == 0


@pytest.mark.asyncio
async def test_stream_retains_limit_until_client_releases_it():
    async def record_metrics(*args):
        pass

    actor = SimpleNamespace(
        _request_limits=1,
        _serve_count=0,
        _metrics_labels={},
        record_metrics=record_metrics,
        model_uid=lambda: "test",
    )

    @request_limit
    async def generate(self):
        async def stream():
            yield "chunk"

        return stream()

    stream = await generate(actor)
    assert actor._serve_count == 1
    with pytest.raises(RuntimeError, match="Rate limit reached"):
        await generate(actor)
    assert actor._serve_count == 1
    assert [chunk async for chunk in stream] == ["chunk"]
    assert actor._serve_count == 1


class _BlockingMetricsActor(xo.StatelessActor):
    def __init__(self):
        self.entered = False
        self.calls = 0

    async def record_metrics(self, *args):
        self.calls += 1
        if self.calls == 1:
            self.entered = True
            await asyncio.Event().wait()

    def is_blocked(self):
        return self.entered


class _LimitedRequestActor(xo.StatelessActor):
    record_metrics = ModelActor.record_metrics

    def __init__(self, worker):
        self._worker = worker
        self._serve_count = 0
        self._request_limits = 1
        self._metrics_labels = {}

    async def _get_worker_ref(self):
        return self._worker

    def model_uid(self):
        return "test-model"

    def count(self):
        return self._serve_count

    @request_limit
    async def generate(self):
        return "result"


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["test://", ""])
async def test_metrics_rpc_cancellation_releases_limit(transport):
    pool = await xo.create_actor_pool(
        f"{transport}127.0.0.1:{xo.utils.get_next_port()}", n_process=0
    )
    async with pool:
        worker = await xo.create_actor(
            _BlockingMetricsActor, address=pool.external_address
        )
        actor = await xo.create_actor(
            _LimitedRequestActor, worker, address=pool.external_address
        )

        async def generate():
            return await actor.generate()

        task = asyncio.create_task(generate())
        try:

            async def wait_until_blocked():
                while not await worker.is_blocked():
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_until_blocked(), timeout=5)
            assert await actor.count() == 1
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert await actor.count() == 0
            assert await actor.generate() == "result"
            assert await actor.count() == 0
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
