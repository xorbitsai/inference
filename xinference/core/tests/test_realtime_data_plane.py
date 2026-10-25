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
import os
import threading
from unittest.mock import AsyncMock

import aiohttp
import pytest

from ...api.tests.test_realtime_api import serve_gateway
from ..model import ModelActor, request_limit
from ..realtime.protocol import RealtimeError
from ..utils import build_replica_model_uid
from .test_realtime_runtime import CONFIG, FakeModel, wait_empty


class RealtimeTestActor(ModelActor):
    """A real xoscar subprocess with no Supervisor metrics dependencies."""

    async def __post_create__(self):
        pass

    async def __pre_destroy__(self):
        if self._realtime_runtime is not None:
            await self._realtime_runtime.stop()

    async def record_metrics(self, *args, **kwargs):
        pass

    def observations(self):
        return {
            "pid": os.getpid(),
            "loads": self._model.loads,
            "sessions": len(self._model.sessions),
            "active": self._realtime_runtime.active_count,
        }


@pytest.mark.parametrize("frames", [10, 100, 1000])
async def test_frame_count_does_not_increase_control_rpc(frames):
    async with serve_gateway() as service:
        async with aiohttp.ClientSession() as client:
            ws = await client.ws_connect(service.url)
            await ws.send_json(CONFIG)
            await ws.receive_json()
            for _ in range(frames):
                await ws.send_bytes(b"1234")
                await ws.receive_json()
            await ws.send_json({"event": "input.end"})
            assert (await ws.receive_json())["data"]["received_bytes"] == frames * 4
            await ws.receive_json()
            await ws.close()
        await wait_empty(service.runtime)
        # The stub exposes only connect-time discovery. Any attempted media
        # Actor call fails; the actual gateway->runtime WebSocket is not mocked.
        assert service.actor.get_realtime_endpoint.await_count == 1
        assert service.supervisor.get_model.await_count == 1
        assert service.runtime.model.loads == 1


async def test_actor_lifecycle_and_shared_capacity():
    model = FakeModel()
    model.load()
    actor = ModelActor(
        "127.0.0.1:1",
        "127.0.0.1:2",
        model,
        build_replica_model_uid("fake", 0),
        request_limits=1,
    )
    actor.record_metrics = AsyncMock()
    await asyncio.gather(actor.wait_for_load(), actor.wait_for_load())
    runtime = actor._realtime_runtime
    assert runtime.model is model
    assert runtime._inference_lock is actor._lock
    assert runtime.describe()["model_uid"] == "fake"
    try:
        async with aiohttp.ClientSession() as client:
            descriptor = actor.get_realtime_endpoint("u", "binary.echo")
            ws = await client.ws_connect(
                descriptor["endpoint"],
                headers={"Authorization": "Bearer " + descriptor["ticket"]},
            )
            await ws.send_json(CONFIG)
            await ws.receive_json()

            @request_limit
            async def file_request(self):
                return None

            with pytest.raises(RuntimeError, match="Rate limit"):
                await file_request(actor)
            await ws.close()
            await wait_empty(runtime)
            # File requests already admitted must also block realtime admission.
            actor._serve_count = 1
            descriptor = actor.get_realtime_endpoint("u", "binary.echo")
            with pytest.raises(aiohttp.WSServerHandshakeError):
                await client.ws_connect(
                    descriptor["endpoint"],
                    headers={"Authorization": "Bearer " + descriptor["ticket"]},
                )
            actor._serve_count = 0
        assert model.loads == 1
    finally:
        await runtime.stop()
    assert not runtime.describe()["readiness"]
    with pytest.raises(RealtimeError):
        runtime.issue_ticket("u", "binary.echo")


async def test_stop_drains_and_releases_session():
    from dataclasses import replace

    async with serve_gateway() as service:
        runtime = service.runtime
        runtime.limits = replace(runtime.limits, drain_timeout=0.01)
        async with aiohttp.ClientSession() as client:
            ws = await client.ws_connect(service.url)
            await ws.send_json(CONFIG)
            await ws.receive_json()
            await asyncio.wait_for(runtime.stop(), 2)
            await ws.close()
        assert runtime.active_count == 0
        assert runtime.model.sessions[0]._closed


async def test_real_actor_subprocess_reuses_loaded_model():
    import xoscar as xo
    from xoscar.backends.allocate_strategy import ProcessIndex

    pool = await xo.create_actor_pool(
        "127.0.0.1", n_process=1, subprocess_start_method="spawn"
    )
    async with pool:
        actor = await xo.create_actor(
            RealtimeTestActor,
            "127.0.0.1:1",
            "127.0.0.1:2",
            FakeModel(),
            build_replica_model_uid("fake", 0),
            address=pool.external_address,
            allocate_strategy=ProcessIndex(1),
        )
        try:
            await actor.load()
            await actor.wait_for_load()
            before = await actor.observations()
            assert before["pid"] != os.getpid()
            async with aiohttp.ClientSession() as client:
                for _ in range(2):
                    descriptor = await actor.get_realtime_endpoint(
                        "user", "binary.echo"
                    )
                    assert descriptor["pid"] == before["pid"]
                    ws = await client.ws_connect(
                        descriptor["endpoint"],
                        headers={"Authorization": "Bearer " + descriptor["ticket"]},
                    )
                    await ws.send_json(CONFIG)
                    await ws.receive_json()
                    for _ in range(10):
                        await ws.send_bytes(b"12")
                        await ws.receive_json()
                    await ws.send_json({"event": "input.end"})
                    await ws.receive_json()
                    await ws.receive_json()
                    await ws.close()
                    # No Actor calls inside the media loop, including output.
                for _ in range(100):
                    after = await actor.observations()
                    if after["active"] == 0:
                        break
                    await asyncio.sleep(0.01)
            assert after == {
                "pid": before["pid"],
                "loads": 1,
                "sessions": 2,
                "active": 0,
            }
        finally:
            await xo.destroy_actor(actor)


@pytest.mark.parametrize("unload", [False, True])
@pytest.mark.parametrize("native_failure", [False, True])
async def test_repeated_file_cancel_retains_lock_capacity_and_weights(
    unload, native_failure
):
    entered, release, finished, stopped = [threading.Event() for _ in range(4)]
    model = FakeModel()

    def stop():
        assert finished.is_set()
        stopped.set()

    model.stop = stop
    actor = ModelActor("127.0.0.1:1", "127.0.0.1:2", model, "fake-0")
    actor.record_metrics = AsyncMock()
    await actor.wait_for_load()
    runtime = actor._realtime_runtime

    def native():
        entered.set()
        try:
            assert release.wait(5)
            if native_failure:
                raise ValueError("native failure after cancellation")
            return b"transcript"
        finally:
            finished.set()

    @request_limit
    async def file_request(self):
        return await self._call_wrapper("binary", native)

    task = asyncio.create_task(file_request(actor))
    successor = None
    try:

        async def wait_native():
            while not entered.is_set():
                await asyncio.sleep(0.001)

        await asyncio.wait_for(wait_native(), 2)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done()
            assert actor._lock.locked()
            assert actor._serve_count == 1
            assert not finished.is_set()
        successor = asyncio.create_task(
            actor.stop() if unload else runtime.invoke(lambda: finished.is_set())
        )
        await asyncio.sleep(0.01)
        assert not successor.done()
        assert not stopped.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        result = await asyncio.wait_for(successor, 2)
        assert actor._serve_count == 0
        assert finished.is_set()
        if unload:
            assert stopped.is_set()
            assert actor._realtime_runtime is None
            await actor.__pre_destroy__()
        else:
            assert result is True
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        if successor is not None:
            await asyncio.gather(successor, return_exceptions=True)
        await runtime.stop()


async def test_batching_adapter_cannot_bypass_shared_inference_lock():
    model = FakeModel()
    model.allow_batch = True
    actor = ModelActor("127.0.0.1:1", "127.0.0.1:2", model, "fake-0")
    with pytest.raises(ValueError, match="serialized inference"):
        await actor.wait_for_load()
    assert actor._realtime_runtime is None


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "outcome", ["complete", "cancel", "error", "stop", "handler_cancel"]
)
async def test_file_stream_retains_inference_lease(asynchronous, outcome):
    from ..realtime.runtime import RealtimeRuntime

    entered, release, finished, closed = [threading.Event() for _ in range(4)]
    entered_async = asyncio.Event()
    loop = asyncio.get_running_loop()
    model = FakeModel()
    stopped = []

    def native():
        entered.set()
        loop.call_soon_threadsafe(entered_async.set)
        assert release.wait(10)
        if outcome == "error":
            raise ValueError("file stream failed")

    def sync_stream(**kwargs):
        try:
            native()
            yield {"text": "file"}
            assert outcome not in ("cancel", "stop", "handler_cancel")
            yield {"text": "tail"}
        finally:
            finished.set()
            closed.set()

    async def async_stream(**kwargs):
        try:
            await asyncio.to_thread(native)
            yield {"text": "file"}
            assert outcome not in ("cancel", "stop", "handler_cancel")
            yield {"text": "tail"}
        finally:
            finished.set()
            closed.set()

    def stop():
        assert closed.is_set()
        stopped.append(True)

    model.stop = stop
    actor = ModelActor("127.0.0.1:1", "127.0.0.1:2", model, "fake-0")
    actor.record_metrics = AsyncMock()
    await actor.wait_for_load()
    runtime: RealtimeRuntime = actor._realtime_runtime
    handler = asyncio.create_task(actor._handle_pending_requests())

    @request_limit
    async def file_request(self):
        return await self._call_wrapper_json(
            async_stream if asynchronous else sync_stream, request_id="file-stream"
        )

    stream = await asyncio.create_task(file_request(actor))
    successor = count_release = None
    try:
        await asyncio.wait_for(entered_async.wait(), 2)
        assert entered.is_set() and actor._lock.locked()
        assert actor._serve_count == 1
        actor._request_limits = 1
        with pytest.raises(RuntimeError, match="Rate limit"):
            await file_request(actor)
        if outcome == "cancel":
            for _ in range(3):
                assert await actor.abort_request("file-stream") == "DONE"
        if outcome == "handler_cancel":
            for _ in range(3):
                handler.cancel()
                await asyncio.sleep(0)
                assert not handler.done() and actor._lock.locked()
        successor = asyncio.create_task(
            actor.stop()
            if outcome == "stop"
            else runtime.invoke(lambda: closed.is_set())
        )
        count_release = asyncio.create_task(actor.decrease_serve_count())
        await asyncio.sleep(0.01)
        assert not successor.done() and not count_release.done()
        assert actor._serve_count == 1
        assert not stopped and not finished.is_set()
        release.set()
        result = await asyncio.wait_for(successor, 2)
        await asyncio.wait_for(count_release, 2)
        assert actor._serve_count == 0 and closed.is_set()
        if outcome == "stop":
            assert stopped
            await actor.__pre_destroy__()
        else:
            assert result is True

        async def collect():
            if asynchronous:
                return [item async for item in stream]
            return await asyncio.to_thread(list, stream)

        if outcome == "error":
            with pytest.raises(ValueError, match="file stream failed"):
                await collect()
        else:
            assert len(await collect()) == (2 if outcome == "complete" else 0)
        assert not actor._realtime_file_streams
    finally:
        release.set()
        for task in (successor, count_release):
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
        handler.cancel()
        await asyncio.gather(handler, return_exceptions=True)
        await runtime.stop()


class StopSequenceActor(ModelActor):
    """Use the real ModelActor startup, stop and destroy hooks with local weights."""

    def __init__(self):
        super().__init__("127.0.0.1:1", "127.0.0.1:2", StopSequenceModel(), "fake-0")

    async def record_metrics(self, *args, **kwargs):
        pass

    async def prepare(self):
        from dataclasses import replace

        await self.wait_for_load()
        self._model.runtime = self._realtime_runtime
        self._realtime_runtime.limits = replace(
            self._realtime_runtime.limits, drain_timeout=0.01
        )

    def observations(self):
        return {
            "entered": self._model.entered.is_set(),
            "finished": self._model.finished.is_set(),
            "stops": list(self._model.stops),
            "state": self._model_state,
        }

    def release_native(self):
        self._model.release.set()


class StopSequenceModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.entered, self.release, self.finished = [
            threading.Event() for _ in range(3)
        ]
        self.stops = []

    def native(self):
        self.entered.set()
        try:
            assert self.release.wait(10)
        finally:
            self.finished.set()

    def stop(self):
        assert self.finished.is_set()
        assert self.runtime.active_count == 0
        assert not self.runtime._inference_lock.locked()
        self.stops.append(True)

    def create_realtime_session(self, config, runtime):
        from .test_realtime_runtime import FakeSession

        class NativeSession(FakeSession):
            async def receive_input(session, frame):
                await runtime.invoke(self.native)

        return NativeSession(runtime)


async def test_stop_then_destroy_drains_real_actor_realtime_session():
    import xoscar as xo

    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        actor = await xo.create_actor(StopSequenceActor, address=pool.external_address)
        stopper = None
        try:
            await actor.prepare()
            descriptor = await actor.get_realtime_endpoint("u", "binary.echo")
            async with aiohttp.ClientSession() as client:
                ws = await client.ws_connect(
                    descriptor["endpoint"],
                    headers={"Authorization": "Bearer " + descriptor["ticket"]},
                )
                await ws.send_json(CONFIG)
                await ws.receive_json()
                await ws.send_bytes(b"native")

                async def wait_state(predicate):
                    while True:
                        if stopper is not None and stopper.done():
                            await stopper
                        state = await actor.observations()
                        if predicate(state):
                            return state
                        await asyncio.sleep(0)

                await asyncio.wait_for(wait_state(lambda s: s["entered"]), 2)
                stopper = asyncio.create_task(actor.stop())
                state = await asyncio.wait_for(
                    wait_state(lambda s: s["state"] == "draining"), 2
                )
                assert not state["finished"] and not state["stops"]
                assert not stopper.done()
                await actor.release_native()
                await asyncio.wait_for(stopper, 2)
                state = await actor.observations()
                assert state["finished"] and state["stops"] == [True]
                await ws.close()
        finally:
            await actor.release_native()
            if stopper is not None:
                await asyncio.gather(stopper, return_exceptions=True)
            # Same ordering as WorkerActor.terminate_model(), using the actual
            # destruction hook rather than calling only __pre_destroy__ directly.
            await xo.destroy_actor(actor)
