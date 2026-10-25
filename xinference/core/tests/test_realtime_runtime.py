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
from dataclasses import replace

import aiohttp
import pytest
import pytest_asyncio

from ...model.realtime import RealtimeSession
from ..realtime.protocol import (
    RealtimeError,
    RealtimeLimits,
    parse_event,
    validate_start,
)
from ..realtime.runtime import RealtimeRuntime

CONFIG = {
    "event": "session.start",
    "protocol_version": 1,
    "task": "binary.echo",
    "input": {"bytes": {}},
    "parameters": {},
}


class FakeSession(RealtimeSession):
    def __init__(self, runtime):
        super().__init__(runtime.limits.output_queue_size)
        self.received = 0

    async def receive_input(self, frame):
        self.received += len(frame)
        await self.emit({"received_bytes": self.received})

    async def finish_input(self):
        await self.emit({"received_bytes": self.received, "done": True})
        await self.end_outputs()


class FakeFamily:
    def to_description(self):
        return {"model_type": "audio", "model_name": "fake"}


class FakeModel:
    def __init__(self):
        self.loads = 0
        self.sessions = []
        self.model_family = FakeFamily()

    def model_uid(self):
        return "fake"

    def load(self):
        self.loads += 1

    def get_realtime_capabilities(self):
        return {"tasks": {"binary.echo": {}, "audio.transcription": {}}}

    def create_realtime_session(self, config, runtime):
        session = FakeSession(runtime)
        self.sessions.append(session)
        return session


@pytest_asyncio.fixture
async def runtime():
    model = FakeModel()
    model.load()
    runtime = RealtimeRuntime(model, "fake", "fake-0")
    await runtime.start("127.0.0.1")
    try:
        yield runtime
    finally:
        await runtime.stop()


async def connect(client, runtime, config=None, descriptor=None):
    config = config or CONFIG
    descriptor = descriptor or runtime.issue_ticket("user-1", config["task"])
    ws = await client.ws_connect(
        descriptor["endpoint"],
        headers={"Authorization": "Bearer " + descriptor["ticket"]},
    )
    await ws.send_json(config)
    return ws


async def wait_empty(runtime):
    for _ in range(100):
        if runtime.active_count == 0:
            return
        await asyncio.sleep(0.01)
    assert runtime.active_count == 0


@pytest.mark.parametrize("task", ["binary.echo", "audio.transcription"])
async def test_common_session(runtime, task):
    async with aiohttp.ClientSession() as client:
        ws = await connect(client, runtime, {**CONFIG, "task": task})
        started = await ws.receive_json()
        assert started["event"] == "session.started"
        await ws.send_bytes(b"123")
        output = await ws.receive_json()
        assert output["seq"] == 1 and output["data"] == {"received_bytes": 3}
        await ws.send_json({"event": "input.end"})
        assert (await ws.receive_json())["data"]["done"]
        assert (await ws.receive_json())["reason"] == "completed"
        await ws.close()
    await wait_empty(runtime)
    assert runtime.model.loads == 1
    assert runtime.describe()["pid"] == os.getpid()


async def test_internal_auth_replay_and_generation(runtime):
    async with aiohttp.ClientSession() as client:
        for headers in ({}, {"Authorization": "Bearer invalid"}):
            with pytest.raises(aiohttp.WSServerHandshakeError) as error:
                await client.ws_connect(runtime.endpoint, headers=headers)
            assert error.value.status == 403
        descriptor = runtime.issue_ticket("u", "binary.echo")
        ws = await connect(client, runtime, descriptor=descriptor)
        await ws.receive_json()
        await ws.close()
        await wait_empty(runtime)
        with pytest.raises(aiohttp.WSServerHandshakeError):
            await connect(client, runtime, descriptor=descriptor)
        descriptor = runtime.issue_ticket("u", "binary.echo")
        runtime.generation = "new-generation"
        with pytest.raises(aiohttp.WSServerHandshakeError):
            await connect(client, runtime, descriptor=descriptor)
    assert runtime.active_count == 0


async def test_expired_and_cross_runtime_ticket(runtime):
    runtime.limits = replace(runtime.limits, ticket_ttl=-1)
    expired = runtime.issue_ticket("u", "binary.echo")
    async with aiohttp.ClientSession() as client:
        with pytest.raises(aiohttp.WSServerHandshakeError):
            await connect(client, runtime, descriptor=expired)
        other = RealtimeRuntime(FakeModel(), "other", "other-0")
        await other.start("127.0.0.1")
        try:
            foreign = other.issue_ticket("u", "binary.echo")
            foreign["endpoint"] = runtime.endpoint
            with pytest.raises(aiohttp.WSServerHandshakeError):
                await connect(client, runtime, descriptor=foreign)
        finally:
            await other.stop()


async def test_capacity_cancel_disconnect_and_draining(runtime):
    async with aiohttp.ClientSession() as client:
        ws = await connect(client, runtime)
        await ws.receive_json()
        with pytest.raises(aiohttp.WSServerHandshakeError) as error:
            await connect(client, runtime)
        assert error.value.status == 503
        await ws.send_json({"event": "session.cancel"})
        assert (await ws.receive_json())["reason"] == "cancelled"
        await ws.close()
        await wait_empty(runtime)
        ws = await connect(client, runtime)
        await ws.receive_json()
        await ws.close()
        await wait_empty(runtime)
        runtime.draining = True
        with pytest.raises(RealtimeError):
            runtime.issue_ticket("u", "binary.echo")
    assert all(session._closed for session in runtime.model.sessions)


@pytest.mark.parametrize(
    "event",
    [
        {"event": "session.start"},
        {"event": "unknown"},
        {"event": "input.end", "unexpected": True},
    ],
)
async def test_invalid_control(runtime, event):
    async with aiohttp.ClientSession() as client:
        ws = await connect(client, runtime)
        await ws.receive_json()
        await ws.send_json(event)
        assert (await ws.receive_json())["event"] == "error"
        await ws.close()
    await wait_empty(runtime)


async def test_media_before_start_and_task_binding(runtime):
    async with aiohttp.ClientSession() as client:
        descriptor = runtime.issue_ticket("u", "binary.echo")
        ws = await connect(
            client, runtime, {**CONFIG, "task": "audio.transcription"}, descriptor
        )
        assert (await ws.receive_json())["code"] == "unsupported_task"
        await ws.close()
        await wait_empty(runtime)
        descriptor = runtime.issue_ticket("u", "binary.echo")
        ws = await client.ws_connect(
            runtime.endpoint,
            headers={"Authorization": "Bearer " + descriptor["ticket"]},
        )
        await ws.send_bytes(b"audio")
        assert (await ws.receive_json())["code"] == "invalid_state"
        await ws.close()
    await wait_empty(runtime)


def test_protocol_rejections():
    import pickle

    error = pickle.loads(
        pickle.dumps(RealtimeError("unsupported_task", "unsupported", 1008))
    )
    assert error.code == "unsupported_task" and error.detail == "unsupported"
    for text in ("[]", "null", "START", '{"event":1}'):
        with pytest.raises(RealtimeError):
            parse_event(text)
    for change in (
        {"protocol_version": True},
        {"protocol_version": 2},
        {"unknown": 1},
        {"input": []},
    ):
        with pytest.raises(RealtimeError):
            validate_start({**CONFIG, **change})
    with pytest.raises(RealtimeError):
        parse_event('{"event":"x"}', max_bytes=2)


async def test_invoke_serialization_and_cancellation(runtime):
    import threading

    entered, release = threading.Event(), threading.Event()

    def work():
        entered.set()
        release.wait(5)

    task = asyncio.create_task(runtime.invoke(work))
    while not entered.is_set():
        await asyncio.sleep(0.001)
    task.cancel()
    await asyncio.sleep(0.01)
    assert runtime._inference_lock.locked()
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not runtime._inference_lock.locked()


async def test_idle_timeout_and_bounded_input(runtime):
    runtime.limits = replace(RealtimeLimits(), idle_timeout=0.05, input_queue_size=2)
    async with aiohttp.ClientSession() as client:
        ws = await connect(client, runtime)
        await ws.receive_json()
        assert (await ws.receive_json())["code"] == "timeout"
        await ws.close()
    await wait_empty(runtime)


async def test_slow_model_backpressure_times_out_without_unbounded_queues(runtime):
    runtime.limits = replace(runtime.limits, input_queue_size=2, queue_timeout=0.03)

    class SlowSession(FakeSession):
        async def receive_input(self, frame):
            await asyncio.Event().wait()

    runtime.model.create_realtime_session = lambda config, owner: SlowSession(owner)
    async with aiohttp.ClientSession() as client:
        ws = await connect(client, runtime)
        await ws.receive_json()
        for _ in range(10):
            await ws.send_bytes(b"1234")
        error = await asyncio.wait_for(ws.receive_json(), 2)
        assert error["code"] == "timeout"
        assert runtime.input_queue_peak == 2
        await ws.close()
    await wait_empty(runtime)


async def test_final_deadline_includes_queued_audio(runtime):
    runtime.limits = replace(runtime.limits, final_timeout=0.03)

    class SlowSession(FakeSession):
        async def receive_input(self, frame):
            await asyncio.Event().wait()

    runtime.model.create_realtime_session = lambda config, owner: SlowSession(owner)
    async with aiohttp.ClientSession() as client:
        ws = await connect(client, runtime)
        await ws.receive_json()
        await ws.send_bytes(b"1234")
        await ws.send_json({"event": "input.end"})
        error = await asyncio.wait_for(ws.receive_json(), 2)
        assert error["code"] == "timeout"
        await ws.close()
    await wait_empty(runtime)


async def test_final_timeout_closes_socket_but_retains_native_capacity(runtime):
    import threading

    runtime.limits = replace(runtime.limits, final_timeout=0.03)
    release = threading.Event()

    class NativeSession(FakeSession):
        async def receive_input(self, frame):
            await runtime.invoke(release.wait, 5)

    runtime.model.create_realtime_session = lambda config, owner: NativeSession(owner)
    try:
        async with aiohttp.ClientSession() as client:
            ws = await connect(client, runtime)
            await ws.receive_json()
            await ws.send_bytes(b"12")
            await ws.send_json({"event": "input.end"})
            error = await asyncio.wait_for(ws.receive_json(), 1)
            assert error["code"] == "timeout"
            assert runtime.active_count == 1
            assert runtime._inference_lock.locked()
            await ws.close()
    finally:
        release.set()
        await wait_empty(runtime)


async def test_repeated_invoke_cancellation_keeps_lock_until_native_end(runtime):
    import threading

    entered, release = threading.Event(), threading.Event()
    entered_async = asyncio.Event()
    loop = asyncio.get_running_loop()

    def work():
        entered.set()
        loop.call_soon_threadsafe(entered_async.set)
        release.wait(5)
        raise RuntimeError("native failure must still be retrieved")

    task = asyncio.create_task(runtime.invoke(work))
    try:
        await asyncio.wait_for(entered_async.wait(), 1)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
            assert runtime._inference_lock.locked() and not task.done()
        assert entered.is_set()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not runtime._inference_lock.locked()
    assert await runtime.invoke(lambda: 42) == 42


async def test_end_hint_precedes_draining_all_accepted_frames(runtime):
    hint, release, entered = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class HintSession(FakeSession):
        def notify_input(self, size):
            self.notified = getattr(self, "notified", 0) + size

        def request_input_end(self):
            assert self.received == 0
            hint.set()

        async def receive_input(self, frame):
            entered.set()
            await release.wait()
            await super().receive_input(frame)

        async def finish_input(self):
            assert hint.is_set() and self.received == 6 and self.notified == 6
            await super().finish_input()

    runtime.model.create_realtime_session = lambda config, owner: HintSession(owner)
    async with aiohttp.ClientSession() as client:
        ws = await connect(client, runtime)
        await ws.receive_json()
        await ws.send_bytes(b"123")
        await entered.wait()
        await ws.send_bytes(b"456")
        await ws.send_json({"event": "input.end"})
        await asyncio.wait_for(hint.wait(), 1)
        release.set()
        assert (await ws.receive_json())["data"]["received_bytes"] == 3
        assert (await ws.receive_json())["data"]["received_bytes"] == 6
        assert (await ws.receive_json())["data"]["done"]
        assert (await ws.receive_json())["reason"] == "completed"
        await ws.close()
    await wait_empty(runtime)
    events = [e["stage"] for e in runtime.recent_summaries[-1]["events"]]
    assert events.index("input_end") < events.index("input_drained")


async def test_native_cancel_ack_fences_output_and_retains_capacity_on_stop(runtime):
    import threading

    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def native():
        runtime.record_event("native_start", kind="partial")
        loop.call_soon_threadsafe(entered.set)
        release.wait(5)
        runtime.record_event("native_end", kind="partial")

    class NativeSession(FakeSession):
        async def receive_input(self, frame):
            await runtime.invoke(native)
            await self.emit({"late": True})

    runtime.model.create_realtime_session = lambda config, owner: NativeSession(owner)
    runtime.limits = replace(runtime.limits, drain_timeout=0.01)
    stopper = None
    try:
        async with aiohttp.ClientSession() as client:
            ws = await connect(client, runtime)
            await ws.receive_json()
            await ws.send_bytes(b"12")
            await asyncio.wait_for(entered.wait(), 1)
            await ws.send_json({"event": "session.cancel"})
            assert (await asyncio.wait_for(ws.receive_json(), 1))[
                "reason"
            ] == "cancelled"
            assert runtime._inference_lock.locked() and runtime.active_count == 1
            with pytest.raises(aiohttp.WSServerHandshakeError):
                await connect(client, runtime)
            stopper = asyncio.create_task(runtime.stop())
            # Let the drain timeout cancel a handler already draining a cancelled
            # processor. It must not abandon the native lease on repeated cancel.
            await asyncio.sleep(0.03)
            assert runtime._inference_lock.locked() and runtime.active_count == 1
            assert not stopper.done()
            await ws.close()
    finally:
        release.set()
        if stopper is not None:
            await asyncio.wait_for(stopper, 2)
    assert runtime.active_count == 0 and not runtime._inference_lock.locked()
    summary = runtime.recent_summaries[-1]
    events = {e["stage"]: e["at_ms"] for e in summary["events"]}
    assert events["native_start"] < events["cancel_processed"] < events["native_end"]
    assert "invoke" in summary["durations"]
    assert "session_id" not in str(summary)


async def test_independent_byte_budget_bounds_inflight_media(runtime):
    entered = asyncio.Event()
    runtime.limits = replace(
        runtime.limits, max_input_queue_bytes=8, queue_timeout=0.03
    )

    class SlowSession(FakeSession):
        async def receive_input(self, frame):
            entered.set()
            await asyncio.Event().wait()

    runtime.model.create_realtime_session = lambda config, owner: SlowSession(owner)
    async with aiohttp.ClientSession() as client:
        ws = await connect(client, runtime)
        await ws.receive_json()
        await ws.send_bytes(b"12345678")
        await entered.wait()
        await ws.send_bytes(b"1")
        error = await asyncio.wait_for(ws.receive_json(), 1)
        assert error["code"] == "timeout"
        await ws.close()
    await wait_empty(runtime)
    assert runtime.input_bytes_peak == 8


def test_trace_is_bounded_and_uses_no_raw_session_id():
    from ..realtime.telemetry import RealtimeTrace

    trace = RealtimeTrace("not-a-public-session-id")
    for _ in range(1000):
        trace.event("native_end", elapsed_ms=1)
        trace.duration("invoke", 0.001)
    snapshot = trace.snapshot()
    assert len(snapshot["events"]) == 128
    assert snapshot["durations"]["invoke"]["count"] == 1000
    assert "not-a-public-session-id" not in str(snapshot)


async def test_cancel_waiting_for_file_lock_never_starts_native_work(runtime):
    called = []
    async with runtime._inference_lock:  # Same lock held by an admitted file API call.
        job = asyncio.create_task(runtime.invoke(lambda: called.append(True)))
        await asyncio.sleep(0)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        assert not called
        assert runtime._inference_lock.locked()
    assert await runtime.invoke(lambda: "next request") == "next request"


async def test_stuck_native_drain_quarantines_without_releasing_lease(runtime):
    import threading

    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    runtime.limits = replace(runtime.limits, drain_timeout=0.02)

    def native():
        loop.call_soon_threadsafe(entered.set)
        release.wait(5)

    class StuckSession(FakeSession):
        async def receive_input(self, frame):
            await runtime.invoke(native)

    runtime.model.create_realtime_session = lambda config, owner: StuckSession(owner)
    try:
        async with aiohttp.ClientSession() as client:
            ws = await connect(client, runtime)
            await ws.receive_json()
            await ws.send_bytes(b"12")
            await asyncio.wait_for(entered.wait(), 1)
            await ws.send_json({"event": "session.cancel"})
            assert (await ws.receive_json())["reason"] == "cancelled"

            async def quarantined():
                while not runtime.draining:
                    await asyncio.sleep(0.001)

            await asyncio.wait_for(quarantined(), 1)
            assert runtime.active_count == 1 and runtime._inference_lock.locked()
            assert not runtime.describe()["readiness"]
            with pytest.raises(RealtimeError, match="draining"):
                runtime.issue_ticket("u", "binary.echo")
            await ws.close()
    finally:
        release.set()
        await wait_empty(runtime)
    assert runtime.draining  # Recovery must be explicit, not silent re-admission.
