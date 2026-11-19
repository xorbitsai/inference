# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# http://www.apache.org/licenses/LICENSE-2.0

# CI: linux

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import APIRouter, FastAPI, HTTPException, Response

from ...core.exceptions import ModelNotReadyError
from ...core.supervisor import SupervisorActor
from ...core.tests.test_model_replica_scale import _FakeScaleWorker, _make_supervisor
from ...core.tests.test_realtime_runtime import FakeModel, RealtimeRuntime
from ...core.utils import build_replica_model_uid
from ...core.worker import ModelStatus, WorkerActor
from ..routers.audio import register_routes


def audio_api(supervisor):
    async def unused() -> Response:
        return Response()

    api = SimpleNamespace(
        _router=APIRouter(),
        _auth_service=None,
        is_authenticated=lambda: False,
        _check_model_access=Mock(),
        _is_ip_allowed=lambda ip: True,
        _get_supervisor_ref=AsyncMock(return_value=supervisor),
        _report_error_event=AsyncMock(),
        create_audio_embedding=unused,
        create_transcriptions=unused,
        create_translations=unused,
        create_speech=unused,
    )
    register_routes(api)
    app = FastAPI()
    app.include_router(api._router)
    return api, app


@asynccontextmanager
async def replicated_audio(replica_count=2):
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._pd_model_mapping = {}
    supervisor._replica_autoscalers = {}
    supervisor._model_uid_to_replica_info = {
        "custom": supervisor._build_replica_info(replica_count)
    }
    supervisor._replica_model_uid_to_worker = {}
    supervisor._replica_model_uid_to_worker_shards = {}
    supervisor._model_replica_locks = {}
    supervisor._model_reload_tasks = {}
    supervisor._initial_model_launches = {}
    supervisor._status_guard_ref = SimpleNamespace(
        remove_replica_status=AsyncMock(), update_instance_info=AsyncMock()
    )
    supervisor._clear_replica_model_gpu_memory = Mock()
    supervisor._unexpected_down_replicas = {}
    supervisor._invalidate_list_models_debounce_cache = Mock()
    runtimes = []
    capabilities = {"asr": {"streaming": True, "server_vad": True}}
    try:
        for replica in range(replica_count):
            uid = build_replica_model_uid("custom", replica)
            model = FakeModel()
            model.load()
            runtime = RealtimeRuntime(model, "custom", uid)
            runtimes.append(runtime)
            await runtime.start("127.0.0.1")
            actor = SimpleNamespace(
                uid=uid.encode(),
                get_audio_capabilities=AsyncMock(return_value=capabilities),
                get_realtime_endpoint=AsyncMock(side_effect=runtime.issue_ticket),
            )
            supervisor._replica_model_uid_to_worker[uid] = SimpleNamespace(
                get_model=AsyncMock(return_value=actor),
                describe_model=AsyncMock(return_value={"model_type": "audio"}),
            )
        api, app = audio_api(supervisor)
        yield SimpleNamespace(
            api=api,
            app=app,
            supervisor=supervisor,
            runtimes=runtimes,
            capabilities=capabilities,
        )
    finally:
        for runtime in runtimes:
            await runtime.stop()


async def test_capabilities_preserve_model_access_before_actor_call():
    capabilities = {"asr": {"streaming": True, "server_vad": True}}
    actor = SimpleNamespace(
        uid=b"custom-rep0",
        get_audio_capabilities=AsyncMock(return_value=capabilities),
    )
    supervisor = SimpleNamespace(get_model=AsyncMock(return_value=actor))
    api, app = audio_api(supervisor)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.get("/v1/models/custom/audio/capabilities")
        assert response.status_code == 200 and response.json() == capabilities
        assert api._check_model_access.call_args.args[1:] == ("custom", "audio")
        api._check_model_access.side_effect = HTTPException(403, "Model access denied")
        response = await client.get("/v1/models/custom/audio/capabilities")
        assert response.status_code == 403
        actor.get_audio_capabilities.assert_awaited_once()


@pytest.mark.parametrize("preflight", [False, True])
async def test_capabilities_preserve_realtime_routing(preflight):
    async with replicated_audio() as service:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(service.app), base_url="http://test"
        ) as client:
            selected = []
            for _ in range(4):
                if preflight:
                    response = await client.get("/v1/models/custom/audio/capabilities")
                    assert response.status_code == 200
                    assert response.json() == service.capabilities
                descriptor = await service.supervisor.get_realtime_endpoint(
                    "custom", "anonymous", "binary.echo"
                )
                selected.append(descriptor["replica_model_uid"])
            assert selected == ["custom-rep0", "custom-rep1"] * 2


@pytest.mark.parametrize(
    "draining, avoided", [(False, ()), (True, ()), (False, (0,)), (False, (1,))]
)
async def test_capabilities_try_ready_replica_without_advancing_scheduler(
    draining, avoided
):
    async with replicated_audio() as service:
        supervisor = service.supervisor
        if draining or avoided:
            supervisor._replica_autoscalers["custom"] = SimpleNamespace(
                draining={0} if draining else set(),
                should_avoid=lambda replica: replica in avoided,
            )
        worker = supervisor._replica_model_uid_to_worker["custom-rep0"]
        worker.get_model.side_effect = ModelNotReadyError("Loading")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(service.app), base_url="http://test"
        ) as client:
            for _ in range(4):
                response = await client.get("/v1/models/custom/audio/capabilities")
                assert response.status_code == 200
                assert response.json() == service.capabilities
        assert worker.get_model.await_count == (0 if draining or 0 in avoided else 4)
        ready_worker = supervisor._replica_model_uid_to_worker["custom-rep1"]
        assert ready_worker.get_model.await_count == 4
        worker.get_model.side_effect = None
        descriptor = await supervisor.get_realtime_endpoint(
            "custom", "anonymous", "binary.echo"
        )
        expected = 1 if draining or 0 in avoided else 0
        assert descriptor["replica_model_uid"] == f"custom-rep{expected}"


def real_worker_lookups(supervisor):
    workers = []
    for uid, worker_ref in supervisor._replica_model_uid_to_worker.items():
        worker = WorkerActor.__new__(WorkerActor)
        worker._model_uid_to_model_status = {uid: ModelStatus(model_state="ready")}
        worker._model_uid_to_model = {uid: worker_ref.get_model.return_value}
        worker._model_uid_launching_guard = {}
        worker._status_guard_ref = None
        worker_ref.get_model.side_effect = worker.get_model
        worker_ref.get_model_status = AsyncMock(side_effect=worker.get_model_status)
        workers.append(worker)
    return workers


async def reject_delete(supervisor, replica):
    route = supervisor._replica_model_uid_to_worker[f"custom-rep{replica}"]
    route.terminate_model = AsyncMock(side_effect=RuntimeError("Call rejected"))
    with pytest.raises(RuntimeError, match="Call rejected"):
        await supervisor.terminate_model_replica("custom", replica)


def slow_rpcs(route, delay=0.55):
    for name in ("get_model", "get_model_status"):
        rpc = getattr(route, name)

        async def slow(*args, rpc=rpc, **kwargs):
            await asyncio.sleep(delay)
            return await rpc(*args, **kwargs)

        setattr(route, name, AsyncMock(side_effect=slow))


@pytest.mark.parametrize("outcome", ["success", "failure", "suppressed"])
async def test_capabilities_skip_stopping_replica_without_advancing_scheduler(outcome):
    async with replicated_audio() as service:
        supervisor = service.supervisor
        workers = real_worker_lookups(supervisor)
        info = supervisor._model_uid_to_replica_info["custom"]
        entered, release = asyncio.Event(), asyncio.Event()

        async def terminate(model_uid):
            await workers[0]._update_model_state(model_uid, "stopping")
            entered.set()
            await release.wait()
            if outcome in ("failure", "suppressed"):
                raise RuntimeError("Termination failed")

        stopping = supervisor._replica_model_uid_to_worker["custom-rep0"]
        stopping.terminate_model = AsyncMock(side_effect=terminate)
        termination = asyncio.create_task(
            supervisor.terminate_model_replica(
                "custom", 0, suppress_exception=outcome == "suppressed"
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 1)
            assert (
                workers[0]._model_uid_to_model_status["custom-rep0"].model_state
                == "stopping"
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(service.app), base_url="http://test"
            ) as client:
                for _ in range(4):
                    response = await client.get("/v1/models/custom/audio/capabilities")
                    assert response.status_code == 200
                    assert response.json() == service.capabilities
            ready = supervisor._replica_model_uid_to_worker["custom-rep1"]
            assert ready.get_model.await_count == 4
            assert next(info.scheduler) == 0
        finally:
            release.set()
            if outcome == "failure":
                with pytest.raises(RuntimeError, match="Termination failed"):
                    await termination
            else:
                await termination
        assert ("custom-rep0" in supervisor._replica_model_uid_to_worker) == (
            outcome == "failure"
        )


async def test_capabilities_skip_replica_still_stopping_after_cancelled_delete():
    async with replicated_audio() as service:
        supervisor = service.supervisor
        uid = "custom-rep0"
        stopping, ready = supervisor._replica_model_uid_to_worker.values()
        worker, _ = real_worker_lookups(supervisor)
        recovered_capabilities = {"asr": {"streaming": False, "server_vad": False}}
        stopping.get_model.return_value.get_audio_capabilities.return_value = (
            recovered_capabilities
        )
        info = supervisor._model_uid_to_replica_info["custom"]
        scheduler = info.scheduler
        entered = asyncio.Event()

        async def pause_at_supervisor_lookup():
            entered.set()
            await asyncio.Future()

        worker.get_supervisor_ref = pause_at_supervisor_lookup
        stopping.terminate_model = AsyncMock(side_effect=worker.terminate_model)
        deletion = asyncio.create_task(supervisor.terminate_model_replica("custom", 0))
        try:
            await asyncio.wait_for(entered.wait(), 2)
        finally:
            deletion.cancel()
            with pytest.raises(asyncio.CancelledError):
                await deletion
        assert worker.get_model_status(uid).model_state == "stopping"
        assert supervisor._replica_model_uid_to_worker[uid] is stopping

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(service.app), base_url="http://test"
        ) as client:
            for _ in range(4):
                response = await client.get("/v1/models/custom/audio/capabilities")
                assert response.status_code == 200
                assert response.json() == service.capabilities
            assert ready.get_model.await_count == 4
            assert stopping.get_model_status.await_count == 4
            ready.get_model_status.assert_not_awaited()
            assert info.scheduler is scheduler
            assert [next(scheduler), next(scheduler)] == [0, 1]

            await worker._update_model_state(uid, "ready")
            response = await client.get("/v1/models/custom/audio/capabilities")
            assert response.status_code == 200
            assert response.json() == recovered_capabilities


async def test_capabilities_skip_replica_published_during_scale_up_rollback():
    capabilities = {
        "demo-rep0": {"asr": {"streaming": True, "server_vad": False}},
        "demo-rep1": {"asr": {"streaming": True, "server_vad": True}},
    }
    worker = _FakeScaleWorker("worker-0:9978", {"n_gpu": None})
    supervisor = _make_supervisor([worker], worker.launch_args)
    local = WorkerActor.__new__(WorkerActor)
    local._model_uid_to_model = {}
    local._model_uid_to_model_status = {}
    local._model_uid_launching_guard = {}
    local._status_guard_ref = None

    def host(uid):
        local._model_uid_to_model[uid] = SimpleNamespace(
            uid=uid.encode(),
            get_audio_capabilities=AsyncMock(return_value=capabilities[uid]),
        )
        local._model_uid_to_model_status[uid] = ModelStatus(model_state="ready")

    async def launch_then_fail_to_load(**kwargs):
        host(kwargs["model_uid"])

    async def fail_to_load(model_uid):
        raise RuntimeError("load failed")

    host("demo-rep0")
    worker.launch_builtin_model = launch_then_fail_to_load
    worker.wait_for_load = fail_to_load
    worker.get_model = AsyncMock(side_effect=local.get_model)
    worker.get_model_status = AsyncMock(side_effect=local.get_model_status)
    worker.terminate_model = AsyncMock(side_effect=local.terminate_model)
    entered = asyncio.Event()

    async def replay_at_supervisor_lookup():
        # The worker has set the new replica `stopping`. Its supervisor lookup
        # re-enters registration, which publishes the replica from inventory.
        supervisor._rebuild_worker_replica_state(
            worker, [{"replica_model_uid": uid} for uid in local._model_uid_to_model]
        )
        entered.set()
        await asyncio.Future()

    local.get_supervisor_ref = replay_at_supervisor_lookup
    rollback = asyncio.create_task(supervisor.add_model_replica("demo"))
    try:
        await asyncio.wait_for(entered.wait(), 2)
    finally:
        rollback.cancel()
        with pytest.raises(asyncio.CancelledError):
            await rollback
    assert local.get_model_status("demo-rep1").model_state == "stopping"
    assert "demo-rep1" in supervisor._replica_model_uid_to_worker
    info = supervisor._model_uid_to_replica_info["demo"]
    # An earlier session lookup took turn 0, so the stopping replica 1 is next.
    assert next(info.scheduler) == 0

    _, app = audio_api(supervisor)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        for _ in range(4):
            response = await client.get("/v1/models/demo/audio/capabilities")
            assert response.status_code == 200
            assert response.json() == capabilities["demo-rep0"]
    looked_up = [call.kwargs["model_uid"] for call in worker.get_model.await_args_list]
    assert looked_up.count("demo-rep0") == 4
    assert [next(info.scheduler), next(info.scheduler)] == [1, 0]


async def test_capabilities_serve_a_slow_sole_replica_after_a_rejected_delete():
    async with replicated_audio(replica_count=1) as service:
        supervisor = service.supervisor
        real_worker_lookups(supervisor)
        (route,) = supervisor._replica_model_uid_to_worker.values()
        await reject_delete(supervisor, 0)
        slow_rpcs(route)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(service.app), base_url="http://test"
        ) as client:
            for _ in range(3):
                response = await client.get("/v1/models/custom/audio/capabilities")
                assert response.status_code == 200
                assert response.json() == service.capabilities
        assert route.get_model.await_count == 3
        route.get_model_status.assert_not_awaited()


async def test_capabilities_return_from_a_ready_first_candidate_without_later_ones():
    async with replicated_audio() as service:
        supervisor = service.supervisor
        real_worker_lookups(supervisor)
        await reject_delete(supervisor, 1)
        later = supervisor._replica_model_uid_to_worker["custom-rep1"]
        release = asyncio.Event()

        async def blocked(*args, **kwargs):
            await release.wait()

        later.get_model = AsyncMock(side_effect=blocked)
        later.get_model_status = AsyncMock(side_effect=blocked)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(service.app), base_url="http://test"
            ) as client:
                response = await asyncio.wait_for(
                    client.get("/v1/models/custom/audio/capabilities"), 5
                )
        finally:
            release.set()
        assert response.status_code == 200
        assert response.json() == service.capabilities
        later.get_model.assert_not_awaited()
        later.get_model_status.assert_not_awaited()


@pytest.mark.parametrize("reply", ["error", "success", "timeout", "status"])
async def test_capabilities_skip_replica_whose_route_disappears_during_lookup(reply):
    async with replicated_audio() as service:
        supervisor = service.supervisor
        removed, ready = supervisor._replica_model_uid_to_worker.values()
        removed.address, ready.address = "worker-0", "worker-1"
        actor = removed.get_model.return_value
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed_reply(*args, **kwargs):
            entered.set()
            await release.wait()
            if reply == "error":
                raise ValueError("Model not found, uid: custom-rep0")
            if reply == "timeout":
                raise asyncio.TimeoutError()
            if reply == "status":
                return None
            return actor

        if reply == "status":
            removed.get_model.side_effect = ValueError("Model not found")
            removed.get_model_status = AsyncMock(side_effect=delayed_reply)
        else:
            removed.get_model.side_effect = delayed_reply
        probe = asyncio.create_task(supervisor.get_model("custom", for_inference=False))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            supervisor._rebuild_worker_replica_state(removed, [])
        finally:
            release.set()
        assert (await asyncio.wait_for(probe, 1)).uid == b"custom-rep1"


@pytest.mark.parametrize("sibling", [None, "loading"])
async def test_capabilities_report_a_stopping_replica_when_none_can_serve(sibling):
    async with replicated_audio(replica_count=1 if sibling is None else 2) as service:
        supervisor = service.supervisor
        workers = real_worker_lookups(supervisor)
        await workers[0]._update_model_state("custom-rep0", "stopping")
        if sibling:
            other = supervisor._replica_model_uid_to_worker["custom-rep1"]
            other.get_model.side_effect = ModelNotReadyError("Loading")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(service.app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/models/custom/audio/capabilities")
            assert response.status_code == 500
            assert response.json() == {
                "detail": "Model custom-rep0 is in stopping state"
            }
            await workers[0]._update_model_state("custom-rep0", "ready")
            response = await client.get("/v1/models/custom/audio/capabilities")
            assert response.status_code == 200
            assert response.json() == service.capabilities


async def test_inference_lookup_preserves_stopping_runtime_error():
    async with replicated_audio() as service:
        supervisor = service.supervisor
        workers = real_worker_lookups(supervisor)
        workers[0]._model_uid_to_model_status["custom-rep0"].model_state = "stopping"
        with pytest.raises(RuntimeError) as exc_info:
            await supervisor.get_model("custom")
        assert type(exc_info.value) is RuntimeError
        assert str(exc_info.value) == "Model custom-rep0 is in stopping state"
        descriptor = await supervisor.get_realtime_endpoint(
            "custom", "anonymous", "binary.echo"
        )
        assert descriptor["replica_model_uid"] == "custom-rep1"


async def test_capabilities_preserve_unrelated_worker_error():
    async with replicated_audio() as service:
        supervisor = service.supervisor
        worker = supervisor._replica_model_uid_to_worker["custom-rep0"]
        error = RuntimeError("Unrelated worker failure")
        worker.get_model.side_effect = error
        with pytest.raises(RuntimeError) as exc_info:
            await supervisor.get_model("custom", for_inference=False)
        assert exc_info.value is error
        ready = supervisor._replica_model_uid_to_worker["custom-rep1"]
        ready.get_model.assert_not_awaited()


@pytest.mark.parametrize("state", ["ready", "error"])
async def test_capabilities_surface_a_failed_lookup_that_is_not_a_stop(state):
    async with replicated_audio() as service:
        supervisor = service.supervisor
        workers = real_worker_lookups(supervisor)
        failing, ready = supervisor._replica_model_uid_to_worker.values()
        workers[0]._model_uid_to_model_status["custom-rep0"].model_state = state
        error = RuntimeError("Unrelated worker failure")
        if state == "ready":
            failing.get_model.side_effect = error
        with pytest.raises(RuntimeError) as exc_info:
            await supervisor.get_model("custom", for_inference=False)
        assert str(exc_info.value) == (
            str(error) if state == "ready" else "Model custom-rep0 is in error state"
        )
        ready.get_model.assert_not_awaited()


@pytest.mark.parametrize("replicas", [1, 2])
async def test_capabilities_lookup_timeout_on_stopping_worker(replicas, monkeypatch):
    monkeypatch.setattr("xinference.constants.XINFERENCE_GET_MODEL_RPC_TIMEOUT", 0.05)
    async with replicated_audio(replica_count=replicas) as service:
        supervisor = service.supervisor
        workers = real_worker_lookups(supervisor)
        uid = "custom-rep0"
        await workers[0]._update_model_state(uid, "stopping")
        first = supervisor._replica_model_uid_to_worker[uid]

        async def no_reply(**kwargs):
            await asyncio.Future()

        first.get_model.side_effect = no_reply
        if replicas == 1:
            with pytest.raises(asyncio.TimeoutError):
                await supervisor.get_model("custom", for_inference=False)
            first.get_model_status.assert_not_awaited()
        else:
            model = await supervisor.get_model("custom", for_inference=False)
            assert model.uid == b"custom-rep1"
            first.get_model_status.assert_awaited_once_with(uid)


@pytest.mark.parametrize("unanswered", ["lookup", "status"])
async def test_capabilities_surface_the_error_when_a_worker_does_not_answer(
    unanswered, monkeypatch
):
    monkeypatch.setattr(
        "xinference.core.supervisor.XINFERENCE_GET_MODEL_RPC_TIMEOUT",
        0.05,
        raising=False,
    )
    async with replicated_audio() as service:
        supervisor = service.supervisor
        real_worker_lookups(supervisor)
        failing, ready = supervisor._replica_model_uid_to_worker.values()
        error = (
            asyncio.TimeoutError()
            if unanswered == "lookup"
            else RuntimeError("Unrelated worker failure")
        )
        failing.get_model.side_effect = error
        release = asyncio.Event()

        async def blocked(model_uid):
            await release.wait()

        failing.get_model_status.side_effect = blocked
        try:
            with pytest.raises(type(error)) as exc_info:
                await asyncio.wait_for(
                    supervisor.get_model("custom", for_inference=False), 2
                )
        finally:
            release.set()
        assert exc_info.value is error
        failing.get_model_status.assert_awaited_once_with("custom-rep0")
        ready.get_model.assert_not_awaited()


@pytest.mark.parametrize("draining", [False, True])
async def test_capabilities_keep_existing_error_when_no_replica_is_ready(draining):
    async with replicated_audio() as service:
        supervisor = service.supervisor
        if draining:
            supervisor._replica_autoscalers["custom"] = SimpleNamespace(
                draining={0, 1}, should_avoid=lambda replica: False
            )
        for worker in supervisor._replica_model_uid_to_worker.values():
            worker.get_model.side_effect = ModelNotReadyError("Loading")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(service.app), base_url="http://test"
        ) as client:
            for _ in range(4):
                response = await client.get("/v1/models/custom/audio/capabilities")
                assert response.status_code == 503
                assert response.headers["retry-after"] == "30"
                error = (
                    "Model custom has no available replica"
                    if draining
                    else "Model custom is loading, please retry later"
                )
                assert response.json() == {
                    "detail": f"Model is loading, please retry later: {error}"
                }
        for worker in supervisor._replica_model_uid_to_worker.values():
            assert worker.get_model.await_count == (0 if draining else 4)
            worker.get_model.side_effect = None
        supervisor._replica_autoscalers.clear()
        descriptor = await supervisor.get_realtime_endpoint(
            "custom", "anonymous", "binary.echo"
        )
        assert descriptor["replica_model_uid"] == "custom-rep0"


async def test_capabilities_preserve_concurrent_affinity_cursor():
    async with replicated_audio() as service:
        supervisor = service.supervisor
        worker = supervisor._replica_model_uid_to_worker["custom-rep0"]
        ready_worker = supervisor._replica_model_uid_to_worker["custom-rep1"]
        ready_model = ready_worker.get_model.return_value
        ready_model.can_accept_chat_request = AsyncMock(return_value=True)
        probing, release = asyncio.Event(), asyncio.Event()

        async def loading(**kwargs):
            probing.set()
            await release.wait()
            raise ModelNotReadyError("Loading")

        worker.get_model.side_effect = loading
        lookup = asyncio.create_task(
            supervisor.get_model("custom", routing_key="session")
        )
        try:
            await asyncio.wait_for(probing.wait(), 1)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(service.app), base_url="http://test"
            ) as client:
                response = await client.get("/v1/models/custom/audio/capabilities")
                assert response.status_code == 200
                assert response.json() == service.capabilities
        finally:
            release.set()
            selected = await asyncio.wait_for(lookup, 1)
        assert selected is ready_model
        worker.get_model.side_effect = None
        descriptor = await supervisor.get_realtime_endpoint(
            "custom", "anonymous", "binary.echo"
        )
        assert descriptor["replica_model_uid"] == "custom-rep0"
