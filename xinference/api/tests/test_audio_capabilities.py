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
async def replicated_audio():
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._pd_model_mapping = {}
    supervisor._replica_autoscalers = {}
    supervisor._model_uid_to_replica_info = {
        "custom": supervisor._build_replica_info(2)
    }
    supervisor._replica_model_uid_to_worker = {}
    runtimes = []
    capabilities = {"asr": {"streaming": True, "server_vad": True}}
    try:
        for replica in range(2):
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
        worker._status_guard_ref = None
        worker_ref.get_model.side_effect = worker.get_model
        workers.append(worker)
    return workers


@pytest.mark.parametrize("outcome", ["success", "failure", "suppressed", "cancelled"])
async def test_capabilities_skip_stopping_replica_without_advancing_scheduler(outcome):
    async with replicated_audio() as service:
        supervisor = service.supervisor
        supervisor._model_replica_locks = {}
        supervisor._model_reload_tasks = {}
        supervisor._replica_model_uid_to_worker_shards = {}
        supervisor._status_guard_ref = SimpleNamespace(
            remove_replica_status=AsyncMock(), update_instance_info=AsyncMock()
        )
        supervisor._clear_replica_model_gpu_memory = Mock()
        supervisor._unexpected_down_replicas = {}
        supervisor._invalidate_list_models_debounce_cache = Mock()
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
            assert info.terminating == {0}
            stopping.get_model.assert_not_awaited()
            ready = supervisor._replica_model_uid_to_worker["custom-rep1"]
            assert ready.get_model.await_count == 4
            workers[0]._model_uid_to_model_status["custom-rep0"].model_state = "ready"
            assert (await supervisor.get_model("custom")).uid == b"custom-rep0"
        finally:
            if outcome == "cancelled":
                termination.cancel()
            release.set()
            if outcome in ("failure", "cancelled"):
                expected = (
                    RuntimeError if outcome == "failure" else asyncio.CancelledError
                )
                with pytest.raises(expected):
                    await termination
            else:
                await termination
        assert not info.terminating


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
