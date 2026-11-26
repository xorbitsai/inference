# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from ...deploy import api_host
from ..oauth2.advanced.auth_service import AdvancedAuthService
from ..oauth2.advanced.crypto import get_password_hash
from ..replica_drain import ReplicaDrainMiddleware, ReplicaDrainRequests
from ..restful_api import RESTfulAPI
from ..routers import admin


@pytest.fixture(params=["api", "supervisor"])
def endpoint(monkeypatch, request):
    call = AsyncMock(return_value={"operation_id": "one", "status": "starting"})
    monkeypatch.setattr(api_host, "host_request", call)
    app = FastAPI()
    api = SimpleNamespace(
        _router=APIRouter(), _auth_service=None, is_authenticated=lambda: False
    )
    app.state.api = api
    admin.register_routes(api)
    app.include_router(api._router)
    return TestClient(app), call, request.param


@pytest.mark.parametrize(
    "body",
    [
        [],
        {"unexpected": 1},
        {"python_executable": "python"},
        {"request_id": True},
        {"version": True},
        {"version": "4.2rc1"},
        {"version": "4.2", "python_executable": "/python"},
    ],
)
def test_invalid_api_relaunch_does_not_execute_host_command(endpoint, body):
    client, call, component = endpoint
    assert client.post(f"/v1/{component}/relaunch", json=body).status_code == 400
    call.assert_not_called()


def test_operation_routes_and_errors(endpoint):
    client, call, component = endpoint
    assert (
        client.post(f"/v1/{component}/relaunch", json={"request_id": "one"}).status_code
        == 202
    )
    call.assert_called_once_with(
        "relaunch" if component == "api" else "supervisor-relaunch", request_id="one"
    )
    assert client.get(f"/v1/{component}-relaunches/one").json()["operation_id"] == "one"
    assert client.get(f"/v1/{component}-relaunches").status_code == 200
    call.side_effect = KeyError("missing")
    assert client.get(f"/v1/{component}-relaunches/missing").status_code == 404
    call.side_effect = RuntimeError(
        "API relaunch requires a hosted xinference-local or xinference-supervisor service"
    )
    assert client.post(f"/v1/{component}/relaunch").status_code == 409


def test_package_update_uses_component_host_and_existing_operation_routes(endpoint):
    client, call, component = endpoint
    assert client.get(f"/v1/{component}/updates").status_code == 200
    call.assert_called_once_with(
        "updates" if component == "api" else "supervisor-updates"
    )
    call.reset_mock()
    assert (
        client.post(f"/v1/{component}/relaunch", json={"version": "4.2.0"}).status_code
        == 202
    )
    call.assert_called_once_with(
        "relaunch" if component == "api" else "supervisor-relaunch", version="4.2.0"
    )


@pytest.mark.parametrize("component", ["api", "supervisor"])
@pytest.mark.parametrize("identity", ["administrator", "user", "admin-api-key"])
def test_every_route_requires_an_administrator_session(
    tmp_path, monkeypatch, identity, component
):
    auth = AdvancedAuthService(str(tmp_path / "auth.db"), "secret", "encryption")
    permissions = ["admin"] if identity != "user" else ["models:write"]
    uid = auth.db.create_user(
        "alice", get_password_hash("password"), permissions=permissions
    )
    if identity == "admin-api-key":
        token = auth.create_api_key_for_user(uid, name="key")["key"]
    else:
        token = auth.login("alice", "password")["access_token"]
    call = AsyncMock(return_value={"operation_id": "one"})
    monkeypatch.setattr(api_host, "host_request", call)
    app = FastAPI()
    api = SimpleNamespace(
        _router=APIRouter(), _auth_service=auth, is_authenticated=lambda: True
    )
    app.state.api = api
    admin.register_routes(api)
    app.include_router(api._router)
    client = TestClient(app, headers={"Authorization": "Bearer " + token})
    for method, path in [
        ("GET", f"/v1/{component}/updates"),
        ("POST", f"/v1/{component}/relaunch"),
        ("GET", f"/v1/{component}-relaunches"),
        ("GET", f"/v1/{component}-relaunches/one"),
    ]:
        response = client.request(method, path)
        expected = (
            (202 if method == "POST" else 200) if identity == "administrator" else 403
        )
        assert response.status_code == expected, response.text
    if identity != "administrator":
        call.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_at", ["request_body", "host_admission"])
async def test_api_relaunch_and_replica_drain_admission(monkeypatch, pause_at):
    release = asyncio.Event()
    relaunch_paused = asyncio.Event()
    drain_paused = asyncio.Event()

    async def request_body():
        if pause_at == "request_body":
            relaunch_paused.set()
            await release.wait()
        yield b'{"request_id": "api-restart"}'

    async def host_request(command, **options):
        if command == "list":
            return []
        assert command == "relaunch"
        if pause_at == "host_admission":
            relaunch_paused.set()
            await release.wait()
        return {"operation_id": "one", "status": "starting"}

    async def begin_drain(*args):
        drain_paused.set()
        if pause_at == "request_body":
            await finish_drain.wait()
        return {"status": "draining"}

    finish_drain = asyncio.Event()
    monkeypatch.setenv(api_host.API_HOST_ENV, "hosted")
    host = AsyncMock(side_effect=host_request)
    monkeypatch.setattr(api_host, "host_request", host)
    supervisor = SimpleNamespace(
        begin_manual_replica_drain=AsyncMock(side_effect=begin_drain),
        confirm_manual_replica_drain=AsyncMock(return_value=True),
        cancel_manual_replica_drain=AsyncMock(),
    )
    api = RESTfulAPI.__new__(RESTfulAPI)
    api._replica_drain_requests = ReplicaDrainRequests()
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    app = FastAPI()
    app.state.api = api
    app.add_api_route("/v1/api/relaunch", admin.relaunch_api, methods=["POST"])
    app.add_api_route(
        "/v1/models/{model_uid}/replicas/{replica_id}/drain",
        api.begin_model_replica_drain,
        methods=["POST"],
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        relaunch = asyncio.create_task(
            client.post("/v1/api/relaunch", content=request_body())
        )
        tasks = [relaunch]
        try:
            await asyncio.wait_for(relaunch_paused.wait(), timeout=2)
            drain_url = "/v1/models/demo/replicas/1/drain?request_id=r1"
            if pause_at == "request_body":
                drain = asyncio.create_task(client.post(drain_url))
                tasks.append(drain)
                await asyncio.wait_for(drain_paused.wait(), timeout=2)
                release.set()
                assert (await relaunch).status_code == 409
                host.assert_awaited_once_with("list", require_single_api=True)
                finish_drain.set()
                assert (await drain).json()["status"] == "drained"
            else:
                assert (await client.post(drain_url)).status_code == 409
                supervisor.begin_manual_replica_drain.assert_not_awaited()
                assert not api._replica_drain_requests.windows
                release.set()
                assert (await relaunch).status_code == 202
        finally:
            release.set()
            finish_drain.set()
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_api_replacement_must_finish_completion_before_replica_drain(
    monkeypatch,
):
    activated = asyncio.Event()
    admitted = asyncio.Event()
    finish_completion = asyncio.Event()

    async def generate(*args, **kwargs):
        admitted.set()
        await finish_completion.wait()
        return b'{"choices": [{"text": "finished"}]}'

    target_replica = SimpleNamespace(
        uid=b"demo-rep1",
        is_vllm_backend=AsyncMock(return_value=False),
        generate=generate,
    )
    supervisor = SimpleNamespace(
        get_model=AsyncMock(return_value=target_replica),
        describe_model=AsyncMock(return_value={"model_engine": "vllm"}),
        begin_manual_replica_drain=AsyncMock(return_value={"status": "draining"}),
        confirm_manual_replica_drain=AsyncMock(return_value=True),
        terminate_drained_model_replica=AsyncMock(return_value=1),
    )

    def make_app():
        api = RESTfulAPI.__new__(RESTfulAPI)
        api._advanced_auth_service = None
        api._get_supervisor_ref = AsyncMock(return_value=supervisor)
        api._replica_drain_requests = ReplicaDrainRequests()
        app = api._app = FastAPI()
        app.state.api = api
        app.add_middleware(ReplicaDrainMiddleware, tracker=api._replica_drain_requests)
        app.add_api_route("/v1/api/relaunch", admin.relaunch_api, methods=["POST"])
        app.add_api_route("/v1/completions", api.create_completion, methods=["POST"])
        replica_path = "/v1/models/{model_uid}/replicas/{replica_id}"
        app.add_api_route(
            replica_path + "/drain", api.begin_model_replica_drain, methods=["POST"]
        )
        app.add_api_route(replica_path, api.terminate_model_replica, methods=["DELETE"])
        return app

    async def activate(command):
        assert command == "activate"
        activated.set()
        await asyncio.wait_for(admitted.wait(), timeout=2)

    host = api_host.APIHost(MagicMock(), {}, MagicMock())
    loop = asyncio.get_running_loop()
    source, replacement = (
        api_host.APIProcess(
            name,
            MagicMock(pid=pid, **{"poll.return_value": None}),
            sys.executable,
            {},
            api_host.APIDiagnostics(),
            loop.create_future(),
            writer=MagicMock(),
            version="4.2.0",
        )
        for name, pid in (("source", 1), ("replacement", 2))
    )
    host.active = source
    host.children[source.identity] = source

    async def spawn(executable, environment):
        host.children[replacement.identity] = replacement
        return replacement

    host.spawn = spawn
    replacement.send = activate
    replacement.wait = AsyncMock(
        side_effect=[None, RuntimeError("Serving acknowledgement failed")]
    )

    server = await asyncio.start_server(host._connection, "127.0.0.1", 0)
    monkeypatch.setenv(
        api_host.API_HOST_ENV,
        json.dumps({"port": server.sockets[0].getsockname()[1], "token": host.token}),
    )
    async with (
        server,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=make_app()), base_url="http://source"
        ) as source_client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=make_app()), base_url="http://replacement"
        ) as replacement_client,
    ):

        async def complete():
            await activated.wait()
            return await replacement_client.post(
                "/v1/completions", json={"model": "demo", "prompt": "continue"}
            )

        completion = asyncio.create_task(complete())
        try:
            response = await source_client.post("/v1/api/relaunch")
            assert response.status_code == 202
            await asyncio.wait_for(host.operation_task, timeout=3)
            operation = host.operations[response.json()["operation_id"]]
            assert operation["status"] == "failed"
            replacement.writer.close.assert_called_once()
            assert admitted.is_set() and not completion.done()

            replica_url = "/v1/models/demo/replicas/1"
            drain_url = replica_url + "/drain?request_id=r1"
            assert (await source_client.post(drain_url)).status_code == 409
            supervisor.begin_manual_replica_drain.assert_not_awaited()

            finish_completion.set()
            response = await asyncio.wait_for(completion, timeout=2)
            assert response.status_code == 200
            assert response.json()["choices"][0]["text"] == "finished"
            assert (await source_client.post(drain_url)).status_code == 409
            replacement.process.poll.return_value = 0
            replacement.exited.set_result(0)
            assert (await source_client.post(drain_url)).json()["status"] == "drained"
            response = await source_client.delete(replica_url + "?drain_id=r1")
            assert response.json() == {"remaining_replicas": 1}
        finally:
            finish_completion.set()
            if not activated.is_set():
                completion.cancel()
            await asyncio.gather(completion, return_exceptions=True)


@pytest.mark.asyncio
async def test_unmanaged_api_cannot_start_a_replacement(monkeypatch):
    monkeypatch.delenv(api_host.API_HOST_ENV, raising=False)
    with pytest.raises(
        RuntimeError, match="hosted xinference-local or xinference-supervisor"
    ):
        await api_host.host_request("relaunch")
