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
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest
from fastapi import APIRouter, FastAPI, HTTPException
from starlette.requests import Request

from ...core.tests.test_realtime_runtime import (
    CONFIG,
    FakeModel,
    RealtimeRuntime,
    wait_empty,
)
from ..routers.realtime import register_routes


@asynccontextmanager
async def serve_gateway(authenticated=False, backend_model=None, api_instance=None):
    import socket

    import uvicorn

    model = backend_model or FakeModel()
    if backend_model is None:
        model.load()
    runtime = RealtimeRuntime(model, "fake", "fake-0")
    await runtime.start("127.0.0.1")
    actor = SimpleNamespace(
        get_realtime_endpoint=AsyncMock(side_effect=runtime.issue_ticket)
    )
    supervisor = SimpleNamespace(
        describe_model=AsyncMock(return_value={"model_type": "audio"}),
        get_model=AsyncMock(return_value=actor),
    )
    api = SimpleNamespace(
        _router=APIRouter(),
        _get_supervisor_ref=AsyncMock(return_value=supervisor),
        is_authenticated=lambda: authenticated,
        _check_model_access=lambda *args: None,
        _is_ip_allowed=lambda ip: True,
        _auth_service=AsyncMock(return_value={"id": 7}),
    )
    if api_instance is None:
        register_routes(api)
        app = FastAPI()
        app.include_router(api._router)
    else:
        api = api_instance
        api._get_supervisor_ref = AsyncMock(return_value=supervisor)
        api.serve(server_factory=lambda config: SimpleNamespace(run=lambda: None))
        app = api._app
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical", ws="websockets"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        if task.done():
            await task
        await asyncio.sleep(0.01)
    try:
        yield SimpleNamespace(
            api=api,
            runtime=runtime,
            actor=actor,
            supervisor=supervisor,
            url=f"ws://127.0.0.1:{sock.getsockname()[1]}/v1/models/fake/realtime",
        )
    finally:
        server.should_exit = True
        await task
        await runtime.stop()


@pytest.mark.parametrize("task_name", ["binary.echo", "audio.transcription"])
async def test_gateway_common_protocol(task_name):
    async with serve_gateway() as service:
        async with aiohttp.ClientSession() as client:
            ws = await client.ws_connect(service.url)
            await ws.send_json({**CONFIG, "task": task_name})
            assert (await ws.receive_json())["event"] == "session.started"
            await ws.send_bytes(b"1234")
            assert (await ws.receive_json())["data"]["received_bytes"] == 4
            await ws.send_json({"event": "input.end"})
            assert (await ws.receive_json())["data"]["done"]
            assert (await ws.receive_json())["event"] == "session.closed"
            await ws.close()
        await wait_empty(service.runtime)
        service.supervisor.get_model.assert_awaited_once_with("fake")
        service.actor.get_realtime_endpoint.assert_awaited_once()


async def test_auth_reuses_http_policy_and_model_permissions():
    async with serve_gateway(authenticated=True) as service:
        async with aiohttp.ClientSession() as client:
            ws = await client.ws_connect(
                service.url, headers={"Authorization": "Bearer secret"}
            )
            await ws.send_json(CONFIG)
            await ws.receive_json()
            call = service.api._auth_service.call_args.kwargs
            assert isinstance(call["request"], Request)
            assert call["request"].method == "GET"
            assert call["token"] == "secret"
            assert call["security_scopes"].scopes == ["models:read"]
            await ws.close()
            service.actor.get_realtime_endpoint.assert_awaited_once_with(
                "7", "binary.echo"
            )


@pytest.mark.parametrize("failure", ["auth", "permission", "origin"])
async def test_reject_before_routing(failure, monkeypatch):
    monkeypatch.delenv("XINFERENCE_REALTIME_ALLOWED_ORIGINS", raising=False)
    async with serve_gateway(authenticated=True) as service:
        if failure == "auth":
            service.api._auth_service.side_effect = HTTPException(401, "denied")
        if failure == "permission":

            def deny(*args):
                raise HTTPException(403, "denied")

            service.api._check_model_access = deny
        headers = {"Origin": "https://evil.invalid"} if failure == "origin" else {}
        async with aiohttp.ClientSession() as client:
            with pytest.raises(aiohttp.WSServerHandshakeError):
                await client.ws_connect(service.url, headers=headers)
        service.supervisor.get_model.assert_not_awaited()


async def test_gateway_disconnect_releases_runtime():
    async with serve_gateway() as service:
        async with aiohttp.ClientSession() as client:
            ws = await client.ws_connect(service.url)
            await ws.send_json(CONFIG)
            await ws.receive_json()
            await ws.close()
        await wait_empty(service.runtime)


@pytest.mark.parametrize("allowed", [False, True])
async def test_ip_allowlist_through_real_api_stack(monkeypatch, tmp_path, allowed):
    from ...core import system_settings_store
    from .. import restful_api

    monkeypatch.setattr(restful_api, "is_auth_advanced", lambda: False)
    monkeypatch.setattr(restful_api, "is_metrics_disabled", lambda: True)
    monkeypatch.setenv("XINFERENCE_GUARD_ENABLED", "false")
    for name, filename in (
        ("XINFERENCE_AUTH_DB_PATH", "auth.db"),
        ("XINFERENCE_LAUNCH_HISTORY_DB_PATH", "history.db"),
        ("XINFERENCE_MONITOR_CONFIG_DB_PATH", "monitor.db"),
        ("XINFERENCE_SYSTEM_SETTINGS_PATH", "settings.json"),
    ):
        monkeypatch.setattr(restful_api, name, str(tmp_path / filename))
    monkeypatch.setattr(restful_api.RESTfulAPI, "_allowed_ip_list", None)
    monkeypatch.setattr(system_settings_store, "_store", None)
    monkeypatch.setattr(
        restful_api,
        "XINFERENCE_ALLOWED_IPS",
        "127.0.0.0/8" if allowed else "192.0.2.0/24",
    )
    api = restful_api.RESTfulAPI("127.0.0.1:1", "127.0.0.1", 0)
    async with serve_gateway(api_instance=api) as service:
        async with aiohttp.ClientSession() as client:
            # Use the HTTP middleware as a control, not a simplified route-only app.
            http_url = service.url.replace("ws://", "http://").split("/v1/")[0]
            async with client.get(http_url + "/docs") as response:
                assert response.status == (200 if allowed else 403)
            if allowed:
                ws = await client.ws_connect(service.url)
                await ws.send_json(CONFIG)
                assert (await ws.receive_json())["event"] == "session.started"
                await ws.close()
                await wait_empty(service.runtime)
            else:
                with pytest.raises(aiohttp.WSServerHandshakeError) as error:
                    await client.ws_connect(service.url)
                assert error.value.status == 403
                service.api._get_supervisor_ref.assert_not_awaited()
                service.supervisor.describe_model.assert_not_awaited()
                service.supervisor.get_model.assert_not_awaited()
                assert service.runtime.active_count == 0
