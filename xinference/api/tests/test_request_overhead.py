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
import ipaddress
import subprocess
import sys
from unittest.mock import Mock

import httpx
import pytest
from fastapi import APIRouter, FastAPI, Request, Response
from starlette.responses import StreamingResponse

from xinference.api import restful_api
from xinference.api.restful_api import RESTfulAPI


@pytest.fixture
def api(monkeypatch):
    api = RESTfulAPI.__new__(RESTfulAPI)
    api._app = FastAPI()
    api._router = APIRouter()
    api._advanced_auth_service = None
    api._monitor_config_store = None
    api._system_settings_store = None
    api._host, api._port = "127.0.0.1", 0
    api._record_admin_audit = Mock()
    api._record_audit = Mock()
    monkeypatch.setattr(RESTfulAPI, "_allowed_ip_list", [])
    monkeypatch.setattr(restful_api, "Server", Mock())
    monkeypatch.setattr(restful_api, "mount_frontend", lambda *a: True)
    monkeypatch.setattr(restful_api, "is_metrics_disabled", lambda: True)
    monkeypatch.setattr(restful_api, "XINFERENCE_ENABLE_OTEL", False)
    monkeypatch.setattr("xinference.api.routers.register_all_routes", lambda api: None)
    api.serve()
    return api


@pytest.mark.asyncio
@pytest.mark.parametrize("audit", [False, True])
@pytest.mark.parametrize("allowed", [False, True])
async def test_ip_decision_request_id_and_audit(api, monkeypatch, audit, allowed):
    api._advanced_auth_service = object() if audit else None
    monkeypatch.setattr(
        RESTfulAPI,
        "_allowed_ip_list",
        [ipaddress.ip_network("127.0.0.0/8" if allowed else "192.0.2.0/24")],
    )
    reached = []

    @api._app.get("/v1/models")
    async def endpoint():
        reached.append(True)
        return Response("ok")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api._app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/v1/models", headers={"X-Request-ID": "test-request"}
        )
    assert response.status_code == (200 if allowed else 403)
    assert response.headers["X-Request-ID"] == "test-request"
    assert bool(reached) == allowed
    assert api._record_admin_audit.call_count == int(audit)
    if audit:
        call = api._record_admin_audit.call_args
        assert call.args[1] == ("success" if allowed else "error")
        assert call.kwargs["status_code"] == response.status_code


@pytest.mark.asyncio
async def test_streaming_and_already_recorded_audit(api):
    api._advanced_auth_service = object()
    closed = []

    @api._app.get("/v1/chat/completions")
    async def stream(request: Request):
        request.state.audit_recorded = True

        async def chunks():
            try:
                yield b"data: first\n\n"
                await asyncio.sleep(0)
                yield b"data: second\n\n"
            finally:
                closed.append(True)

        return StreamingResponse(chunks(), media_type="text/event-stream")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api._app), base_url="http://test"
    ) as client:
        response = await client.get("/v1/chat/completions")
    assert response.text == "data: first\n\ndata: second\n\n"
    assert response.headers["X-Request-ID"]
    assert closed == [True]
    api._record_admin_audit.assert_not_called()


@pytest.mark.asyncio
async def test_endpoint_exception_is_still_audited(api):
    api._advanced_auth_service = object()

    @api._app.get("/v1/models")
    async def fail():
        raise ValueError("test failure")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api._app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.get("/v1/models")
    assert response.status_code == 500
    api._record_admin_audit.assert_called_once()
    assert api._record_admin_audit.call_args.kwargs["status_code"] == 500


@pytest.mark.asyncio
async def test_stream_disconnect_closes_generator(api):
    first_chunk = asyncio.Event()
    closed = asyncio.Event()

    @api._app.get("/stream")
    async def stream():
        async def chunks():
            try:
                yield b"first"
                await asyncio.Event().wait()
            finally:
                closed.set()

        return StreamingResponse(chunks())

    sent_request = False

    async def receive():
        nonlocal sent_request
        if not sent_request:
            sent_request = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await first_chunk.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            first_chunk.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/stream",
        "raw_path": b"/stream",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 123),
        "server": ("test", 80),
    }
    await asyncio.wait_for(api._app(scope, receive, send), 2)
    assert first_chunk.is_set() and closed.is_set()


@pytest.mark.parametrize("mode", ["enabled", "disabled", "already-frozen"])
def test_lifespan_collects_new_cycles_and_restores_gc_state(mode):
    # GC freeze is process-wide: exercise the real collector in a subprocess
    # instead of changing the test runner's heap or mocking collection away.
    script = r"""
import asyncio, gc, sys, weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock
from xinference.api import restful_api

restful_api.is_metrics_disabled = lambda: True
mode = sys.argv[1]
if mode == "disabled":
    gc.disable()
if mode == "already-frozen":
    gc.collect()
    gc.freeze()
was_enabled, frozen_before = gc.isenabled(), gc.get_freeze_count()
api = SimpleNamespace(_cluster_metrics_task=None,
    _close_elasticsearch_client=AsyncMock(), _close_token_router_client=AsyncMock())
class Cycle:
    pass
async def run():
    try:
        async with restful_api.RESTfulAPI._lifespan(api, None):
            assert gc.get_freeze_count() > 0
            assert gc.isenabled() == was_enabled
            cycle = Cycle()
            cycle.self = cycle
            ref = weakref.ref(cycle)
            del cycle
            gc.collect()
            assert ref() is None, "request cycles must remain collectible"
            raise RuntimeError("shutdown after failure")
    except RuntimeError:
        pass
    assert gc.isenabled() == was_enabled
    assert (gc.get_freeze_count() > 0) == (frozen_before > 0)
    api._close_elasticsearch_client.assert_awaited_once()
    api._close_token_router_client.assert_awaited_once()
asyncio.run(run())
"""
    result = subprocess.run(
        [sys.executable, "-c", script, mode], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
