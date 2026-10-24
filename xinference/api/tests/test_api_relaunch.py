# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from ...deploy import api_host
from ..oauth2.advanced.auth_service import AdvancedAuthService
from ..oauth2.advanced.crypto import get_password_hash
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
    [[], {"unexpected": 1}, {"python_executable": "python"}, {"request_id": True}],
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
        "API relaunch requires a POSIX xinference-local or xinference-supervisor service"
    )
    assert client.post(f"/v1/{component}/relaunch").status_code == 409


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
async def test_unmanaged_api_cannot_start_a_replacement(monkeypatch):
    monkeypatch.delenv(api_host.API_HOST_ENV, raising=False)
    with pytest.raises(
        RuntimeError, match="POSIX xinference-local or xinference-supervisor"
    ):
        await api_host.host_request("relaunch")
