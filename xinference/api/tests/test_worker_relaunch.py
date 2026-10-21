# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from ..routers import admin


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"", b"{}", b'{"request_id":"one"}'])
async def test_relaunch_returns_pollable_operation(body):
    supervisor = SimpleNamespace(
        relaunch_worker=AsyncMock(
            return_value={"operation_id": "one", "status": "starting"}
        )
    )
    api = SimpleNamespace(_get_supervisor_ref=AsyncMock(return_value=supervisor))
    response = await admin.relaunch_worker(
        "127.0.0.1:9978", SimpleNamespace(body=AsyncMock(return_value=body)), api
    )
    assert response.status_code == 202
    assert json.loads(response.body)["operation_id"] == "one"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b"[]",
        b'{"unknown":1}',
        b'{"python_executable":"python"}',
        b'{"request_id":true}',
        b"{",
    ],
)
async def test_invalid_request_never_starts_worker(body):
    api = SimpleNamespace(_get_supervisor_ref=AsyncMock())
    with pytest.raises(HTTPException) as error:
        await admin.relaunch_worker(
            "old", SimpleNamespace(body=AsyncMock(return_value=body)), api
        )
    assert error.value.status_code == 400
    api._get_supervisor_ref.assert_not_called()


@pytest.mark.asyncio
async def test_missing_operation_and_conflicting_request():
    supervisor = SimpleNamespace(
        get_worker_relaunch=AsyncMock(side_effect=KeyError("missing")),
        relaunch_worker=AsyncMock(side_effect=RuntimeError("already running")),
    )
    api = SimpleNamespace(_get_supervisor_ref=AsyncMock(return_value=supervisor))
    with pytest.raises(HTTPException) as error:
        await admin.get_worker_relaunch("missing", api)
    assert error.value.status_code == 404
    with pytest.raises(HTTPException) as error:
        await admin.relaunch_worker(
            "old", SimpleNamespace(body=AsyncMock(return_value=b"{}")), api
        )
    assert error.value.status_code == 409


@pytest.mark.parametrize("authenticated", [False, True])
def test_worker_relaunch_routes_require_admin_scope(authenticated):
    api = MagicMock()
    api.is_authenticated.return_value = authenticated
    admin.register_routes(api)
    routes = {
        call.args[0]: call.kwargs for call in api._router.add_api_route.call_args_list
    }
    for path in (
        "/v1/workers/{worker_address}/relaunch",
        "/v1/worker-relaunches",
        "/v1/worker-relaunches/{operation_id}",
    ):
        dependencies = routes[path]["dependencies"]
        assert (
            [dependency.scopes for dependency in dependencies] if dependencies else []
        ) == ([["admin"]] if authenticated else [])
