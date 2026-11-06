# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

from ..restful_api import RESTfulAPI
from ..routers import models


@pytest.mark.parametrize("authenticated", [True, False])
def test_runtime_route_uses_model_list_permission(authenticated):
    api = MagicMock()
    api.is_authenticated.return_value = authenticated
    models.register_routes(api)
    route = next(
        call
        for call in api._router.add_api_route.call_args_list
        if call.args[0] == "/v1/models/{model_uid:path}/runtime"
    )
    assert route.kwargs["methods"] == ["GET"]
    dependencies = route.kwargs["dependencies"]
    assert (
        dependencies[0].scopes == ["models:list"]
        if authenticated
        else dependencies is None
    )


@pytest.mark.asyncio
async def test_runtime_endpoint_is_separate_from_compatibility_metadata():
    expected = {"model_uid": "demo", "runtime_context_length": 4096, "replicas": []}
    supervisor = SimpleNamespace(
        get_model_runtime_info=AsyncMock(return_value=expected),
        describe_running_model=AsyncMock(
            return_value={"model_name": "demo", "context_length": 131072}
        ),
    )
    api = SimpleNamespace(_get_supervisor_ref=AsyncMock(return_value=supervisor))
    app = FastAPI()
    app.add_api_route(
        "/v1/models/{model_uid:path}/runtime",
        RESTfulAPI.get_model_runtime_info.__get__(api),
    )
    app.add_api_route("/v1/models/{model_uid}", RESTfulAPI.describe_model.__get__(api))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/v1/models/demo/runtime")
        assert response.status_code == 200 and response.json() == expected
        response = await client.get("/v1/models/demo")
        assert response.json() == {"model_name": "demo", "context_length": 131072}
        supervisor.get_model_runtime_info.side_effect = ValueError("Model not found")
        assert (await client.get("/v1/models/missing/runtime")).status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_uid,encoded_uid",
    [
        ("model/name", "model%2Fname"),
        ("org/model/name", "org%2Fmodel%2Fname"),
        ("model%2Fname", "model%252Fname"),
    ],
    ids=["slash", "nested_slash", "literal_percent"],
)
async def test_registered_runtime_route_preserves_encoded_uid(model_uid, encoded_uid):
    expected = {
        "model_uid": model_uid,
        "runtime_context_length": 4096,
        "replicas": [],
    }
    supervisor = SimpleNamespace(
        get_model_runtime_info=AsyncMock(return_value=expected)
    )
    api = MagicMock()
    api.is_authenticated.return_value = False
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    api.get_model_runtime_info = RESTfulAPI.get_model_runtime_info.__get__(api)
    models.register_routes(api)
    route = next(
        call
        for call in api._router.add_api_route.call_args_list
        if call.args[1] is api.get_model_runtime_info
    )
    app = FastAPI()
    app.add_api_route(*route.args, **route.kwargs)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/v1/models/{encoded_uid}/runtime")
    assert response.status_code == 200
    assert response.json() == expected
    supervisor.get_model_runtime_info.assert_awaited_once_with(model_uid)
