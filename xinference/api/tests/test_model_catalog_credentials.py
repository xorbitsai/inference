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

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import APIRouter, FastAPI, HTTPException
from starlette.exceptions import HTTPException as StarletteHTTPException
from xoscar.serialization import deserialize, serialize

from xinference import extensions
from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService
from xinference.api.restful_api import RESTfulAPI
from xinference.api.routers import models
from xinference.core.exceptions import (
    ModelCatalogUnavailableError,
    ModelCatalogUpdateError,
)
from xinference.extensions import ModelCatalogCredentials

TOKEN = "test-catalog-secret"


@pytest.fixture
def api(tmp_path, monkeypatch):
    api = RESTfulAPI.__new__(RESTfulAPI)
    api._app = FastAPI()
    api._router = APIRouter()
    service = AdvancedAuthService(
        db_path=str(tmp_path / "auth.db"),
        jwt_secret_key="catalog-test-jwt-secret-key-32-chars",
        encryption_key="catalog-test-encryption-key",
    )
    api._auth_service = api._advanced_auth_service = service
    api._get_supervisor_ref = AsyncMock(
        return_value=SimpleNamespace(update_model_type=AsyncMock())
    )
    hook = Mock(return_value=ModelCatalogCredentials(TOKEN))
    monkeypatch.setattr(
        extensions,
        "get_extensions",
        lambda: (SimpleNamespace(name="catalog", model_catalog_credentials=hook),),
    )
    models.register_routes(api)
    api._app.include_router(api._router)
    return api, hook


def bearer(api, permissions):
    name = "user-" + str(len(permissions))
    user_id = api._auth_service.db.create_user(
        name, None, permissions=permissions, must_change_password=0
    )
    token = api._auth_service.create_access_token(user_id, name, permissions)
    return {"Authorization": "Bearer " + token}


async def update(api, headers, *, single=False, **request_kwargs):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api._app), base_url="http://test"
    ) as client:
        return await client.post(
            "/v1/models/update_spec" if single else "/v1/models/update_type",
            headers=headers,
            **(
                request_kwargs
                or {
                    "json": {
                        "model_type": "llm",
                        **({"model_name": "wanted"} if single else {}),
                    }
                }
            ),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("permissions, status", [(None, 401), (["models:read"], 403)])
async def test_authentication_and_scope_checks_precede_provider(
    api, permissions, status, single
):
    runtime, hook = api
    headers = bearer(runtime, permissions) if permissions is not None else {}
    result = await update(runtime, headers, single=single)
    assert result.status_code == status
    hook.assert_not_called()
    runtime._get_supervisor_ref.assert_not_awaited()


@pytest.mark.asyncio
async def test_authenticated_provider_and_unchanged_success_response(api):
    runtime, hook = api
    result = await update(runtime, bearer(runtime, ["models:register"]))
    assert result.status_code == 200
    assert result.json() == {
        "data": {"model_type": "llm", "message": "Successfully updated model type: llm"}
    }
    request, model_type = hook.call_args.args
    assert request.headers["authorization"].startswith("Bearer ")
    assert model_type == "llm"
    supervisor = await runtime._get_supervisor_ref()
    supervisor.update_model_type.assert_awaited_once_with(
        "llm", credentials=ModelCatalogCredentials(TOKEN)
    )
    assert TOKEN not in result.text


@pytest.mark.asyncio
async def test_anonymous_refresh_without_extension_keeps_rpc_shape(api, monkeypatch):
    runtime, _ = api
    runtime._advanced_auth_service = None
    runtime._router = APIRouter()
    runtime._app = FastAPI()
    models.register_routes(runtime)
    runtime._app.include_router(runtime._router)
    monkeypatch.setattr(extensions, "get_extensions", lambda: ())
    result = await update(runtime, {})
    assert result.status_code == 200
    supervisor = await runtime._get_supervisor_ref()
    supervisor.update_model_type.assert_awaited_once_with("llm")


@pytest.mark.asyncio
@pytest.mark.parametrize("exception_type", [HTTPException, StarletteHTTPException])
@pytest.mark.parametrize("status", [401, 403])
async def test_provider_denial_preserves_http_status_headers_and_never_calls_supervisor(
    api, exception_type, status
):
    runtime, hook = api
    hook.side_effect = exception_type(
        status, "Catalog access denied", headers={"WWW-Authenticate": "Bearer"}
    )
    result = await update(runtime, bearer(runtime, ["models:register"]))
    assert result.status_code == status
    assert result.json() == {"detail": "Catalog access denied"}
    assert result.headers["WWW-Authenticate"] == "Bearer"
    runtime._get_supervisor_ref.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"content": '{"model_type":', "headers": {"Content-Type": "application/json"}},
        {"json": {}},
        {"json": None},
        {"json": [TOKEN]},
        {"json": {"model_type": []}},
    ],
)
async def test_invalid_body_is_400_before_provider_or_rpc(api, caplog, request_kwargs):
    runtime, hook = api
    caplog.set_level(logging.DEBUG, logger="xinference.api.restful_api")
    headers = bearer(runtime, ["models:register"])
    headers.update(request_kwargs.get("headers", {}))
    payload = {k: v for k, v in request_kwargs.items() if k != "headers"}
    result = await update(runtime, headers, **payload)
    assert result.status_code == 400
    assert result.json() == {"detail": "Invalid request body"}
    assert TOKEN not in result.text
    assert TOKEN not in caplog.text
    hook.assert_not_called()
    runtime._get_supervisor_ref.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("credentialed", [False, True])
async def test_single_model_refresh_uses_the_existing_provider_and_permission_path(
    api, monkeypatch, credentialed
):
    runtime, hook = api
    if not credentialed:
        monkeypatch.setattr(extensions, "get_extensions", lambda: ())
    result = await update(
        runtime,
        bearer(runtime, ["models:register"]),
        single=True,
        json={"model_type": "llm", "model_name": "wanted"},
    )
    assert result.status_code == 200
    assert result.json() == {
        "data": {
            "model_type": "llm",
            "model_name": "wanted",
            "message": "Successfully updated model: wanted",
        }
    }
    expected: dict = {"model_name": "wanted"}
    if credentialed:
        expected["credentials"] = ModelCatalogCredentials(TOKEN)
        hook.assert_called_once()
    supervisor = await runtime._get_supervisor_ref()
    supervisor.update_model_type.assert_awaited_once_with("llm", **expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name", [None, "", " padded", "a\nb", "a\x7fb", 42, False, [], "a" * 513]
)
async def test_single_model_name_is_required_and_validated_before_credentials(
    api, name
):
    runtime, hook = api
    result = await update(
        runtime,
        bearer(runtime, ["models:register"]),
        single=True,
        json={"model_type": "llm", "model_name": name},
    )
    assert result.status_code == 400
    assert result.json() == {"detail": "Invalid request body"}
    hook.assert_not_called()
    runtime._get_supervisor_ref.assert_not_awaited()


@pytest.mark.asyncio
async def test_single_model_name_cannot_be_omitted(api):
    runtime, hook = api
    result = await update(
        runtime,
        bearer(runtime, ["models:register"]),
        single=True,
        json={"model_type": "llm"},
    )
    assert result.status_code == 400
    hook.assert_not_called()
    runtime._get_supervisor_ref.assert_not_awaited()


@pytest.mark.asyncio
async def test_single_model_api_reaches_supervisor_worker_and_persisted_catalog(
    api, tmp_path, monkeypatch
):
    import json
    from functools import partial

    import requests

    from xinference.core.model_catalog import store_model_catalog
    from xinference.core.supervisor import SupervisorActor
    from xinference.core.worker import WorkerActor

    runtime, _ = api

    def record(name, timestamp):
        return {
            "version": 2,
            "model_name": name,
            "context_length": 4096,
            "model_lang": ["en"],
            "model_ability": ["generate"],
            "updated_at": timestamp,
            "model_specs": [
                {
                    "model_format": "pytorch",
                    "model_size_in_billions": 1,
                    "model_src": {
                        "huggingface": {
                            "model_id": "org/demo",
                            "model_revision": "main",
                            "quantizations": ["none"],
                        }
                    },
                }
            ],
        }

    monkeypatch.setattr("xinference.constants.XINFERENCE_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr("xinference.model.llm.register_builtin_model", Mock())
    path = tmp_path / "v2" / "builtin" / "llm" / "llm_models.json"
    keep, fresh = record("keep", 1), record("wanted", 2)
    store_model_catalog(path, [keep, record("wanted", 1)])
    worker = SimpleNamespace(_custom_register_type_to_cls={"LLM": None})
    worker._store_complete_model_configurations = partial(
        WorkerActor._store_complete_model_configurations, worker
    )
    supervisor = SupervisorActor()
    supervisor._worker_address_to_worker = {
        "worker": SimpleNamespace(
            update_model_type=partial(WorkerActor.update_model_type, worker)
        )
    }
    runtime._get_supervisor_ref = AsyncMock(return_value=supervisor)
    download = Mock(
        return_value=SimpleNamespace(
            status_code=200,
            raise_for_status=Mock(),
            json=lambda: fresh,
        )
    )
    monkeypatch.setattr(requests, "get", download)
    result = await update(
        runtime,
        bearer(runtime, ["models:register"]),
        single=True,
        json={"model_type": "llm", "model_name": "wanted"},
    )
    assert result.status_code == 200
    assert json.loads(path.read_text(encoding="utf-8")) == [keep, fresh]
    assert download.call_args.kwargs["headers"] == {"X-Models-Token": TOKEN}
    assert TOKEN not in result.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source, failure, status, detail",
    [
        ("provider", ValueError, 500, "Failed to resolve model catalog credentials"),
        ("provider", TypeError, 500, "Failed to resolve model catalog credentials"),
        ("provider", RuntimeError, 500, "Failed to resolve model catalog credentials"),
        ("supervisor", ModelCatalogUpdateError, 502, "Failed to update model catalog"),
        (
            "supervisor",
            ModelCatalogUnavailableError,
            503,
            "No workers available to update model catalog",
        ),
        ("supervisor", ConnectionError, 503, "Model catalog supervisor unavailable"),
        ("supervisor", RuntimeError, 503, "Model catalog supervisor unavailable"),
    ],
)
async def test_errors_are_classified_without_echoing_catalog_credentials(
    api, caplog, failure, status, detail, source
):
    runtime, hook = api
    caplog.set_level(logging.DEBUG, logger="xinference.api.restful_api")
    error = failure(TOKEN)
    if source == "provider":
        hook.side_effect = error
    else:
        # Actor exceptions must retain their type across serialization while
        # the API still ignores potentially sensitive exception messages.
        header, buffers = serialize(error)
        runtime._get_supervisor_ref.return_value.update_model_type.side_effect = (
            deserialize(header, buffers)
        )
    result = await update(runtime, bearer(runtime, ["models:register"]))
    assert result.status_code == status
    assert result.json() == {"detail": detail}
    assert TOKEN not in result.text
    assert TOKEN not in caplog.text
    if source == "provider":
        runtime._get_supervisor_ref.assert_not_awaited()
    else:
        runtime._get_supervisor_ref.return_value.update_model_type.assert_awaited_once()


@pytest.mark.asyncio
async def test_multiple_providers_are_a_server_error_without_rpc(api, monkeypatch):
    runtime, hook = api
    monkeypatch.setattr(
        extensions,
        "get_extensions",
        lambda: (
            SimpleNamespace(name="first", model_catalog_credentials=hook),
            SimpleNamespace(name="second", model_catalog_credentials=hook),
        ),
    )
    result = await update(runtime, bearer(runtime, ["models:register"]))
    assert result.status_code == 500
    assert result.json() == {"detail": "Failed to resolve model catalog credentials"}
    assert TOKEN not in result.text
    runtime._get_supervisor_ref.assert_not_awaited()
