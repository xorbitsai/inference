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

# CI: linux

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import APIRouter, FastAPI
from xoscar.serialization import deserialize, serialize

from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService
from xinference.api.restful_api import RESTfulAPI
from xinference.api.routers import models
from xinference.core.exceptions import (
    ModelCatalogUnavailableError,
    ModelCatalogUpdateError,
)

TOKEN = "test-catalog-secret"


@pytest.fixture
def api(tmp_path):
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
    models.register_routes(api)
    api._app.include_router(api._router)
    return api


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
async def test_refresh_requires_authentication_and_registration_scope(
    api, permissions, status, single
):
    runtime = api
    headers = bearer(runtime, permissions) if permissions is not None else {}
    result = await update(runtime, headers, single=single)
    assert result.status_code == status
    runtime._get_supervisor_ref.assert_not_awaited()


@pytest.mark.asyncio
async def test_authenticated_refresh_keeps_success_response(api):
    runtime = api
    result = await update(runtime, bearer(runtime, ["models:register"]))
    assert result.status_code == 200
    assert result.json() == {
        "data": {"model_type": "llm", "message": "Successfully updated model type: llm"}
    }
    supervisor = await runtime._get_supervisor_ref()
    supervisor.update_model_type.assert_awaited_once_with("llm")
    assert TOKEN not in result.text


@pytest.mark.asyncio
async def test_anonymous_refresh_keeps_rpc_shape(api):
    runtime = api
    runtime._advanced_auth_service = None
    runtime._router = APIRouter()
    runtime._app = FastAPI()
    models.register_routes(runtime)
    runtime._app.include_router(runtime._router)
    result = await update(runtime, {})
    assert result.status_code == 200
    supervisor = await runtime._get_supervisor_ref()
    supervisor.update_model_type.assert_awaited_once_with("llm")


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
async def test_invalid_body_is_400_before_rpc(api, caplog, request_kwargs):
    runtime = api
    caplog.set_level(logging.DEBUG, logger="xinference.api.restful_api")
    headers = bearer(runtime, ["models:register"])
    headers.update(request_kwargs.get("headers", {}))
    payload = {k: v for k, v in request_kwargs.items() if k != "headers"}
    result = await update(runtime, headers, **payload)
    assert result.status_code == 400
    assert result.json() == {"detail": "Invalid request body"}
    assert TOKEN not in result.text
    assert TOKEN not in caplog.text
    runtime._get_supervisor_ref.assert_not_awaited()


@pytest.mark.asyncio
async def test_single_model_refresh_reaches_supervisor(api):
    runtime = api
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
    supervisor = await runtime._get_supervisor_ref()
    supervisor.update_model_type.assert_awaited_once_with("llm", model_name="wanted")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name", [None, "", " padded", "a\nb", "a\x7fb", 42, False, [], "a" * 513]
)
async def test_single_model_name_is_required_and_validated_before_rpc(api, name):
    runtime = api
    result = await update(
        runtime,
        bearer(runtime, ["models:register"]),
        single=True,
        json={"model_type": "llm", "model_name": name},
    )
    assert result.status_code == 400
    assert result.json() == {"detail": "Invalid request body"}
    runtime._get_supervisor_ref.assert_not_awaited()


@pytest.mark.asyncio
async def test_single_model_name_cannot_be_omitted(api):
    runtime = api
    result = await update(
        runtime,
        bearer(runtime, ["models:register"]),
        single=True,
        json={"model_type": "llm"},
    )
    assert result.status_code == 400
    runtime._get_supervisor_ref.assert_not_awaited()


@pytest.mark.asyncio
async def test_single_model_api_reaches_supervisor_worker_and_persisted_catalog(
    api, tmp_path, monkeypatch
):
    from functools import partial

    from xinference.core import model_catalog
    from xinference.core.supervisor import SupervisorActor
    from xinference.core.worker import WorkerActor

    runtime = api

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
    path = model_catalog.github_catalog_path("llm")
    commit = "b" * 40
    keep, fresh = record("keep", 1), record("wanted", 2)
    model_catalog.store_model_catalog(
        path, [keep, record("wanted", 1)], github_commit="a" * 40
    )
    worker = SimpleNamespace()
    supervisor = SupervisorActor()
    supervisor._worker_address_to_worker = {
        "worker": SimpleNamespace(
            update_model_type=partial(WorkerActor.update_model_type, worker)
        )
    }
    runtime._get_supervisor_ref = AsyncMock(return_value=supervisor)
    download = Mock(return_value=(commit, "wanted", [fresh]))
    monkeypatch.setattr(model_catalog, "download_github_catalog", download)
    result = await update(
        runtime,
        bearer(runtime, ["models:register"]),
        single=True,
        json={"model_type": "llm", "model_name": "wanted"},
    )
    assert result.status_code == 200
    assert model_catalog.read_github_catalog(path) == (True, [keep, fresh])
    download.assert_called_once_with("llm", "wanted")
    assert TOKEN not in result.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure, status, detail",
    [
        (ModelCatalogUpdateError, 502, "Failed to update model catalog"),
        (
            ModelCatalogUnavailableError,
            503,
            "No workers available to update model catalog",
        ),
        (ConnectionError, 503, "Model catalog supervisor unavailable"),
        (RuntimeError, 503, "Model catalog supervisor unavailable"),
    ],
)
async def test_errors_are_classified_without_echoing_actor_exceptions(
    api, caplog, failure, status, detail
):
    caplog.set_level(logging.DEBUG, logger="xinference.api.restful_api")
    header, buffers = serialize(failure(TOKEN))
    api._get_supervisor_ref.return_value.update_model_type.side_effect = deserialize(
        header, buffers
    )
    result = await update(api, bearer(api, ["models:register"]))
    assert result.status_code == status
    assert result.json() == {"detail": detail}
    assert TOKEN not in result.text
    assert TOKEN not in caplog.text
