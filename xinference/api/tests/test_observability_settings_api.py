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

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, FastAPI, Header, HTTPException
from fastapi.security import SecurityScopes
from fastapi.testclient import TestClient

from xinference.api.routers import observability_settings, system_settings
from xinference.core.observability_settings_store import ObservabilitySettingsStore
from xinference.core.system_settings_store import SystemSettingsStore


@pytest.fixture
def stores(tmp_path):
    path = tmp_path / "system-settings.json"
    return (
        SystemSettingsStore(str(path), environ={}),
        ObservabilitySettingsStore(
            str(path),
            environ={"XINFERENCE_OTLP_API_KEY": "otel_abcdefgh12345678"},
        ),
    )


@pytest.fixture
def scoped_client(stores):
    download_store, observability_store = stores
    token_scopes = {
        "reader": {"settings:read"},
        "writer": {"settings:write"},
    }

    async def auth_service(
        security_scopes: SecurityScopes,
        authorization: str = Header(default=""),
    ):
        token = authorization.removeprefix("Bearer ")
        scopes = token_scopes.get(token)
        if scopes is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        if not set(security_scopes.scopes).issubset(scopes):
            raise HTTPException(status_code=403, detail="Forbidden")
        return {"username": token}

    router = APIRouter()
    api = SimpleNamespace(
        _router=router,
        _auth_service=auth_service,
        _get_supervisor_ref=AsyncMock(return_value=AsyncMock()),
        is_authenticated=lambda: True,
    )
    app = FastAPI()
    app.state.api = api
    app.state.system_settings_store = download_store
    app.state.observability_settings_store = observability_store
    system_settings.register_routes(api)
    observability_settings.register_routes(api)
    app.include_router(router)
    return TestClient(app)


def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def test_observability_routes_use_settings_scopes(scoped_client):
    path = "/v1/cluster/observability_settings"
    assert scoped_client.get(path).status_code == 401
    assert scoped_client.get(path, headers=_auth_headers("writer")).status_code == 403
    assert scoped_client.get(path, headers=_auth_headers("reader")).status_code == 200

    payload = scoped_client.get(path, headers=_auth_headers("reader")).json()
    assert (
        scoped_client.put(
            path, headers=_auth_headers("reader"), json=payload
        ).status_code
        == 403
    )
    assert (
        scoped_client.put(
            path, headers=_auth_headers("writer"), json=payload
        ).status_code
        == 200
    )


def test_download_and_observability_apis_save_independently(scoped_client, stores):
    download_store, observability_store = stores
    download_path = "/v1/cluster/system_settings"
    observability_path = "/v1/cluster/observability_settings"

    download_payload = download_store.get_public()
    download_payload["download_source"] = "modelscope"
    response = scoped_client.put(
        download_path,
        headers=_auth_headers("writer"),
        json=download_payload,
    )
    assert response.status_code == 200
    assert download_store.get().download_source == "modelscope"
    assert observability_store.get().opentelemetry.enable_otel is False

    observability_payload = observability_store.get_public()
    observability_payload["opentelemetry"]["enable_otel"] = True
    response = scoped_client.put(
        observability_path,
        headers=_auth_headers("writer"),
        json=observability_payload,
    )
    assert response.status_code == 200
    assert response.json()["opentelemetry"]["otlp_api_key"] == "otel********5678"
    assert observability_store.get().opentelemetry.enable_otel is True
    assert download_store.get().download_source == "modelscope"
    document = json.loads(Path(download_store.path).read_text(encoding="utf-8"))
    assert document["settings"]["download_source"] == "modelscope"
    assert document["observability"]["opentelemetry"]["enable_otel"] is True


def test_entities_reject_fields_from_the_other_section(scoped_client, stores):
    download_store, observability_store = stores
    download_payload = download_store.get_public()
    download_payload["enable_otel"] = True
    assert (
        scoped_client.put(
            "/v1/cluster/system_settings",
            headers=_auth_headers("writer"),
            json=download_payload,
        ).status_code
        == 422
    )

    observability_payload = observability_store.get_public()
    observability_payload["download_source"] = "modelscope"
    assert (
        scoped_client.put(
            "/v1/cluster/observability_settings",
            headers=_auth_headers("writer"),
            json=observability_payload,
        ).status_code
        == 422
    )


def test_observability_reset_does_not_reset_download(scoped_client, stores):
    download_store, observability_store = stores
    download_payload = download_store.get_public()
    download_payload["download_source"] = "modelscope"
    download_store.save_public(download_payload)
    observability_payload = observability_store.get_public()
    observability_payload["opentelemetry"]["enable_otel"] = True
    observability_store.save_public(observability_payload)

    response = scoped_client.post(
        "/v1/cluster/observability_settings/reset",
        headers=_auth_headers("writer"),
    )

    assert response.status_code == 200
    assert response.json()["opentelemetry"]["enable_otel"] is False
    assert download_store.get().download_source == "modelscope"
    document = json.loads(Path(download_store.path).read_text(encoding="utf-8"))
    assert document["settings"]["download_source"] == "modelscope"
    assert "observability" not in document
