# Copyright 2022-2026 XProbe Inc.
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

"""Unit tests for admin router handlers."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from xinference.api.routers import admin


def _json_body(response):
    return json.loads(response.body.decode())


@pytest.fixture
def mock_supervisor():
    supervisor = AsyncMock()
    supervisor.get_status = AsyncMock(return_value={"running_models": 0})
    supervisor.get_cluster_device_info = AsyncMock(
        return_value={"devices": [], "workers": []}
    )
    supervisor.get_devices_count = AsyncMock(return_value={"cpu": 8, "gpu": 0})
    supervisor.get_workers_info = AsyncMock(return_value=[])
    supervisor.get_supervisor_info = AsyncMock(
        return_value={"supervisor_address": "127.0.0.1:9999"}
    )
    supervisor.abort_cluster = AsyncMock(return_value=True)
    supervisor.list_cached_models = AsyncMock(return_value=[])
    supervisor.list_deletable_models = AsyncMock(return_value=[])
    supervisor.confirm_and_remove_model = AsyncMock(return_value=True)
    supervisor.list_virtual_envs = AsyncMock(return_value=[])
    supervisor.remove_virtual_env = AsyncMock(return_value=True)
    supervisor.get_progress = AsyncMock(return_value=0.5)
    return supervisor


@pytest.fixture
def mock_api(mock_supervisor):
    api = MagicMock()
    api._supervisor_address = "127.0.0.1:9999"
    api._get_supervisor_ref = AsyncMock(return_value=mock_supervisor)
    api._auth_service = MagicMock()
    api._auth_service.generate_token_for_user = MagicMock(
        return_value={"access_token": "test-token", "token_type": "bearer"}
    )
    api.is_authenticated = MagicMock(return_value=False)
    return api


@pytest.mark.asyncio
async def test_get_status_returns_200_and_data(mock_api, mock_supervisor):
    mock_supervisor.get_status.return_value = {"running_models": 2}
    response = await admin.get_status(api=mock_api)
    assert response.status_code == 200
    assert _json_body(response) == {"running_models": 2}


@pytest.mark.asyncio
async def test_get_status_raises_500_on_supervisor_error(mock_api, mock_supervisor):
    mock_supervisor.get_status.side_effect = RuntimeError("supervisor down")
    with pytest.raises(HTTPException) as exc_info:
        await admin.get_status(api=mock_api)
    assert exc_info.value.status_code == 500
    assert "supervisor down" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_address_returns_supervisor_address(mock_api):
    mock_api._supervisor_address = "10.0.0.1:12345"
    response = await admin.get_address(api=mock_api)
    assert response.status_code == 200
    assert _json_body(response) == "10.0.0.1:12345"


@pytest.mark.asyncio
async def test_get_cluster_version_returns_version():
    response = await admin.get_cluster_version()
    assert response.status_code == 200
    data = _json_body(response)
    assert "version" in data or "git" in data or len(data) >= 1


@pytest.mark.asyncio
async def test_is_cluster_authenticated_returns_auth_flag(mock_api):
    mock_api.is_authenticated.return_value = True
    response = await admin.is_cluster_authenticated(api=mock_api)
    assert response.status_code == 200
    assert _json_body(response) == {"auth": True}

    mock_api.is_authenticated.return_value = False
    response = await admin.is_cluster_authenticated(api=mock_api)
    assert _json_body(response) == {"auth": False}


@pytest.mark.asyncio
async def test_login_for_access_token_returns_token(mock_api):
    request = MagicMock()
    request.json = AsyncMock(return_value={"username": "user", "password": "pass"})
    mock_api._auth_service.generate_token_for_user.return_value = {
        "access_token": "jwt-xxx",
        "token_type": "bearer",
    }
    response = await admin.login_for_access_token(request=request, api=mock_api)
    assert response.status_code == 200
    assert _json_body(response)["access_token"] == "jwt-xxx"
    mock_api._auth_service.generate_token_for_user.assert_called_once_with(
        "user", "pass"
    )


@pytest.mark.asyncio
async def test_get_cluster_device_info_returns_data(mock_api, mock_supervisor):
    mock_supervisor.get_cluster_device_info.return_value = {
        "devices": ["gpu-0"],
        "workers": [{"ip": "127.0.0.1"}],
    }
    response = await admin.get_cluster_device_info(api=mock_api, detailed=True)
    assert response.status_code == 200
    data = _json_body(response)
    assert data["devices"] == ["gpu-0"]
    mock_supervisor.get_cluster_device_info.assert_called_once_with(detailed=True)


@pytest.mark.asyncio
async def test_get_devices_count_returns_data(mock_api, mock_supervisor):
    mock_supervisor.get_devices_count.return_value = {"cpu": 16, "gpu": 2}
    response = await admin.get_devices_count(api=mock_api)
    assert response.status_code == 200
    assert _json_body(response) == {"cpu": 16, "gpu": 2}


@pytest.mark.asyncio
async def test_get_workers_info_returns_data(mock_api, mock_supervisor):
    mock_supervisor.get_workers_info.return_value = [
        {"worker_id": "w1", "status": "running"}
    ]
    response = await admin.get_workers_info(api=mock_api)
    assert response.status_code == 200
    assert _json_body(response) == [{"worker_id": "w1", "status": "running"}]


@pytest.mark.asyncio
async def test_get_supervisor_info_returns_data(mock_api, mock_supervisor):
    mock_supervisor.get_supervisor_info.return_value = {"address": "0.0.0.0"}
    response = await admin.get_supervisor_info(api=mock_api)
    assert response.status_code == 200
    assert _json_body(response) == {"address": "0.0.0.0"}


@pytest.mark.asyncio
async def test_abort_cluster_returns_result_and_does_not_kill_in_test(
    mock_api, mock_supervisor
):
    mock_supervisor.abort_cluster.return_value = True
    with patch("xinference.api.routers.admin.os.kill", MagicMock()):
        response = await admin.abort_cluster(api=mock_api)
    assert response.status_code == 200
    assert _json_body(response) == {"result": True}


@pytest.mark.asyncio
async def test_list_cached_models_returns_list(mock_api, mock_supervisor):
    mock_supervisor.list_cached_models.return_value = ["model1", "model2"]
    response = await admin.list_cached_models(
        api=mock_api, model_name="qwen", worker_ip=None
    )
    assert response.status_code == 200
    assert _json_body(response) == {"list": ["model1", "model2"]}
    mock_supervisor.list_cached_models.assert_called_once_with("qwen", None)


@pytest.mark.asyncio
async def test_list_model_files_returns_paths(mock_api, mock_supervisor):
    mock_supervisor.list_deletable_models.return_value = ["/path/a", "/path/b"]
    response = await admin.list_model_files(
        api=mock_api,
        model_version="1.0",
        worker_ip="10.0.0.1",
    )
    assert response.status_code == 200
    data = _json_body(response)
    assert data["model_version"] == "1.0"
    assert data["worker_ip"] == "10.0.0.1"
    assert data["paths"] == ["/path/a", "/path/b"]


@pytest.mark.asyncio
async def test_confirm_and_remove_model_returns_result(mock_api, mock_supervisor):
    mock_supervisor.confirm_and_remove_model.return_value = True
    response = await admin.confirm_and_remove_model(
        api=mock_api, model_version="1.0", worker_ip=None
    )
    assert response.status_code == 200
    assert _json_body(response) == {"result": True}


@pytest.mark.asyncio
async def test_list_virtual_envs_returns_list(mock_api, mock_supervisor):
    mock_supervisor.list_virtual_envs.return_value = [{"name": "venv1"}]
    response = await admin.list_virtual_envs(
        api=mock_api,
        model_name="qwen",
        model_engine="vllm",
        worker_ip=None,
    )
    assert response.status_code == 200
    assert _json_body(response) == {"list": [{"name": "venv1"}]}


@pytest.mark.asyncio
async def test_remove_virtual_env_requires_model_name(mock_api):
    with pytest.raises(HTTPException) as exc_info:
        await admin.remove_virtual_env(
            api=mock_api,
            model_name=None,
            model_engine=None,
            python_version=None,
            worker_ip=None,
        )
    assert exc_info.value.status_code == 400
    assert "model_name" in exc_info.value.detail


@pytest.mark.asyncio
async def test_remove_virtual_env_returns_result(mock_api, mock_supervisor):
    mock_supervisor.remove_virtual_env.return_value = True
    response = await admin.remove_virtual_env(
        api=mock_api,
        model_name="qwen",
        model_engine=None,
        python_version=None,
        worker_ip=None,
    )
    assert response.status_code == 200
    assert _json_body(response) == {"result": True}


@pytest.mark.asyncio
async def test_get_progress_returns_progress(mock_api, mock_supervisor):
    mock_supervisor.get_progress.return_value = 0.75
    response = await admin.get_progress(request_id="req-123", api=mock_api)
    assert response.status_code == 200
    assert _json_body(response) == {"progress": 0.75}
    mock_supervisor.get_progress.assert_called_once_with("req-123")


@pytest.mark.asyncio
async def test_get_progress_raises_400_on_key_error(mock_api, mock_supervisor):
    mock_supervisor.get_progress.side_effect = KeyError("req-missing")
    with pytest.raises(HTTPException) as exc_info:
        await admin.get_progress(request_id="req-missing", api=mock_api)
    assert exc_info.value.status_code == 400
