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
from unittest.mock import AsyncMock, MagicMock

import pytest

from xinference.api.routers import system_settings
from xinference.core.system_settings_store import SystemSettingsStore


def _json_body(response):
    return json.loads(response.body.decode())


@pytest.fixture
def store(tmp_path):
    return SystemSettingsStore(
        str(tmp_path / "system-settings.json"),
        environ={
            "XINFERENCE_MODEL_SRC": "auto",
            "HUGGING_FACE_HUB_TOKEN": "hf_abcdefgh12345678",
        },
    )


@pytest.fixture
def mock_request(store):
    request = MagicMock()
    request.app.state.system_settings_store = store
    supervisor_ref = AsyncMock()
    request.app.state.api._get_supervisor_ref = AsyncMock(return_value=supervisor_ref)
    return request


@pytest.mark.asyncio
async def test_get_returns_full_masked_settings(mock_request):
    response = await system_settings.get_system_settings(request=mock_request)

    assert response.status_code == 200
    assert _json_body(response) == {
        "download_source": "auto",
        "hf_endpoint": "",
        "hf_token": "hf_a********5678",
        "pip_index_url": "",
        "download_max_attempts": 3,
        "hub_detect_timeout": 3.0,
        "model_download_workers": 2,
    }


@pytest.mark.asyncio
async def test_put_full_get_payload_preserves_token(mock_request, store):
    current = _json_body(
        await system_settings.get_system_settings(request=mock_request)
    )
    current["download_source"] = "modelscope"
    current["download_max_attempts"] = 5
    body = system_settings.SystemSettingsPayload(**current)

    response = await system_settings.update_system_settings(
        request=mock_request, body=body
    )

    assert response.status_code == 200
    assert _json_body(response)["hf_token"] == "hf_a********5678"
    assert store.get().hf_token == "hf_abcdefgh12345678"
    assert store.get().download_source == "modelscope"
    assert store.get().download_max_attempts == 5
    supervisor_ref = mock_request.app.state.api._get_supervisor_ref.return_value
    supervisor_ref.update_system_settings.assert_awaited_once_with(
        store.get().to_dict()
    )


@pytest.mark.asyncio
async def test_put_replaces_and_clears_token(mock_request, store):
    payload = store.get_public()
    payload["hf_token"] = "hf_newtoken12345678"
    response = await system_settings.update_system_settings(
        request=mock_request,
        body=system_settings.SystemSettingsPayload(**payload),
    )
    assert _json_body(response)["hf_token"] == "hf_n********5678"
    assert store.get().hf_token == "hf_newtoken12345678"

    payload = _json_body(response)
    payload["hf_token"] = ""
    await system_settings.update_system_settings(
        request=mock_request,
        body=system_settings.SystemSettingsPayload(**payload),
    )
    assert store.get().hf_token == ""


@pytest.mark.asyncio
async def test_reset_returns_startup_settings(mock_request, store):
    payload = store.get_public()
    payload["download_source"] = "huggingface"
    store.save_public(payload)

    response = await system_settings.reset_system_settings(request=mock_request)

    assert response.status_code == 200
    restored = _json_body(response)
    assert restored["download_source"] == "auto"
    assert restored["hf_token"] == "hf_a********5678"
    supervisor_ref = mock_request.app.state.api._get_supervisor_ref.return_value
    supervisor_ref.update_system_settings.assert_awaited_once_with(
        store.get_startup().to_dict()
    )
