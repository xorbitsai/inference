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

"""Unit tests for admin router handlers."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from xinference.api.routers import admin, models
from xinference.core.virtual_env_manager import VirtualEnvConflictError


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
    supervisor.cache_builtin_model = AsyncMock(
        return_value={"cache_uid": "cache-1", "model_name": "qwen"}
    )
    supervisor.get_cache_builtin_model_progress_details = AsyncMock(
        return_value={"progress": 0.5, "stage": "downloading"}
    )
    supervisor.cancel_cache_builtin_model = AsyncMock()
    supervisor.delete_cache_builtin_model = AsyncMock(
        return_value={"removed_bytes": 1024}
    )
    supervisor.pause_cache_builtin_model = AsyncMock(
        return_value={"cache_uid": "cache-1", "status": "paused"}
    )
    supervisor.resume_cache_builtin_model = AsyncMock(
        return_value={"cache_uid": "cache-1", "status": "resuming"}
    )
    supervisor.list_model_downloads = AsyncMock(return_value=[])
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
async def test_get_cluster_version_full_revisionid():
    import re

    from xinference import __version__

    response = await admin.get_cluster_version()
    assert response.status_code == 200
    data = _json_body(response)
    assert data["version"] == __version__
    try:
        from xinference._commit import full_revisionid
    except ImportError:
        pytest.skip("no build-time commit metadata (plain source tree)")
    # normal VCS builds must expose the full 40-character SHA, matching the
    # versioneer-era full-revisionid contract
    assert data["full-revisionid"] == full_revisionid
    assert re.fullmatch(r"[0-9a-f]{40}", data["full-revisionid"])


@pytest.mark.asyncio
async def test_is_cluster_authenticated_returns_auth_flag(mock_api):
    mock_api.is_authenticated.return_value = True
    response = await admin.is_cluster_authenticated(api=mock_api)
    assert response.status_code == 200
    assert _json_body(response) == {"auth": True}

    mock_api.is_authenticated.return_value = False
    response = await admin.is_cluster_authenticated(api=mock_api)
    assert _json_body(response) == {"auth": False}


@pytest.mark.parametrize("is_auth", [False, True])
def test_cache_model_reuses_launch_model_permission(is_auth):
    def capture_routes(register_routes):
        captured = {}

        def add_api_route(path, endpoint, methods=None, **kwargs):
            captured[(path, tuple(methods or []))] = kwargs

        api = MagicMock()
        api._router.add_api_route.side_effect = add_api_route
        api._auth_service = MagicMock()
        api.is_authenticated.return_value = is_auth
        register_routes(api)
        return captured

    model_routes = capture_routes(models.register_routes)
    admin_routes = capture_routes(admin.register_routes)
    launch_dependencies = model_routes[("/v1/models", ("POST",))]["dependencies"]
    cache_dependencies = admin_routes[("/v1/cache/models", ("POST",))]["dependencies"]
    download_delete_dependencies = admin_routes[
        ("/v1/downloads/{cache_uid}", ("DELETE",))
    ]["dependencies"]

    if not is_auth:
        assert launch_dependencies is None
        assert cache_dependencies is None
        assert download_delete_dependencies is None
        return

    assert launch_dependencies[0].scopes == ["models:write"]
    assert cache_dependencies[0].scopes == launch_dependencies[0].scopes
    assert download_delete_dependencies[0].scopes == ["cache:delete"]


@pytest.mark.asyncio
async def test_get_cluster_device_info_returns_data(mock_api, mock_supervisor):
    mock_supervisor.get_cluster_device_info.return_value = {
        "devices": ["gpu-0"],
        "workers": [{"ip": "127.0.0.1"}],
    }
    response = await admin.get_cluster_device_info(
        api=mock_api, detailed=True, include_routers=True
    )
    assert response.status_code == 200
    data = _json_body(response)
    assert data["devices"] == ["gpu-0"]
    mock_supervisor.get_cluster_device_info.assert_called_once_with(
        detailed=True, include_routers=True
    )


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
async def test_cache_model_forwards_only_download_inputs(mock_api, mock_supervisor):
    request = MagicMock()
    request.json = AsyncMock(
        return_value={
            "cache_uid": "cache-1",
            "model_name": "qwen",
            "model_type": "LLM",
            "model_engine": "transformers",
            "model_format": "pytorch",
            "quantization": "none",
            "n_gpu": 2,
            "replica": 3,
            "enable_mtp": True,
            "draft_quantization": "q4_k_m",
        }
    )

    response = await admin.cache_model(request=request, api=mock_api)

    assert response.status_code == 200
    assert _json_body(response)["cache_uid"] == "cache-1"
    call_kwargs = mock_supervisor.cache_builtin_model.await_args.kwargs
    assert "n_gpu" not in call_kwargs
    assert "replica" not in call_kwargs
    assert call_kwargs["enable_mtp"] is True
    assert call_kwargs["draft_quantization"] == "q4_k_m"


@pytest.mark.parametrize(
    ("internal_key", "value"),
    [
        ("_resume", True),
        ("_download_repositories", [{"path": "/tmp/client-controlled"}]),
    ],
)
@pytest.mark.asyncio
async def test_cache_model_rejects_internal_fields(
    mock_api, mock_supervisor, internal_key, value
):
    request = MagicMock()
    request.json = AsyncMock(
        return_value={
            "model_name": "qwen",
            "model_type": "LLM",
            "model_engine": "transformers",
            internal_key: value,
        }
    )

    with pytest.raises(HTTPException) as exc_info:
        await admin.cache_model(request=request, api=mock_api)

    assert exc_info.value.status_code == 400
    assert internal_key in exc_info.value.detail
    mock_supervisor.cache_builtin_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_cache_model_progress_and_cancel(mock_api, mock_supervisor):
    response = await admin.get_cache_model_progress(cache_uid="cache-1", api=mock_api)
    assert _json_body(response) == {"progress": 0.5, "stage": "downloading"}

    cancel_response = await admin.cancel_cache_model(cache_uid="cache-1", api=mock_api)
    assert cancel_response.status_code == 200
    mock_supervisor.cancel_cache_builtin_model.assert_awaited_once_with("cache-1")

    delete_response = await admin.delete_cache_download(
        cache_uid="cache-1", api=mock_api
    )
    assert _json_body(delete_response) == {"removed_bytes": 1024}
    mock_supervisor.delete_cache_builtin_model.assert_awaited_once_with("cache-1")


@pytest.mark.asyncio
async def test_list_model_downloads_returns_list(mock_api, mock_supervisor):
    downloads = [{"model_uid": "qwen", "stage": "downloading"}]
    mock_supervisor.list_model_downloads.return_value = downloads

    response = await admin.list_model_downloads(api=mock_api)

    assert response.status_code == 200
    assert _json_body(response) == {"list": downloads}
    mock_supervisor.list_model_downloads.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_pause_and_resume_cache_model(mock_api, mock_supervisor):
    pause_response = await admin.pause_cache_model(cache_uid="cache-1", api=mock_api)
    assert _json_body(pause_response)["status"] == "paused"
    mock_supervisor.pause_cache_builtin_model.assert_awaited_once_with("cache-1")

    resume_response = await admin.resume_cache_model(cache_uid="cache-1", api=mock_api)
    assert _json_body(resume_response)["status"] == "resuming"
    mock_supervisor.resume_cache_builtin_model.assert_awaited_once_with("cache-1")


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
async def test_remove_virtual_env_returns_conflict_for_active_model(
    mock_api, mock_supervisor
):
    mock_supervisor.remove_virtual_env.side_effect = VirtualEnvConflictError(
        "environment is used by qwen-rep0"
    )

    with pytest.raises(HTTPException) as exc_info:
        await admin.remove_virtual_env(
            api=mock_api,
            model_name="Qwen3.8-Flash-Next",
            model_engine="vllm",
            python_version="3.12",
            worker_ip=None,
        )

    assert exc_info.value.status_code == 409
    assert "qwen-rep0" in exc_info.value.detail


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


@pytest.mark.asyncio
async def test_list_audit_filter_options_from_file(tmp_path, monkeypatch):
    to_thread = AsyncMock(wraps=admin.asyncio.to_thread)
    monkeypatch.setattr(admin.asyncio, "to_thread", to_thread)
    audit_entries = [
        {
            "@timestamp": "2026-08-03T08:24:16+00:00",
            "user": "zoe",
            "api_key_name": "robot",
            "model_id": "sense-voice",
            "model_name": "SenseVoiceSmall",
            "client_ip": "192.168.1.10",
        },
        {
            "@timestamp": "2026-08-03T08:25:16+00:00",
            "user": "Admin",
            "api_key_name": "assistant",
            "model_id": "qwen",
            "model_name": "Qwen3",
            "client_ip": "10.0.0.2",
        },
    ]
    (tmp_path / "audit.log").write_text(
        "\n".join(
            [
                json.dumps(audit_entries[0]),
                "null",
                "[1, 2]",
                json.dumps(audit_entries[1]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("xinference.constants.XINFERENCE_LOG_DIR", str(tmp_path))

    response = await admin._list_audit_filter_options_from_file(
        time_from="", time_to=""
    )

    assert _json_body(response) == {
        "user": ["Admin", "zoe"],
        "api_key_name": ["assistant", "robot"],
        "model_id": ["qwen", "sense-voice"],
        "model_name": ["Qwen3", "SenseVoiceSmall"],
        "client_ip": ["10.0.0.2", "192.168.1.10"],
    }
    to_thread.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_audit_filter_options_from_file_is_bounded_and_sorted(
    tmp_path, monkeypatch
):
    entries = [
        json.dumps({"user": f"user-{index:03d}"})
        for index in reversed(range(admin._AUDIT_FILTER_OPTION_LIMIT + 100))
    ]
    (tmp_path / "audit.log").write_text("\n".join(entries) + "\n", encoding="utf-8")
    monkeypatch.setattr("xinference.constants.XINFERENCE_LOG_DIR", str(tmp_path))

    response = await admin._list_audit_filter_options_from_file(
        time_from="", time_to=""
    )

    users = _json_body(response)["user"]
    assert len(users) == admin._AUDIT_FILTER_OPTION_LIMIT
    assert users == [
        f"user-{index:03d}" for index in range(admin._AUDIT_FILTER_OPTION_LIMIT)
    ]


def _audit_field_caps(direct_indices=(), keyword_indices=()):
    all_indices = [*direct_indices, *keyword_indices]
    fields = {}
    for field_name in admin._AUDIT_TEXT_FILTER_FIELDS:
        base_capabilities = {}
        if direct_indices:
            base_capabilities["keyword"] = {
                "aggregatable": True,
                "indices": list(direct_indices),
            }
        if keyword_indices:
            base_capabilities["text"] = {
                "aggregatable": False,
                "indices": list(keyword_indices),
            }
        fields[field_name] = base_capabilities
        if keyword_indices:
            fields[f"{field_name}.keyword"] = {
                "keyword": {
                    "aggregatable": True,
                    "indices": list(keyword_indices),
                }
            }
    return {"indices": all_indices, "fields": fields}


@pytest.mark.asyncio
async def test_list_audit_filter_options_from_elasticsearch(monkeypatch):
    captured = []
    responses = [
        _audit_field_caps(direct_indices=("audit-direct",)),
        {
            "aggregations": {
                field_name: {"buckets": [{"key": f"{field_name}-value"}]}
                for field_name in admin._AUDIT_TEXT_FILTER_FIELDS
            }
        },
    ]

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def json(self):
            return responses.pop(0)

    class FakeClientSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        def post(self, url, headers, json=None):
            captured.append({"url": url, "body": json})
            return FakeResponse()

    monkeypatch.setenv("XINFERENCE_ES_URL", "http://elasticsearch:9200")
    monkeypatch.setattr(admin.aiohttp, "ClientSession", FakeClientSession)

    response = await admin.list_audit_filter_options(time_from="now-6h", time_to="now")

    assert _json_body(response) == {
        field_name: [f"{field_name}-value"]
        for field_name in admin._AUDIT_TEXT_FILTER_FIELDS
    }
    assert captured[0]["url"].startswith(
        "http://elasticsearch:9200/xinference-audit-*/_field_caps?"
    )
    assert captured[0]["body"] is None
    assert captured[1] == {
        "url": "http://elasticsearch:9200/audit-direct/_search",
        "body": {
            "size": 0,
            "query": {"range": {"@timestamp": {"gte": "now-6h", "lte": "now"}}},
            "aggs": {
                field_name: {
                    "terms": {
                        "field": field_name,
                        "size": admin._AUDIT_FILTER_OPTION_LIMIT,
                    }
                }
                for field_name in admin._AUDIT_TEXT_FILTER_FIELDS
            },
        },
    }


@pytest.mark.asyncio
async def test_list_audit_filter_options_merges_mixed_mapping_indices(monkeypatch):
    captured = []
    responses = [
        _audit_field_caps(
            direct_indices=("audit-direct",), keyword_indices=("audit-dynamic",)
        ),
        {
            "aggregations": {
                field_name: {"buckets": [{"key": "common"}, {"key": "recent-direct"}]}
                for field_name in admin._AUDIT_TEXT_FILTER_FIELDS
            }
        },
        {
            "aggregations": {
                field_name: {"buckets": [{"key": "common"}, {"key": "legacy-dynamic"}]}
                for field_name in admin._AUDIT_TEXT_FILTER_FIELDS
            }
        },
    ]

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def json(self):
            return responses.pop(0)

    class FakeClientSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        def post(self, url, headers, json=None):
            captured.append({"url": url, "body": json})
            return FakeResponse()

    monkeypatch.setenv("XINFERENCE_ES_URL", "http://elasticsearch:9200")
    monkeypatch.setattr(admin.aiohttp, "ClientSession", FakeClientSession)

    response = await admin.list_audit_filter_options()

    assert _json_body(response) == {
        field_name: ["common", "legacy-dynamic", "recent-direct"]
        for field_name in admin._AUDIT_TEXT_FILTER_FIELDS
    }
    assert [request["url"] for request in captured[1:]] == [
        "http://elasticsearch:9200/audit-direct/_search",
        "http://elasticsearch:9200/audit-dynamic/_search",
    ]
    assert captured[1]["body"]["aggs"] == {
        field_name: {
            "terms": {
                "field": field_name,
                "size": admin._AUDIT_FILTER_OPTION_LIMIT,
            }
        }
        for field_name in admin._AUDIT_TEXT_FILTER_FIELDS
    }
    assert captured[2]["body"]["aggs"] == {
        field_name: {
            "terms": {
                "field": f"{field_name}.keyword",
                "size": admin._AUDIT_FILTER_OPTION_LIMIT,
            }
        }
        for field_name in admin._AUDIT_TEXT_FILTER_FIELDS
    }


@pytest.mark.asyncio
async def test_list_audit_filter_options_returns_502_when_field_caps_fails(monkeypatch):
    captured_urls = []

    class FakeResponse:
        status = 503

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def text(self):
            return '{"error":{"reason":"all shards failed"}}'

    class FakeClientSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        def post(self, url, headers, json=None):
            captured_urls.append(url)
            return FakeResponse()

    monkeypatch.setenv("XINFERENCE_ES_URL", "http://elasticsearch:9200")
    monkeypatch.setattr(admin.aiohttp, "ClientSession", FakeClientSession)

    with pytest.raises(HTTPException) as exc_info:
        await admin.list_audit_filter_options()

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Elasticsearch query failed"
    assert len(captured_urls) == 1


@pytest.mark.asyncio
async def test_list_audit_filter_options_returns_502_when_group_search_fails(
    monkeypatch,
):
    captured_urls = []
    responses = [
        (200, "", _audit_field_caps(direct_indices=("audit-direct",))),
        (503, '{"error":{"reason":"all shards failed"}}', {}),
    ]

    class FakeResponse:
        def __init__(self, status, text, data):
            self.status = status
            self._text = text
            self._data = data

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def text(self):
            return self._text

        async def json(self):
            return self._data

    class FakeClientSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        def post(self, url, headers, json=None):
            captured_urls.append(url)
            return FakeResponse(*responses.pop(0))

    monkeypatch.setenv("XINFERENCE_ES_URL", "http://elasticsearch:9200")
    monkeypatch.setattr(admin.aiohttp, "ClientSession", FakeClientSession)

    with pytest.raises(HTTPException) as exc_info:
        await admin.list_audit_filter_options()

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Elasticsearch query failed"
    assert len(captured_urls) == 2
    assert captured_urls[1] == "http://elasticsearch:9200/audit-direct/_search"
