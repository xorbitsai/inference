# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# CI: linux

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from ...model.llm.weight_cache import RELOAD_FIELDS
from ..restful_api import RESTfulAPI


def make_api(engine="vllm"):
    api = RESTfulAPI.__new__(RESTfulAPI)
    supervisor = MagicMock()
    supervisor.get_model_reload_config = AsyncMock(
        return_value={"engine": engine, "parameters": RELOAD_FIELDS[engine]}
    )
    supervisor.reload_model = AsyncMock(
        return_value={"operation_id": "one", "status": "reloading"}
    )
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    return api, supervisor


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        [],
        {},
        {"model_config": []},
        {"model_config": {"max_num_seqs": True}},
        {"model_config": {"dtype": "float16"}},
        {"model_config": {"reasoning_content": "true"}},
        {"model_config": {"limit_mm_per_prompt": {"image": 0}}},
        {"model_config": {"limit_mm_per_prompt": {"image": True}}},
        {"model_config": {"max_num_seqs": 8}, "drain_timeout": True},
        {"model_config": {"max_num_seqs": 8}, "drain_timeout": float("inf")},
        {"model_config": {"max_num_seqs": 8}, "ignored": 1},
    ],
)
async def test_invalid_reload_is_not_scheduled(body):
    api, supervisor = make_api()
    request = MagicMock(json=AsyncMock(return_value=body))
    with pytest.raises(HTTPException) as exc:
        await api.reload_model("test", request)
    assert exc.value.status_code == 400
    supervisor.reload_model.assert_not_called()


@pytest.mark.asyncio
async def test_reload_accepted_and_duplicate_conflict():
    api, supervisor = make_api()
    request = MagicMock(
        json=AsyncMock(return_value={"model_config": {"max_num_seqs": 8}})
    )
    response = await api.reload_model("test", request)
    assert response.status_code == 202
    assert json.loads(response.body)["operation_id"] == "one"
    supervisor.reload_model.assert_awaited_once_with(
        "test", {"max_num_seqs": 8}, 300, python_executable=None
    )
    supervisor.reload_model.side_effect = RuntimeError("already reloading")
    with pytest.raises(HTTPException) as exc:
        await api.reload_model("test", request)
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_environment_upgrade_accepts_empty_parameter_patch():
    api, supervisor = make_api()
    request = MagicMock(
        json=AsyncMock(return_value={"python_executable": "/opt/new/bin/python"})
    )
    response = await api.reload_model("test", request)
    assert response.status_code == 202
    supervisor.reload_model.assert_awaited_once_with(
        "test", {}, 300, python_executable="/opt/new/bin/python"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["python", True, "/bad\x00python"])
async def test_invalid_environment_never_schedules_upgrade(value):
    api, supervisor = make_api()
    request = MagicMock(json=AsyncMock(return_value={"python_executable": value}))
    with pytest.raises(HTTPException) as error:
        await api.reload_model("test", request)
    assert error.value.status_code == 400
    supervisor.reload_model.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "engine,patch",
    [
        ("vllm", {"reasoning_content": True}),
        ("vllm", {"limit_mm_per_prompt": {"image": 8}}),
        ("sglang", {"reasoning_content": True}),
        ("sglang", {"limit_mm_data_per_request": {"image": 8}}),
    ],
)
async def test_runtime_settings_are_forwarded_without_coercion(engine, patch):
    api, supervisor = make_api(engine)
    request = MagicMock(json=AsyncMock(return_value={"model_config": patch}))
    response = await api.reload_model("test", request)
    assert response.status_code == 202
    supervisor.reload_model.assert_awaited_once_with(
        "test", patch, 300, python_executable=None
    )


@pytest.mark.asyncio
async def test_reasoning_patch_absent_from_metadata_is_rejected():
    api, supervisor = make_api()
    supervisor.get_model_reload_config.return_value = {
        "engine": "vllm",
        "parameters": {"max_num_seqs": "positive_int"},
    }
    request = MagicMock(
        json=AsyncMock(return_value={"model_config": {"reasoning_content": True}})
    )
    with pytest.raises(HTTPException) as exc:
        await api.reload_model("test", request)
    assert exc.value.status_code == 400
    supervisor.reload_model.assert_not_called()


@pytest.mark.asyncio
async def test_loading_model_returns_retryable_status():
    from ...core.exceptions import ModelNotReadyError

    api, supervisor = make_api()
    supervisor.reload_model.side_effect = ModelNotReadyError("Model is loading")
    request = MagicMock(
        json=AsyncMock(return_value={"model_config": {"max_num_seqs": 8}})
    )
    with pytest.raises(HTTPException) as exc:
        await api.reload_model("test", request)
    assert exc.value.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("admin", [False, True])
async def test_environment_upgrade_requires_admin(admin):
    from starlette.requests import Request

    api, supervisor = make_api()
    api._advanced_auth_service = AsyncMock(
        side_effect=None if admin else HTTPException(403, "Not enough permissions")
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/reload",
            "headers": [(b"authorization", b"Bearer token")],
        }
    )
    request._json = {"python_executable": "/new/python"}
    if admin:
        assert (await api.reload_model("test", request)).status_code == 202
    else:
        with pytest.raises(HTTPException) as error:
            await api.reload_model("test", request)
        assert error.value.status_code == 403
        supervisor.reload_model.assert_not_called()
    assert api._advanced_auth_service.await_args.args[1].scopes == ["admin"]


@pytest.mark.asyncio
@pytest.mark.parametrize("admin", [False, True])
async def test_reload_metadata_only_discloses_interpreter_to_admin(admin):
    from starlette.requests import Request

    api, supervisor = make_api()
    supervisor.get_model_reload_config.return_value["python_executable"] = (
        "/private/python"
    )
    api._advanced_auth_service = AsyncMock(
        return_value={"permissions": ["admin"] if admin else ["models:list"]}
    )
    request = Request(
        {"type": "http", "method": "GET", "path": "/config", "headers": []}
    )
    response = await api.get_model_reload_config("test", request)
    metadata = json.loads(response.body)
    assert ("python_executable" in metadata) is admin
    assert metadata["can_upgrade_environment"] is admin


@pytest.mark.asyncio
async def test_launch_cannot_override_persisted_model_interpreter():
    api, supervisor = make_api()
    api._uid_to_model_name = {}
    supervisor.launch_builtin_model = AsyncMock(return_value="test")
    request = MagicMock(
        json=AsyncMock(
            return_value={
                "model_uid": "test",
                "model_name": "test",
                "model_engine": "vllm",
                "model_python_executable": "/arbitrary/executable",
            }
        )
    )
    assert (await api.launch_model(request, wait_ready=True)).status_code == 200
    assert (
        "model_python_executable"
        not in supervisor.launch_builtin_model.await_args.kwargs
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("scheme", ["Bearer", "bearer"])
async def test_admin_owned_api_key_cannot_upgrade_or_see_interpreter(tmp_path, scheme):
    from starlette.requests import Request

    from ..oauth2.advanced.auth_service import AdvancedAuthService
    from ..oauth2.advanced.crypto import get_password_hash

    auth = AdvancedAuthService(
        str(tmp_path / "auth.db"), "test-secret", "test-encryption"
    )
    user_id = auth.db.create_user(
        "admin", get_password_hash("test-password"), permissions=["admin"]
    )
    key = auth.create_api_key_for_user(user_id, name="model-key")["key"]
    api, supervisor = make_api()
    api._advanced_auth_service = auth
    supervisor.get_model_reload_config.return_value["python_executable"] = (
        "/private/python"
    )

    def request(method):
        return Request(
            {
                "type": "http",
                "method": method,
                "path": "/v1/models/test/reload",
                "headers": [(b"authorization", (scheme + " " + key).encode())],
            }
        )

    metadata = json.loads(
        (await api.get_model_reload_config("test", request("GET"))).body
    )
    assert metadata["can_upgrade_environment"] is False
    assert "python_executable" not in metadata
    post = request("POST")
    post._json = {"python_executable": "/new/python"}
    with pytest.raises(HTTPException) as error:
        await api.reload_model("test", post)
    assert error.value.status_code == 403
    assert "API keys can only access" in error.value.detail
    supervisor.reload_model.assert_not_called()


@pytest.mark.asyncio
async def test_autostart_rest_rejects_internal_interpreter():
    from ...core.supervisor import SupervisorActor

    api, _ = make_api()
    supervisor = SupervisorActor.__new__(SupervisorActor)
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    request = MagicMock(
        json=AsyncMock(
            return_value={
                "launch": {
                    "model_name": "test",
                    "model_python_executable": "/arbitrary/python",
                }
            }
        )
    )
    with pytest.raises(HTTPException) as error:
        await api.upsert_autostart_model(request, {"username": "ordinary"})
    assert error.value.status_code == 400
    assert "reserved for internal recovery" in error.value.detail
