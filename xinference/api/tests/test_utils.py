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

import pytest
from fastapi import HTTPException

from ..utils import _MODEL_NOT_FOUND_CACHE, require_model


@pytest.fixture(autouse=True)
def _clear_negative_cache():
    """Clear the negative cache before each test to avoid cross-test pollution."""
    _MODEL_NOT_FOUND_CACHE.clear()
    yield
    _MODEL_NOT_FOUND_CACHE.clear()


class DummyModel:
    """Dummy model for testing."""

    def __init__(self, uid: str):
        self.uid = uid


class DummySupervisor:
    """Dummy supervisor for testing."""

    def __init__(self, models=None):
        self._models = models or {}

    async def get_model(self, model_uid: str):
        if model_uid not in self._models:
            raise ValueError(f"Model {model_uid} not found")
        return self._models[model_uid]


class TestRequireModel:
    """Tests for require_model function."""

    @pytest.mark.asyncio
    async def test_successful_get(self):
        """Test successful model retrieval."""
        model = DummyModel("test-model")
        supervisor = DummySupervisor(models={"test-model": model})

        async def get_supervisor():
            return supervisor

        result = await require_model(get_supervisor, "test-model")
        assert result is model

    @pytest.mark.asyncio
    async def test_model_not_found_raises_400(self):
        """Test that ValueError raises 400."""
        supervisor = DummySupervisor(models={})

        async def get_supervisor():
            return supervisor

        with pytest.raises(HTTPException) as exc_info:
            await require_model(get_supervisor, "missing-model")

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_unexpected_error_raises_500(self):
        """Test that unexpected errors raise 500."""

        async def get_supervisor():
            raise RuntimeError("Unexpected error")

        with pytest.raises(HTTPException) as exc_info:
            await require_model(get_supervisor, "any-model")

        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_reports_error_event(self):
        """Test that error events are reported."""
        supervisor = DummySupervisor(models={})
        reported = []

        async def get_supervisor():
            return supervisor

        async def report_error(model_uid, message):
            reported.append((model_uid, message))

        with pytest.raises(HTTPException):
            await require_model(
                get_supervisor, "missing-model", report_error_event=report_error
            )

        assert len(reported) == 1
        assert reported[0][0] == "missing-model"

    @pytest.mark.asyncio
    async def test_no_report_error_event_when_none(self):
        """Test that no error is reported when report_error_event is None."""
        supervisor = DummySupervisor(models={})

        async def get_supervisor():
            return supervisor

        # Should not raise any error when report_error_event is None
        with pytest.raises(HTTPException):
            await require_model(get_supervisor, "missing-model")


@pytest.mark.asyncio
async def test_require_model_forwards_correlation_id_without_changing_signature():
    from ..model_request_logging import (
        _reset_current_model_request_id,
        _set_current_model_request_id,
    )

    calls = []
    model = DummyModel("test-model")

    class CorrelatedSupervisor:
        async def get_model(self, model_uid: str, **kwargs):
            calls.append((model_uid, kwargs))
            return model

    async def get_supervisor():
        return CorrelatedSupervisor()

    token = _set_current_model_request_id("http-request-id")
    try:
        result = await require_model(get_supervisor, "test-model")
    finally:
        _reset_current_model_request_id(token)

    assert result.uid == model.uid
    assert len(calls) == 1
    model_uid, kwargs = calls[0]
    assert model_uid == "test-model"
    metadata = kwargs["__xinf_rpc_metadata__"]
    assert metadata["version"] == 1
    assert metadata["correlation_id"] == "http-request-id"
    assert metadata["actor_call_id"]
    assert "operation_request_id" not in metadata


@pytest.mark.asyncio
async def test_launch_model_by_version_clears_negative_cache():
    from unittest.mock import AsyncMock, MagicMock

    from ..restful_api import RESTfulAPI

    supervisor = DummySupervisor()

    async def get_supervisor_ref():
        return supervisor

    with pytest.raises(HTTPException) as exc_info:
        await require_model(get_supervisor_ref, "my-model")
    assert exc_info.value.status_code == 404
    assert "my-model" in _MODEL_NOT_FOUND_CACHE

    api = RESTfulAPI.__new__(RESTfulAPI)
    launcher = MagicMock()
    launcher.launch_model_by_version = AsyncMock(return_value="my-model")
    api._get_supervisor_ref = AsyncMock(return_value=launcher)
    request = MagicMock(
        json=AsyncMock(
            return_value={
                "model_uid": "my-model",
                "model_type": "LLM",
                "model_version": "qwen3--8B--pytorch--none",
            }
        )
    )
    response = await api.launch_model_by_version(request, wait_ready=True)
    assert response.status_code == 200
    assert "my-model" not in _MODEL_NOT_FOUND_CACHE

    supervisor._models["my-model"] = DummyModel("my-model")
    assert (await require_model(get_supervisor_ref, "my-model")).uid == "my-model"
