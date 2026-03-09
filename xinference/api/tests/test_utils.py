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

import pytest
from fastapi import HTTPException

from ..utils import require_model


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


class TestGetModelOrError:
    """Tests for get_model_or_error function."""

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

        assert exc_info.value.status_code == 400

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
