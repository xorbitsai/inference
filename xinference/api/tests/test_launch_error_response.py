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
"""Tests for the launch error response body.

``detail`` must stay a plain string so existing clients keep working, with the
traceback carried in a separate, optional field.
"""
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from ...core.error_utils import format_error_summary, format_error_traceback
from ..restful_api import DetailedHTTPException


def _build_app() -> FastAPI:
    """Mirror the handler registration in ``RESTfulAPI.serve``."""
    app = FastAPI()

    @app.exception_handler(DetailedHTTPException)
    async def detailed_http_exception_handler(
        request: Request, exc: DetailedHTTPException
    ):
        body: Dict[str, Any] = {"detail": exc.detail}
        if exc.tb:
            body["traceback"] = exc.tb
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.get("/with-traceback")
    async def with_traceback():
        try:
            raise FileNotFoundError("no such file: /nonexistent/model")
        except Exception as e:
            raise DetailedHTTPException(
                500, format_error_summary(e), format_error_traceback(e)
            )

    @app.get("/without-traceback")
    async def without_traceback():
        raise DetailedHTTPException(499, "Cancelled by user")

    @app.get("/bad-request")
    async def bad_request():
        try:
            raise ValueError("bad model path")
        except Exception as e:
            raise DetailedHTTPException(
                400, format_error_summary(e), format_error_traceback(e)
            )

    return app


def test_detail_is_a_string_and_traceback_is_separate():
    with TestClient(_build_app(), raise_server_exceptions=False) as client:
        response = client.get("/with-traceback")

    assert response.status_code == 500
    body = response.json()
    # The Web UI interceptor and the Python client both read `detail` as a
    # string; making it an object would break them.
    assert isinstance(body["detail"], str)
    assert body["detail"] == "FileNotFoundError: no such file: /nonexistent/model"
    assert "Traceback (most recent call last)" in body["traceback"]


def test_traceback_key_omitted_when_absent():
    with TestClient(_build_app(), raise_server_exceptions=False) as client:
        response = client.get("/without-traceback")

    assert response.status_code == 499
    body = response.json()
    assert body == {"detail": "Cancelled by user"}
    assert "traceback" not in body


def test_status_code_is_preserved():
    with TestClient(_build_app(), raise_server_exceptions=False) as client:
        response = client.get("/bad-request")

    assert response.status_code == 400
    assert response.json()["detail"] == "ValueError: bad model path"


def test_traceback_suppressed_by_env(monkeypatch):
    from ...constants import XINFERENCE_ENV_DISABLE_ERROR_TRACEBACK

    monkeypatch.setenv(XINFERENCE_ENV_DISABLE_ERROR_TRACEBACK, "1")
    with TestClient(_build_app(), raise_server_exceptions=False) as client:
        response = client.get("/with-traceback")

    body = response.json()
    assert "traceback" not in body
    # The summary is still reported; only the traceback is withheld.
    assert body["detail"].startswith("FileNotFoundError")


def test_exception_subclasses_http_exception():
    exc = DetailedHTTPException(503, "boom")
    assert exc.status_code == 503
    assert exc.detail == "boom"
    assert exc.tb is None
