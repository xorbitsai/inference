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
"""Regression tests for the ``/v1/admin/setup`` first-run route.

Covers Gemini review findings on PR #5144:

* Once setup has completed, ``setup_admin`` must reject immediately --
  before validating the password or computing its bcrypt hash -- so the
  endpoint doesn't stay a permanently unauthenticated, CPU-expensive route
  after the first admin exists.
* A bootstrap ``setup_token`` (printed to the server log, or provided via
  ``XINFERENCE_AUTH_SETUP_TOKEN``) is required to create the first admin,
  so reaching the public endpoint over the network isn't by itself enough
  to win full admin access.

    pytest xinference/api/tests/test_admin_setup_route.py -v
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from xinference import constants as xconst
from xinference.api.oauth2.advanced.auth_service import (
    PASSWORD_MIN_LENGTH,
    AdvancedAuthService,
)
from xinference.api.oauth2.advanced.routes import register_advanced_auth_routes


@pytest.fixture(autouse=True)
def _isolated_auth_dir(tmp_path, monkeypatch):
    # setup_token generation/deletion reads XINFERENCE_AUTH_DIR from
    # xinference.constants; isolate it so tests don't touch the real
    # ~/.xinference/auth directory or collide with each other.
    monkeypatch.setattr(xconst, "XINFERENCE_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.setattr(
        xconst,
        "XINFERENCE_AUTH_SETUP_TOKEN_PATH",
        str(tmp_path / "auth" / "setup_token"),
    )
    monkeypatch.delenv("XINFERENCE_AUTH_SETUP_TOKEN", raising=False)
    yield


@pytest.fixture
def auth_service(tmp_path):
    db_path = str(tmp_path / "auth.db")
    return AdvancedAuthService(
        db_path=db_path,
        jwt_secret_key="unit-test-secret",
        encryption_key="unit-test-encryption-key",
    )


def _make_client(auth_service):
    app = FastAPI()
    router = APIRouter()
    app.state.advanced_auth = auth_service
    register_advanced_auth_routes(SimpleNamespace(_app=app, _router=router))
    app.include_router(router)
    return TestClient(app)


def test_setup_succeeds_with_valid_token(auth_service):
    client = _make_client(auth_service)
    token = xconst.get_or_create_setup_token()

    response = client.post(
        "/v1/admin/setup",
        json={
            "username": "admin",
            "password": "a-strong-password",
            "setup_token": token,
        },
    )
    assert response.status_code == 201
    assert response.json()["username"] == "admin"


def test_setup_rejects_missing_token(auth_service):
    client = _make_client(auth_service)
    xconst.get_or_create_setup_token()

    response = client.post(
        "/v1/admin/setup",
        json={"username": "admin", "password": "a-strong-password"},
    )
    assert response.status_code == 403


def test_setup_rejects_wrong_token(auth_service):
    client = _make_client(auth_service)
    xconst.get_or_create_setup_token()

    response = client.post(
        "/v1/admin/setup",
        json={
            "username": "admin",
            "password": "a-strong-password",
            "setup_token": "definitely-not-the-right-token",
        },
    )
    assert response.status_code == 403


def test_setup_rejects_non_string_password(auth_service):
    """A non-string password (e.g. an int) must be rejected with 400, not
    crash the endpoint with an unhandled TypeError from len(password).
    """
    client = _make_client(auth_service)
    token = xconst.get_or_create_setup_token()

    response = client.post(
        "/v1/admin/setup",
        json={"username": "admin", "password": 12345, "setup_token": token},
    )
    assert response.status_code == 400


def test_setup_rejects_non_string_setup_token(auth_service):
    """A non-string setup_token must be rejected with 403, not crash with
    a TypeError from secrets.compare_digest.
    """
    client = _make_client(auth_service)
    xconst.get_or_create_setup_token()

    response = client.post(
        "/v1/admin/setup",
        json={
            "username": "admin",
            "password": "a-strong-password",
            "setup_token": 12345,
        },
    )
    assert response.status_code == 403


def test_setup_token_deleted_after_successful_setup(auth_service):
    client = _make_client(auth_service)
    token = xconst.get_or_create_setup_token()

    response = client.post(
        "/v1/admin/setup",
        json={
            "username": "admin",
            "password": "a-strong-password",
            "setup_token": token,
        },
    )
    assert response.status_code == 201
    assert not __import__("os").path.exists(xconst.XINFERENCE_AUTH_SETUP_TOKEN_PATH)


def test_env_setup_token_is_honored(auth_service, monkeypatch):
    monkeypatch.setenv("XINFERENCE_AUTH_SETUP_TOKEN", "env-provided-token")
    client = _make_client(auth_service)

    wrong = client.post(
        "/v1/admin/setup",
        json={
            "username": "admin",
            "password": "a-strong-password",
            "setup_token": "wrong",
        },
    )
    assert wrong.status_code == 403

    right = client.post(
        "/v1/admin/setup",
        json={
            "username": "admin",
            "password": "a-strong-password",
            "setup_token": "env-provided-token",
        },
    )
    assert right.status_code == 201


def test_setup_rejects_second_call_after_completion(auth_service):
    client = _make_client(auth_service)
    token = xconst.get_or_create_setup_token()
    first = client.post(
        "/v1/admin/setup",
        json={
            "username": "admin",
            "password": "a-strong-password",
            "setup_token": token,
        },
    )
    assert first.status_code == 201

    # A second attempt gets a fresh token (the first was deleted), so this
    # exercises the needs_setup() guard, not the token check.
    second_token = xconst.get_or_create_setup_token()
    second = client.post(
        "/v1/admin/setup",
        json={
            "username": "someone-else",
            "password": "another-password",
            "setup_token": second_token,
        },
    )
    assert second.status_code == 403


def test_setup_rejects_before_hashing_password_once_completed(auth_service):
    """The 403 for a completed setup must be raised before bcrypt hashing
    (or password-length / token validation) runs, so a completed
    deployment isn't stuck paying CPU cost on every call to this still
    -public endpoint.
    """
    client = _make_client(auth_service)
    token = xconst.get_or_create_setup_token()
    first = client.post(
        "/v1/admin/setup",
        json={
            "username": "admin",
            "password": "a-strong-password",
            "setup_token": token,
        },
    )
    assert first.status_code == 201

    with patch("xinference.api.oauth2.advanced.routes.get_password_hash") as mock_hash:
        response = client.post(
            "/v1/admin/setup",
            json={"username": "someone-else", "password": "another-password"},
        )
    assert response.status_code == 403
    mock_hash.assert_not_called()


def test_setup_status_reflects_completion(auth_service):
    client = _make_client(auth_service)

    before = client.get("/v1/admin/setup/status")
    assert before.json() == {
        "needs_setup": True,
        "initialized": False,
        "password_min_length": PASSWORD_MIN_LENGTH,
    }

    token = xconst.get_or_create_setup_token()
    client.post(
        "/v1/admin/setup",
        json={
            "username": "admin",
            "password": "a-strong-password",
            "setup_token": token,
        },
    )

    after = client.get("/v1/admin/setup/status")
    assert after.json() == {
        "needs_setup": False,
        "initialized": True,
        "password_min_length": PASSWORD_MIN_LENGTH,
    }
