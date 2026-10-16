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

* Creating the first admin requires the operator-only setup token.
* The token survives restarts and is shared by processes using one auth DB.
* Once setup has completed, ``setup_admin`` must reject immediately --
  before validating the password or computing its bcrypt hash -- so the
  endpoint doesn't stay a permanently unauthenticated, CPU-expensive route
  after the first admin exists.

    pytest xinference/api/tests/test_admin_setup_route.py -v
"""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from xinference.api.oauth2.advanced.auth_service import (
    PASSWORD_MIN_LENGTH,
    AdvancedAuthService,
)
from xinference.api.oauth2.advanced.database import Database
from xinference.api.oauth2.advanced.routes import register_advanced_auth_routes


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


def test_setup_succeeds(auth_service):
    client = _make_client(auth_service)

    response = client.post(
        "/v1/admin/setup",
        json={
            "setup_token": auth_service.setup_token,
            "username": "admin",
            "password": "a-strong-password",
        },
    )
    assert response.status_code == 201
    assert response.json()["username"] == "admin"


@pytest.mark.parametrize("setup_token", [None, "", "wrong-token", 123, [], "无效令牌"])
def test_setup_rejects_missing_or_invalid_token_before_hashing(
    auth_service, setup_token
):
    client = _make_client(auth_service)
    payload = {"username": "attacker", "password": "a-strong-password"}
    if setup_token is not None:
        payload["setup_token"] = setup_token
    with patch(
        "xinference.api.oauth2.advanced.routes.get_password_hash", return_value="unused"
    ) as mock_hash:
        response = client.post("/v1/admin/setup", json=payload)
    assert response.status_code == 403
    mock_hash.assert_not_called()
    assert auth_service.db.user_count() == 0


def test_setup_rejects_short_password(auth_service):
    client = _make_client(auth_service)

    response = client.post(
        "/v1/admin/setup",
        json={
            "setup_token": auth_service.setup_token,
            "username": "admin",
            "password": "short",
        },
    )
    assert response.status_code == 400


def test_setup_rejects_non_string_password(auth_service):
    """A non-string password (e.g. an int) must be rejected with 400, not
    crash the endpoint with an unhandled TypeError from len(password).
    """
    client = _make_client(auth_service)

    response = client.post(
        "/v1/admin/setup",
        json={
            "setup_token": auth_service.setup_token,
            "username": "admin",
            "password": 12345,
        },
    )
    assert response.status_code == 400


def test_setup_rejects_second_call_after_completion(auth_service):
    client = _make_client(auth_service)
    first = client.post(
        "/v1/admin/setup",
        json={
            "setup_token": auth_service.setup_token,
            "username": "admin",
            "password": "a-strong-password",
        },
    )
    assert first.status_code == 201

    second = client.post(
        "/v1/admin/setup",
        json={
            "setup_token": auth_service.setup_token,
            "username": "someone-else",
            "password": "another-password",
        },
    )
    assert second.status_code == 403


def test_setup_rejects_before_hashing_password_once_completed(auth_service):
    """The 403 for a completed setup must be raised before bcrypt hashing
    (or password-length validation) runs, so a completed deployment isn't
    stuck paying CPU cost on every call to this still-public endpoint.
    """
    client = _make_client(auth_service)
    first = client.post(
        "/v1/admin/setup",
        json={
            "setup_token": auth_service.setup_token,
            "username": "admin",
            "password": "a-strong-password",
        },
    )
    assert first.status_code == 201

    with patch("xinference.api.oauth2.advanced.routes.get_password_hash") as mock_hash:
        response = client.post(
            "/v1/admin/setup",
            json={
                "setup_token": auth_service.setup_token,
                "username": "someone-else",
                "password": "another-password",
            },
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

    client.post(
        "/v1/admin/setup",
        json={
            "setup_token": auth_service.setup_token,
            "username": "admin",
            "password": "a-strong-password",
        },
    )

    after = client.get("/v1/admin/setup/status")
    assert after.json() == {
        "needs_setup": False,
        "initialized": True,
        "password_min_length": PASSWORD_MIN_LENGTH,
    }


def _reopen_auth_service(db_path):
    return AdvancedAuthService(
        db_path=str(db_path),
        jwt_secret_key="unit-test-secret",
        encryption_key="unit-test-encryption-key",
    )


def test_setup_token_survives_restart_and_is_consumed(auth_service):
    token = auth_service.setup_token
    assert isinstance(token, str) and len(token) >= 43
    reopened = _reopen_auth_service(auth_service.db.path)
    assert reopened.setup_token == token
    response = _make_client(reopened).post(
        "/v1/admin/setup",
        json={
            "setup_token": token,
            "username": "operator",
            "password": "strong-password",
        },
    )
    assert response.status_code == 201
    assert "setup_token" not in response.json()
    assert reopened.db.get_config("setup_token") is None
    assert auth_service.setup_token is None
    assert not auth_service.verify_setup_token(token)
    assert _reopen_auth_service(auth_service.db.path).setup_token is None


def test_setup_token_is_not_exposed_by_status_or_errors(auth_service, tmp_path):
    client = _make_client(auth_service)
    token = auth_service.setup_token
    assert token not in client.get("/v1/admin/setup/status").text
    other = _reopen_auth_service(tmp_path / "other.db")
    response = client.post(
        "/v1/admin/setup",
        json={
            "setup_token": other.setup_token,
            "username": "attacker",
            "password": "strong-password",
        },
    )
    assert response.status_code == 403
    assert token not in response.text
    assert other.setup_token not in response.text
    assert auth_service.db.user_count() == 0


def test_independent_services_share_token_and_only_one_setup_wins(tmp_path):
    db_path = tmp_path / "concurrent.db"
    Database(str(db_path))
    with ThreadPoolExecutor(max_workers=2) as executor:
        services = list(executor.map(_reopen_auth_service, [db_path, db_path]))
    auth_service = services[0]
    token = auth_service.setup_token
    assert all(service.setup_token == token for service in services)

    def setup(item):
        index, service = item
        return (
            _make_client(service)
            .post(
                "/v1/admin/setup",
                json={
                    "setup_token": token,
                    "username": f"operator-{index}",
                    "password": "strong-password",
                },
            )
            .status_code
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        statuses = list(executor.map(setup, enumerate(services)))
    assert sorted(statuses) == [201, 403]
    assert auth_service.db.user_count() == 1
    assert auth_service.db.get_config("setup_token") is None


def test_initialized_database_never_generates_setup_token(auth_service):
    auth_service.db.create_first_user("operator", "unused", permissions=["admin"])
    with patch(
        "xinference.api.oauth2.advanced.database.secrets.token_urlsafe"
    ) as generate:
        reopened = _reopen_auth_service(auth_service.db.path)
    generate.assert_not_called()
    assert reopened.setup_token is None
    response = _make_client(reopened).post("/v1/admin/setup", json={})
    assert response.status_code == 403
    assert "already completed" in response.json()["detail"]
