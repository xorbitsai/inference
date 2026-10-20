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
"""HTTP regressions for malformed login requests returning server errors."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService
from xinference.api.oauth2.advanced.crypto import get_password_hash
from xinference.api.oauth2.advanced.routes import advanced_login


@pytest.fixture
def login_client():
    auth = SimpleNamespace(login=Mock())
    app = FastAPI()
    app.state.advanced_auth = auth
    app.add_api_route("/token", advanced_login, methods=["POST"])
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, auth


@pytest.mark.parametrize(
    "content,content_type",
    [
        (b"username=alice&password=secret", "application/x-www-form-urlencoded"),
        (b"", "application/json"),
        (b'{"username":', "application/json"),
        (b'\xff{"username":"alice"}', "application/json"),
        pytest.param(
            b'{"username":' + b"1" * 5000 + b"}",
            "application/json",
            id="oversized-integer",
        ),
        pytest.param(
            b"[" * 2000 + b"0" + b"]" * 2000,
            "application/json",
            id="excessive-nesting",
        ),
        (b"null", "application/json"),
        (b"[]", "application/json"),
    ],
)
def test_login_rejects_invalid_body(login_client, content, content_type):
    client, auth = login_client

    response = client.post(
        "/token", content=content, headers={"Content-Type": content_type}
    )

    assert response.status_code == 400
    assert response.json()["detail"]
    auth.login.assert_not_called()


@pytest.mark.parametrize("field", ["username", "password"])
@pytest.mark.parametrize("value", [None, 123, ["alice"]])
def test_login_rejects_non_string_credentials(login_client, field, value):
    client, auth = login_client
    body = {"username": "alice", "password": "secret"}
    body[field] = value

    response = client.post("/token", json=body)

    assert response.status_code == 400
    assert response.json() == {"detail": "username and password must be strings"}
    auth.login.assert_not_called()


def test_login_preserves_credentials(login_client):
    client, auth = login_client
    tokens = {
        "access_token": "access",
        "refresh_token": "refresh",
        "token_type": "bearer",
    }
    auth.login.return_value = tokens

    response = client.post("/token", json={"username": "用户", "password": " secret "})

    assert response.status_code == 200
    assert response.json() == tokens
    auth.login.assert_called_once_with("用户", " secret ")


def test_login_valid_and_incorrect_password(tmp_path):
    auth = AdvancedAuthService(
        db_path=str(tmp_path / "auth.db"),
        jwt_secret_key="unit-test-secret",
        encryption_key="unit-test-encryption-key",
    )
    auth.db.create_first_user("alice", get_password_hash("correct-password"), [])
    app = FastAPI()
    app.state.advanced_auth = auth
    app.add_api_route("/token", advanced_login, methods=["POST"])
    with TestClient(app) as client:
        success = client.post(
            "/token", json={"username": "alice", "password": "correct-password"}
        )
        assert success.status_code == 200
        assert success.json()["token_type"] == "bearer"
        assert (
            auth.verify_access_token(success.json()["access_token"])["sub"] == "alice"
        )
        assert success.json()["refresh_token"]

        for body in [
            {"username": "alice", "password": "incorrect-password"},
            {"username": "alice"},
            {"password": "correct-password"},
            {},
        ]:
            failure = client.post("/token", json=body)
            assert failure.status_code == 401
            assert failure.headers["WWW-Authenticate"] == "Bearer"
