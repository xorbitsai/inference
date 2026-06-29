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

import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.security import SecurityScopes

from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService
from xinference.api.oauth2.advanced.database import Database
from xinference.api.oauth2.advanced.routes import (
    list_api_keys,
    reveal_api_key,
    rotate_api_key,
)


def _json_body(response):
    return json.loads(response.body.decode())


def _auth_service(tmp_path):
    return AdvancedAuthService(
        db_path=str(tmp_path / "auth.db"),
        jwt_secret_key="test-jwt-secret",
        encryption_key="test-encryption-key",
    )


def _request(auth, body, user_id, username, scopes):
    token = auth.create_access_token(user_id, username, scopes)
    request = MagicMock()
    request.app.state.advanced_auth = auth
    request.headers = {"Authorization": f"Bearer {token}"}
    request.json = AsyncMock(return_value=body)
    request.client = SimpleNamespace(host="127.0.0.1")
    request.url = SimpleNamespace(path="/v1/admin/keys")
    request.method = "POST"
    return request


def _admin_request(auth, body):
    admin = auth.db.get_user_by_username("admin")
    return _request(
        auth,
        body,
        admin["id"],
        admin["username"],
        ["admin", "keys:create", "keys:manage"],
    )


def _create_user(auth, username, scopes):
    user_id = auth.db.create_user(
        username=username,
        password_hash=None,
        permissions=scopes,
    )
    return auth.db.get_user_by_id(user_id)


def _create_key(auth, user_id):
    return auth.create_api_key_for_user(
        user_id=user_id,
        name="rotating key",
        description="preserve me",
        token_budget=100,
        token_renewal="daily",
        request_rate_limit_enabled=True,
        request_rate_limit_requests=10,
        request_rate_limit_window_seconds=60,
        model_permissions=[
            {"permission_type": "model_id", "permission_value": "test-model"}
        ],
    )


async def _authenticate_with_api_key(auth, api_key):
    request = MagicMock()
    request.headers = {"Authorization": f"Bearer {api_key}"}
    request.client = SimpleNamespace(host="127.0.0.1")
    request.url = SimpleNamespace(path="/v1/models")
    request.method = "GET"
    request.state = SimpleNamespace()
    return await auth(request, SecurityScopes(scopes=["models:list"]), api_key)


@pytest.mark.asyncio
async def test_rotate_api_key_returns_new_secret_once_and_invalidates_old_secret(
    tmp_path,
):
    auth = _auth_service(tmp_path)
    admin = auth.db.get_user_by_username("admin")
    created = _create_key(auth, admin["id"])
    old_secret = created["key"]
    old_key = auth.db.get_api_key_by_id(created["id"])

    response = await rotate_api_key(created["id"], _admin_request(auth, {}))
    rotated = _json_body(response)

    assert rotated["id"] == created["id"]
    assert rotated["key"].startswith(("sk-", "xf-"))
    assert rotated["key"] != old_secret
    assert rotated["key_prefix"] == rotated["key"][:7]
    assert rotated["rotated_at"] is not None
    assert "key" not in _json_body(await list_api_keys(_admin_request(auth, {})))[0]

    with pytest.raises(HTTPException) as old_exc:
        await _authenticate_with_api_key(auth, old_secret)
    assert old_exc.value.status_code == 401

    user = await _authenticate_with_api_key(auth, rotated["key"])
    assert user["id"] == admin["id"]

    new_key = auth.db.get_api_key_by_id(created["id"])
    assert new_key["id"] == old_key["id"]
    assert new_key["user_id"] == old_key["user_id"]
    assert new_key["name"] == old_key["name"]
    assert new_key["description"] == old_key["description"]
    assert new_key["token_budget"] == old_key["token_budget"]
    assert new_key["token_renewal"] == old_key["token_renewal"]
    assert (
        new_key["request_rate_limit_enabled"] == old_key["request_rate_limit_enabled"]
    )
    assert new_key["model_permissions"] == old_key["model_permissions"]
    assert new_key["key_hash"] != old_key["key_hash"]
    assert new_key["key_prefix"] == rotated["key_prefix"]
    assert new_key["rotated_at"] == rotated["rotated_at"]


@pytest.mark.asyncio
async def test_reveal_api_key_does_not_return_plaintext_secret(tmp_path):
    auth = _auth_service(tmp_path)
    admin = auth.db.get_user_by_username("admin")
    created = _create_key(auth, admin["id"])

    with pytest.raises(HTTPException) as exc_info:
        await reveal_api_key(created["id"], _admin_request(auth, {}))

    assert exc_info.value.status_code == 410
    assert "only displayed when it is created or rotated" in exc_info.value.detail


@pytest.mark.asyncio
async def test_owner_can_rotate_own_api_key(tmp_path):
    auth = _auth_service(tmp_path)
    owner = _create_user(auth, "owner", ["keys:create", "keys:manage"])
    created = _create_key(auth, owner["id"])

    response = await rotate_api_key(
        created["id"],
        _request(
            auth,
            {},
            owner["id"],
            owner["username"],
            ["keys:create", "keys:manage"],
        ),
    )

    assert _json_body(response)["id"] == created["id"]


@pytest.mark.asyncio
async def test_non_owner_without_admin_scope_cannot_rotate_api_key(tmp_path):
    auth = _auth_service(tmp_path)
    owner = _create_user(auth, "owner", ["keys:create", "keys:manage"])
    other = _create_user(auth, "other", ["keys:create", "keys:manage"])
    created = _create_key(auth, owner["id"])

    with pytest.raises(HTTPException) as exc_info:
        await rotate_api_key(
            created["id"],
            _request(
                auth,
                {},
                other["id"],
                other["username"],
                ["keys:create", "keys:manage"],
            ),
        )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_rotate_missing_api_key_returns_404(tmp_path):
    auth = _auth_service(tmp_path)

    with pytest.raises(HTTPException) as exc_info:
        await rotate_api_key(999, _admin_request(auth, {}))

    assert exc_info.value.status_code == 404


def test_api_key_rotation_column_is_added_to_existing_databases(tmp_path):
    db_path = tmp_path / "old-auth.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL,
                password_hash TEXT,
                source TEXT NOT NULL DEFAULT 'local',
                enabled INTEGER DEFAULT 1,
                must_change_password INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE api_keys (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                key_hash TEXT UNIQUE NOT NULL,
                key_encrypted TEXT NOT NULL,
                key_prefix TEXT NOT NULL,
                name TEXT,
                enabled INTEGER DEFAULT 1,
                expires_at TIMESTAMP,
                encryption_version INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

    Database(str(db_path))

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(api_keys)")}

    assert "rotated_at" in columns
