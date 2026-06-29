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
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService
from xinference.api.oauth2.advanced.database import Database
from xinference.api.oauth2.advanced.routes import create_api_key, list_api_keys


def _json_body(response):
    return json.loads(response.body.decode())


def _auth_service(tmp_path):
    return AdvancedAuthService(
        db_path=str(tmp_path / "auth.db"),
        jwt_secret_key="test-jwt-secret",
        encryption_key="test-encryption-key",
    )


def _admin_request(auth, body):
    admin = auth.db.get_user_by_username("admin")
    token = auth.create_access_token(
        admin["id"],
        admin["username"],
        ["admin", "keys:create", "keys:manage"],
    )
    request = MagicMock()
    request.app.state.advanced_auth = auth
    request.headers = {"Authorization": f"Bearer {token}"}
    request.json = AsyncMock(return_value=body)
    return request


def _create_key(auth, **kwargs):
    admin = auth.db.get_user_by_username("admin")
    return auth.create_api_key_for_user(
        user_id=admin["id"],
        name="renewal key",
        token_budget=kwargs.pop("token_budget", 10),
        **kwargs,
    )


def _set_renewal_state(auth, key_id, **fields):
    auth.db.update_api_key(key_id, **fields)


def test_no_renewal_never_resets_token_usage(tmp_path):
    auth = _auth_service(tmp_path)
    key = _create_key(auth, token_renewal="none", token_budget=5)
    auth.db.increment_api_key_token_usage(key["id"], 5)

    with pytest.raises(HTTPException) as exc_info:
        auth.ensure_api_key_token_budget(key["id"], now=datetime(2026, 1, 3, 0, 0, 0))

    stored = auth.db.get_api_key_by_id(key["id"])
    assert exc_info.value.status_code == 429
    assert stored["token_usage"] == 5
    assert stored["token_renewal_next_at"] is None


def test_daily_renewal_resets_usage_and_advances_next_boundary(tmp_path):
    auth = _auth_service(tmp_path)
    key = _create_key(auth, token_renewal="daily", token_budget=5)
    _set_renewal_state(
        auth,
        key["id"],
        token_usage=5,
        token_renewed_at="2026-01-01T00:00:00",
        token_renewal_next_at="2026-01-02T00:00:00",
    )

    auth.ensure_api_key_token_budget(key["id"], now=datetime(2026, 1, 2, 0, 0, 0))

    stored = auth.db.get_api_key_by_id(key["id"])
    assert stored["token_usage"] == 0
    assert stored["token_renewed_at"] == "2026-01-02T00:00:00"
    assert stored["token_renewal_next_at"] == "2026-01-03T00:00:00"


def test_repeated_checks_in_same_period_do_not_reset_usage_again(tmp_path):
    auth = _auth_service(tmp_path)
    key = _create_key(auth, token_renewal="daily", token_budget=10)
    _set_renewal_state(
        auth,
        key["id"],
        token_usage=10,
        token_renewed_at="2026-01-01T00:00:00",
        token_renewal_next_at="2026-01-02T00:00:00",
    )

    auth.ensure_api_key_token_budget(key["id"], now=datetime(2026, 1, 2, 0, 0, 0))
    auth.db.increment_api_key_token_usage(key["id"], 3)
    auth.ensure_api_key_token_budget(key["id"], now=datetime(2026, 1, 2, 12, 0, 0))

    stored = auth.db.get_api_key_by_id(key["id"])
    assert stored["token_usage"] == 3
    assert stored["token_renewed_at"] == "2026-01-02T00:00:00"
    assert stored["token_renewal_next_at"] == "2026-01-03T00:00:00"


def test_stale_token_renewal_does_not_overwrite_new_period_usage(tmp_path):
    auth = _auth_service(tmp_path)
    key = _create_key(auth, token_renewal="daily", token_budget=10)
    _set_renewal_state(
        auth,
        key["id"],
        token_usage=10,
        token_renewed_at="2026-01-01T00:00:00",
        token_renewal_next_at="2026-01-02T00:00:00",
    )
    stale_state = auth.db.get_api_key_token_usage_state(key["id"])

    auth.ensure_api_key_token_budget(key["id"], now=datetime(2026, 1, 2, 0, 0, 0))
    auth.db.increment_api_key_token_usage(key["id"], 4)
    auth._renew_api_key_token_budget_if_needed(
        key["id"], stale_state, datetime(2026, 1, 2, 0, 0, 1)
    )

    stored = auth.db.get_api_key_by_id(key["id"])
    assert stored["token_usage"] == 4
    assert stored["token_renewed_at"] == "2026-01-02T00:00:00"
    assert stored["token_renewal_next_at"] == "2026-01-03T00:00:00"


def test_custom_renewal_uses_configured_interval_days(tmp_path):
    auth = _auth_service(tmp_path)
    key = _create_key(
        auth,
        token_renewal="custom",
        token_renewal_interval_days=14,
        token_budget=10,
    )
    _set_renewal_state(
        auth,
        key["id"],
        token_usage=9,
        token_renewed_at="2026-01-01T00:00:00",
        token_renewal_next_at="2026-01-15T00:00:00",
    )

    auth.ensure_api_key_token_budget(key["id"], now=datetime(2026, 1, 15, 0, 0, 0))

    stored = auth.db.get_api_key_by_id(key["id"])
    assert stored["token_usage"] == 0
    assert stored["token_renewed_at"] == "2026-01-15T00:00:00"
    assert stored["token_renewal_next_at"] == "2026-01-29T00:00:00"


def test_disabled_key_does_not_renew_or_reenable(tmp_path):
    auth = _auth_service(tmp_path)
    key = _create_key(auth, token_renewal="daily", token_budget=10)
    _set_renewal_state(
        auth,
        key["id"],
        enabled=0,
        token_usage=9,
        token_renewed_at="2026-01-01T00:00:00",
        token_renewal_next_at="2026-01-02T00:00:00",
    )

    auth.ensure_api_key_token_budget(key["id"], now=datetime(2026, 1, 2, 0, 0, 0))

    stored = auth.db.get_api_key_by_id(key["id"])
    assert stored["enabled"] == 0
    assert stored["token_usage"] == 9
    assert stored["token_renewal_next_at"] == "2026-01-02T00:00:00"


def test_expired_key_does_not_renew_usage(tmp_path):
    auth = _auth_service(tmp_path)
    key = _create_key(
        auth,
        token_renewal="daily",
        token_budget=5,
        expires_at="2026-01-01T12:00:00",
    )
    _set_renewal_state(
        auth,
        key["id"],
        token_usage=5,
        token_renewed_at="2026-01-01T00:00:00",
        token_renewal_next_at="2026-01-02T00:00:00",
    )

    with pytest.raises(HTTPException):
        auth.ensure_api_key_token_budget(key["id"], now=datetime(2026, 1, 2, 0, 0, 0))

    stored = auth.db.get_api_key_by_id(key["id"])
    assert stored["token_usage"] == 5
    assert stored["token_renewal_next_at"] == "2026-01-02T00:00:00"


@pytest.mark.asyncio
async def test_management_api_exposes_token_renewal_state(tmp_path):
    auth = _auth_service(tmp_path)
    response = await create_api_key(
        _admin_request(
            auth,
            {
                "name": "daily budget",
                "token_budget": 100,
                "token_renewal": "daily",
            },
        )
    )
    created = _json_body(response)

    assert created["token_usage"] == 0
    assert created["token_renewed_at"] is not None
    assert created["token_renewal_next_at"] is not None

    listed_response = await list_api_keys(_admin_request(auth, {}))
    listed = _json_body(listed_response)
    assert listed[0]["token_usage"] == 0
    assert listed[0]["token_renewed_at"] == created["token_renewed_at"]
    assert listed[0]["token_renewal_next_at"] == created["token_renewal_next_at"]


def test_api_key_token_renewal_columns_are_added_to_existing_databases(tmp_path):
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

    assert {"token_renewed_at", "token_renewal_next_at"}.issubset(columns)
