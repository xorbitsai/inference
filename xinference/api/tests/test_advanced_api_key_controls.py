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
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService
from xinference.api.oauth2.advanced.database import Database
from xinference.api.oauth2.advanced.routes import (
    create_api_key,
    list_api_keys,
    update_api_key,
)


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


@pytest.mark.asyncio
async def test_api_key_usage_controls_round_trip_through_create_list_and_update(
    tmp_path,
):
    auth = _auth_service(tmp_path)
    create_request = _admin_request(
        auth,
        {
            "name": "ci pipeline",
            "token_budget": 1_000_000,
            "token_renewal": "monthly",
            "expires_at": "2026-12-31T23:59:59",
            "request_rate_limit_enabled": True,
            "request_rate_limit_requests": 120,
            "request_rate_limit_window_seconds": 60,
        },
    )

    create_response = await create_api_key(create_request)
    created = _json_body(create_response)

    assert created["token_budget"] == 1_000_000
    assert created["token_renewal"] == "monthly"
    assert created["request_rate_limit_enabled"] is True
    assert created["request_rate_limit_requests"] == 120
    assert created["request_rate_limit_window_seconds"] == 60

    list_response = await list_api_keys(_admin_request(auth, {}))
    listed = _json_body(list_response)
    assert listed[0]["token_budget"] == 1_000_000
    assert listed[0]["token_renewal"] == "monthly"
    assert listed[0]["request_rate_limit_enabled"] is True

    update_request = _admin_request(
        auth,
        {
            "token_budget": 2_000_000,
            "token_renewal": "custom",
            "token_renewal_interval_days": 14,
            "request_rate_limit_enabled": False,
            "request_rate_limit_requests": None,
            "request_rate_limit_window_seconds": None,
        },
    )
    update_response = await update_api_key(created["id"], update_request)
    assert _json_body(update_response) == {"ok": True}

    key = auth.db.get_api_key_by_id(created["id"])
    assert key["token_budget"] == 2_000_000
    assert key["token_renewal"] == "custom"
    assert key["token_renewal_interval_days"] == 14
    assert key["request_rate_limit_enabled"] == 0
    assert key["request_rate_limit_requests"] is None
    assert key["request_rate_limit_window_seconds"] is None


def test_api_key_usage_control_columns_are_added_to_existing_databases(tmp_path):
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

    assert {
        "description",
        "token_budget",
        "token_renewal",
        "token_renewal_interval_days",
        "request_rate_limit_enabled",
        "request_rate_limit_requests",
        "request_rate_limit_window_seconds",
    }.issubset(columns)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, message",
    [
        ({"token_budget": -1}, "token_budget must be greater than 0"),
        ({"token_renewal": "weekly"}, "token_renewal must be one of"),
        (
            {"token_renewal": "custom"},
            "token_renewal_interval_days is required",
        ),
        (
            {"request_rate_limit_enabled": True, "request_rate_limit_requests": 10},
            "request_rate_limit_window_seconds is required",
        ),
        (
            {
                "request_rate_limit_enabled": True,
                "request_rate_limit_requests": 0,
                "request_rate_limit_window_seconds": 60,
            },
            "request_rate_limit_requests must be greater than 0",
        ),
    ],
)
async def test_invalid_api_key_usage_control_values_are_rejected(
    tmp_path, body, message
):
    auth = _auth_service(tmp_path)
    request = _admin_request(auth, body)

    with pytest.raises(HTTPException) as exc_info:
        await create_api_key(request)

    assert exc_info.value.status_code == 400
    assert message in exc_info.value.detail
