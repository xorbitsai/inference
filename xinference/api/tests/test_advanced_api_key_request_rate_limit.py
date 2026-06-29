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

import concurrent.futures
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService
from xinference.api.oauth2.advanced.database import Database
from xinference.api.oauth2.advanced.rate_limiter import RateLimitConfig
from xinference.api.oauth2.advanced.routes import create_api_key, list_api_keys
from xinference.api.restful_api import RESTfulAPI


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
        name=kwargs.pop("name", "limited key"),
        **kwargs,
    )


def _api(auth):
    api = object.__new__(RESTfulAPI)
    api._advanced_auth_service = auth
    api._uid_to_model_name = {}
    api._set_trace_model = MagicMock()
    api._set_trace_model_type = MagicMock()
    api._get_supervisor_ref = AsyncMock()
    api._report_error_event = AsyncMock()
    api._get_model_last_error = AsyncMock(side_effect=lambda _uid, exc: exc)
    return api


def _request(api_key, payload, path="/v1/completions"):
    request = MagicMock()
    request.headers = {"Authorization": f"Bearer {api_key}"}
    request.state = SimpleNamespace()
    request.client = SimpleNamespace(host="127.0.0.1")
    request.url = SimpleNamespace(path=path)
    request.json = AsyncMock(return_value=payload)
    return request


class _CompletionModel:
    uid = b"test-model"

    async def generate(self, *_args, **_kwargs):
        return json.dumps({"choices": [{"text": "hello"}]})

    async def is_vllm_backend(self):
        return False


def test_request_rate_limit_disabled_is_noop(tmp_path):
    auth = _auth_service(tmp_path)
    key = _create_key(auth, request_rate_limit_enabled=False)

    auth.ensure_api_key_request_rate_limit(key["id"], now=datetime(2026, 1, 1, 0, 0, 0))
    auth.record_api_key_request_success(key["id"], now=datetime(2026, 1, 1, 0, 0, 1))

    stored = auth.db.get_api_key_by_id(key["id"])
    assert stored["request_rate_limit_count"] == 0
    assert stored["request_rate_limit_window_started_at"] is None


def test_request_rate_limit_rejects_after_successful_window_is_exhausted(tmp_path):
    auth = _auth_service(tmp_path)
    key = _create_key(
        auth,
        request_rate_limit_enabled=True,
        request_rate_limit_requests=2,
        request_rate_limit_window_seconds=60,
    )
    now = datetime(2026, 1, 1, 0, 0, 0)

    auth.ensure_api_key_request_rate_limit(key["id"], now=now)
    auth.record_api_key_request_success(key["id"], now=now)
    auth.ensure_api_key_request_rate_limit(key["id"], now=now + timedelta(seconds=1))
    auth.record_api_key_request_success(key["id"], now=now + timedelta(seconds=1))

    with pytest.raises(HTTPException) as exc_info:
        auth.ensure_api_key_request_rate_limit(
            key["id"], now=now + timedelta(seconds=2)
        )

    state = auth.db.get_api_key_request_rate_limit_state(key["id"])
    assert exc_info.value.status_code == 429
    assert "API key request rate limit exceeded" in exc_info.value.detail
    assert state["request_rate_limit_count"] == 2
    assert state["request_rate_limit_remaining"] == 0
    assert state["request_rate_limit_reset_at"] == "2026-01-01T00:01:00"


def test_request_rate_limit_window_reset_allows_requests_again(tmp_path):
    auth = _auth_service(tmp_path)
    key = _create_key(
        auth,
        request_rate_limit_enabled=True,
        request_rate_limit_requests=1,
        request_rate_limit_window_seconds=10,
    )
    now = datetime(2026, 1, 1, 0, 0, 0)
    auth.record_api_key_request_success(key["id"], now=now)

    auth.ensure_api_key_request_rate_limit(key["id"], now=now + timedelta(seconds=11))
    auth.record_api_key_request_success(key["id"], now=now + timedelta(seconds=11))

    state = auth.db.get_api_key_request_rate_limit_state(key["id"])
    assert state["request_rate_limit_count"] == 1
    assert state["request_rate_limit_window_started_at"] == "2026-01-01T00:00:11"
    assert state["request_rate_limit_reset_at"] == "2026-01-01T00:00:21"


def test_request_rate_limit_is_scoped_to_api_key_not_client_ip(tmp_path):
    auth = _auth_service(tmp_path)
    first = _create_key(
        auth,
        name="first",
        request_rate_limit_enabled=True,
        request_rate_limit_requests=1,
        request_rate_limit_window_seconds=60,
    )
    second = _create_key(
        auth,
        name="second",
        request_rate_limit_enabled=True,
        request_rate_limit_requests=1,
        request_rate_limit_window_seconds=60,
    )
    now = datetime(2026, 1, 1, 0, 0, 0)

    auth.record_api_key_request_success(first["id"], now=now)

    with pytest.raises(HTTPException):
        auth.ensure_api_key_request_rate_limit(first["id"], now=now)
    auth.ensure_api_key_request_rate_limit(second["id"], now=now)


def test_request_rate_limit_does_not_touch_failed_authentication_bans(tmp_path):
    auth = _auth_service(tmp_path)
    key = _create_key(
        auth,
        request_rate_limit_enabled=True,
        request_rate_limit_requests=5,
        request_rate_limit_window_seconds=60,
    )
    client_ip = "127.0.0.1"
    auth._rate_limiter.record_key_failure(
        client_ip,
        key["id"],
        RateLimitConfig(max_failures=1, window_seconds=60, ban_seconds=60),
    )

    auth.record_api_key_request_success(key["id"], now=datetime.now())

    state = auth.db.get_api_key_request_rate_limit_state(key["id"])
    assert state["request_rate_limit_count"] == 1
    assert auth._rate_limiter.is_key_banned(client_ip, key["id"]) is True


def test_request_rate_limit_success_count_is_concurrency_safe(tmp_path, monkeypatch):
    auth = _auth_service(tmp_path)
    key = _create_key(
        auth,
        request_rate_limit_enabled=True,
        request_rate_limit_requests=5,
        request_rate_limit_window_seconds=60,
    )
    now = datetime(2026, 1, 1, 0, 0, 0)
    original_get = auth.db.get_api_key_request_rate_limit_state
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    read_count = 0

    def synchronized_get(key_id):
        nonlocal read_count
        state = original_get(key_id)
        with lock:
            read_count += 1
            should_wait = read_count <= 2
        if should_wait:
            barrier.wait(timeout=5)
        return state

    monkeypatch.setattr(
        auth.db, "get_api_key_request_rate_limit_state", synchronized_get
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(auth.record_api_key_request_success, key["id"], now)
            for _ in range(2)
        ]
        for future in futures:
            future.result(timeout=5)

    state = original_get(key["id"])
    assert state["request_rate_limit_count"] == 2
    assert state["request_rate_limit_remaining"] == 3


@pytest.mark.asyncio
async def test_management_api_exposes_request_rate_limit_state(tmp_path):
    auth = _auth_service(tmp_path)
    create_response = await create_api_key(
        _admin_request(
            auth,
            {
                "name": "rate limited",
                "request_rate_limit_enabled": True,
                "request_rate_limit_requests": 3,
                "request_rate_limit_window_seconds": 60,
            },
        )
    )
    created = _json_body(create_response)
    now = datetime.now().replace(microsecond=0)
    auth.record_api_key_request_success(created["id"], now=now)

    list_response = await list_api_keys(_admin_request(auth, {}))
    listed = _json_body(list_response)[0]

    assert created["request_rate_limit_count"] == 0
    assert created["request_rate_limit_remaining"] == 3
    assert created["request_rate_limit_reset_at"] is None
    assert listed["request_rate_limit_count"] == 1
    assert listed["request_rate_limit_remaining"] == 2
    assert (
        listed["request_rate_limit_reset_at"]
        == (now + timedelta(seconds=60)).isoformat()
    )


@pytest.mark.asyncio
async def test_management_api_refreshes_elapsed_request_rate_limit_window(tmp_path):
    auth = _auth_service(tmp_path)
    create_response = await create_api_key(
        _admin_request(
            auth,
            {
                "name": "elapsed window",
                "request_rate_limit_enabled": True,
                "request_rate_limit_requests": 1,
                "request_rate_limit_window_seconds": 1,
            },
        )
    )
    created = _json_body(create_response)
    auth.record_api_key_request_success(
        created["id"], now=datetime(2026, 1, 1, 0, 0, 0)
    )

    list_response = await list_api_keys(_admin_request(auth, {}))
    listed = _json_body(list_response)[0]

    assert listed["request_rate_limit_count"] == 0
    assert listed["request_rate_limit_remaining"] == 1
    assert listed["request_rate_limit_reset_at"] is None


@pytest.mark.asyncio
async def test_exhausted_request_rate_limit_rejects_before_model_work(
    tmp_path, monkeypatch
):
    auth = _auth_service(tmp_path)
    key = _create_key(
        auth,
        request_rate_limit_enabled=True,
        request_rate_limit_requests=1,
        request_rate_limit_window_seconds=60,
    )
    auth.record_api_key_request_success(key["id"], now=datetime.now())
    require_model = AsyncMock(return_value=_CompletionModel())
    monkeypatch.setattr("xinference.api.restful_api.require_model", require_model)

    with pytest.raises(HTTPException) as exc_info:
        await _api(auth).create_completion(
            _request(
                key["key"],
                {"model": "test-model", "prompt": "hello", "stream": False},
            )
        )

    assert exc_info.value.status_code == 429
    require_model.assert_not_called()


def test_api_key_request_rate_limit_columns_are_added_to_existing_databases(tmp_path):
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
        "request_rate_limit_count",
        "request_rate_limit_window_started_at",
    }.issubset(columns)
