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
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService
from xinference.api.oauth2.advanced.routes import list_api_keys


def _json_body(response):
    return json.loads(response.body.decode())


def _auth_service(tmp_path):
    return AdvancedAuthService(
        db_path=str(tmp_path / "auth.db"),
        jwt_secret_key="test-jwt-secret",
        encryption_key="test-encryption-key",
    )


def _admin_request(auth, body=None):
    admin = auth.db.get_user_by_username("admin")
    token = auth.create_access_token(
        admin["id"],
        admin["username"],
        ["admin", "keys:create", "keys:manage"],
    )
    request = MagicMock()
    request.app.state.advanced_auth = auth
    request.headers = {"Authorization": f"Bearer {token}"}
    request.json = AsyncMock(return_value=body or {})
    return request


@pytest.mark.asyncio
async def test_key_list_exposes_usage_state_without_plaintext_secret(tmp_path):
    auth = _auth_service(tmp_path)
    admin = auth.db.get_user_by_username("admin")
    created = auth.create_api_key_for_user(
        user_id=admin["id"],
        name="observable key",
        token_budget=10,
        token_renewal="daily",
        request_rate_limit_enabled=True,
        request_rate_limit_requests=5,
        request_rate_limit_window_seconds=60,
    )
    auth.db.increment_api_key_token_usage(created["id"], 7)
    auth.record_api_key_request_success(created["id"], now=datetime.now())

    listed = _json_body(await list_api_keys(_admin_request(auth)))[0]

    assert "key" not in listed
    assert listed["token_usage"] == 7
    assert listed["token_remaining"] == 3
    assert listed["token_budget_exhausted"] is False
    assert listed["token_renewal_next_at"] is not None
    assert listed["request_rate_limit_count"] == 1
    assert listed["request_rate_limit_remaining"] == 4
    assert listed["request_rate_limit_reset_at"] is not None
    assert listed["rotated_at"] is None


@pytest.mark.asyncio
async def test_key_list_marks_exhausted_token_budget(tmp_path):
    auth = _auth_service(tmp_path)
    admin = auth.db.get_user_by_username("admin")
    created = auth.create_api_key_for_user(
        user_id=admin["id"],
        name="exhausted key",
        token_budget=10,
    )
    auth.db.increment_api_key_token_usage(created["id"], 10)

    listed = _json_body(await list_api_keys(_admin_request(auth)))[0]

    assert listed["token_remaining"] == 0
    assert listed["token_budget_exhausted"] is True
