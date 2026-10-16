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
"""Regression tests for non-string payload handling on admin routes.

Gemini review on PR #5144 pointed out that ``create_user`` and
``change_password`` parse ``username``/``password``/``new_password`` from
the raw JSON body without checking they're strings. A client sending a
non-string value (e.g. an int or a list) would otherwise hit
``len(password)`` and crash with an unhandled ``TypeError`` -> 500,
instead of a clean 400. ``setup_admin`` has its own coverage in
test_admin_setup_route.py.

    pytest xinference/api/tests/test_admin_route_input_validation.py -v
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from xinference.api.oauth2.advanced.routes import change_password, create_user


def _make_request(body: dict):
    request = MagicMock()
    request.json = AsyncMock(return_value=body)
    return request


@pytest.mark.asyncio
async def test_create_user_rejects_non_string_password():
    request = _make_request({"username": "alice", "password": 12345})
    with pytest.raises(HTTPException) as exc:
        await create_user(request)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_create_user_rejects_non_string_username():
    request = _make_request({"username": ["alice"], "password": "s3cret!"})
    with pytest.raises(HTTPException) as exc:
        await create_user(request)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_change_password_rejects_non_string_new_password(monkeypatch):
    import xinference.api.oauth2.advanced.routes as routes

    auth = MagicMock()
    auth.db.get_user_by_id.return_value = {
        "id": 2,
        "username": "bob",
        "source": "local",
        "permissions": [],
    }
    monkeypatch.setattr(routes, "get_advanced_auth", lambda request: auth)
    monkeypatch.setattr(
        routes, "_reject_admin_target_takeover", lambda request, auth, uid: None
    )

    request = _make_request({"new_password": ["not", "a", "string"]})
    with pytest.raises(HTTPException) as exc:
        await change_password(user_id=2, request=request)
    assert exc.value.status_code == 400
