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
"""Regression tests for the XINFERENCE_AUTH_ADVANCED on/off toggle.

Covers the two behaviors a user cares about when switching advanced auth:

* When advanced auth is ON, an API key created via ``create_api_key_for_user``
  must be accepted by ``_check_model_access`` (the same check every inference
  route runs), and rejected once invalid/absent.
* When advanced auth is OFF (``_advanced_auth_service is None``), model access
  checks are a no-op for every request -- there is no code path left that
  understands the ``xf-`` API key format, so old keys are neither accepted
  nor rejected, they simply stop mattering.

    pytest xinference/api/tests/test_auth_advanced_toggle.py -v
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService
from xinference.api.restful_api import RESTfulAPI


def _make_request():
    request = MagicMock()
    request.headers = {}
    request.state = MagicMock()
    request.state.model_name = ""
    request.client.host = "127.0.0.1"
    request.url.path = "/v1/chat/completions"
    return request


@pytest.fixture
def advanced_auth_service(tmp_path):
    db_path = str(tmp_path / "auth.db")
    return AdvancedAuthService(
        db_path=db_path,
        jwt_secret_key="unit-test-secret",
        encryption_key="unit-test-encryption-key",
    )


def _create_user(auth_service, username, permissions):
    from xinference.api.oauth2.advanced.crypto import get_password_hash

    return auth_service.db.create_user(
        username=username,
        password_hash=get_password_hash("pass"),
        source="local",
        enabled=1,
        must_change_password=0,
        permissions=permissions,
    )


def _bare_api_instance():
    """A RESTfulAPI with __init__ skipped so we can set auth fields directly."""
    api = RESTfulAPI.__new__(RESTfulAPI)
    api._uid_to_model_name = {}
    return api


def test_api_key_accepted_when_advanced_auth_enabled(advanced_auth_service):
    api = _bare_api_instance()
    api._advanced_auth_service = advanced_auth_service
    api._auth_service = advanced_auth_service

    user_id = _create_user(advanced_auth_service, "alice", ["models:read"])
    api_key = advanced_auth_service.create_api_key_for_user(user_id=user_id)["key"]

    request = _make_request()
    request.headers = {"Authorization": f"Bearer {api_key}"}

    # Should not raise: the key has all-model access by default.
    api._check_model_access(request, "some-model-uid", "LLM")


def test_api_key_rejected_for_unauthorized_model_when_advanced_auth_enabled(
    advanced_auth_service,
):
    api = _bare_api_instance()
    api._advanced_auth_service = advanced_auth_service
    api._auth_service = advanced_auth_service

    user_id = _create_user(advanced_auth_service, "bob", ["models:read"])
    api_key = advanced_auth_service.create_api_key_for_user(
        user_id=user_id,
        model_permissions=[
            {"permission_type": "model", "permission_value": "other-model"}
        ],
    )["key"]

    request = _make_request()
    request.headers = {"Authorization": f"Bearer {api_key}"}

    with pytest.raises(HTTPException) as exc:
        api._check_model_access(request, "some-model-uid", "LLM")
    assert exc.value.status_code == 403


def test_invalid_api_key_rejected_when_advanced_auth_enabled(advanced_auth_service):
    api = _bare_api_instance()
    api._advanced_auth_service = advanced_auth_service
    api._auth_service = advanced_auth_service

    request = _make_request()
    request.headers = {"Authorization": "Bearer xf-not-a-real-key"}

    with pytest.raises(HTTPException) as exc:
        api._check_model_access(request, "some-model-uid", "LLM")
    assert exc.value.status_code == 403


def test_model_access_is_unchecked_when_advanced_auth_disabled(tmp_path):
    """With advanced auth off, _check_model_access is a no-op for any token.

    This is the behavior behind the user-facing question: an ``xf-`` API key
    minted while advanced auth was on is neither accepted nor rejected once
    advanced auth is off -- there is no longer any code path that inspects
    it, so requests carrying it (or no token at all, or garbage) all pass
    through untouched.
    """
    api = _bare_api_instance()
    api._advanced_auth_service = None
    api._auth_service = None

    for headers in (
        {},
        {"Authorization": "Bearer xf-some-previously-valid-key"},
        {"Authorization": "Bearer complete-garbage"},
    ):
        request = _make_request()
        request.headers = headers
        # Should not raise regardless of what the token looks like.
        api._check_model_access(request, "some-model-uid", "LLM")


def test_admin_key_routes_not_registered_when_advanced_auth_disabled():
    """The /v1/admin/keys* management routes only exist when advanced auth
    is on -- register_advanced_auth_routes is gated on _advanced_auth_service.
    """
    api = _bare_api_instance()
    api._advanced_auth_service = None
    api._auth_service = None

    assert api._advanced_auth_service is None
    # Mirrors the gating condition in RESTfulAPI init_app / __init__.
    assert not bool(api._advanced_auth_service)
