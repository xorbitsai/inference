"""Regression tests for #5058 Finding 5: a user still flagged
``must_change_password`` must be blocked server-side from every request
except changing their own password -- enforced in the auth dependency
(``AdvancedAuthService.__call__``), not merely returned to the frontend.

This gate matters for existing databases: an account migrated from an older
release may carry ``must_change_password=1``, and without a server-side check
it could obtain fully-functional (including admin) tokens.

    pytest xinference/api/tests/test_oauth2_must_change_password.py -v
"""

import asyncio

import pytest
from fastapi import HTTPException
from fastapi.security import SecurityScopes
from starlette.requests import Request

from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService


def _make_service(tmp_path):
    return AdvancedAuthService(
        db_path=str(tmp_path / "auth.db"),
        jwt_secret_key="unit-test-secret",
        encryption_key="unit-test-encryption-key",
    )


def _request(method: str, path: str) -> Request:
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 5555),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


def _run(coro):
    return asyncio.run(coro)


def _make_user(service, permissions, must_change_password):
    user_id = service.db.create_user(
        username="u",
        password_hash="x",
        source="local",
        permissions=permissions,
        must_change_password=1 if must_change_password else 0,
    )
    token = service.create_access_token(user_id, "u", permissions)
    return user_id, token


def test_flagged_user_blocked_on_other_endpoints(tmp_path):
    service = _make_service(tmp_path)
    user_id, token = _make_user(service, ["users:manage"], must_change_password=True)
    with pytest.raises(HTTPException) as exc:
        _run(
            service(
                _request("GET", "/v1/admin/users"),
                SecurityScopes(scopes=["users:manage"]),
                token,
            )
        )
    assert exc.value.status_code == 403
    assert "Password change required" in exc.value.detail


def test_flagged_user_allowed_own_password_change(tmp_path):
    service = _make_service(tmp_path)
    user_id, token = _make_user(service, ["users:manage"], must_change_password=True)
    result = _run(
        service(
            _request("PUT", f"/v1/admin/users/{user_id}/password"),
            SecurityScopes(scopes=["users:manage"]),
            token,
        )
    )
    assert result["id"] == user_id


def test_flagged_user_cannot_change_other_users_password(tmp_path):
    service = _make_service(tmp_path)
    user_id, token = _make_user(service, ["users:manage"], must_change_password=True)
    other_id = user_id + 999
    with pytest.raises(HTTPException) as exc:
        _run(
            service(
                _request("PUT", f"/v1/admin/users/{other_id}/password"),
                SecurityScopes(scopes=["users:manage"]),
                token,
            )
        )
    assert exc.value.status_code == 403
    assert "Password change required" in exc.value.detail


def test_flagged_admin_is_gated_before_admin_bypass(tmp_path):
    """An account holding ``admin`` must still be blocked while flagged; the
    gate runs before the admin scope bypass, otherwise the most dangerous
    account would escape enforcement."""
    service = _make_service(tmp_path)
    user_id, token = _make_user(service, ["admin"], must_change_password=True)
    with pytest.raises(HTTPException) as exc:
        _run(
            service(
                _request("GET", "/v1/admin/keys"),
                SecurityScopes(scopes=[]),
                token,
            )
        )
    assert exc.value.status_code == 403
    # ...but the same admin may change their own password.
    result = _run(
        service(
            _request("PUT", f"/v1/admin/users/{user_id}/password"),
            SecurityScopes(scopes=[]),
            token,
        )
    )
    assert result["id"] == user_id


def test_unflagged_user_is_unaffected(tmp_path):
    service = _make_service(tmp_path)
    user_id, token = _make_user(service, ["users:manage"], must_change_password=False)
    result = _run(
        service(
            _request("GET", "/v1/admin/users"),
            SecurityScopes(scopes=["users:manage"]),
            token,
        )
    )
    assert result["id"] == user_id


def test_get_on_password_path_is_not_a_bypass(tmp_path):
    """Only PUT to the own-password endpoint is exempt; a GET to the same path
    must not slip past the gate."""
    service = _make_service(tmp_path)
    user_id, token = _make_user(service, ["users:manage"], must_change_password=True)
    with pytest.raises(HTTPException) as exc:
        _run(
            service(
                _request("GET", f"/v1/admin/users/{user_id}/password"),
                SecurityScopes(scopes=["users:manage"]),
                token,
            )
        )
    assert exc.value.status_code == 403


def test_flagged_account_api_key_is_blocked(tmp_path):
    """An API key owned by a still-flagged account must not reach model
    endpoints (the API-key path does not go through the JWT gate)."""
    service = _make_service(tmp_path)
    user_id = service.db.create_user(
        username="legacy",
        password_hash="x",
        source="local",
        permissions=["models:read"],
        must_change_password=1,
    )
    key = service.create_api_key_for_user(user_id=user_id, name="k")["key"]

    with pytest.raises(HTTPException) as exc:
        _run(
            service(
                _request("POST", "/v1/chat/completions"),
                SecurityScopes(scopes=["models:read"]),
                key,
            )
        )
    assert exc.value.status_code == 403
    assert "Password change required" in exc.value.detail


def test_api_key_works_again_after_flag_cleared(tmp_path):
    """Clearing must_change_password (e.g. the owner changed the password)
    restores the existing API key rather than banning it permanently."""
    service = _make_service(tmp_path)
    user_id = service.db.create_user(
        username="legacy",
        password_hash="x",
        source="local",
        permissions=["models:read"],
        must_change_password=1,
    )
    key = service.create_api_key_for_user(user_id=user_id, name="k")["key"]

    # Flag cleared as a password change would.
    service.db.update_user(user_id, must_change_password=0)

    result = _run(
        service(
            _request("POST", "/v1/chat/completions"),
            SecurityScopes(scopes=["models:read"]),
            key,
        )
    )
    assert result["id"] == user_id
