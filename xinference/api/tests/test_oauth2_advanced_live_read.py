"""Regression tests for advanced-auth live-read permission enforcement.

Verifies that ``AdvancedAuthService.__call__`` checks route scopes against
the user's DB-current permissions rather than the scopes baked into the JWT.
This makes admin permission changes (grant/revoke) take effect on the next
request instead of requiring the user to re-login.

Also covers the refresh path: ``refresh_access_token`` must re-read
permissions from DB and bake them into the new access token, with refresh
token rotation invalidating the old token.

    pytest xinference/api/tests/test_oauth2_advanced_live_read.py -v
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.security import SecurityScopes
from fastapi.testclient import TestClient
from jose import jwt

from xinference.api.oauth2.advanced.auth_service import (
    JWT_ALGORITHM,
    AdvancedAuthService,
)
from xinference.api.oauth2.advanced.routes import register_advanced_auth_routes


def _make_request(path: str = "/v1/admin/keys"):
    request = MagicMock()
    request.url.path = path
    request.method = "GET"
    request.headers = {"content-type": "application/json", "content-length": "0"}
    request.client.host = "127.0.0.1"
    request.body = AsyncMock(return_value=b"{}")
    return request


@pytest.fixture
def auth_service(tmp_path):
    db_path = str(tmp_path / "auth.db")
    service = AdvancedAuthService(
        db_path=db_path,
        jwt_secret_key="unit-test-secret",
        encryption_key="unit-test-encryption-key",
    )
    return service


def _create_user(auth_service, username, permissions):
    from xinference.api.oauth2.advanced.crypto import get_password_hash

    user_id = auth_service.db.create_user(
        username=username,
        password_hash=get_password_hash("pass"),
        source="local",
        enabled=1,
        must_change_password=0,
        permissions=permissions,
    )
    return user_id


def _make_client(auth_service):
    app = FastAPI()
    router = APIRouter()
    app.state.advanced_auth = auth_service
    register_advanced_auth_routes(SimpleNamespace(_app=app, _router=router))
    app.include_router(router)
    return TestClient(app)


@pytest.mark.asyncio
async def test_jwt_scope_check_uses_db_permissions_revoke(auth_service):
    """JWT has ``keys:create`` but DB revoked it → 403.

    Proves the scope check reads DB-current permissions, not the JWT
    snapshot. Without live-read, this request would succeed (JWT still
    has ``keys:create``) and the revocation wouldn't take effect until
    token expiry.
    """
    user_id = _create_user(auth_service, "alice", ["keys:create", "models:read"])
    token = auth_service.create_access_token(
        user_id, "alice", ["keys:create", "models:read"]
    )
    # Admin revokes keys:create in the DB after login.
    auth_service.db.set_user_permissions(user_id, ["models:read"])

    with pytest.raises(HTTPException) as exc:
        await auth_service(
            _make_request(),
            SecurityScopes(scopes=["keys:create"]),
            token,
        )
    assert exc.value.status_code == 403
    assert exc.value.detail == "Not enough permissions"


@pytest.mark.asyncio
async def test_jwt_scope_check_uses_db_permissions_grant(auth_service):
    """JWT lacks ``keys:create`` but DB granted it → 200.

    The mirror of the revoke case: a newly-granted permission takes effect
    immediately on the next request, without waiting for token refresh.
    """
    user_id = _create_user(auth_service, "bob", ["models:read"])
    token = auth_service.create_access_token(user_id, "bob", ["models:read"])
    # Admin grants keys:create in the DB after login.
    auth_service.db.set_user_permissions(user_id, ["models:read", "keys:create"])

    user = await auth_service(
        _make_request(),
        SecurityScopes(scopes=["keys:create"]),
        token,
    )
    assert user["username"] == "bob"


@pytest.mark.asyncio
async def test_admin_bypass_still_uses_jwt(auth_service):
    """Admin bypass at L550 checks JWT scopes, not DB permissions.

    This is by design: the ``admin`` wildcard stays in the JWT and cannot
    be revoked mid-session via DB changes. If a deployment needs to
    revoke admin mid-session, that's a separate issue (e.g., disable the
    user account, which IS checked against DB at L546).
    """
    user_id = _create_user(auth_service, "admin2", ["admin", "models:read"])
    token = auth_service.create_access_token(
        user_id, "admin2", ["admin", "models:read"]
    )
    # Even if DB no longer lists admin, the JWT still carries it.
    auth_service.db.set_user_permissions(user_id, ["models:read"])

    user = await auth_service(
        _make_request(),
        SecurityScopes(scopes=["users:manage"]),
        token,
    )
    assert user["username"] == "admin2"


def test_refresh_re_reads_permissions_from_db(auth_service):
    """``refresh_access_token`` bakes DB-current permissions into the new JWT.

    Independent of the live-read scope check, the refresh path must also
    re-read permissions so that tokens issued post-refresh reflect the
    latest DB state.
    """
    user_id = _create_user(auth_service, "carol", ["models:read"])
    refresh_token = auth_service.create_refresh_token(user_id)
    # Grant keys:create after the initial refresh token was issued.
    auth_service.db.set_user_permissions(user_id, ["models:read", "keys:create"])

    result = auth_service.refresh_access_token(refresh_token)
    assert result is not None
    payload = jwt.decode(
        result["access_token"],
        "unit-test-secret",
        algorithms=[JWT_ALGORITHM],
    )
    assert "keys:create" in payload["scopes"]
    assert "models:read" in payload["scopes"]


def test_refresh_token_rotation_invalidates_old(auth_service):
    """Refresh token rotation: the old refresh token is invalidated on use.

    A second refresh with the same (now-rotated) refresh token must fail.
    """
    user_id = _create_user(auth_service, "dave", ["models:read"])
    refresh_token = auth_service.create_refresh_token(user_id)

    first = auth_service.refresh_access_token(refresh_token)
    assert first is not None

    second = auth_service.refresh_access_token(refresh_token)
    assert second is None


def test_key_reveal_route_uses_db_permissions_grant(auth_service):
    """The key reveal route should honor DB-current ``keys:manage`` grants."""
    user_id = _create_user(auth_service, "gina", [])
    token = auth_service.create_access_token(user_id, "gina", [])
    auth_service.db.set_user_permissions(user_id, ["keys:manage"])
    api_key = auth_service.create_api_key_for_user(user_id=user_id)

    response = _make_client(auth_service).get(
        f"/v1/admin/keys/{api_key['id']}/reveal",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["key"] == api_key["key"]


def test_keys_read_route_uses_db_permissions_grant(auth_service):
    """The custom OR dependency for key reads must also live-read DB scopes."""
    user_id = _create_user(auth_service, "harry", [])
    other_user_id = _create_user(auth_service, "harry-other", [])
    token = auth_service.create_access_token(user_id, "harry", [])
    auth_service.db.set_user_permissions(user_id, ["keys:manage"])
    own_key = auth_service.create_api_key_for_user(user_id=user_id)
    other_key = auth_service.create_api_key_for_user(user_id=other_user_id)

    response = _make_client(auth_service).get(
        "/v1/admin/keys", headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 200
    assert {item["id"] for item in response.json()} == {own_key["id"], other_key["id"]}


def test_key_get_route_uses_db_permissions_grant(auth_service):
    """``keys:manage`` granted after login should allow reading any key."""
    user_id = _create_user(auth_service, "jane", [])
    other_user_id = _create_user(auth_service, "jane-other", [])
    token = auth_service.create_access_token(user_id, "jane", [])
    auth_service.db.set_user_permissions(user_id, ["keys:manage"])
    other_key = auth_service.create_api_key_for_user(user_id=other_user_id)

    response = _make_client(auth_service).get(
        f"/v1/admin/keys/{other_key['id']}",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["id"] == other_key["id"]


def test_key_create_route_uses_db_permissions_grant_for_owner(auth_service):
    """A live-granted key manager can create keys for another owner."""
    user_id = _create_user(auth_service, "kate", [])
    other_user_id = _create_user(auth_service, "kate-other", [])
    token = auth_service.create_access_token(user_id, "kate", [])
    auth_service.db.set_user_permissions(user_id, ["keys:create", "keys:manage"])

    response = _make_client(auth_service).post(
        "/v1/admin/keys",
        json={"owner": other_user_id},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 201
    key = auth_service.db.get_api_key_by_id(response.json()["id"])
    assert key["user_id"] == other_user_id


def test_keys_read_route_rejects_api_key_tokens(auth_service):
    """API keys must not access key-management routes via owner permissions."""
    user_id = _create_user(auth_service, "irene", ["keys:manage"])
    api_key = auth_service.create_api_key_for_user(user_id=user_id)["key"]

    response = _make_client(auth_service).get(
        "/v1/admin/keys", headers={"Authorization": f"Bearer {api_key}"}
    )

    assert response.status_code == 403


def test_validate_model_access_grant_after_login(auth_service):
    """``validate_model_access`` reflects DB-current permissions for JWTs.

    If admin grants ``models:read`` after login, the JWT still lacks it,
    but ``validate_model_access`` must read DB and grant access — otherwise
    the route-level ``Security(scopes=["models:read"])`` would pass while
    ``_check_model_access`` 403s, leaving the grant ineffective for
    inference endpoints.
    """
    user_id = _create_user(auth_service, "erin", [])
    token = auth_service.create_access_token(user_id, "erin", [])
    # Admin grants models:read in DB after login.
    auth_service.db.set_user_permissions(user_id, ["models:read"])

    assert auth_service.validate_model_access(token, "any-model", "LLM") is True


def test_validate_model_access_revoke_after_login(auth_service):
    """Revocation must also take effect in ``validate_model_access``.

    JWT still carries ``models:read`` (snapshot), but DB revoked it.
    ``validate_model_access`` must return False so the model request is
    denied, consistent with the route-level DB scope check.
    """
    user_id = _create_user(auth_service, "frank", ["models:read"])
    token = auth_service.create_access_token(user_id, "frank", ["models:read"])
    # Admin revokes models:read in DB after login.
    auth_service.db.set_user_permissions(user_id, [])

    assert auth_service.validate_model_access(token, "any-model", "LLM") is False
