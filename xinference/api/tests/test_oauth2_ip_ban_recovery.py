# CI: linux
"""IP bans must remain recoverable without treating expired sessions as attacks."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Security
from fastapi.testclient import TestClient
from jose import jwt

from xinference.api.oauth2.advanced import rate_limiter
from xinference.api.oauth2.advanced.auth_service import (
    JWT_ALGORITHM,
    AdvancedAuthService,
)
from xinference.api.oauth2.advanced.crypto import get_password_hash
from xinference.api.oauth2.advanced.routes import register_advanced_auth_routes

CLIENT_IP = "testclient"
SECRET = "ip-ban-regression-secret"


@pytest.fixture(scope="module")
def password_hash():
    return get_password_hash("test-password")


@pytest.fixture
def setup(tmp_path, password_hash):
    auth = AdvancedAuthService(
        db_path=str(tmp_path / "auth.db"),
        jwt_secret_key=SECRET,
        encryption_key="ip-ban-regression-encryption",
    )
    auth._rate_limiter.update_ip_config(max_failures=3)
    user_id = auth.db.create_user(
        "admin", password_hash=password_hash, permissions=["admin"]
    )
    token = auth.create_access_token(user_id, "admin", ["admin"])
    app = FastAPI()
    app.state.advanced_auth = auth
    register_advanced_auth_routes(SimpleNamespace(_app=app, _router=app.router))

    @app.get("/protected")
    async def protected(_user=Security(auth, scopes=["models:read"])):
        return {"ok": True}

    with TestClient(app) as client:
        yield auth, client, user_id, token


def _headers(token):
    return {"Authorization": f"Bearer {token}"}


def _expired_token(token, *, secret=SECRET, token_type="access"):
    claims = jwt.get_unverified_claims(token)
    claims["exp"] = datetime.utcnow() - timedelta(minutes=1)
    claims["type"] = token_type
    return jwt.encode(claims, secret, algorithm=JWT_ALGORITHM)


def _ban_ip(auth):
    for _ in range(auth._rate_limiter._ip_config.max_failures):
        auth._rate_limiter.record_invalid_key(CLIENT_IP)
    assert auth._rate_limiter.is_ip_banned(CLIENT_IP)


def test_expired_session_can_refresh_without_banning_ip(setup):
    auth, client, user_id, token = setup
    expired = _expired_token(token)
    for _ in range(12):
        response = client.get("/protected", headers=_headers(expired))
        assert response.status_code == 401
    assert not auth._rate_limiter.is_ip_banned(CLIENT_IP)
    refresh = auth.create_refresh_token(user_id)
    response = client.post("/v1/auth/refresh", json={"refresh_token": refresh})
    assert response.status_code == 200
    assert (
        client.get(
            "/protected", headers=_headers(response.json()["access_token"])
        ).status_code
        == 200
    )


@pytest.mark.parametrize(
    "credential", ["invalid_key", "malformed", "forged", "refresh"]
)
def test_invalid_credentials_still_trigger_ip_ban(setup, credential):
    auth, client, _, token = setup
    invalid = {
        "invalid_key": "xf-invalid-key",
        "malformed": "not-a-jwt",
        "forged": _expired_token(token, secret="wrong-secret"),
        "refresh": _expired_token(token, token_type="refresh"),
    }[credential]
    for _ in range(3):
        assert client.get("/protected", headers=_headers(invalid)).status_code == 401
    response = client.get("/protected", headers=_headers(token))
    assert response.status_code == 429
    assert response.json() == {
        "detail": "Too many failed attempts. IP temporarily banned."
    }
    assert response.headers["x-xinference-error-code"] == "ip_banned"
    assert 0 < int(response.headers["retry-after"]) <= 3600


def test_banned_admin_can_log_in_and_unban_but_cannot_use_other_routes(setup):
    auth, client, _, _ = setup
    _ban_ip(auth)
    response = client.post(
        "/token", json={"username": "admin", "password": "test-password"}
    )
    assert response.status_code == 200
    headers = _headers(response.json()["access_token"])
    for path in ("/protected", "/v1/admin/users"):
        assert client.get(path, headers=headers).status_code == 429
    response = client.get("/v1/auth/me", headers=headers)
    assert response.status_code == 200
    assert response.json()["permissions"] == ["admin"]
    for path in ("rate-limit", "banned-ips", "banned-keys"):
        response = client.get(f"/v1/admin/security/{path}", headers=headers)
        assert response.status_code == 200, response.text
    response = client.post(
        "/v1/admin/security/unban-ip", headers=headers, json={"ip": CLIENT_IP}
    )
    assert response.status_code == 200
    assert not auth._rate_limiter.is_ip_banned(CLIENT_IP)
    assert client.get("/protected", headers=headers).status_code == 200


@pytest.mark.parametrize(
    "credential",
    [
        "missing",
        "forged",
        "expired",
        "non_admin",
        "revoked",
        "disabled",
        "password_change",
        "api_key",
    ],
)
def test_ban_recovery_requires_a_current_enabled_admin_session(setup, credential):
    auth, client, user_id, token = setup
    if credential == "forged":
        claims = jwt.get_unverified_claims(token)
        token = jwt.encode(claims, "wrong-secret", algorithm=JWT_ALGORITHM)
    elif credential == "expired":
        token = _expired_token(token)
    elif credential == "non_admin":
        # The JWT still says admin; live permissions must win.
        auth.db.set_user_permissions(user_id, ["models:read"])
    elif credential == "revoked":
        auth.db.update_password_and_revoke_tokens(user_id, "new-password-hash")
    elif credential == "disabled":
        auth.db.update_user(user_id, enabled=0)
    elif credential == "password_change":
        auth.db.update_user(user_id, must_change_password=1)
    elif credential == "api_key":
        token = auth.create_api_key_for_user(user_id=user_id, name="admin-key")["key"]
    _ban_ip(auth)
    headers = {} if credential == "missing" else _headers(token)
    response = client.post(
        "/v1/admin/security/unban-ip", headers=headers, json={"ip": CLIENT_IP}
    )
    assert response.status_code == 429
    assert auth._rate_limiter.is_ip_banned(CLIENT_IP)
    assert client.get("/v1/auth/me", headers=headers).status_code == 429


def test_key_ban_has_a_distinct_error_code(setup):
    auth, client, user_id, _ = setup
    created = auth.create_api_key_for_user(user_id=user_id, name="test-key")
    key_id = auth.db.list_api_keys()[0]["id"]
    for _ in range(auth._rate_limiter._key_config.max_failures):
        auth._rate_limiter.record_key_failure(CLIENT_IP, key_id)
    response = client.get("/protected", headers=_headers(created["key"]))
    assert response.status_code == 429
    assert response.headers["x-xinference-error-code"] == "key_banned"
    assert int(response.headers["retry-after"]) > 0


@pytest.mark.parametrize("ban_kind", ["ip", "key"])
@pytest.mark.parametrize("elapsed, expected", [(0.1, 60), (1.0, 59), (59.9, 1)])
def test_retry_after_covers_the_remaining_ban(
    setup, monkeypatch, ban_kind, elapsed, expected
):
    auth, client, user_id, token = setup
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(rate_limiter, "time", SimpleNamespace(time=lambda: clock.now))
    if ban_kind == "ip":
        auth._rate_limiter.update_ip_config(ban_seconds=60)
        _ban_ip(auth)
    else:
        created = auth.create_api_key_for_user(user_id=user_id, name="retry-key")
        token = created["key"]
        key_id = auth.db.list_api_keys()[0]["id"]
        auth._rate_limiter.update_key_config(ban_seconds=60)
        for _ in range(auth._rate_limiter._key_config.max_failures):
            auth._rate_limiter.record_key_failure(CLIENT_IP, key_id)

    clock.now += elapsed
    response = client.get("/protected", headers=_headers(token))
    assert response.status_code == 429
    assert response.headers["x-xinference-error-code"] == f"{ban_kind}_banned"
    assert int(response.headers["retry-after"]) == expected

    # A client following the advertised delay must not hit the same ban again.
    clock.now += int(response.headers["retry-after"])
    assert client.get("/protected", headers=_headers(token)).status_code == 200
