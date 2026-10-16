"""Exercise account recovery and live distribution policies through HTTP."""

from types import SimpleNamespace

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService
from xinference.api.oauth2.advanced.crypto import get_password_hash
from xinference.api.oauth2.advanced.routes import register_advanced_auth_routes


@pytest.fixture
def account(tmp_path):
    auth = AdvancedAuthService(
        str(tmp_path / "auth.db"), "test-secret", "test-encryption"
    )
    user_id = auth.db.create_user(
        "alice", get_password_hash("old-password"), permissions=["models:read"]
    )
    app = FastAPI()
    router = APIRouter()
    app.state.advanced_auth = auth
    register_advanced_auth_routes(SimpleNamespace(_app=app, _router=router))
    app.include_router(router)
    client = TestClient(app)
    login = auth.login("alice", "old-password")
    client.headers["Authorization"] = "Bearer " + login["access_token"]
    return auth, user_id, client, login


def test_ordinary_user_changes_only_own_password(account):
    auth, user_id, client, login = account
    other = auth.db.create_user(
        "bob", get_password_hash("bob-password"), permissions=[]
    )
    assert (
        client.put(
            f"/v1/admin/users/{other}/password", json={"new_password": "new-password"}
        ).status_code
        == 403
    )
    assert (
        client.put(
            f"/v1/admin/users/{user_id}/password",
            json={"current_password": "wrong", "new_password": "new-password"},
        ).status_code
        == 403
    )
    assert (
        client.put(
            f"/v1/admin/users/{user_id}/password",
            json={"current_password": "old-password", "new_password": "new-password"},
        ).status_code
        == 200
    )
    assert auth.verify_access_token(login["access_token"]) is None
    assert auth.refresh_access_token(login["refresh_token"]) is None
    assert auth.login("alice", "new-password")["access_token"]


def test_forced_change_works_without_users_manage(account):
    auth, user_id, client, _ = account
    auth.db.update_user(user_id, must_change_password=1)
    assert client.get("/v1/admin/users").status_code == 403
    assert (
        client.put(
            f"/v1/admin/users/{user_id}/password", json={"new_password": "new-password"}
        ).status_code
        == 200
    )
    assert not auth.db.get_user_by_id(user_id)["must_change_password"]


def test_permission_resolver_is_live_for_every_identity_read(account):
    auth, user_id, client, login = account
    effective = ["admin"]
    auth.db.permission_resolver = lambda uid, default: (
        effective if uid == user_id else default
    )
    assert client.get("/v1/auth/me").json()["permissions"] == ["admin"]
    assert client.get("/v1/admin/users").status_code == 200
    assert auth.db.list_users()[0]["permissions"] == ["admin"]
    assert auth.db.get_user_by_username("alice")["permissions"] == ["admin"]
    effective[:] = ["models:read"]
    assert client.get("/v1/auth/me").json()["permissions"] == ["models:read"]
    assert client.get("/v1/admin/users").status_code == 403
    rotated = auth.refresh_access_token(login["refresh_token"])
    assert auth.verify_access_token(rotated["access_token"])["scopes"] == [
        "models:read"
    ]


def test_stale_admin_cannot_grant_permissions_or_manage_foreign_keys(account):
    auth, user_id, client, _ = account
    auth.db.set_user_permissions(user_id, ["admin"])
    token = auth.login("alice", "old-password")["access_token"]
    client.headers["Authorization"] = "Bearer " + token
    auth.db.set_user_permissions(user_id, ["users:manage", "keys:create"])
    other = auth.db.create_user(
        "bob", get_password_hash("bob-password"), permissions=[]
    )
    assert (
        client.post(
            "/v1/admin/users",
            json={
                "username": "mallory",
                "password": "new-password",
                "permissions": ["admin"],
            },
        ).status_code
        == 403
    )
    response = client.post("/v1/admin/keys", json={"name": "owned", "owner": other})
    assert response.status_code == 201
    assert auth.db.list_api_keys()[0]["user_id"] == user_id


def test_resolver_failures_do_not_restore_snapshot_permissions(account):
    auth, _, client, _ = account

    def unavailable(*args):
        raise RuntimeError("policy store unavailable")

    auth.db.permission_resolver = unavailable
    with pytest.raises(RuntimeError, match="policy store unavailable"):
        client.get("/v1/auth/me")
