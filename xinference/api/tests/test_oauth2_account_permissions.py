"""Exercise account recovery and live distribution policies through HTTP."""

from types import SimpleNamespace

import pytest
from fastapi import APIRouter, FastAPI, Security
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

    @router.get("/v1/readable", dependencies=[Security(auth, scopes=["models:read"])])
    def readable():
        return {"ok": True}

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
            f"/v1/admin/users/{user_id}/password", json={"new_password": "new-password"}
        ).status_code
        == 403
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
    assert client.get("/v1/readable").status_code == 200
    auth.db.update_user(user_id, must_change_password=1)
    denied = client.get("/v1/readable")
    assert denied.status_code == 403
    assert "Password change required" in denied.json()["detail"]
    assert client.get("/v1/auth/me").status_code == 403
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


def test_current_account_resolves_once_and_reports_source(account):
    auth, _, client, _ = account
    calls = []
    auth.db.permission_resolver = lambda uid, stored: calls.append(uid) or stored
    profile = client.get("/v1/auth/me")
    assert profile.status_code == 200
    assert len(calls) == 1
    assert profile.json()["source"] == "local"
    assert "must_change_password" not in profile.json()
    key = auth.create_api_key_for_user(profile.json()["id"], name="profile-key")["key"]
    assert (
        client.get(
            "/v1/auth/me", headers={"Authorization": "Bearer " + key}
        ).status_code
        == 401
    )
    assert (
        client.get(
            "/v1/auth/me", headers={"Authorization": "bearer " + key}
        ).status_code
        == 401
    )


def test_oidc_account_cannot_change_local_password(account):
    auth, _, client, _ = account
    uid = auth.db.create_user("oidc-user", None, source="oidc", permissions=[])
    token = auth.create_access_token(uid, "oidc-user", [])
    headers = {"Authorization": "Bearer " + token}
    assert client.get("/v1/auth/me", headers=headers).json()["source"] == "oidc"
    response = client.put(
        f"/v1/admin/users/{uid}/password",
        json={"current_password": "whatever", "new_password": "new-password"},
        headers=headers,
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Only local users can change password"


def test_key_read_rejects_revoked_admin_without_key_grants(account):
    auth, uid, client, _ = account
    auth.db.set_user_permissions(uid, ["admin"])
    token = auth.login("alice", "old-password")["access_token"]
    headers = {"Authorization": "Bearer " + token}
    auth.create_api_key_for_user(uid, name="owned")
    key_id = auth.db.list_api_keys()[0]["id"]
    assert client.get("/v1/admin/keys", headers=headers).status_code == 200
    auth.db.set_user_permissions(uid, ["models:read"])
    for path in ("/v1/admin/keys", f"/v1/admin/keys/{key_id}"):
        assert client.get(path, headers=headers).status_code == 403


def test_privileged_self_change_checks_supplied_password_but_keeps_reset_contract(
    account,
):
    auth, uid, client, _ = account
    auth.db.set_user_permissions(uid, ["admin"])
    path = f"/v1/admin/users/{uid}/password"
    assert (
        client.put(
            path, json={"new_password": "new-password", "current_password": "wrong"}
        ).status_code
        == 403
    )
    assert client.put(path, json={"new_password": "new-password"}).status_code == 200


def test_non_admin_user_manager_can_reset_another_local_user(account):
    auth, uid, client, _ = account
    auth.db.set_user_permissions(uid, ["users:manage"])
    other = auth.db.create_user(
        "bob", get_password_hash("bob-password"), permissions=["models:read"]
    )
    login = auth.login("bob", "bob-password")
    response = client.put(
        f"/v1/admin/users/{other}/password", json={"new_password": "new-password"}
    )
    assert response.status_code == 200
    assert auth.verify_access_token(login["access_token"]) is None
    assert auth.refresh_access_token(login["refresh_token"]) is None
    assert auth.login("bob", "new-password")["access_token"]


@pytest.mark.parametrize("scheme", ["Bearer", "bearer"])
def test_api_key_cannot_change_password_even_with_admin_owner(account, scheme):
    auth, uid, client, _ = account
    auth.db.set_user_permissions(uid, ["admin"])
    key = auth.create_api_key_for_user(uid, name="password-key")["key"]
    response = client.put(
        f"/v1/admin/users/{uid}/password",
        json={"current_password": "old-password", "new_password": "new-password"},
        headers={"Authorization": scheme + " " + key},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Cannot change another user's password"
    assert auth.login("alice", "old-password")["access_token"]
