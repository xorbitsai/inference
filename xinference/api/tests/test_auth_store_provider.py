# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Exercise store selection through the real auth and recovery entry points."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import APIRouter, FastAPI, Request, Security
from fastapi.testclient import TestClient

from xinference import extensions
from xinference.api.oauth2.advanced import migrate, reset_password
from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService
from xinference.api.oauth2.advanced.auth_store import AuthStore, create_auth_store
from xinference.api.oauth2.advanced.crypto import verify_password
from xinference.api.oauth2.advanced.database import Database
from xinference.api.oauth2.advanced.routes import register_advanced_auth_routes


class AlternateStore:
    """A non-Database provider forwarding operations to an isolated test store."""

    def __init__(self, path):
        self._database = Database(str(path))

    def __getattr__(self, name):
        return getattr(self._database, name)

    @property
    def path(self):
        return None

    @property
    def permission_resolver(self):
        return self._database.permission_resolver

    @permission_resolver.setter
    def permission_resolver(self, resolver):
        self._database.permission_resolver = resolver


@pytest.fixture(autouse=True)
def clear_extensions(monkeypatch):
    monkeypatch.delenv("XINFERENCE_EXTENSIONS", raising=False)
    monkeypatch.setattr(extensions, "_loaded", None)


def install(monkeypatch, provider):
    module = ModuleType("auth_store_test_extension")
    module.create = lambda: SimpleNamespace(
        name="test-store",
        api_version=1,
        create_auth_store=provider,
        configure_api=lambda context: None,
        worker_metadata=lambda: {},
        filter_worker_resources=lambda worker, workers: worker["gpu_devices"],
        control_operation=lambda operation, payload: None,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("XINFERENCE_EXTENSIONS", module.__name__ + ":create")


@pytest.mark.parametrize("purpose", ["service", "reset_password", "migrate"])
def test_default_stays_sqlite(tmp_path, purpose):
    path = tmp_path / "auth.db"
    db = create_auth_store(str(path), purpose=purpose)
    assert isinstance(db, Database)
    assert db.path == path
    db.create_user("alice", None)
    assert create_auth_store(str(path)).get_user_by_username("alice") is not None


def test_old_extension_needs_no_store_hook(monkeypatch, tmp_path):
    monkeypatch.setattr(
        extensions, "get_extensions", lambda: (SimpleNamespace(name="old-extension"),)
    )
    assert isinstance(create_auth_store(str(tmp_path / "auth.db")), Database)


def test_provider_can_explicitly_choose_default(monkeypatch, tmp_path):
    install(monkeypatch, lambda context: None)
    assert isinstance(create_auth_store(str(tmp_path / "auth.db")), Database)


@pytest.fixture
def shared_services(monkeypatch, tmp_path):
    def provide(context):
        store = AlternateStore(tmp_path / "shared.db")
        store.api_key_cache_enabled = False
        return store

    install(monkeypatch, provide)
    first = AdvancedAuthService("unused.db", "shared-secret", "shared-encryption")
    user = first.db.create_user("reader", None, permissions=["models:read"])
    key = first.create_api_key_for_user(user, name="existing")
    peer = AdvancedAuthService("unused.db", "shared-secret", "shared-encryption")
    return first, peer, user, key


def peer_client(auth, **kwargs):
    app = FastAPI()

    @app.get("/model/{model}")
    def model(model: str, user=Security(auth, scopes=["models:read"])):
        return {"user_id": user["id"]}

    return TestClient(app, **kwargs)


def test_shared_store_observes_keys_created_after_peer_start(shared_services):
    first, peer, user, _ = shared_services
    key = first.create_api_key_for_user(user, name="new")
    assert peer.validate_api_key(key["key"]) is not None
    with peer_client(peer) as client:
        response = client.get(
            "/model/allowed", headers={"Authorization": "Bearer " + key["key"]}
        )
    assert response.status_code == 200, response.text
    assert response.json()["user_id"] == user


@pytest.mark.parametrize("change", ["disable", "delete", "expire", "disable_owner"])
def test_shared_store_observes_key_and_owner_revocation(shared_services, change):
    first, peer, user, key = shared_services
    assert peer.validate_api_key(key["key"]) is not None
    if change == "disable":
        first.db.update_api_key(key["id"], enabled=0)
    elif change == "delete":
        first.db.delete_api_key(key["id"])
    elif change == "expire":
        first.db.update_api_key(key["id"], expires_at="2000-01-01T00:00:00")
    else:
        first.disable_user(user)
    assert peer.validate_api_key(key["key"]) is None
    assert not peer.validate_model_access(key["key"], "any-model", "LLM")
    with peer_client(peer) as client:
        response = client.get(
            "/model/allowed", headers={"Authorization": "Bearer " + key["key"]}
        )
    assert response.status_code == 401, response.text


def test_shared_store_observes_model_grant_narrowing_and_restoration(shared_services):
    first, peer, user, key = shared_services
    assert peer.validate_model_access(key["key"], "removed", "LLM")
    first.db.set_api_key_model_permissions(
        key["id"], [{"permission_type": "model_id", "permission_value": "allowed"}]
    )
    assert peer.validate_api_key(key["key"]) is not None
    assert not peer.validate_model_access(key["key"], "removed", "LLM")
    assert peer.validate_model_access(key["key"], "allowed", "LLM")
    first.db.set_api_key_model_permissions(key["id"], [{"permission_type": "all"}])
    assert peer.validate_model_access(key["key"], "removed", "LLM")


@pytest.mark.parametrize("operation", ["get_api_key_by_hash", "get_user_by_id"])
def test_shared_store_failure_does_not_reuse_positive_key(
    shared_services, monkeypatch, operation
):
    first, peer, user, key = shared_services
    assert peer.validate_api_key(key["key"]) is not None
    monkeypatch.setattr(
        peer.db, operation, Mock(side_effect=RuntimeError("unavailable"))
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        peer.validate_api_key(key["key"])
    with pytest.raises(RuntimeError, match="unavailable"):
        peer.validate_model_access(key["key"], "allowed", "LLM")
    with peer_client(peer, raise_server_exceptions=False) as client:
        response = client.get(
            "/model/allowed", headers={"Authorization": "Bearer " + key["key"]}
        )
    assert response.status_code == 500, response.text


def test_shared_store_rejects_key_with_missing_owner(shared_services, monkeypatch):
    first, peer, user, key = shared_services
    assert peer.validate_api_key(key["key"]) is not None
    monkeypatch.setattr(peer.db, "get_user_by_id", Mock(return_value=None))
    assert peer.validate_api_key(key["key"]) is None
    assert not peer.validate_model_access(key["key"], "allowed", "LLM")
    with peer_client(peer) as client:
        response = client.get(
            "/model/allowed", headers={"Authorization": "Bearer " + key["key"]}
        )
    assert response.status_code == 401, response.text


@pytest.mark.parametrize("explicit_cache_policy", [False, True])
def test_provider_keeps_api_key_caching_by_default(
    monkeypatch, tmp_path, explicit_cache_policy
):
    store = AlternateStore(tmp_path / "cached.db")
    if explicit_cache_policy:
        store.api_key_cache_enabled = True
    install(monkeypatch, lambda context: store)
    reader = AdvancedAuthService("unused.db", "shared-secret", "shared-encryption")
    user = store.create_user("reader", None, permissions=["models:read"])
    writer = AdvancedAuthService("unused.db", "shared-secret", "shared-encryption")
    key = writer.create_api_key_for_user(user, name="created-after-reader-start")
    assert reader.validate_api_key(key["key"]) is None
    with peer_client(reader) as client:
        response = client.get(
            "/model/allowed", headers={"Authorization": "Bearer " + key["key"]}
        )
        assert response.status_code == 401, response.text
        reader.cache.reload()
        assert reader.validate_api_key(key["key"]) is not None
        assert (
            client.get(
                "/model/allowed", headers={"Authorization": "Bearer " + key["key"]}
            ).status_code
            == 200
        )


@pytest.mark.parametrize("model_uid", ["allowed", "denied"])
@pytest.mark.parametrize(
    "change", ["none", "revoke", "storage_before_access", "storage_before_audit"]
)
def test_shared_store_resolves_api_key_once_per_request(
    shared_services, monkeypatch, model_uid, change
):
    from xinference.api.oauth2.advanced import audit
    from xinference.api.restful_api import RESTfulAPI

    first, peer, user, key = shared_services
    first.db.set_api_key_model_permissions(
        key["id"], [{"permission_type": "model_id", "permission_value": "allowed"}]
    )
    lookups = {}
    for operation in ("get_api_key_by_hash", "get_user_by_id"):
        lookup = Mock(wraps=getattr(peer.db, operation))
        monkeypatch.setattr(peer.db, operation, lookup)
        lookups[operation] = lookup
    record = Mock()
    monkeypatch.setattr(audit, "record_audit_event", record)
    api = RESTfulAPI.__new__(RESTfulAPI)
    api._advanced_auth_service = peer
    api._uid_to_model_name = {}
    app = FastAPI()
    app.middleware("http")(api._audit_middleware)

    def fail_store():
        for lookup in lookups.values():
            lookup.side_effect = RuntimeError("unavailable after authentication")

    @app.post("/v1/chat/completions")
    async def infer(request: Request, owner=Security(peer, scopes=["models:read"])):
        if change == "revoke":
            first.db.delete_api_key(key["id"])
        elif change == "storage_before_access":
            fail_store()
        api._check_model_access(request, model_uid, "LLM")
        if change == "storage_before_audit":
            fail_store()
        return {"user_id": owner["id"]}

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer " + key["key"]},
            json={"model": model_uid},
        )
        expected_status = 200 if model_uid == "allowed" else 403
        assert response.status_code == expected_status, response.text
        for lookup in lookups.values():
            lookup.assert_called_once()
        record.assert_called_once()
        event = record.call_args.kwargs
        assert event["category"] == "inference"
        assert event["status"] == ("success" if expected_status == 200 else "denied")
        assert event["status_code"] == expected_status
        assert event["user"] == "reader"
        assert event["api_key_name"] == "existing"
        assert event["api_key_prefix"] == key["prefix"]
        if change == "revoke":
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer " + key["key"]},
                json={"model": model_uid},
            )
            assert response.status_code == 401, response.text


@pytest.mark.parametrize("enabled", [None, 0, "false"])
def test_provider_rejects_invalid_api_key_cache_policy(monkeypatch, tmp_path, enabled):
    store = AlternateStore(tmp_path / "provided.db")
    store.api_key_cache_enabled = enabled
    install(monkeypatch, lambda context: store)
    with pytest.raises(TypeError, match="api_key_cache_enabled"):
        create_auth_store(str(tmp_path / "unused.db"))


@pytest.mark.parametrize("callable_operation", [False, True])
def test_real_routes_use_provider_and_live_permissions(
    monkeypatch, tmp_path, callable_operation
):
    store = AlternateStore(tmp_path / "provided.db")
    if callable_operation:
        store.update_password_and_revoke_tokens = Mock(
            wraps=store.update_password_and_revoke_tokens
        )
    requested = tmp_path / "must-not-create.db"
    contexts = []

    def provide(context):
        contexts.append(context)
        return store

    install(monkeypatch, provide)
    auth = AdvancedAuthService(
        str(requested), "store-test-jwt", "store-test-encryption"
    )
    app, router = FastAPI(), APIRouter()
    app.state.advanced_auth = auth
    register_advanced_auth_routes(SimpleNamespace(_app=app, _router=router))
    app.include_router(router)
    with TestClient(app) as client:
        setup_token = store.get_or_create_setup_token()
        response = client.post(
            "/v1/admin/setup",
            json={
                "username": "admin",
                "password": "test-password",
                "setup_token": setup_token,
            },
        )
        assert response.status_code == 201, response.text
        login = client.post(
            "/token", json={"username": "admin", "password": "test-password"}
        )
        assert login.status_code == 200, login.text
        tokens = login.json()
        headers = {"Authorization": "Bearer " + tokens["access_token"]}
        assert client.get("/v1/admin/users", headers=headers).status_code == 200
        store.permission_resolver = lambda user_id, permissions: []
        assert client.get("/v1/admin/users", headers=headers).status_code == 403
        store.permission_resolver = None
        reset_password.reset_admin_password(store, "admin", "replacement-password")
        assert auth.verify_access_token(tokens["access_token"]) is None
        assert auth.refresh_access_token(tokens["refresh_token"]) is None
        assert (
            client.post(
                "/token", json={"username": "admin", "password": "replacement-password"}
            ).status_code
            == 200
        )
    assert auth.db is store
    assert contexts == [extensions.AuthStoreContext(str(requested), "service")]
    assert not requested.exists()


def test_migration_and_password_recovery_use_selected_store(monkeypatch, tmp_path):
    store = AlternateStore(tmp_path / "provided.db")
    requested = tmp_path / "must-not-create.db"
    contexts = []

    def provide(context):
        contexts.append(context)
        return store

    install(monkeypatch, provide)
    old = tmp_path / "old-auth.json"
    old.write_text(
        json.dumps(
            {
                "user_config": [
                    {
                        "username": "admin",
                        "password": "original-password",
                        "permissions": ["admin"],
                        "api_keys": ["sk-store-test"],
                    }
                ]
            }
        )
    )
    migrate.migrate(str(old), str(requested), "store-test-encryption")
    assert store.user_count() == 1
    assert len(store.get_all_api_keys_with_users()) == 1
    user = store.get_user_by_username("admin")
    store.create_refresh_token(user["id"], "old-refresh-token", "2099-01-01T00:00:00")
    monkeypatch.setattr(
        reset_password,
        "parse_args",
        lambda: argparse.Namespace(
            db_path=str(requested),
            username="admin",
            password="replacement-password",
            yes=True,
        ),
    )
    reset_password.main()
    updated = store.get_user_by_username("admin")
    assert verify_password("replacement-password", updated["password_hash"])
    assert store.get_refresh_token("old-refresh-token") is None
    assert updated["token_version"] == user["token_version"] + 1
    assert contexts == [
        extensions.AuthStoreContext(str(requested), "migrate"),
        extensions.AuthStoreContext(str(requested), "reset_password"),
    ]
    assert not requested.exists()


def test_migration_dry_run_does_not_initialize_provider(monkeypatch, tmp_path):
    provider = Mock(side_effect=AssertionError("Dry run must not connect"))
    install(monkeypatch, provider)
    old = tmp_path / "old-auth.json"
    old.write_text(json.dumps({"user_config": []}))
    migrate.migrate(str(old), str(tmp_path / "unused.db"), "test-key", dry_run=True)
    provider.assert_not_called()


def test_provider_error_does_not_open_sqlite(monkeypatch, tmp_path):
    def unavailable(context):
        raise RuntimeError("Auth store unavailable")

    install(monkeypatch, unavailable)
    path = tmp_path / "must-not-create.db"
    with pytest.raises(RuntimeError, match="Auth store unavailable"):
        AdvancedAuthService(str(path), "store-test-jwt", "store-test-encryption")
    assert not path.exists()


def test_duplicate_providers_rejected_before_initialization(monkeypatch, tmp_path):
    first, second = Mock(), Mock()
    monkeypatch.setattr(
        extensions,
        "get_extensions",
        lambda: (
            SimpleNamespace(name="a", create_auth_store=first),
            SimpleNamespace(name="b", create_auth_store=second),
        ),
    )
    with pytest.raises(ValueError, match="Multiple auth store providers"):
        create_auth_store(str(tmp_path / "unused.db"))
    first.assert_not_called()
    second.assert_not_called()


def test_async_factory_is_rejected(monkeypatch, tmp_path):
    async def provide(context):
        return AlternateStore(tmp_path / "provided.db")

    install(monkeypatch, provide)
    with pytest.raises(TypeError, match="must be synchronous"):
        create_auth_store(str(tmp_path / "unused.db"))
    assert not (tmp_path / "provided.db").exists()


@pytest.mark.parametrize("omit_operation", [None, "update_password_and_revoke_tokens"])
def test_explicit_protocol_subclass_requires_every_operation(
    monkeypatch, tmp_path, omit_operation
):
    backing = AlternateStore(tmp_path / "provided.db")
    implementations = {
        name: getattr(backing, name)
        for name, member in vars(AuthStore).items()
        if not name.startswith("_") and callable(member) and name != omit_operation
    }
    # Supply every operation except the selected one, which inherits the stub.
    store = type(
        "ExplicitStore",
        (AuthStore,),
        dict(implementations, path=None, permission_resolver=None),
    )()
    install(monkeypatch, lambda context: store)
    requested = tmp_path / "must-not-create.db"
    if omit_operation is None:
        auth = AdvancedAuthService(str(requested), "test-jwt", "test-encryption")
        assert auth.db is store
    else:
        with pytest.raises(TypeError, match=omit_operation):
            AdvancedAuthService(str(requested), "test-jwt", "test-encryption")
    assert not requested.exists()


@pytest.mark.parametrize(
    "invalid",
    ["missing", "async", "async_callable", "path", "resolver", "resolver_callable"],
)
def test_invalid_store_rejected_at_startup(monkeypatch, tmp_path, invalid):
    store = AlternateStore(tmp_path / "provided.db")

    async def async_operation(*args):
        pass

    class AsyncOperation:
        async def __call__(self, *args):
            pass

    if invalid == "missing":
        store.create_user = None
    elif invalid == "async":
        store.rotate_refresh_token = async_operation
    elif invalid == "async_callable":
        store.update_password_and_revoke_tokens = AsyncOperation()
    elif invalid == "path":
        store = SimpleNamespace(
            **{
                name: getattr(store, name)
                for name in dir(Database)
                if not name.startswith("_")
            },
            permission_resolver=None,
        )
        store.path = "not-a-path"
    elif invalid == "resolver":
        store.permission_resolver = async_operation
    else:
        store.permission_resolver = AsyncOperation()
    install(monkeypatch, lambda context: store)
    requested = tmp_path / "unused.db"
    with pytest.raises(TypeError, match="Auth store"):
        create_auth_store(str(requested))
    assert not requested.exists()


def test_provider_loaded_in_fresh_process(monkeypatch, tmp_path):
    module = tmp_path / "process_auth_store.py"
    module.write_text(
        "from types import SimpleNamespace\n"
        "from xinference.api.oauth2.advanced.database import Database\n"
        "def create():\n"
        "    def provide(context):\n"
        "        assert context.purpose == 'service'\n"
        "        return Database(context.db_path + '.provided')\n"
        "    return SimpleNamespace(name='process-store', api_version=1, create_auth_store=provide,\n"
        "        configure_api=lambda context: None, worker_metadata=lambda: {},\n"
        "        filter_worker_resources=lambda worker, workers: worker['gpu_devices'],\n"
        "        control_operation=lambda operation, payload: None)\n"
    )
    environment = dict(os.environ, XINFERENCE_EXTENSIONS="process_auth_store:create")
    environment["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(Path.cwd())])
    requested = tmp_path / "must-not-create.db"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService; import sys; auth=AdvancedAuthService(sys.argv[1],'test-jwt','test-encryption'); assert auth.db.path.name.endswith('.provided')",
            str(requested),
        ],
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert not requested.exists()
    assert (tmp_path / (requested.name + ".provided")).exists()
