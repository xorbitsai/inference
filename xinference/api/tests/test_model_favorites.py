# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Storage contracts and real authenticated HTTP entry points for favorites."""

import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from xinference import extensions
from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService
from xinference.api.oauth2.advanced.database import SCHEMA_SQL, Database
from xinference.api.routers import model_favorites
from xinference.core.model_favorites_store import (
    ModelFavoritesStore,
    SQLiteModelFavoritesStore,
    create_model_favorites_store,
)


@pytest.fixture(autouse=True)
def isolate_extensions(monkeypatch):
    monkeypatch.delenv("XINFERENCE_EXTENSIONS", raising=False)
    monkeypatch.setattr(extensions, "_loaded", None)


def client_for(store, auth=None, *, raise_server_exceptions=False):
    app = FastAPI()
    api = SimpleNamespace(
        _router=APIRouter(),
        _auth_service=auth,
        _model_favorites_store=store,
        is_authenticated=lambda: auth is not None,
    )
    app.state.api = api
    model_favorites.register_routes(api)
    app.include_router(api._router)

    @app.get("/loop-thread")
    async def loop_thread():
        return threading.get_ident()

    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def test_sqlite_persists_idempotent_changes_and_isolates_owners(tmp_path):
    path = str(tmp_path / "nested" / "favorites.db")
    store = create_model_favorites_store(path, None)
    assert isinstance(store, SQLiteModelFavoritesStore)
    store.add(1, "Qwen/team")
    store.add(1, "qwen/team")
    store.add(1, "Qwen/team")
    store.add(2, "separate")
    store.add(None, "public")
    reopened = SQLiteModelFavoritesStore(path)
    assert reopened.list(1) == ["Qwen/team", "qwen/team"]
    reopened.remove(1, "Qwen/team")
    reopened.remove(1, "Qwen/team")
    assert store.list(1) == ["qwen/team"]
    assert store.list(2) == ["separate"]
    assert store.list(None) == ["public"]


def test_independent_connections_do_not_lose_concurrent_additions(tmp_path):
    path = str(tmp_path / "favorites.db")
    first, peer = SQLiteModelFavoritesStore(path), SQLiteModelFavoritesStore(path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda i: peer.add(1, f"model-{i}"), range(30)))
    assert set(first.list(1)) == {f"model-{i}" for i in range(30)}


def test_sqlite_writes_do_not_wait_for_active_readers(tmp_path):
    path = str(tmp_path / "favorites.db")
    SQLiteModelFavoritesStore(path).add(1, "old")
    # Reopen an existing database created with the original journal mode.
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA journal_mode=DELETE")
    store = SQLiteModelFavoritesStore(path)

    with closing(sqlite3.connect(path)) as reader:
        reader.execute("BEGIN")
        query = "SELECT model_name FROM model_favorites WHERE owner=1"
        assert reader.execute(query).fetchall() == [("old",)]
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(store.add, 1, "new").result(timeout=15)
            executor.submit(store.remove, 1, "old").result(timeout=15)
        assert store.list(1) == ["new"]
        assert reader.execute(query).fetchall() == [("old",)]
        reader.rollback()
        assert reader.execute(query).fetchall() == [("new",)]


def test_committed_changes_are_visible_to_another_process(tmp_path):
    path = str(tmp_path / "favorites.db")
    store = SQLiteModelFavoritesStore(path)
    store.add(1, "first")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from xinference.core.model_favorites_store import SQLiteModelFavoritesStore; "
            "s=SQLiteModelFavoritesStore(sys.argv[1]); "
            "assert s.list(1) == ['first']; s.add(1, 'second')",
            path,
        ],
        check=True,
        timeout=30,
    )
    assert store.list(1) == ["first", "second"]


@pytest.mark.parametrize("path", ["", ":memory:"])
def test_default_requires_a_persistent_file(path):
    with pytest.raises(ValueError, match="persistent SQLite file"):
        create_model_favorites_store(path, None)


def test_provider_receives_selected_auth_store_without_creating_sqlite(
    monkeypatch, tmp_path
):
    auth, store = object(), Mock(spec=ModelFavoritesStore)
    provider = Mock(return_value=store)
    monkeypatch.setattr(
        extensions,
        "get_extensions",
        lambda: (
            SimpleNamespace(name="storage", create_model_favorites_store=provider),
        ),
    )
    path = tmp_path / "unused.db"
    assert create_model_favorites_store(str(path), auth) is store
    context = provider.call_args.args[0]
    assert context.auth_store is auth and context.db_path == str(path)
    assert not path.exists()


@pytest.mark.parametrize("kind", ["none", "old"])
def test_provider_can_preserve_default(monkeypatch, tmp_path, kind):
    extension = SimpleNamespace(name="storage")
    if kind == "none":
        extension.create_model_favorites_store = lambda context: None
    monkeypatch.setattr(extensions, "get_extensions", lambda: (extension,))
    assert isinstance(
        create_model_favorites_store(str(tmp_path / "favorites.db"), None),
        SQLiteModelFavoritesStore,
    )


def test_remote_auth_store_requires_a_favorites_provider(tmp_path):
    path = tmp_path / "must-not-create.db"
    with pytest.raises(RuntimeError, match="non-SQLite auth storage"):
        create_model_favorites_store(str(path), SimpleNamespace(path=None))
    assert not path.exists()


@pytest.mark.parametrize(
    "kind",
    [
        "failure",
        "duplicate",
        "async",
        "incomplete",
        "stub",
        "async_operation",
        "async_callable",
    ],
)
def test_invalid_provider_never_opens_fallback(monkeypatch, tmp_path, kind):
    async def asynchronous(*args):
        return None

    class AsyncCallable:
        async def __call__(self, *args):
            return None

    class Stub(ModelFavoritesStore):
        pass

    provider = Mock(side_effect=RuntimeError("unavailable"))
    if kind == "async":
        provider = asynchronous
    elif kind in {"incomplete", "stub", "async_operation", "async_callable"}:
        result = Stub() if kind == "stub" else SimpleNamespace()
        if kind.startswith("async_"):
            result = Mock(spec=ModelFavoritesStore)
            result.add = asynchronous if kind == "async_operation" else AsyncCallable()
        provider = Mock(return_value=result)
    extension = SimpleNamespace(name="storage", create_model_favorites_store=provider)
    monkeypatch.setattr(
        extensions,
        "get_extensions",
        lambda: (extension,) * (2 if kind == "duplicate" else 1),
    )
    path = tmp_path / "unused.db"
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        create_model_favorites_store(str(path), None)
    assert not path.exists()
    if kind == "duplicate":
        provider.assert_not_called()


@pytest.fixture
def authenticated(tmp_path):
    auth = AdvancedAuthService(
        str(tmp_path / "auth.db"), "jwt-secret", "encryption-secret"
    )
    alice = auth.db.create_user("same-name", None, permissions=["models:list"])
    bob = auth.db.create_user(
        "same-name",
        None,
        source="oidc",
        oidc_sub="subject",
        permissions=["models:list"],
    )
    store = create_model_favorites_store(str(tmp_path / "unused.db"), auth.db)
    with client_for(store, auth) as client:
        yield auth, alice, bob, store, client


def bearer(auth, user):
    row = auth.db.get_user_by_id(user)
    return {
        "Authorization": "Bearer "
        + auth.create_access_token(user, row["username"], row["permissions"])
    }


def test_http_uses_live_account_id_and_key_owner(authenticated):
    auth, alice, bob, store, client = authenticated
    url = "/v1/model_favorites"
    a, b = bearer(auth, alice), bearer(auth, bob)
    assert client.get(url).status_code == 401
    assert (
        client.put(url, json={"model_name": "org/model"}, headers=a).status_code == 200
    )
    assert client.get(url, headers=a).json() == ["org/model"]
    assert client.get(url + "?owner=user:1", headers=b).json() == []
    assert (
        client.put(
            url, json={"model_name": "other", "owner": f"user:{alice}"}, headers=b
        ).status_code
        == 422
    )
    assert (
        client.request(
            "DELETE", url, json={"model_name": "org/model"}, headers=b
        ).status_code
        == 200
    )
    key = auth.create_api_key_for_user(alice, name="favorites")
    assert client.get(
        url, headers={"Authorization": "Bearer " + key["key"]}
    ).json() == ["org/model"]
    key_headers = {"Authorization": "Bearer " + key["key"]}
    for method in ("PUT", "DELETE"):
        assert (
            client.request(
                method, url, json={"model_name": "org/model"}, headers=key_headers
            ).status_code
            == 403
        )
    assert client.get(url, headers=a).json() == ["org/model"]
    auth.db.update_user(alice, username="renamed")
    assert client.get(url, headers=a).json() == ["org/model"]
    auth.db.set_user_permissions(alice, [])
    assert client.get(url, headers=a).status_code == 403
    auth.db.update_user(bob, enabled=0)
    assert client.get(url, headers=b).status_code == 401


@pytest.mark.parametrize("source", ["local", "oidc"])
def test_recreated_account_does_not_inherit_deleted_owner_favorites(
    authenticated, source
):
    auth, alice, bob, store, client = authenticated
    old_headers = bearer(auth, bob)
    assert (
        client.put(
            "/v1/model_favorites",
            json={"model_name": "private-old-model"},
            headers=old_headers,
        ).status_code
        == 200
    )
    auth.db.delete_user(bob)
    replacement = auth.db.create_user(
        "replacement",
        None,
        source=source,
        oidc_sub="replacement-sub" if source == "oidc" else None,
        permissions=["models:list"],
    )
    assert (
        client.get("/v1/model_favorites", headers=bearer(auth, replacement)).json()
        == []
    )
    assert replacement != bob
    assert client.get("/v1/model_favorites", headers=old_headers).status_code == 401


def test_existing_sqlite_database_keeps_monotonic_ids_after_all_users_are_deleted(
    tmp_path,
):
    path = str(tmp_path / "old-auth.db")
    # This is the existing non-AUTOINCREMENT schema, with no sequence state.
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.execute(
            "INSERT INTO users(id, username, source) VALUES (41, 'original', 'local')"
        )
        conn.execute(
            "INSERT INTO user_permissions(user_id, permission) VALUES(41, 'admin')"
        )
    database = Database(path)
    assert database.get_user_by_id(41)["permissions"] == ["admin"]
    database.delete_user(41)
    replacement = Database(path).create_first_user("replacement", None)
    assert replacement > 41
    database.delete_user(replacement)
    assert (
        Database(path).create_user("oidc", None, source="oidc", oidc_sub="new-sub")
        > replacement
    )
    # Explicit IDs imported after initialization must also raise the watermark.
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO users(id, username, source) VALUES (99, 'imported', 'local')"
        )
    database.delete_user(99)
    assert Database(path).create_user("after-import", None) > 99


def test_independent_auth_instances_allocate_non_reused_ids_atomically(tmp_path):
    path = str(tmp_path / "auth.db")
    first, peer = Database(path), Database(path)
    removed = first.create_user("removed", None)
    peer.delete_user(removed)
    with ThreadPoolExecutor(max_workers=4) as executor:
        assigned = list(
            executor.map(
                lambda i: (first if i % 2 else peer).create_user(f"new-{i}", None),
                range(12),
            )
        )
    assert len(set(assigned)) == 12
    assert min(assigned) > removed


@pytest.mark.parametrize(
    "name", ["", " ", " model", "model\n", "x\x00y", "x" * 513, 123, None]
)
@pytest.mark.parametrize("method", ["PUT", "DELETE"])
def test_http_rejects_invalid_names(tmp_path, name, method):
    with client_for(
        SQLiteModelFavoritesStore(str(tmp_path / "favorites.db"))
    ) as client:
        assert (
            client.request(
                method, "/v1/model_favorites", json={"model_name": name}
            ).status_code
            == 422
        )


def test_anonymous_collection_persists_and_operations_run_outside_event_loop(tmp_path):
    store = SQLiteModelFavoritesStore(str(tmp_path / "favorites.db"))
    worker_threads = []
    read = store.list

    def list_names(owner):
        worker_threads.append(threading.get_ident())
        return read(owner)

    store.list = list_names
    with client_for(store) as client:
        assert (
            client.put("/v1/model_favorites", json={"model_name": "模型"}).status_code
            == 200
        )
        assert client.get("/v1/model_favorites").json() == ["模型"]
        assert worker_threads[0] != client.get("/loop-thread").json()
    with client_for(SQLiteModelFavoritesStore(store._path)) as peer:
        assert peer.get("/v1/model_favorites").json() == ["模型"]
        for _ in range(2):
            assert (
                peer.request(
                    "DELETE", "/v1/model_favorites", json={"model_name": "模型"}
                ).status_code
                == 200
            )


@pytest.mark.parametrize(
    "method,operation", [("GET", "list"), ("PUT", "add"), ("DELETE", "remove")]
)
def test_storage_failure_never_returns_empty_success_or_writes_fallback(
    tmp_path, monkeypatch, method, operation
):
    from xinference.core import model_favorites_store as stores

    store = Mock(spec=ModelFavoritesStore)
    getattr(store, operation).side_effect = RuntimeError("offline")
    fallback = Mock(side_effect=AssertionError("No runtime fallback"))
    monkeypatch.setattr(stores, "SQLiteModelFavoritesStore", fallback)
    monkeypatch.setattr(
        extensions,
        "get_extensions",
        lambda: (
            SimpleNamespace(
                name="provider", create_model_favorites_store=lambda context: store
            ),
        ),
    )
    path = tmp_path / "unused.db"
    selected = create_model_favorites_store(str(path), None)
    with client_for(selected) as client:
        options = {} if method == "GET" else {"json": {"model_name": "wanted"}}
        assert (
            client.request(method, "/v1/model_favorites", **options).status_code == 500
        )
    getattr(store, operation).assert_called_once()
    fallback.assert_not_called()
    assert not path.exists()


@pytest.mark.parametrize("result", ["wrong", {}, [123], ["valid", None]])
def test_bad_provider_result_is_rejected(result):
    store = Mock(spec=ModelFavoritesStore)
    store.list.return_value = result
    with client_for(store, raise_server_exceptions=True) as client:
        with pytest.raises(TypeError, match="list of names"):
            client.get("/v1/model_favorites")


@pytest.mark.parametrize(
    "user", [None, {}, {"id": 0}, {"id": -1}, {"id": True}, {"id": "1"}]
)
def test_invalid_authenticated_owner_cannot_access_anonymous_favorites(tmp_path, user):
    async def invalid_auth():
        return user

    store = SQLiteModelFavoritesStore(str(tmp_path / "favorites.db"))
    store.add(None, "anonymous-only")
    with client_for(store, invalid_auth) as client:
        for method in ("GET", "PUT", "DELETE"):
            options = (
                {} if method == "GET" else {"json": {"model_name": "anonymous-only"}}
            )
            assert (
                client.request(method, "/v1/model_favorites", **options).status_code
                == 403
            )
    assert store.list(None) == ["anonymous-only"]


@pytest.mark.parametrize("custom", [False, True])
def test_real_api_constructor_initializes_favorites_provider(
    monkeypatch, tmp_path, custom
):
    from xinference.api import restful_api
    from xinference.core import system_settings_store

    monkeypatch.setattr(restful_api, "is_auth_advanced", lambda: custom)
    selected_auth = object()
    provider = Mock(return_value=Mock(spec=ModelFavoritesStore))
    if custom:
        from xinference.api.oauth2.advanced import auth_service

        monkeypatch.setattr(
            auth_service,
            "AdvancedAuthService",
            lambda **kw: SimpleNamespace(db=selected_auth, setup_token=None),
        )
        monkeypatch.setattr(
            restful_api, "get_auth_jwt_secret_key", lambda: "jwt-secret"
        )
        monkeypatch.setattr(
            restful_api, "get_auth_encryption_key", lambda: "encryption-secret"
        )
        monkeypatch.setattr(
            extensions,
            "get_extensions",
            lambda: (
                SimpleNamespace(
                    name="favorites", create_model_favorites_store=provider
                ),
            ),
        )
    monkeypatch.setattr(
        restful_api, "XINFERENCE_AUTH_DB_PATH", str(tmp_path / "auth.db")
    )
    monkeypatch.setattr(
        restful_api, "XINFERENCE_LAUNCH_HISTORY_DB_PATH", str(tmp_path / "history.db")
    )
    monkeypatch.setattr(
        restful_api, "XINFERENCE_MONITOR_CONFIG_DB_PATH", str(tmp_path / "monitor.db")
    )
    monkeypatch.setattr(
        restful_api, "XINFERENCE_SYSTEM_SETTINGS_PATH", str(tmp_path / "settings.json")
    )
    monkeypatch.setattr(system_settings_store, "_store", None)
    api = restful_api.RESTfulAPI("127.0.0.1:1", "127.0.0.1", 0)
    if custom:
        assert api._model_favorites_store is provider.return_value
        assert provider.call_args.args[0].auth_store is selected_auth
        assert provider.call_args.args[0].db_path == str(tmp_path / "auth.db")
        assert not (tmp_path / "auth.db").exists()
    else:
        assert isinstance(api._model_favorites_store, SQLiteModelFavoritesStore)
        assert api._model_favorites_store._path == str(tmp_path / "auth.db")


@pytest.mark.parametrize(
    "authenticated,custom_path", [(False, False), (True, False), (True, True)]
)
def test_api_favorites_reuse_the_selected_database_without_enabling_auth(
    monkeypatch, tmp_path, authenticated, custom_path
):
    from xinference.api import restful_api
    from xinference.core import system_settings_store

    configured = tmp_path / "configured-auth.db"
    selected = tmp_path / "selected-auth.db" if custom_path else configured
    monkeypatch.setattr(restful_api, "is_auth_advanced", lambda: authenticated)
    secret = Mock(return_value="test-secret")
    monkeypatch.setattr(restful_api, "get_auth_jwt_secret_key", secret)
    monkeypatch.setattr(restful_api, "get_auth_encryption_key", secret)
    for name, path in {
        "XINFERENCE_AUTH_DB_PATH": configured,
        "XINFERENCE_LAUNCH_HISTORY_DB_PATH": tmp_path / "history.db",
        "XINFERENCE_MONITOR_CONFIG_DB_PATH": tmp_path / "monitor.db",
        "XINFERENCE_SYSTEM_SETTINGS_PATH": tmp_path / "settings.json",
    }.items():
        monkeypatch.setattr(restful_api, name, str(path))
    monkeypatch.setattr(system_settings_store, "_store", None)
    if custom_path:
        monkeypatch.setattr(
            extensions,
            "get_extensions",
            lambda: (
                SimpleNamespace(
                    name="storage",
                    create_auth_store=lambda context: Database(str(selected)),
                ),
            ),
        )
    api = restful_api.RESTfulAPI("127.0.0.1:1", "127.0.0.1", 0)
    assert api.is_authenticated() == authenticated
    assert api._model_favorites_store._path == str(selected)
    api._model_favorites_store.add(None, "shared")
    with sqlite3.connect(selected) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
        assert "model_favorites" in tables
        assert ("users" in tables) == authenticated
    if authenticated:
        auth = api._advanced_auth_service
        user = auth.db.create_user("reader", None, permissions=["models:list"])
        with client_for(api._model_favorites_store, auth) as client:
            headers = bearer(auth, user)
            assert client.get("/v1/model_favorites", headers=headers).json() == []
            assert (
                client.put(
                    "/v1/model_favorites",
                    json={"model_name": "private"},
                    headers=headers,
                ).status_code
                == 200
            )
        assert auth.db.path == selected
    else:
        secret.assert_not_called()
        assert api._advanced_auth_service is None
    # A no-auth process sees only the shared collection in that same database.
    with client_for(SQLiteModelFavoritesStore(str(selected))) as client:
        assert client.get("/v1/model_favorites").json() == ["shared"]
    assert not (tmp_path / "model_favorites.db").exists()
    if custom_path:
        assert not configured.exists()
