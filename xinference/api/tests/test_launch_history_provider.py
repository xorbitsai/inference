# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Store selection through API, supervisor and authenticated history routes."""

import asyncio
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from xinference import extensions
from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService
from xinference.api.oauth2.advanced.database import Database
from xinference.api.routers import launch_history
from xinference.core import launch_history_store
from xinference.core.launch_history_store import (
    LaunchHistoryStorage,
    LaunchHistoryStore,
    create_launch_history_store,
)


@pytest.fixture(autouse=True)
def isolate_extensions(monkeypatch):
    monkeypatch.delenv("XINFERENCE_EXTENSIONS", raising=False)
    monkeypatch.setattr(extensions, "_loaded", None)


def install(monkeypatch, provider):
    monkeypatch.setattr(
        extensions,
        "get_extensions",
        lambda: (
            SimpleNamespace(name="history", create_launch_history_store=provider),
        ),
    )


@pytest.mark.parametrize("service", ["api", "supervisor"])
@pytest.mark.parametrize("kind", ["unset", "old", "none"])
def test_default_path_and_existing_sqlite_history_are_preserved(
    monkeypatch, tmp_path, service, kind
):
    path = str(tmp_path / "history.db")
    LaunchHistoryStore(path).upsert("qwen", "uid", {"n_gpu": 1}, "alice")
    if kind != "unset":
        extension = SimpleNamespace(name="old")
        if kind == "none":
            extension.create_launch_history_store = lambda context: None
        monkeypatch.setattr(extensions, "get_extensions", lambda: (extension,))
    store = create_launch_history_store(path, service=service)
    assert isinstance(store, LaunchHistoryStore)
    assert store.list(username="alice")[0]["data"] == {"n_gpu": 1}


@pytest.mark.parametrize("service", ["api", "supervisor"])
def test_provider_receives_context_without_creating_default_file(
    monkeypatch, tmp_path, service
):
    store = Mock(spec=LaunchHistoryStorage)
    provider = Mock(return_value=store)
    install(monkeypatch, provider)
    path, auth = tmp_path / "unused.db", object()
    assert (
        create_launch_history_store(str(path), service=service, auth_store=auth)
        is store
    )
    context = provider.call_args.args[0]
    assert context.db_path == str(path)
    assert context.service == service
    assert context.auth_store is auth
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
        "async_generator",
    ],
)
def test_provider_errors_never_initialize_sqlite(monkeypatch, tmp_path, kind):
    path = tmp_path / "must-not-create.db"
    default = Mock(side_effect=AssertionError("SQLite must not be initialized"))
    monkeypatch.setattr(launch_history_store, "LaunchHistoryStore", default)
    store = Mock(spec=LaunchHistoryStorage)

    async def asynchronous(*args, **kwargs):
        return store

    class AsyncCallable:
        async def __call__(self, *args, **kwargs):
            return None

    async def generator(*args, **kwargs):
        yield None

    provider = Mock(return_value=store)
    if kind == "failure":
        provider.side_effect = RuntimeError("unavailable")
    elif kind == "async":
        provider = asynchronous
    elif kind == "incomplete":
        provider.return_value = object()
    elif kind == "stub":

        class StubStore(LaunchHistoryStorage):
            pass

        provider.return_value = StubStore()
    elif kind == "async_operation":
        store.upsert = asynchronous
    elif kind == "async_callable":
        store.upsert = AsyncCallable()
    elif kind == "async_generator":
        store.list_autostart = generator
    if kind == "duplicate":
        extension = SimpleNamespace(
            name="history", create_launch_history_store=provider
        )
        monkeypatch.setattr(
            extensions, "get_extensions", lambda: (extension, extension)
        )
    else:
        install(monkeypatch, provider)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        create_launch_history_store(str(path), service="api")
    if kind == "duplicate":
        provider.assert_not_called()
    default.assert_not_called()
    assert not path.exists()


@pytest.mark.parametrize("authenticated", [False, True])
def test_real_api_constructor_passes_selected_auth_store(
    monkeypatch, tmp_path, authenticated
):
    from xinference.api import restful_api
    from xinference.api.oauth2.advanced import auth_service
    from xinference.core import system_settings_store

    selected_auth = Database(str(tmp_path / "auth.db")) if authenticated else None
    monkeypatch.setattr(restful_api, "is_auth_advanced", lambda: authenticated)
    monkeypatch.setattr(restful_api, "get_auth_jwt_secret_key", lambda: "test-secret")
    monkeypatch.setattr(
        restful_api, "get_auth_encryption_key", lambda: "test-encryption"
    )
    monkeypatch.setattr(
        auth_service,
        "AdvancedAuthService",
        lambda **kwargs: SimpleNamespace(db=selected_auth, setup_token=None),
    )
    for name, filename in (
        ("XINFERENCE_AUTH_DB_PATH", "auth.db"),
        ("XINFERENCE_LAUNCH_HISTORY_DB_PATH", "history.db"),
        ("XINFERENCE_MONITOR_CONFIG_DB_PATH", "monitor.db"),
        ("XINFERENCE_SYSTEM_SETTINGS_PATH", "settings.json"),
    ):
        monkeypatch.setattr(restful_api, name, str(tmp_path / filename))
    monkeypatch.setattr(system_settings_store, "_store", None)
    provider = Mock(return_value=Mock(spec=LaunchHistoryStorage))
    install(monkeypatch, provider)
    api = restful_api.RESTfulAPI("127.0.0.1:1", "127.0.0.1", 0)
    assert api._launch_history_store is provider.return_value
    context = provider.call_args.args[0]
    assert context.service == "api" and context.auth_store is selected_auth
    assert context.db_path == str(tmp_path / "history.db")
    assert not (tmp_path / "history.db").exists()


def test_real_supervisor_uses_provider_for_autostart(monkeypatch, tmp_path):
    from xinference.core import supervisor, system_settings_store

    for name, filename in (
        ("XINFERENCE_LAUNCH_HISTORY_DB_PATH", "history.db"),
        ("XINFERENCE_TOKEN_ROUTER_DB_PATH", "router.db"),
        ("XINFERENCE_DOWNLOAD_TASK_DB_PATH", "downloads.db"),
    ):
        monkeypatch.setattr(supervisor, name, str(tmp_path / filename))
    monkeypatch.setenv(
        "XINFERENCE_SYSTEM_SETTINGS_PATH", str(tmp_path / "settings.json")
    )
    monkeypatch.setattr(system_settings_store, "_store", None)
    store = Mock(spec=LaunchHistoryStorage)
    store.list_autostart.return_value = [
        {"launch": {"model_name": "qwen", "model_uid": "uid"}, "priority": 5}
    ]
    provider = Mock(return_value=store)
    install(monkeypatch, provider)
    actor = supervisor.SupervisorActor()
    assert actor._launch_history_store is store
    context = provider.call_args.args[0]
    assert context.service == "supervisor" and context.auth_store is None
    assert not (tmp_path / "history.db").exists()

    async def load():
        return await actor._load_autostart_entries("alice")

    entries = asyncio.run(load())
    assert entries[0]["launch"]["model_uid"] == "uid"
    assert entries[0]["priority"] == 5
    store.list_autostart.assert_called_once_with("alice")


def client_for(store, auth=None):
    app = FastAPI()
    api = SimpleNamespace(
        _router=APIRouter(),
        _auth_service=auth,
        _launch_history_store=store,
        is_authenticated=lambda: auth is not None,
    )
    app.state.api = api
    launch_history.register_routes(api)
    app.include_router(api._router)

    @app.get("/loop-thread")
    async def loop_thread():
        return threading.get_ident()

    return TestClient(app, raise_server_exceptions=False)


def test_authenticated_routes_preserve_ownership_redaction_and_delete_scope(
    monkeypatch, tmp_path
):
    store = LaunchHistoryStore(str(tmp_path / "provided.db"))
    install(monkeypatch, lambda context: store)
    selected = create_launch_history_store(str(tmp_path / "unused.db"), service="api")
    auth = AdvancedAuthService(
        str(tmp_path / "auth.db"), "test-secret", "test-encryption"
    )
    headers = {}
    for name in ("alice", "bob"):
        user_id = auth.db.create_user(
            name, None, permissions=["models:list", "models:write"]
        )
        headers[name] = {
            "Authorization": "Bearer "
            + auth.create_access_token(user_id, name, ["models:list", "models:write"])
        }
    with client_for(selected, auth) as client:
        payload = {
            "model_name": "qwen",
            "model_uid": "uid",
            "created_by": "bob",
            "data": {"model_engine": "vllm", "envs": {"TOKEN": "private"}},
        }
        assert (
            client.post(
                "/v1/launch_history", json=payload, headers=headers["alice"]
            ).status_code
            == 200
        )
        assert (
            client.get("/v1/launch_history", headers=headers["alice"]).json()[0]["data"]
            == payload["data"]
        )
        foreign = client.get("/v1/launch_history", headers=headers["bob"]).json()[0]
        assert foreign["data"] == {"model_engine": "vllm"} and not foreign["is_owner"]
        assert (
            client.get("/v1/launch_history?scope=mine", headers=headers["bob"]).json()
            == []
        )
        assert (
            client.delete(
                "/v1/launch_history/qwen/uid", headers=headers["bob"]
            ).status_code
            == 404
        )
        assert (
            client.delete(
                "/v1/launch_history/qwen/uid", headers=headers["alice"]
            ).status_code
            == 200
        )
        assert client.get("/v1/launch_history", headers=headers["alice"]).json() == []
        assert client.get("/v1/launch_history").status_code == 401
    assert not (tmp_path / "unused.db").exists()


def test_history_operations_run_outside_api_event_loop():
    store = Mock(spec=LaunchHistoryStorage)
    threads = []
    store.list.side_effect = (
        lambda **kwargs: threads.append(threading.get_ident()) or []
    )
    store.upsert.side_effect = lambda **kwargs: threads.append(threading.get_ident())
    store.delete.side_effect = (
        lambda *args: threads.append(threading.get_ident()) or True
    )
    with client_for(store) as client:
        loop = client.get("/loop-thread").json()
        assert (
            client.post("/v1/launch_history", json={"model_name": "qwen"}).status_code
            == 200
        )
        assert client.get("/v1/launch_history").status_code == 200
        assert client.delete("/v1/launch_history/qwen/uid").status_code == 200
    assert len(threads) == 3 and all(thread != loop for thread in threads)


@pytest.mark.parametrize("operation", ["list", "upsert", "delete"])
def test_storage_failures_are_http_errors_without_fallback(
    monkeypatch, tmp_path, operation
):
    store = Mock(spec=LaunchHistoryStorage)
    getattr(store, operation).side_effect = RuntimeError("storage unavailable")
    install(monkeypatch, lambda context: store)
    selected = create_launch_history_store(str(tmp_path / "unused.db"), service="api")
    with client_for(selected) as client:
        response = {
            "list": lambda: client.get("/v1/launch_history"),
            "upsert": lambda: client.post(
                "/v1/launch_history", json={"model_name": "qwen"}
            ),
            "delete": lambda: client.delete("/v1/launch_history/qwen/uid"),
        }[operation]()
    assert response.status_code == 500
    assert not (tmp_path / "unused.db").exists()


def test_spawned_supervisor_reads_committed_api_history(monkeypatch, tmp_path):
    shared = tmp_path / "shared.db"
    module = tmp_path / "history_test_provider.py"
    module.write_text(
        "import os\nfrom types import SimpleNamespace\n"
        "from xinference.core.launch_history_store import LaunchHistoryStore\n"
        "def create():\n"
        " return SimpleNamespace(name='history-test',api_version=1,create_launch_history_store=lambda context: LaunchHistoryStore(os.environ['HISTORY_TEST_PATH']))\n"
    )
    env = {
        **os.environ,
        "XINFERENCE_EXTENSIONS": "history_test_provider:create",
        "HISTORY_TEST_PATH": str(shared),
        "PYTHONPATH": os.pathsep.join([str(tmp_path), *sys.path]),
    }
    script = (
        "import json,sys; from xinference.core.launch_history_store import create_launch_history_store; "
        "s=create_launch_history_store(sys.argv[2],service=sys.argv[1]); "
    )
    api = subprocess.run(
        [
            sys.executable,
            "-c",
            script
            + "s.upsert_autostart({'launch':{'model_name':'qwen','model_uid':'uid'}},username='alice')",
            "api",
            str(tmp_path / "unused-api.db"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert api.returncode == 0, api.stderr
    peer = subprocess.run(
        [
            sys.executable,
            "-c",
            script + "print(json.dumps(s.list_autostart())); s.remove_autostart('uid')",
            "supervisor",
            str(tmp_path / "unused-supervisor.db"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert peer.returncode == 0, peer.stderr
    assert json.loads(peer.stdout)[0]["created_by"] == "alice"
    store = LaunchHistoryStore(str(shared))
    assert store.list_autostart() == []
    assert len(store.list(username="alice")) == 1
    assert not (tmp_path / "unused-api.db").exists()
    assert not (tmp_path / "unused-supervisor.db").exists()
