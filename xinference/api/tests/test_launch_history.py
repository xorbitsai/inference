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

"""Unit tests for the launch history store and router handlers."""

import inspect
import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from xinference.api.routers import launch_history
from xinference.core.autostart import normalize_autostart_model_entry
from xinference.core.launch_history_store import LaunchHistoryStore


def _json_body(response):
    return json.loads(response.body.decode())


@pytest.fixture
def store(tmp_path):
    return LaunchHistoryStore(str(tmp_path / "launch_history.db"))


# --- Store tests ---


def test_upsert_inserts_then_lists(store):
    store.upsert("llama", "", {"model_engine": "vllm"}, username="alice")
    rows = store.list()
    assert len(rows) == 1
    row = rows[0]
    assert row["model_name"] == "llama"
    assert row["model_uid"] == ""
    assert row["data"] == {"model_engine": "vllm"}
    assert row["created_by"] == "alice"
    assert row["updated_by"] == "alice"


def test_upsert_same_user_updates_in_place(store):
    store.upsert("llama", "", {"v": 1}, username="alice")
    store.upsert("llama", "", {"v": 2}, username="alice")
    rows = store.list()
    assert len(rows) == 1
    row = rows[0]
    assert row["data"] == {"v": 2}
    assert row["created_by"] == "alice"
    assert row["updated_by"] == "alice"


def test_upsert_distinct_users_create_separate_rows(store):
    # Each user owns a separate row per (model_name, model_uid).
    store.upsert("llama", "", {"v": 1}, username="alice")
    store.upsert("llama", "", {"v": 2}, username="bob")
    rows = store.list()
    assert len(rows) == 2
    assert {r["created_by"] for r in rows} == {"alice", "bob"}


def test_upsert_distinct_uids_are_separate_rows(store):
    store.upsert("llama", "uid-1", {"v": 1})
    store.upsert("llama", "uid-2", {"v": 2})
    assert len(store.list()) == 2


def test_list_filters_by_model_name(store):
    store.upsert("llama", "", {"v": 1})
    store.upsert("qwen", "", {"v": 2})
    rows = store.list(model_name="qwen")
    assert len(rows) == 1
    assert rows[0]["model_name"] == "qwen"


def test_list_owner_only_filters_by_username(store):
    store.upsert("llama", "", {"owner": "alice"}, username="alice")
    store.upsert("qwen", "", {"owner": "bob"}, username="bob")

    rows = store.list(username="alice", owner_only=True)

    assert len(rows) == 1
    assert rows[0]["created_by"] == "alice"
    assert rows[0]["data"] == {"owner": "alice"}


def test_list_owner_only_combines_with_model_name(store):
    store.upsert("llama", "alice-uid", {"owner": "alice"}, username="alice")
    store.upsert("llama", "bob-uid", {"owner": "bob"}, username="bob")
    store.upsert("qwen", "alice-uid", {"owner": "alice"}, username="alice")

    rows = store.list(model_name="llama", username="alice", owner_only=True)

    assert len(rows) == 1
    assert rows[0]["model_name"] == "llama"
    assert rows[0]["model_uid"] == "alice-uid"
    assert rows[0]["created_by"] == "alice"


def test_list_owner_only_supports_anonymous_records(store):
    store.upsert("llama", "anonymous", {"owner": "anonymous"})
    store.upsert("llama", "alice", {"owner": "alice"}, username="alice")

    rows = store.list(username="", owner_only=True)

    assert len(rows) == 1
    assert rows[0]["model_uid"] == "anonymous"
    assert rows[0]["created_by"] == ""


def test_list_emits_utc_z_timestamps(store):
    store.upsert("llama", "", {"v": 1})
    row = store.list()[0]
    for key in ("created_at", "updated_at"):
        assert row[key].endswith("Z")
        assert "T" in row[key]


def test_delete_removes_row(store):
    store.upsert("llama", "", {"v": 1})
    assert store.delete("llama", "") is True
    assert store.list() == []


def test_delete_missing_returns_false(store):
    assert store.delete("nope", "") is False


def test_delete_scoped_to_owner(store):
    store.upsert("llama", "", {"v": 1}, username="alice")
    # A different user cannot delete someone else's row.
    assert store.delete("llama", "", "bob") is False
    assert len(store.list()) == 1
    # The owner can delete their own row.
    assert store.delete("llama", "", "alice") is True
    assert store.list() == []


def test_list_redacts_other_users_sensitive_fields(store):
    store.upsert(
        "llama",
        "",
        {
            "model_engine": "vllm",
            "envs": {"HF_TOKEN": "secret"},
            "api_key": "sk-should-not-leak",
        },
        username="alice",
    )
    rows = store.list(model_name="llama", username="bob")
    assert len(rows) == 1
    # Only whitelisted fields survive for another user.
    assert rows[0]["data"] == {"model_engine": "vllm"}
    assert "envs" not in rows[0]["data"]
    assert "api_key" not in rows[0]["data"]


def test_list_returns_full_data_for_owner(store):
    payload = {"model_engine": "vllm", "envs": {"HF_TOKEN": "secret"}}
    store.upsert("llama", "", payload, username="alice")
    rows = store.list(model_name="llama", username="alice")
    assert rows[0]["data"] == payload


def test_list_without_username_is_unredacted(store):
    payload = {"model_engine": "vllm", "envs": {"HF_TOKEN": "secret"}}
    store.upsert("llama", "", payload, username="alice")
    # username=None means a trusted/unscoped call: no redaction.
    rows = store.list(model_name="llama")
    assert rows[0]["data"] == payload


def test_empty_username_defaults_to_blank(store):
    store.upsert("llama", "", {"v": 1})
    row = store.list()[0]
    assert row["created_by"] == ""
    assert row["updated_by"] == ""


def test_store_persists_across_instances(tmp_path):
    db = str(tmp_path / "lh.db")
    LaunchHistoryStore(db).upsert("llama", "", {"v": 1}, username="alice")
    rows = LaunchHistoryStore(db).list()
    assert len(rows) == 1
    assert rows[0]["created_by"] == "alice"


def test_store_migrates_autostart_columns(tmp_path):
    db = str(tmp_path / "old.db")
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE launch_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_uid TEXT NOT NULL DEFAULT '',
                data TEXT NOT NULL,
                created_by TEXT DEFAULT '',
                updated_by TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE UNIQUE INDEX idx_launch_history_model
                ON launch_history(model_name, model_uid, created_by);
            INSERT INTO launch_history (model_name, model_uid, data)
                VALUES ('llama', 'uid-1', '{"model_name": "llama"}');
            """
        )

    store = LaunchHistoryStore(db)
    row = store.list()[0]
    assert row["autostart_enabled"] is False
    assert row["autostart_priority"] == 100
    assert row["ui_data"] == {}
    assert store.list_autostart() == []


@pytest.mark.parametrize(
    ("ui_n_gpu", "launch_n_gpu"),
    [("GPU", "auto"), ("CPU", None)],
)
def test_history_device_metadata_does_not_corrupt_enabled_autostart(
    store, ui_n_gpu, launch_n_gpu
):
    launch = {
        "model_name": "llama",
        "model_uid": "uid-1",
        "model_type": "embedding",
        "n_gpu": launch_n_gpu,
    }
    store.upsert_autostart({"launch": launch}, username="alice")

    store.upsert(
        "llama",
        "uid-1",
        launch,
        username="alice",
        ui_data={"n_gpu": ui_n_gpu},
    )

    row = store.list(model_name="llama", username="alice")[0]
    assert row["data"]["n_gpu"] == launch_n_gpu
    assert row["ui_data"] == {"n_gpu": ui_n_gpu}
    autostart_entry = store.list_autostart()[0]
    assert autostart_entry["launch"]["n_gpu"] == launch_n_gpu
    normalized_launch = normalize_autostart_model_entry(autostart_entry)["launch"]
    if launch_n_gpu is None:
        assert "n_gpu" not in normalized_launch
    else:
        assert normalized_launch["n_gpu"] == launch_n_gpu


def test_save_with_autostart_preserves_ui_metadata_in_either_write_order(store):
    launch = {
        "model_name": "llama",
        "model_uid": "uid-1",
        "model_type": "embedding",
        "n_gpu": "auto",
    }

    # History can finish before the independently saved autostart request.
    store.upsert(
        "llama",
        "uid-1",
        launch,
        username="alice",
        ui_data={"n_gpu": "GPU"},
    )
    store.upsert_autostart({"launch": launch}, username="alice")
    row = store.list(model_name="llama", username="alice")[0]
    assert row["data"]["n_gpu"] == "auto"
    assert row["ui_data"] == {"n_gpu": "GPU"}

    # A later history save updates only the UI intent with normalized launch data.
    cpu_launch = {**launch, "n_gpu": None}
    store.upsert_autostart({"launch": cpu_launch}, username="alice")
    store.upsert(
        "llama",
        "uid-1",
        cpu_launch,
        username="alice",
        ui_data={"n_gpu": "CPU"},
    )
    row = store.list(model_name="llama", username="alice")[0]
    assert row["data"]["n_gpu"] is None
    assert row["ui_data"] == {"n_gpu": "CPU"}
    assert store.list_autostart()[0]["launch"]["n_gpu"] is None


def test_legacy_history_update_preserves_existing_ui_metadata(store):
    store.upsert(
        "llama",
        "uid-1",
        {"model_name": "llama", "model_uid": "uid-1", "n_gpu": "auto"},
        username="alice",
        ui_data={"n_gpu": "GPU"},
    )

    store.upsert(
        "llama",
        "uid-1",
        {"model_name": "llama", "model_uid": "uid-1", "n_gpu": "auto"},
        username="alice",
    )

    assert store.list(model_name="llama", username="alice")[0]["ui_data"] == {
        "n_gpu": "GPU"
    }


def test_upsert_autostart_inserts_and_lists_unredacted_payload(store):
    store.upsert_autostart(
        {
            "enabled": True,
            "priority": 10,
            "max_retries": 5,
            "retry_interval_seconds": 60,
            "launch": {
                "model_name": "llama",
                "model_uid": "uid-1",
                "model_engine": "vllm",
                "envs": {"HF_TOKEN": "secret"},
            },
        },
        username="alice",
    )

    entries = store.list_autostart()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["priority"] == 10
    assert entry["max_retries"] == 5
    assert entry["retry_interval_seconds"] == 60
    assert entry["created_by"] == "alice"
    assert entry["launch"]["envs"] == {"HF_TOKEN": "secret"}


def test_list_autostart_redacts_other_users_non_shareable_fields(store):
    store.upsert_autostart(
        {
            "launch": {
                "model_name": "llama",
                "model_uid": "uid-1",
                "model_engine": "vllm",
                "n_gpu": "auto",
                "model_path": "/private/alice/model",
                "worker_ip": "10.0.0.8",
                "envs": {"HF_TOKEN": "secret"},
                "custom_private_kwarg": "private",
            },
        },
        username="alice",
    )

    entries = store.list_autostart(username="bob")

    assert len(entries) == 1
    assert entries[0]["created_by"] == "alice"
    assert entries[0]["launch"] == {
        "model_name": "llama",
        "model_uid": "uid-1",
        "model_engine": "vllm",
        "n_gpu": "auto",
    }

    owner_entry = store.list_autostart(username="alice")[0]
    assert owner_entry["launch"]["model_path"] == "/private/alice/model"
    assert owner_entry["launch"]["worker_ip"] == "10.0.0.8"

    scheduler_entry = store.list_autostart()[0]
    assert scheduler_entry["launch"]["custom_private_kwarg"] == "private"


def test_upsert_autostart_keeps_model_uid_globally_unique(store):
    store.upsert_autostart(
        {
            "launch": {
                "model_name": "llama",
                "model_uid": "shared-uid",
                "model_engine": "vllm",
            }
        },
        username="alice",
    )
    store.upsert_autostart(
        {
            "launch": {
                "model_name": "qwen",
                "model_uid": "shared-uid",
                "model_engine": "transformers",
            }
        },
        username="bob",
    )

    entries = store.list_autostart()
    assert len(entries) == 1
    assert entries[0]["created_by"] == "bob"
    assert entries[0]["launch"]["model_name"] == "qwen"
    rows = store.list()
    assert len(rows) == 2
    assert sum(1 for row in rows if row["autostart_enabled"]) == 1


def test_upsert_autostart_disabled_entry_clears_existing_autostart(store):
    store.upsert_autostart(
        {"launch": {"model_name": "llama", "model_uid": "uid-1"}},
        username="alice",
    )
    store.upsert_autostart(
        {
            "enabled": False,
            "launch": {"model_name": "llama", "model_uid": "uid-1"},
        },
        username="alice",
    )

    assert store.list_autostart() == []
    rows = store.list()
    assert len(rows) == 1
    assert rows[0]["autostart_enabled"] is False


def test_remove_autostart_preserves_history_row(store):
    store.upsert_autostart(
        {"launch": {"model_name": "llama", "model_uid": "uid-1"}},
        username="alice",
    )

    assert store.remove_autostart("uid-1") is True
    assert store.list_autostart() == []
    rows = store.list()
    assert len(rows) == 1
    assert rows[0]["autostart_enabled"] is False


def test_delete_history_removes_autostart_source(store):
    store.upsert_autostart(
        {"launch": {"model_name": "llama", "model_uid": "uid-1"}},
        username="alice",
    )

    assert store.delete("llama", "uid-1", "alice") is True
    assert store.list() == []
    assert store.list_autostart() == []


# --- Router handler tests ---


@pytest.fixture
def mock_api():
    api = MagicMock()
    api._launch_history_store = MagicMock()
    return api


def _request_with_json(body):
    request = MagicMock()
    request.json = AsyncMock(return_value=body)
    return request


def test_list_handler_returns_store_data(mock_api):
    mock_api._launch_history_store.list.return_value = [{"model_name": "llama"}]
    response = launch_history.list_launch_history(model_name="llama", api=mock_api)
    assert _json_body(response) == [{"model_name": "llama", "is_owner": True}]
    mock_api._launch_history_store.list.assert_called_once_with(
        model_name="llama", username="", owner_only=False
    )


def test_list_handler_marks_owner_and_other_users(mock_api):
    mock_api._launch_history_store.list.return_value = [
        {"model_name": "llama", "created_by": "alice", "data": {"envs": {"A": "1"}}},
        {"model_name": "llama", "created_by": "bob", "data": {"model_engine": "vllm"}},
    ]
    response = launch_history.list_launch_history(
        model_name="llama", api=mock_api, user={"username": "alice"}
    )
    body = _json_body(response)
    assert body[0]["is_owner"] is True
    assert body[1]["is_owner"] is False
    assert body[0]["data"] == {"envs": {"A": "1"}}
    assert body[1]["data"] == {"model_engine": "vllm"}


def test_list_handler_marks_anonymous_record_as_owner(mock_api):
    mock_api._launch_history_store.list.return_value = [
        {"model_name": "llama", "created_by": ""}
    ]
    response = launch_history.list_launch_history(model_name="llama", api=mock_api)
    assert _json_body(response)[0]["is_owner"] is True


def test_list_handler_scope_mine_forwards_owner_filter(mock_api):
    mock_api._launch_history_store.list.return_value = []

    response = launch_history.list_launch_history(
        model_name="llama",
        scope="mine",
        api=mock_api,
        user={"username": "alice"},
    )

    assert _json_body(response) == []
    mock_api._launch_history_store.list.assert_called_once_with(
        model_name="llama", username="alice", owner_only=True
    )


def test_list_handler_raises_500_on_error(mock_api):
    mock_api._launch_history_store.list.side_effect = RuntimeError("boom")
    with pytest.raises(HTTPException) as exc:
        launch_history.list_launch_history(api=mock_api)
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_create_handler_upserts_with_username(mock_api):
    request = _request_with_json(
        {
            "model_name": "llama",
            "model_uid": "",
            "data": {"model_engine": "vllm"},
            "ui_data": {"n_gpu": "GPU"},
        }
    )
    response = await launch_history.create_launch_history(
        request=request, api=mock_api, user={"username": "alice"}
    )
    assert _json_body(response) == {"status": "ok"}
    mock_api._launch_history_store.upsert.assert_called_once_with(
        model_name="llama",
        model_uid="",
        data={"model_engine": "vllm"},
        username="alice",
        ui_data={"n_gpu": "GPU"},
    )


@pytest.mark.asyncio
async def test_create_handler_blank_username_without_user(mock_api):
    request = _request_with_json({"model_name": "llama", "data": {}})
    await launch_history.create_launch_history(request=request, api=mock_api)
    _, kwargs = mock_api._launch_history_store.upsert.call_args
    assert kwargs["username"] == ""
    assert kwargs["ui_data"] is None


@pytest.mark.asyncio
async def test_create_handler_requires_model_name(mock_api):
    request = _request_with_json({"data": {"x": 1}})
    with pytest.raises(HTTPException) as exc:
        await launch_history.create_launch_history(request=request, api=mock_api)
    assert exc.value.status_code == 400
    mock_api._launch_history_store.upsert.assert_not_called()


def test_delete_handler_ok(mock_api):
    mock_api._launch_history_store.delete.return_value = True
    response = launch_history.delete_launch_history(
        model_name="llama", model_uid="", api=mock_api
    )
    assert _json_body(response) == {"status": "ok"}


def test_delete_handler_404_when_missing(mock_api):
    mock_api._launch_history_store.delete.return_value = False
    with pytest.raises(HTTPException) as exc:
        launch_history.delete_launch_history(
            model_name="nope", model_uid="", api=mock_api
        )
    assert exc.value.status_code == 404


# --- register_routes tests ---


def _register_and_capture(is_auth):
    captured = {}

    def add_api_route(path, endpoint, methods=None, **kwargs):
        captured[(path, tuple(methods or []))] = endpoint

    api = MagicMock()
    api._router.add_api_route.side_effect = add_api_route
    api._auth_service = MagicMock()
    api.is_authenticated.return_value = is_auth
    api._launch_history_store = MagicMock()
    launch_history.register_routes(api)
    return api, captured


def test_auth_off_post_handler_exposes_no_user_param():
    # Regression: when auth is disabled the POST handler must not expose a
    # `user: Optional[dict]` parameter, otherwise FastAPI treats it as a request
    # body field and rejects normal launches with HTTP 422.
    _, captured = _register_and_capture(is_auth=False)
    post_handler = captured[("/v1/launch_history", ("POST",))]
    assert "user" not in inspect.signature(post_handler).parameters


@pytest.mark.asyncio
async def test_auth_off_post_handler_forwards_none_user():
    api, captured = _register_and_capture(is_auth=False)
    post_handler = captured[("/v1/launch_history", ("POST",))]
    request = _request_with_json({"model_name": "llama", "data": {"x": 1}})
    response = await post_handler(request=request, api_=api)
    assert _json_body(response) == {"status": "ok"}
    _, kwargs = api._launch_history_store.upsert.call_args
    assert kwargs["username"] == ""


def test_auth_off_get_handler_exposes_no_user_param():
    _, captured = _register_and_capture(is_auth=False)
    get_handler = captured[("/v1/launch_history", ("GET",))]
    assert "user" not in inspect.signature(get_handler).parameters


def test_auth_off_get_handler_forwards_blank_username():
    api, captured = _register_and_capture(is_auth=False)
    api._launch_history_store.list.return_value = []
    get_handler = captured[("/v1/launch_history", ("GET",))]
    get_handler(model_name="llama", api_=api)
    api._launch_history_store.list.assert_called_once_with(
        model_name="llama", username="", owner_only=False
    )


def test_auth_on_get_handler_scopes_by_username():
    api, captured = _register_and_capture(is_auth=True)
    api._launch_history_store.list.return_value = []
    get_handler = captured[("/v1/launch_history", ("GET",))]
    get_handler(model_name="llama", user={"username": "alice"}, api_=api)
    api._launch_history_store.list.assert_called_once_with(
        model_name="llama", username="alice", owner_only=False
    )


def test_auth_off_get_handler_forwards_scope_mine():
    api, captured = _register_and_capture(is_auth=False)
    api._launch_history_store.list.return_value = []
    get_handler = captured[("/v1/launch_history", ("GET",))]

    get_handler(model_name=None, scope="mine", api_=api)

    api._launch_history_store.list.assert_called_once_with(
        model_name=None, username="", owner_only=True
    )


def test_auth_on_get_handler_forwards_scope_mine():
    api, captured = _register_and_capture(is_auth=True)
    api._launch_history_store.list.return_value = []
    get_handler = captured[("/v1/launch_history", ("GET",))]

    get_handler(model_name=None, scope="mine", user={"username": "alice"}, api_=api)

    api._launch_history_store.list.assert_called_once_with(
        model_name=None, username="alice", owner_only=True
    )


def test_get_handler_rejects_unknown_scope():
    api, captured = _register_and_capture(is_auth=False)
    app = FastAPI()
    app.state.api = api
    app.add_api_route(
        "/v1/launch_history",
        captured[("/v1/launch_history", ("GET",))],
        methods=["GET"],
    )

    response = TestClient(app).get("/v1/launch_history", params={"scope": "all"})

    assert response.status_code == 422
    api._launch_history_store.list.assert_not_called()


def test_auth_off_delete_handler_exposes_no_user_param():
    _, captured = _register_and_capture(is_auth=False)
    delete_handler = captured[("/v1/launch_history/{model_name}", ("DELETE",))]
    assert "user" not in inspect.signature(delete_handler).parameters


def test_auth_on_delete_handler_scopes_by_username():
    api, captured = _register_and_capture(is_auth=True)
    api._launch_history_store.delete.return_value = True
    delete_handler = captured[("/v1/launch_history/{model_name}", ("DELETE",))]
    delete_handler(
        model_name="llama", model_uid="", user={"username": "alice"}, api_=api
    )
    args, _ = api._launch_history_store.delete.call_args
    assert args == ("llama", "", "alice")


@pytest.mark.parametrize("target", ["history", "autostart"])
def test_history_cannot_replace_trusted_interpreter_in_enabled_autostart(store, target):
    launch = {"model_uid": "uid-1", "model_name": "test"}
    store.upsert_autostart({"launch": launch}, username="alice")
    store.update_autostart_launch_config(
        "uid-1", {"model_python_executable": "/admin-selected/python"}
    )
    payload = {**launch, "model_python_executable": "/arbitrary/python"}
    with pytest.raises(ValueError, match="reserved for internal recovery"):
        if target == "history":
            store.upsert("test", "uid-1", payload, username="alice")
        else:
            store.upsert_autostart({"launch": payload}, username="alice")
    assert (
        store.list_autostart()[0]["launch"]["model_python_executable"]
        == "/admin-selected/python"
    )


@pytest.mark.asyncio
async def test_history_route_rejects_internal_interpreter_before_persisting(mock_api):
    request = MagicMock(
        json=AsyncMock(
            return_value={
                "model_name": "test",
                "data": {"model_python_executable": "/arbitrary/python"},
            }
        )
    )
    with pytest.raises(HTTPException) as error:
        await launch_history.create_launch_history(
            request, api=mock_api, user={"username": "alice"}
        )
    assert error.value.status_code == 400
    mock_api._launch_history_store.upsert.assert_not_called()


def _runtime_api(store):
    import asyncio
    from types import SimpleNamespace

    from xinference.core.supervisor import SupervisorActor

    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._launch_history_store = store
    supervisor._autostart_store_lock = asyncio.Lock()
    supervisor._autostart_model_states = {}
    supervisor._status_guard_ref = SimpleNamespace(
        get_instance_info=AsyncMock(return_value=[])
    )
    supervisor._run_in_executor = AsyncMock(side_effect=lambda func, *args: func(*args))
    supervisor._schedule_autostart = MagicMock()
    supervisor._launch_builtin_model = AsyncMock(return_value="uid-1")
    api = SimpleNamespace(
        _launch_history_store=store,
        _get_supervisor_ref=AsyncMock(return_value=supervisor),
    )
    return api, supervisor


def _save_runtime(store, username="alice"):
    store.upsert_autostart(
        {"launch": {"model_name": "test", "model_uid": "uid-1", "max_num_seqs": 16}},
        username=username,
    )
    store.update_autostart_launch_config(
        "uid-1", {"model_python_executable": "/admin-selected/python"}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_ready", [False, True])
async def test_fresh_launch_clears_previous_runtime_but_autostart_replay_keeps_it(
    store, wait_ready
):
    _save_runtime(store)
    _, supervisor = _runtime_api(store)
    entry = (await supervisor._load_autostart_entries())[0]
    await supervisor._launch_autostart_model(entry["launch"])
    assert "model_python_executable" in store.list_autostart()[0]["launch"]

    # Terminate the loaded instance, preserving its enabled autostart policy.
    supervisor._get_model_replica_lock = lambda _: supervisor._autostart_store_lock
    supervisor._model_reload_tasks = {}
    supervisor._terminate_model = AsyncMock()
    await supervisor.terminate_model("uid-1")
    await supervisor.launch_builtin_model(
        model_name="test", model_uid="uid-1", wait_ready=wait_ready
    )
    public_entry = store.list_autostart(username="alice")[0]
    store.upsert_autostart(public_entry, username="alice")
    supervisor._launch_history_store = LaunchHistoryStore(store._db_path)
    launch = (await supervisor._load_autostart_entries())[0]["launch"]
    assert "model_python_executable" not in launch
    await supervisor._launch_autostart_model(launch)
    assert (
        "model_python_executable"
        not in supervisor._launch_builtin_model.await_args.kwargs
    )


@pytest.mark.asyncio
async def test_rejected_fresh_launch_keeps_current_runtime(store):
    _save_runtime(store)
    _, supervisor = _runtime_api(store)
    supervisor._launch_builtin_model.side_effect = ValueError("already running")
    with pytest.raises(ValueError, match="already running"):
        await supervisor.launch_builtin_model(model_name="test", model_uid="uid-1")
    assert (
        store.list_autostart()[0]["launch"]["model_python_executable"]
        == "/admin-selected/python"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("username", ["alice", ""])
async def test_owner_history_replay_and_save_preserves_runtime(store, username):
    from xinference.api.restful_api import RESTfulAPI

    _save_runtime(store, username)
    api, supervisor = _runtime_api(store)
    user = {"username": username}
    history = _json_body(
        launch_history.list_launch_history(
            model_name=None, scope="mine", api=api, user=user
        )
    )[0]
    assert history["is_owner"]
    assert "model_python_executable" not in json.dumps(history)
    assert "runtime_python_executable" not in json.dumps(history)
    history["data"]["max_num_seqs"] = 32
    response = await launch_history.create_launch_history(
        _request_with_json(history), api=api, user=user
    )
    assert response.status_code == 200
    response = await RESTfulAPI.upsert_autostart_model(
        api, _request_with_json({"launch": history["data"]}), user=user
    )
    assert response.status_code == 200
    saved = (await supervisor._load_autostart_entries())[0]["launch"]
    assert saved["max_num_seqs"] == 32
    assert saved["model_python_executable"] == "/admin-selected/python"


@pytest.mark.asyncio
@pytest.mark.parametrize("username", ["alice", "bob"])
@pytest.mark.parametrize(
    "field,value", [("priority", 5), ("max_retries", 7), ("retry_interval_seconds", 90)]
)
async def test_autostart_get_policy_edit_post_preserves_runtime(
    store, field, value, username
):
    from xinference.api.restful_api import RESTfulAPI

    _save_runtime(store)
    api, supervisor = _runtime_api(store)
    user = {"username": username}
    entry = _json_body(await RESTfulAPI.get_autostart_config(api, user))["models"][0]
    assert "model_python_executable" not in entry["launch"]
    entry[field] = value
    response = await RESTfulAPI.upsert_autostart_model(
        api, _request_with_json(entry), user
    )
    assert response.status_code == 200
    # Exercise a fresh store instance, as Supervisor restart does.
    supervisor._launch_history_store = LaunchHistoryStore(store._db_path)
    restored = (await supervisor._load_autostart_entries())[0]
    assert restored[field] == value
    assert restored["launch"]["model_python_executable"] == "/admin-selected/python"
    await supervisor._launch_autostart_model(restored["launch"])
    assert (
        supervisor._launch_builtin_model.await_args.kwargs["model_python_executable"]
        == "/admin-selected/python"
    )


@pytest.mark.asyncio
async def test_legacy_autostart_json_cannot_select_interpreter_after_migration(
    tmp_path,
):
    db = tmp_path / "legacy.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE launch_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_uid TEXT NOT NULL DEFAULT '',
                data TEXT NOT NULL,
                autostart_enabled INTEGER NOT NULL DEFAULT 0,
                created_by TEXT DEFAULT '', updated_by TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.execute(
            "INSERT INTO launch_history (model_name, model_uid, data, autostart_enabled, created_by) VALUES (?, ?, ?, 1, ?)",
            (
                "test",
                "uid-1",
                json.dumps(
                    {
                        "model_name": "test",
                        "model_python_executable": "/untrusted/python",
                    }
                ),
                "alice",
            ),
        )
    store = LaunchHistoryStore(str(db))
    _, supervisor = _runtime_api(store)
    entries = await supervisor._load_autostart_entries()
    assert len(entries) == 1
    assert "model_python_executable" not in entries[0]["launch"]
    await supervisor._launch_autostart_model(entries[0]["launch"])
    assert (
        "model_python_executable"
        not in supervisor._launch_builtin_model.await_args.kwargs
    )
    for username in (None, "alice", "bob", ""):
        assert "model_python_executable" not in json.dumps(
            store.list(username=username)
        )
    store.update_autostart_launch_config("uid-1", {"max_num_seqs": 32})
    assert "model_python_executable" not in store.list_autostart()[0]["launch"]
    # Only a subsequent authorized commit may select an interpreter.
    store.update_autostart_launch_config(
        "uid-1", {"model_python_executable": "/authorized/python"}
    )
    restarted = LaunchHistoryStore(str(db))
    assert (
        restarted.list_autostart()[0]["launch"]["model_python_executable"]
        == "/authorized/python"
    )


def test_runtime_config_survives_parameter_updates_and_can_be_cleared(store):
    _save_runtime(store)
    patch = {"max_num_seqs": 32}
    store.update_autostart_launch_config("uid-1", patch)
    assert patch == {"max_num_seqs": 32}
    assert (
        store.list_autostart()[0]["launch"]["model_python_executable"]
        == "/admin-selected/python"
    )
    assert "model_python_executable" not in store.list()[0]["data"]
    store.update_autostart_launch_config("uid-1", {"model_python_executable": None})
    restored = LaunchHistoryStore(store._db_path).list_autostart()[0]["launch"]
    assert restored["max_num_seqs"] == 32
    assert "model_python_executable" not in restored


@pytest.mark.parametrize("username", [None, "alice", "bob", ""])
def test_every_history_projection_excludes_server_runtime(store, username):
    _save_runtime(store)
    entries = store.list(username=username)
    assert len(entries) == 1
    assert "model_python_executable" not in json.dumps(entries)
    assert "/admin-selected/python" not in json.dumps(entries)


def test_policy_disable_and_reenable_preserves_runtime_until_history_deleted(store):
    _save_runtime(store)
    entry = {"launch": {"model_name": "test", "model_uid": "uid-1"}, "enabled": False}
    store.upsert_autostart(entry, username="alice")
    assert store.list_autostart() == []
    entry["enabled"] = True
    store.upsert_autostart(entry, username="alice")
    assert (
        store.list_autostart()[0]["launch"]["model_python_executable"]
        == "/admin-selected/python"
    )
    assert store.delete("test", "uid-1", username="alice")
    store.upsert_autostart(entry, username="alice")
    assert "model_python_executable" not in store.list_autostart()[0]["launch"]


@pytest.mark.parametrize("new_name,new_uid", [("other", "uid-1"), ("test", "uid-2")])
def test_runtime_is_not_shared_with_another_model(store, new_name, new_uid):
    _save_runtime(store)
    store.upsert_autostart(
        {"launch": {"model_name": new_name, "model_uid": new_uid}}, username="alice"
    )
    entry = next(
        e
        for e in store.list_autostart()
        if e["launch"]["model_uid"] == new_uid and e["launch"]["model_name"] == new_name
    )
    assert "model_python_executable" not in entry["launch"]
