# Copyright 2022-2026 XProbe Inc.
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
from fastapi import HTTPException

from xinference.api.routers import launch_history
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
    assert store.list_autostart() == []


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
    assert _json_body(response) == [{"model_name": "llama"}]
    mock_api._launch_history_store.list.assert_called_once_with(
        model_name="llama", username=""
    )


def test_list_handler_raises_500_on_error(mock_api):
    mock_api._launch_history_store.list.side_effect = RuntimeError("boom")
    with pytest.raises(HTTPException) as exc:
        launch_history.list_launch_history(api=mock_api)
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_create_handler_upserts_with_username(mock_api):
    request = _request_with_json(
        {"model_name": "llama", "model_uid": "", "data": {"model_engine": "vllm"}}
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
    )


@pytest.mark.asyncio
async def test_create_handler_blank_username_without_user(mock_api):
    request = _request_with_json({"model_name": "llama", "data": {}})
    await launch_history.create_launch_history(request=request, api=mock_api)
    _, kwargs = mock_api._launch_history_store.upsert.call_args
    assert kwargs["username"] == ""


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
    _, kwargs = api._launch_history_store.list.call_args
    assert kwargs["username"] == ""


def test_auth_on_get_handler_scopes_by_username():
    api, captured = _register_and_capture(is_auth=True)
    api._launch_history_store.list.return_value = []
    get_handler = captured[("/v1/launch_history", ("GET",))]
    get_handler(model_name="llama", user={"username": "alice"}, api_=api)
    _, kwargs = api._launch_history_store.list.call_args
    assert kwargs["username"] == "alice"


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
