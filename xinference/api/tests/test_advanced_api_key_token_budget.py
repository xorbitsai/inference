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

import json
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService
from xinference.api.oauth2.advanced.database import Database
from xinference.api.restful_api import RESTfulAPI


def _auth_service(tmp_path):
    return AdvancedAuthService(
        db_path=str(tmp_path / "auth.db"),
        jwt_secret_key="test-jwt-secret",
        encryption_key="test-encryption-key",
    )


def _api(auth):
    api = object.__new__(RESTfulAPI)
    api._advanced_auth_service = auth
    api._uid_to_model_name = {}
    api._set_trace_model = MagicMock()
    api._set_trace_model_type = MagicMock()
    api._get_supervisor_ref = AsyncMock()
    api._report_error_event = AsyncMock()
    api._get_model_last_error = AsyncMock(side_effect=lambda _uid, exc: exc)
    return api


def _api_key(auth, token_budget=None):
    admin = auth.db.get_user_by_username("admin")
    return auth.create_api_key_for_user(
        user_id=admin["id"],
        name="budgeted key",
        token_budget=token_budget,
    )


def _request(api_key, payload, path="/v1/completions"):
    request = MagicMock()
    request.headers = {"Authorization": f"Bearer {api_key}"}
    request.state = SimpleNamespace()
    request.client = SimpleNamespace(host="127.0.0.1")
    request.url = SimpleNamespace(path=path)
    request.json = AsyncMock(return_value=payload)
    return request


class _CompletionModel:
    uid = b"test-model"

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def generate(self, *_args, **_kwargs):
        self.calls += 1
        return json.dumps(self.payload)

    async def chat(self, *_args, **_kwargs):
        self.calls += 1
        return json.dumps(self.payload)

    async def is_vllm_backend(self):
        return False

    async def create_embedding(self, *_args, **_kwargs):
        self.calls += 1
        return json.dumps(self.payload)

    async def rerank(self, *_args, **_kwargs):
        self.calls += 1
        return json.dumps(self.payload)


class _Supervisor:
    async def describe_model(self, _model_uid):
        return {"model_family": "qwen2"}


class _StreamingCompletionModel:
    uid = b"test-model"

    async def generate(self, *_args, **_kwargs):
        async def _iter():
            yield {"data": json.dumps({"choices": [{"text": "hello"}]})}
            yield {
                "data": json.dumps(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 2,
                            "completion_tokens": 3,
                            "total_tokens": 5,
                        },
                    }
                )
            }

        return _iter()

    async def decrease_serve_count(self):
        return None


class _SplitSSEStreamingCompletionModel:
    uid = b"test-model"

    async def generate(self, *_args, **_kwargs):
        async def _iter():
            yield 'data: {"choices":[{"text":"hello"}]}\n\n'
            yield 'data: {"choices":[],"usage":{"prompt_tokens":2,'
            yield '"completion_tokens":4,"total_tokens":6}}\n\n'
            yield "data: [DONE]\n\n"

        return _iter()

    async def decrease_serve_count(self):
        return None


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"usage": {"prompt_tokens": 2, "completion_tokens": 3}}, 5),
        ({"usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 7}}, 7),
        ({"usage": {"input_tokens": 4, "output_tokens": 6}}, 10),
        ({"usage": {"total_tokens": 8}}, 8),
        ({"usage": {"total_tokens": 0}}, None),
        ({"data": []}, None),
    ],
)
def test_extract_token_usage_supports_inference_response_shapes(payload, expected):
    assert RESTfulAPI._extract_token_usage(json.dumps(payload)) == expected


@pytest.mark.asyncio
async def test_completion_usage_increments_api_key_token_usage(tmp_path, monkeypatch):
    auth = _auth_service(tmp_path)
    created_key = _api_key(auth, token_budget=10)
    model = _CompletionModel(
        {"usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}
    )
    monkeypatch.setattr(
        "xinference.api.restful_api.require_model", AsyncMock(return_value=model)
    )

    response = await _api(auth).create_completion(
        _request(
            created_key["key"],
            {"model": "test-model", "prompt": "hello", "stream": False},
        )
    )

    assert json.loads(response.body.decode())["usage"]["total_tokens"] == 5
    assert auth.db.get_api_key_by_id(created_key["id"])["token_usage"] == 5


@pytest.mark.asyncio
async def test_unlimited_api_key_still_records_usage(tmp_path, monkeypatch):
    auth = _auth_service(tmp_path)
    created_key = _api_key(auth)
    model = _CompletionModel({"usage": {"total_tokens": 4}})
    monkeypatch.setattr(
        "xinference.api.restful_api.require_model", AsyncMock(return_value=model)
    )

    await _api(auth).create_completion(
        _request(
            created_key["key"],
            {"model": "test-model", "prompt": "hello", "stream": False},
        )
    )

    key = auth.db.get_api_key_by_id(created_key["id"])
    assert key["token_budget"] is None
    assert key["token_usage"] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method_name, payload, response_payload, expected_tokens, path",
    [
        (
            "create_chat_completion",
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
            {
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            },
            7,
            "/v1/chat/completions",
        ),
        (
            "create_embedding",
            {"model": "test-model", "input": "hello"},
            {"data": [], "usage": {"prompt_tokens": 6, "total_tokens": 6}},
            6,
            "/v1/embeddings",
        ),
        (
            "rerank",
            {
                "model": "test-model",
                "query": "hello",
                "documents": ["a", "b"],
            },
            {"results": [], "usage": {"total_tokens": 9}},
            9,
            "/v1/rerank",
        ),
    ],
)
async def test_inference_endpoints_increment_api_key_token_usage(
    tmp_path,
    monkeypatch,
    method_name,
    payload,
    response_payload,
    expected_tokens,
    path,
):
    auth = _auth_service(tmp_path)
    created_key = _api_key(auth, token_budget=100)
    api = _api(auth)
    api._get_supervisor_ref = AsyncMock(return_value=_Supervisor())
    monkeypatch.setattr(
        "xinference.api.restful_api.require_model",
        AsyncMock(return_value=_CompletionModel(response_payload)),
    )

    await getattr(api, method_name)(_request(created_key["key"], payload, path=path))

    assert (
        auth.db.get_api_key_by_id(created_key["id"])["token_usage"] == expected_tokens
    )


@pytest.mark.asyncio
async def test_exhausted_token_budget_rejects_before_model_work(tmp_path, monkeypatch):
    auth = _auth_service(tmp_path)
    created_key = _api_key(auth, token_budget=5)
    auth.db.increment_api_key_token_usage(created_key["id"], 5)
    require_model = AsyncMock()
    monkeypatch.setattr("xinference.api.restful_api.require_model", require_model)

    with pytest.raises(HTTPException) as exc_info:
        await _api(auth).create_completion(
            _request(
                created_key["key"],
                {"model": "test-model", "prompt": "hello", "stream": False},
            )
        )

    assert exc_info.value.status_code == 429
    assert "token budget exhausted" in exc_info.value.detail.lower()
    require_model.assert_not_called()


@pytest.mark.asyncio
async def test_streaming_completion_records_final_usage_once(tmp_path, monkeypatch):
    auth = _auth_service(tmp_path)
    created_key = _api_key(auth, token_budget=20)
    monkeypatch.setattr(
        "xinference.api.restful_api.require_model",
        AsyncMock(return_value=_StreamingCompletionModel()),
    )

    response = await _api(auth).create_completion(
        _request(
            created_key["key"],
            {"model": "test-model", "prompt": "hello", "stream": True},
        )
    )
    async for _chunk in response.body_iterator:
        pass

    assert auth.db.get_api_key_by_id(created_key["id"])["token_usage"] == 5


@pytest.mark.asyncio
async def test_streaming_completion_records_split_sse_usage(tmp_path, monkeypatch):
    auth = _auth_service(tmp_path)
    created_key = _api_key(auth, token_budget=20)
    monkeypatch.setattr(
        "xinference.api.restful_api.require_model",
        AsyncMock(return_value=_SplitSSEStreamingCompletionModel()),
    )

    response = await _api(auth).create_completion(
        _request(
            created_key["key"],
            {"model": "test-model", "prompt": "hello", "stream": True},
        )
    )
    async for _chunk in response.body_iterator:
        pass

    assert auth.db.get_api_key_by_id(created_key["id"])["token_usage"] == 6


def test_api_key_token_usage_column_is_added_to_existing_databases(tmp_path):
    db_path = tmp_path / "old-auth.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL,
                password_hash TEXT,
                source TEXT NOT NULL DEFAULT 'local',
                enabled INTEGER DEFAULT 1,
                must_change_password INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE api_keys (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                key_hash TEXT UNIQUE NOT NULL,
                key_encrypted TEXT NOT NULL,
                key_prefix TEXT NOT NULL,
                name TEXT,
                enabled INTEGER DEFAULT 1,
                expires_at TIMESTAMP,
                encryption_version INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

    Database(str(db_path))

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(api_keys)")}

    assert "token_usage" in columns


def test_api_key_token_usage_increment_is_concurrency_safe(tmp_path):
    auth = _auth_service(tmp_path)
    created_key = _api_key(auth, token_budget=10_000)

    def worker():
        for _ in range(50):
            auth.db.increment_api_key_token_usage(created_key["id"], 2)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert auth.db.get_api_key_by_id(created_key["id"])["token_usage"] == 1_000
