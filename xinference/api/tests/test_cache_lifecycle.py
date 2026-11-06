# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from .. import restful_api
from ..restful_api import RESTfulAPI
from .test_llm_request_abort import _new_api


class _Request:
    client = "test-client"

    def __init__(self, body, credential="one"):
        self.headers = {"Authorization": f"Bearer {credential}"} if credential else {}
        self.body = body
        self.state = SimpleNamespace()

    async def json(self):
        return self.body


def test_cache_session_scope_matches_chat_and_control_and_isolates_credentials():
    body = {"id": "agent", "revision": 1, "phase": "waiting"}
    chat = RESTfulAPI._cache_session_tag(_Request(body), body)
    control = RESTfulAPI._cache_session_tag(_Request(body), body, control=True)
    other = RESTfulAPI._cache_session_tag(_Request(body, "two"), body, control=True)
    assert chat.id == control.id != other.id
    assert "agent" not in chat.id and "one" not in chat.id
    assert chat.phase == "active" and control.phase == "waiting"
    assert chat.scope == control.scope != other.scope


def test_anonymous_clients_share_scope_and_session_ids_must_be_unique():
    body = {"id": "agent", "revision": 0}
    first = RESTfulAPI._cache_session_tag(_Request({}, credential=None), body)
    second = RESTfulAPI._cache_session_tag(_Request({}, credential=None), body)
    other = RESTfulAPI._cache_session_tag(
        _Request({}, credential=None), dict(body, id="other")
    )
    assert first.id == second.id != other.id
    assert first.scope == second.scope == other.scope


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "session",
    [False, 42, {}, {"id": "a", "revision": True}, {"id": " ", "revision": 0}],
)
async def test_chat_invalid_cache_session_returns_400(monkeypatch, session):
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    api = _new_api()
    supervisor = SimpleNamespace(update_cache_session=AsyncMock())
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    model = SimpleNamespace(
        is_vllm_backend=AsyncMock(return_value=True), chat=AsyncMock()
    )
    monkeypatch.setattr(restful_api, "require_model", AsyncMock(return_value=model))
    with pytest.raises(HTTPException) as caught:
        await api.create_chat_completion(
            _Request(
                {
                    "model": "model",
                    "messages": [{"role": "user", "content": "OK"}],
                    "cache_session": session,
                }
            )
        )
    assert caught.value.status_code == 400
    supervisor.update_cache_session.assert_not_awaited()
    model.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_control_authorizes_model_and_reports_stale_revision():
    api = object.__new__(RESTfulAPI)
    api._check_model_access = Mock()
    supervisor = SimpleNamespace(update_cache_session=AsyncMock(return_value=False))
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    request = _Request({"id": "agent", "revision": 1, "phase": "waiting"})
    response = await api.update_cache_session("model", request)
    assert json.loads(response.body) == {"accepted": False}
    api._check_model_access.assert_called_once_with(request, "model", "LLM")
    sent = supervisor.update_cache_session.call_args.args
    assert sent[0] == "model" and sent[1]["id"] != "agent"
    assert sent[1]["phase"] == "waiting"


@pytest.mark.asyncio
async def test_unauthorized_control_never_reaches_tracker():
    api = object.__new__(RESTfulAPI)
    api._check_model_access = Mock(side_effect=HTTPException(403, "Denied"))
    api._get_supervisor_ref = AsyncMock()
    with pytest.raises(HTTPException) as exc:
        await api.update_cache_session(
            "model", _Request({"id": "a", "revision": 0, "phase": "finished"})
        )
    assert exc.value.status_code == 403
    api._get_supervisor_ref.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("routing_key", [None, "same-agent"])
@pytest.mark.parametrize("vllm", [True, False])
@pytest.mark.parametrize(
    "hint_error", [None, asyncio.TimeoutError(), ConnectionError("unavailable")]
)
async def test_chat_scopes_metadata_and_continues_when_hints_are_unavailable(
    monkeypatch, hint_error, routing_key, vllm
):
    from ...model.llm.xavier.cache_lifecycle import CACHE_SESSION_ARG, CacheSessionTag
    from ..restful_api import Response

    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    api = _new_api()
    supervisor = SimpleNamespace(
        describe_model=AsyncMock(
            return_value={"model_family": "test", "model_engine": "vLLM"}
        ),
        update_cache_session=AsyncMock(return_value=True, side_effect=hint_error),
    )
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    model = SimpleNamespace(
        uid=b"model-rep0",
        is_vllm_backend=AsyncMock(return_value=vllm),
        is_transformers_batching_enabled=AsyncMock(return_value=False),
        chat=AsyncMock(return_value=b"{}"),
    )
    require_model = AsyncMock(return_value=model)

    async def run_request(request, actor, request_id, kind, call):
        return Response(await call, media_type="application/json")

    monkeypatch.setattr(restful_api, "require_model", require_model)
    monkeypatch.setattr(restful_api, "_run_non_streaming_request", run_request)
    request = _Request(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "OK"}],
            "cache_session": {"id": "agent", "revision": 0},
            "routing_key": routing_key,
            CACHE_SESSION_ARG: {"id": "spoofed", "revision": 99},
        }
    )
    response = await api.create_chat_completion(request)
    assert response.status_code == 200
    config = model.chat.call_args.args[1]
    raw = model.chat.call_args.kwargs["raw_params"]
    assert "cache_session" not in config and "cache_session" not in raw
    assert "routing_key" not in config and "routing_key" not in raw
    routing_options = require_model.await_args.kwargs
    if routing_key is not None:
        assert routing_options["routing_key"] != routing_key
        supervisor.describe_model.assert_awaited_once_with(
            "model", selected_replica_uid="model-rep0"
        )
    else:
        assert routing_options == {}
    if hint_error is None:
        assert isinstance(config[CACHE_SESSION_ARG], CacheSessionTag)
        assert config[CACHE_SESSION_ARG].id != "agent"
        assert config[CACHE_SESSION_ARG] == raw[CACHE_SESSION_ARG]
    else:
        assert CACHE_SESSION_ARG not in config and CACHE_SESSION_ARG not in raw
