# CI: linux
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

"""Inference 4xx responses leave one content-free WARNING line."""

import json
import logging
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException, Request
from starlette.responses import JSONResponse, Response

from .. import model_request_logging, restful_api
from ..model_request_logging import schema_error_fields, schema_field_names
from .test_llm_request_abort import _new_api
from .test_model_request_logging import _app

# Placed in every request's content.  It must never reach the log.
SENTINEL = "SENTINEL-c0ntent 7f3a9e"
REQUEST_ID = "saas-" + "a" * 64
LOGGER = model_request_logging.logger.name
CHAT = "/v1/chat/completions"


def _messages():
    return [{"role": "user", "content": f"secret prompt {SENTINEL}"}]


def _chat_body(**overrides):
    body = {"model": "m", "request_id": REQUEST_ID, "messages": _messages()}
    body.update(overrides)
    return body


def _api(monkeypatch, *, token_router_runtime=None):
    model = SimpleNamespace(
        uid=b"m-rep0",
        is_vllm_backend=AsyncMock(return_value=False),
        chat=AsyncMock(return_value=b'{"choices": []}'),
        generate=AsyncMock(return_value=b'{"choices": []}'),
    )
    supervisor = SimpleNamespace(
        get_model=AsyncMock(return_value=model),
        describe_model=AsyncMock(return_value={"model_family": "test-family"}),
        resolve_token_router_runtime=AsyncMock(return_value=token_router_runtime),
    )
    api = _new_api()
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    monkeypatch.setattr(
        restful_api,
        "XINFERENCE_TOKEN_ROUTER_ENABLED",
        token_router_runtime is not None,
    )
    return api


async def _post(app, path, content, headers=None):
    if not isinstance(content, str):
        content = json.dumps(content)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        return await client.post(
            path,
            content=content,
            headers={"content-type": "application/json", **(headers or {})},
        )


def _rejections(caplog):
    return [
        record
        for record in caplog.records
        if record.name == LOGGER
        and record.getMessage().startswith("Inference request rejected:")
    ]


def _fields(record):
    message = record.getMessage().split(": ", 1)[1]
    return dict(part.split("=", 1) for part in message.split(" "))


CHAT_CASES = {
    "invalid_json": (
        '{"model": "m", "messages": [{"role": "user", "content": "' + SENTINEL,
        400,
        "JSONDecodeError",
    ),
    "body_not_object": ([SENTINEL], 400, None),
    "schema_validation": (
        _chat_body(messages=SENTINEL, max_tokens=SENTINEL),
        422,
        "ValidationError",
    ),
    "empty_messages": (_chat_body(messages=[]), 400, None),
    "invalid_routing_key": (_chat_body(routing_key=" "), 400, None),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", CHAT_CASES)
async def test_chat_rejection_logs_one_tagged_line(monkeypatch, caplog, reason):
    content, status, cause = CHAT_CASES[reason]
    api = _api(monkeypatch)
    caplog.set_level(logging.DEBUG)

    response = await _post(_app(CHAT, api.create_chat_completion), CHAT, content)

    assert response.status_code == status
    records = _rejections(caplog)
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.WARNING
    fields = _fields(record)
    assert fields["reason"] == reason
    assert fields["status"] == str(status)
    assert fields["endpoint"] == json.dumps(CHAT)
    assert fields.get("cause") == cause
    if isinstance(content, dict):
        assert fields["request_id"] == json.dumps(REQUEST_ID)
        assert fields["model_uid"] == '"m"'
    else:
        # No parsed body: fall back to the HTTP correlation ID.
        assert fields["request_id"] == json.dumps(response.headers["x-request-id"])
    assert SENTINEL not in caplog.text


@pytest.mark.asyncio
async def test_schema_rejection_logs_only_location_and_type(monkeypatch, caplog):
    api = _api(monkeypatch)
    caplog.set_level(logging.DEBUG)
    body = _chat_body(messages=SENTINEL, max_tokens=SENTINEL)

    response = await _post(_app(CHAT, api.create_chat_completion), CHAT, body)

    # The response body still carries the full validation detail.
    assert response.status_code == 422
    assert {tuple(error["loc"]) for error in response.json()["detail"]} >= {
        ("messages",),
        ("max_tokens",),
    }
    fields = _fields(_rejections(caplog)[0])
    errors = fields["schema_errors"].split(",")
    # Error types differ between pydantic v1 and v2; both are identifiers.
    assert {error.split(":")[0] for error in errors} >= {"messages", "max_tokens"}
    assert all(not error.endswith(":*") for error in errors)
    for forbidden in ("input", "msg", "ctx", "detail", SENTINEL):
        assert forbidden not in caplog.text


@pytest.mark.asyncio
async def test_chat_template_kwargs_rejection_is_tagged(monkeypatch, caplog):
    api = _api(monkeypatch, token_router_runtime={"available": True})
    api._proxy_token_router_chat_completion = AsyncMock()
    caplog.set_level(logging.DEBUG)
    body = _chat_body(chat_template_kwargs=f"not json {SENTINEL}")

    response = await _post(_app(CHAT, api.create_chat_completion), CHAT, body)

    assert response.status_code == 400
    api._proxy_token_router_chat_completion.assert_not_awaited()
    records = _rejections(caplog)
    assert len(records) == 1
    fields = _fields(records[0])
    assert fields["reason"] == "chat_template_kwargs"
    assert fields["cause"] == "TokenizationError"
    assert SENTINEL not in caplog.text


@pytest.mark.asyncio
async def test_value_error_detail_is_not_logged_by_the_rejection_line(
    monkeypatch, caplog
):
    api = _api(monkeypatch)
    supervisor = await api._get_supervisor_ref()
    supervisor.describe_model.side_effect = ValueError(f"bad model {SENTINEL}")
    caplog.set_level(logging.DEBUG)

    response = await _post(_app(CHAT, api.create_chat_completion), CHAT, _chat_body())

    assert response.status_code == 400
    record = _rejections(caplog)[0]
    fields = _fields(record)
    assert fields["reason"] == "describe_model"
    assert fields["cause"] == "ValueError"
    assert SENTINEL not in record.getMessage()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content,reason",
    [
        ('{"model": "m", "prompt": "' + SENTINEL, "invalid_json"),
        (
            {"model": "m", "prompt": SENTINEL, "max_tokens": SENTINEL},
            "schema_validation",
        ),
    ],
)
async def test_completion_rejection_is_tagged(monkeypatch, caplog, content, reason):
    api = _api(monkeypatch)
    caplog.set_level(logging.DEBUG)
    path = "/v1/completions"

    response = await _post(_app(path, api.create_completion), path, content)

    assert response.status_code in (400, 422)
    records = _rejections(caplog)
    assert len(records) == 1
    assert _fields(records[0])["reason"] == reason
    assert SENTINEL not in caplog.text


@pytest.mark.asyncio
async def test_returned_responses_rejection_is_tagged(monkeypatch, caplog):
    api = _api(monkeypatch)
    caplog.set_level(logging.DEBUG)
    path = "/v1/responses"

    response = await _post(
        _app(path, api.create_response), path, '{"input": "' + SENTINEL
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Request body must be valid JSON"
    records = _rejections(caplog)
    assert len(records) == 1
    fields = _fields(records[0])
    assert fields["reason"] == "invalid_json"
    assert "exception" not in fields
    assert SENTINEL not in caplog.text


@pytest.mark.asyncio
async def test_untagged_http_exception_is_still_logged(caplog):
    async def endpoint(request: Request):
        raise HTTPException(status_code=400, detail=SENTINEL)

    caplog.set_level(logging.DEBUG)
    response = await _post(_app(CHAT, endpoint), CHAT, "{}")

    assert response.status_code == 400
    assert response.json() == {"detail": SENTINEL}
    fields = _fields(_rejections(caplog)[0])
    assert fields["reason"] == "untagged"
    assert fields["exception"] == "HTTPException"
    assert SENTINEL not in caplog.text


@pytest.mark.asyncio
async def test_route_parameter_validation_is_logged_as_422(caplog):
    async def endpoint(request: Request, limit: int):
        return Response()

    caplog.set_level(logging.DEBUG)
    response = await _post(_app(CHAT, endpoint), CHAT + f"?limit={SENTINEL}", "{}")

    assert response.status_code == 422
    fields = _fields(_rejections(caplog)[0])
    assert fields["status"] == "422"
    assert fields["reason"] == "request_validation"
    assert fields["schema_errors"] == "query.limit:int_parsing"
    assert SENTINEL not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 499, 500, 503])
async def test_non_rejections_are_not_logged(caplog, status):
    async def endpoint(request: Request):
        if status >= 500:
            raise HTTPException(status_code=status, detail="x")
        return JSONResponse({}, status_code=status)

    caplog.set_level(logging.DEBUG)
    response = await _post(_app(CHAT, endpoint), CHAT, "{}")

    assert response.status_code == status
    assert _rejections(caplog) == []


@pytest.mark.asyncio
async def test_untrusted_identifiers_are_quoted(caplog):
    async def endpoint(request: Request):
        model_request_logging.note_inference_request_identity(
            request, {"model": 'm" reason=forged', "request_id": "id x=1"}
        )
        raise HTTPException(status_code=400)

    caplog.set_level(logging.DEBUG)
    await _post(_app(CHAT, endpoint), CHAT, "{}")

    message = _rejections(caplog)[0].getMessage()
    assert 'model_uid="m\\" reason=forged"' in message
    assert 'request_id="id x=1"' in message
    assert message.count(" reason=") == 2  # the forged one stays inside quotes
    assert message.endswith("exception=HTTPException")


def test_schema_error_fields_drop_content_and_mask_untrusted_keys():
    errors = [
        {
            "loc": ("metadata", f"key {SENTINEL}", 0),
            "type": "string_type",
            "msg": SENTINEL,
            "input": SENTINEL,
            "ctx": {"error": SENTINEL},
        },
        {"loc": ("messages", 0, "content"), "type": SENTINEL},
        {"loc": ["max_tokens"], "type": "type_error.integer"},
        {"loc": ("tools\n", 1), "type": "missing\n"},
        {"loc": None},
        "not a dict",
    ]

    fields = schema_error_fields(
        errors, frozenset({"metadata", "messages", "content", "max_tokens", "tools"})
    )

    assert fields == [
        "metadata.*.0:string_type",
        "messages.0.content:*",
        "max_tokens:type_error.integer",
        "*.1:*",
        "*:*",
    ]


def test_schema_field_names_cover_nested_schema_names_only():
    names = schema_field_names(restful_api.CreateChatCompletion)

    assert {"messages", "logit_bias", "tools", "function", "parameters"} <= names
    assert "SamplePrivateLabel" not in names


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,handler,body",
    [
        (CHAT, "create_chat_completion", _chat_body()),
        (
            "/v1/completions",
            "create_completion",
            {"model": "m", "request_id": REQUEST_ID, "prompt": SENTINEL},
        ),
    ],
)
async def test_identifier_shaped_dict_keys_are_masked(
    monkeypatch, caplog, path, handler, body
):
    api = _api(monkeypatch)
    caplog.set_level(logging.DEBUG)
    body = dict(body, logit_bias={"SamplePrivateLabel": "not-an-int"})

    response = await _post(_app(path, getattr(api, handler)), path, body)

    # The response keeps pydantic's original location.
    assert response.status_code == 422
    assert ["logit_bias", "SamplePrivateLabel"] in [
        error["loc"] for error in response.json()["detail"]
    ]
    record = _rejections(caplog)[0]
    errors = _fields(record)["schema_errors"].split(",")
    assert any(error.startswith("logit_bias.*:") for error in errors)
    assert "SamplePrivateLabel" not in caplog.text


@pytest.mark.asyncio
async def test_model_access_denial_bounds_the_logged_model_id(monkeypatch, caplog):
    api = _api(monkeypatch)
    api._advanced_auth_service = SimpleNamespace(
        validate_model_access=lambda *args: False
    )
    api._check_model_access = MethodType(
        restful_api.RESTfulAPI._check_model_access, api
    )
    long_model = "m" * 300 + "TAIL"
    caplog.set_level(logging.DEBUG)

    response = await _post(
        _app(CHAT, api.create_chat_completion),
        CHAT,
        _chat_body(model=long_model),
        headers={"authorization": "Bearer test-key"},
    )

    assert response.status_code == 403
    records = _rejections(caplog)
    assert len(records) == 1
    fields = _fields(records[0])
    assert fields["reason"] == "model_access_denied"
    assert fields["model_uid"] == '""'
    assert fields["request_id"] == json.dumps(REQUEST_ID)
    assert "TAIL" not in records[0].getMessage()
