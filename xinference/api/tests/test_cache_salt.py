# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request

from xinference import extensions
from xinference.api import model_request_logging, restful_api
from xinference.api.model_request_logging import ModelRequestLoggingRoute
from xinference.api.restful_api import RESTfulAPI
from xinference.cache_isolation import validate_cache_salt
from xinference.core.model import ModelActor
from xinference.core.utils import truncate_log_arg
from xinference.types import CreateChatCompletion, CreateCompletion


@pytest.mark.parametrize(
    "salt", ["", 123, False, [], {}, "x" * 257, "a\nb", "a\x00b", "\ud800"]
)
def test_invalid_salt_does_not_echo_secret(salt):
    with pytest.raises(ValueError, match="cache_salt"):
        validate_cache_salt(salt)


def test_schema_preserves_salt_and_logs_redact_nested_configs():
    salt = "customer-secret"
    assert CreateCompletion(model="m", prompt="hi", cache_salt=salt).cache_salt == salt
    assert (
        CreateChatCompletion(model="m", messages=[], cache_salt=salt).cache_salt == salt
    )
    config = {"cache_salt": salt, "nested": [{"cache_salt": salt}], "max_tokens": 3}
    assert salt not in truncate_log_arg((config, {"raw_params": config}))
    assert config["cache_salt"] == salt
    assert salt not in repr(extensions.CacheSalt(salt))


@pytest.mark.parametrize("required", [False, True])
def test_ingress_policy_preserves_unsalted_behavior_and_requires_when_enabled(
    monkeypatch, required
):
    monkeypatch.setattr(extensions, "get_extensions", lambda: ())
    monkeypatch.setattr(restful_api, "XINFERENCE_REQUIRE_CACHE_SALT", required)
    if required:
        with pytest.raises(HTTPException, match="cache_salt is required"):
            RESTfulAPI._apply_cache_salt(object(), {})
    else:
        config = {}
        RESTfulAPI._apply_cache_salt(object(), config)
        assert config == {}
    config, raw = {"cache_salt": "scope"}, {}
    RESTfulAPI._apply_cache_salt(object(), config, raw)
    assert config == raw == {"cache_salt": "scope"}


@pytest.mark.parametrize("result", [None, "wrong-type", RuntimeError("secret")])
def test_provider_failure_never_falls_back_to_client_salt(monkeypatch, result):
    def provider(request):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(
        extensions,
        "get_extensions",
        lambda: (SimpleNamespace(name="test", cache_salt=provider),),
    )
    with pytest.raises(HTTPException) as caught:
        RESTfulAPI._apply_cache_salt(object(), {"cache_salt": "client"})
    assert caught.value.status_code == 500
    assert caught.value.detail == "Cache salt provider failed"


def test_provider_is_authoritative_and_multiple_providers_are_rejected(monkeypatch):
    provider = SimpleNamespace(
        name="test", cache_salt=lambda request: extensions.CacheSalt(request.state.salt)
    )
    monkeypatch.setattr(extensions, "get_extensions", lambda: (provider,))
    request = SimpleNamespace(state=SimpleNamespace(salt="server"))
    source, config, raw = {"cache_salt": "spoofed"}, {}, {"cache_salt": "spoofed"}
    RESTfulAPI._apply_cache_salt(request, source, config, raw)
    assert source == config == raw == {"cache_salt": "server"}
    monkeypatch.setattr(extensions, "get_extensions", lambda: (provider, provider))
    with pytest.raises(HTTPException):
        RESTfulAPI._apply_cache_salt(request, source)


def test_unsupported_engine_rejects_salt():
    actor = SimpleNamespace(_model=object())
    ModelActor._validate_cache_salt(actor, ({},), {})
    with pytest.raises(ValueError, match="not supported"):
        ModelActor._validate_cache_salt(actor, ({"cache_salt": "scope"},), {})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint", ["completions", "chat/completions", "messages", "responses"]
)
@pytest.mark.parametrize("authoritative", [False, True])
@pytest.mark.parametrize(
    "client_salt", ["client-private-cache-scope", {}, [], 123, False]
)
async def test_api_resolves_salt_after_auth_before_model_dispatch(
    monkeypatch, endpoint, authoritative, client_salt
):
    api = object.__new__(RESTfulAPI)
    api._advanced_auth_service = None
    api._uid_to_model_name = {}
    order = []
    requests = []

    def check_access(request, *args):
        requests.append(request)
        order.append("auth")

    api._check_model_access = check_access
    provider_salt = "provider-private-cache-scope"
    expected_salt = provider_salt if authoritative else client_salt
    provider = SimpleNamespace(
        name="test",
        cache_salt=lambda request: (
            order.append("salt"),
            extensions.CacheSalt(provider_salt),
        )[1],
    )
    monkeypatch.setattr(
        extensions, "get_extensions", lambda: (provider,) if authoritative else ()
    )
    events = []
    monkeypatch.setattr(
        model_request_logging, "XINFERENCE_MODEL_REQUEST_LOG_ENABLED", True
    )
    monkeypatch.setattr(
        model_request_logging,
        "_get_model_request_logger",
        lambda: SimpleNamespace(info=lambda value: events.append(json.loads(value))),
    )
    model = SimpleNamespace(
        uid="m",
        is_vllm_backend=AsyncMock(return_value=True),
        get_limit_mm_per_prompt=AsyncMock(return_value=None),
    )
    inference_configs = []
    cached_bodies = []

    async def called(*args, **kwargs):
        order.append("model")
        inference_configs.extend((args[1], kwargs["raw_params"]))
        assert args[1]["cache_salt"] == expected_salt
        assert kwargs["raw_params"]["cache_salt"] == expected_salt
        cached_bodies.append(json.loads(json.dumps(await requests[-1].json())))
        raise HTTPException(418, "Reached engine")

    model.generate = model.chat = called
    model.prepare_anthropic_request = AsyncMock(
        side_effect=lambda messages, config: dict(
            messages=messages,
            generate_config=config,
            input_tokens=1,
            cache_control_applied=False,
        )
    )
    supervisor = SimpleNamespace(
        resolve_token_router_runtime=AsyncMock(return_value=None),
        describe_model=AsyncMock(
            return_value={"model_name": "m", "model_ability": ["chat"]}
        ),
    )
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    api._get_model_last_error = AsyncMock(side_effect=lambda uid, exc: exc)
    api._report_error_event = AsyncMock()
    monkeypatch.setattr(restful_api, "require_model", AsyncMock(return_value=model))
    app = FastAPI()
    app.router.route_class = ModelRequestLoggingRoute
    app.add_api_route(
        "/v1/" + endpoint,
        getattr(
            api,
            {
                "completions": "create_completion",
                "chat/completions": "create_chat_completion",
                "messages": "create_message",
                "responses": "create_response",
            }[endpoint],
        ),
        methods=["POST"],
    )
    body = dict(model="m", cache_salt=client_salt, max_tokens=1)
    body.update(
        {"prompt": "hi"}
        if endpoint == "completions"
        else (
            {"input": "hi"}
            if endpoint == "responses"
            else {"messages": [{"role": "user", "content": "hi"}]}
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/" + endpoint, json=body, headers={"anthropic-version": "2023-06-01"}
        )
    if not authoritative and not isinstance(client_salt, str):
        assert response.status_code == 400
        assert "cache_salt" in response.text
        assert order == ["auth"]
        assert not inference_configs
    else:
        assert order == (
            ["auth", "salt", "model"] if authoritative else ["auth", "model"]
        )
        assert all(
            config["cache_salt"] == expected_salt for config in inference_configs
        )
        assert await requests[0].json() == cached_bodies[0]
    started = next(
        event for event in events if event["event"] == "model_request_started"
    )
    assert started["request_body"]["cache_salt"] == "<redacted>"
    assert started["request_body"]["model"] == "m"
    if isinstance(client_salt, str):
        assert client_salt not in json.dumps(events)
    assert provider_salt not in json.dumps(events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b'{"cache_salt":"private-scope"} trailing',
        b'{"cache_salt":"private-scope',
        b'{"\\u0063ache_salt":"private-scope",',
        b'{"cache_salt":"private-scope", "prompt":"\xff"}',
    ],
)
async def test_request_log_omits_invalid_json_instead_of_logging_raw_salt(
    monkeypatch, body
):
    events = []
    monkeypatch.setattr(
        model_request_logging, "XINFERENCE_MODEL_REQUEST_LOG_ENABLED", True
    )
    monkeypatch.setattr(model_request_logging, "_write_event", events.append)
    app = FastAPI()
    app.router.route_class = ModelRequestLoggingRoute

    @app.post("/v1/completions")
    async def endpoint(request: Request):
        assert await request.body() == body
        raise HTTPException(400, "Request body must be valid JSON")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/completions",
            content=body,
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 400
    assert events[0]["request_body_omitted"] == {"reason": "invalid_json"}
    assert "request_body_raw" not in events[0]
    assert "private-scope" not in json.dumps(events)
