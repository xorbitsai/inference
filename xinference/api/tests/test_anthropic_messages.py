import json
from types import MethodType

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from xinference.api import restful_api as restful_api_module
from xinference.api.restful_api import RESTfulAPI


def make_rest_api():
    api = RESTfulAPI.__new__(RESTfulAPI)
    api._advanced_auth_service = None
    api._uid_to_model_name = {}

    async def get_supervisor_ref(_self):
        return object()

    api._get_supervisor_ref = MethodType(get_supervisor_ref, api)
    return api


def make_app(api):
    app = FastAPI()
    app.add_api_route("/v1/messages", api.create_message, methods=["POST"])
    app.add_api_route("/anthropic/v1/messages", api.create_message, methods=["POST"])
    return app


@pytest.mark.asyncio
async def test_standard_path_requires_version_header():
    app = make_app(make_rest_api())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": "model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "anthropic-version" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_standard_path_rejects_unsupported_version():
    app = make_app(make_rest_api())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/messages",
            headers={"anthropic-version": "2099-01-01"},
            json={
                "model": "model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "2099-01-01" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_physical_non_stream_converts_bytes_response(monkeypatch):
    api = make_rest_api()
    app = make_app(api)
    calls = {}

    class FakeModel:
        uid = "physical-model"

        async def chat(self, messages, kwargs, raw_params=None):
            calls["messages"] = messages
            calls["kwargs"] = kwargs
            calls["raw_params"] = raw_params
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": "physical answer"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 2},
                }
            ).encode()

    async def fake_require_model(*args, **kwargs):
        return FakeModel()

    monkeypatch.setattr(restful_api_module, "require_model", fake_require_model)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/anthropic/v1/messages",
            json={
                "model": "physical-model",
                "system": "Be helpful.",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
            },
        )

    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": "physical answer"}]
    assert calls == {
        "messages": [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "hello"},
        ],
        "kwargs": {"max_tokens": 32, "stream": False},
        "raw_params": {"max_tokens": 32, "stream": False},
    }


@pytest.mark.asyncio
async def test_physical_stream_converts_sse_and_releases_iterator(monkeypatch):
    api = make_rest_api()
    app = make_app(api)
    calls = {"decrease": 0}

    class FakeModel:
        uid = "physical-model"

        async def chat(self, messages, kwargs, raw_params=None):
            calls["kwargs"] = kwargs
            calls["raw_params"] = raw_params

            async def chunks():
                yield (
                    'data: {"choices":[{"delta":{"content":"one"},'
                    '"finish_reason":null}]}\n\n'
                    'data: {"choices":[{"delta":{"content":" two"},'
                    '"finish_reason":"stop"}]}\n\n'
                )
                yield (
                    'data: {"choices":[],"usage":{"prompt_tokens":6,'
                    '"completion_tokens":2}}\n\n'
                )
                yield "data: [DONE]\n\n"

            return chunks()

        async def decrease_serve_count(self):
            calls["decrease"] += 1

    async def fake_require_model(*args, **kwargs):
        return FakeModel()

    monkeypatch.setattr(restful_api_module, "require_model", fake_require_model)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/anthropic/v1/messages",
            json={
                "model": "physical-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert calls["kwargs"] == {
        "max_tokens": 32,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    assert calls["raw_params"] == calls["kwargs"]
    assert calls["decrease"] == 1
    assert '"text": "one"' in response.text
    assert '"text": " two"' in response.text
    assert '"output_tokens": 2' in response.text
    assert response.text.count("event: message_stop") == 1


@pytest.mark.asyncio
async def test_missing_physical_model_returns_anthropic_not_found(monkeypatch):
    app = make_app(make_rest_api())

    async def fake_require_model(*args, **kwargs):
        raise HTTPException(status_code=404, detail="model not found")

    monkeypatch.setattr(restful_api_module, "require_model", fake_require_model)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/anthropic/v1/messages",
            json={
                "model": "missing-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
            },
        )

    assert response.status_code == 404
    assert response.json()["error"] == {
        "type": "not_found_error",
        "message": "model not found",
    }
