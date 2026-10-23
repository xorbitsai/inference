# CI: linux

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

    class FakeSupervisor:
        async def resolve_token_router_runtime(self, model_uid):
            return None

    async def get_supervisor_ref(_self):
        return FakeSupervisor()

    api._get_supervisor_ref = MethodType(get_supervisor_ref, api)
    return api


def make_app(api):
    app = FastAPI()
    app.add_api_route("/v1/messages", api.create_message, methods=["POST"])
    app.add_api_route("/anthropic/v1/messages", api.create_message, methods=["POST"])
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["/v1/messages", "/v1/messages/count_tokens"])
@pytest.mark.parametrize(
    "context_management",
    [
        {},
        {"edits": [{"type": "clear_thinking_20251015"}]},
        {"edits": [{"type": "clear_thinking_20251015", "keep": {"type": "all"}}]},
        {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
    ],
)
async def test_context_management_optional_shapes_are_accepted(
    monkeypatch, endpoint, context_management
):
    api = make_rest_api()
    app = make_app(api)
    app.add_api_route(
        "/v1/messages/count_tokens", api.count_anthropic_tokens, methods=["POST"]
    )

    class Model:
        async def prepare_anthropic_request(self, messages, config):
            return {
                "messages": messages,
                "generate_config": config,
                "input_tokens": 3,
                "cache_control_applied": False,
            }

        async def chat(self, *args, **kwargs):
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            }

    async def require_model(*args, **kwargs):
        return Model()

    monkeypatch.setattr(restful_api_module, "require_model", require_model)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            endpoint,
            headers={"anthropic-version": "2023-06-01"},
            json={
                "model": "local",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
                "context_management": context_management,
            },
        )

    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,status",
    [
        (NotImplementedError("Forced tool choice is not supported"), 501),
        (RuntimeError("Rate limit reached"), 429),
    ],
)
async def test_backend_capability_error_is_reported_before_generation(
    monkeypatch, error, status
):
    class Model:
        async def prepare_anthropic_request(self, messages, config):
            raise error

        async def chat(self, *args, **kwargs):
            pytest.fail("An unsupported request must not start generation")

    async def require_model(*args):
        return Model()

    monkeypatch.setattr(restful_api_module, "require_model", require_model)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=make_app(make_rest_api())),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/messages",
            headers={"anthropic-version": "2023-06-01"},
            json={
                "model": "local",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
                "tool_choice": {"type": "any"},
            },
        )
    assert response.status_code == status
    assert response.json()["error"]["message"] == str(error)


@pytest.mark.asyncio
@pytest.mark.parametrize("root_path", ["", "/xinference", "/anthropic"])
async def test_standard_path_requires_version_header(root_path):
    app = make_app(make_rest_api())
    transport = httpx.ASGITransport(app=app, root_path=root_path)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"{root_path}/v1/messages",
            json={
                "model": "model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
            },
        )

    assert response.status_code == 400
    assert response.headers["request-id"].startswith("req_")
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
@pytest.mark.parametrize("root_path", ["", "/anthropic"])
async def test_physical_non_stream_converts_bytes_response(monkeypatch, root_path):
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
    transport = httpx.ASGITransport(app=app, root_path=root_path)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"{root_path}/anthropic/v1/messages",
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


@pytest.mark.asyncio
async def test_physical_stream_failure_is_reported_as_model_generator(monkeypatch):
    from xinference.api.streaming_outcome import (
        FailureOrigin,
        StreamingOutcomeReporter,
        StreamState,
    )

    api = make_rest_api()
    app = make_app(api)
    reporter = StreamingOutcomeReporter()

    class FakeModel:
        uid = "physical-model"

        async def chat(self, messages, kwargs, raw_params=None):
            async def chunks():
                yield 'data: {"choices":[{"delta":{"content":"one"}}]}\n\n'
                raise RuntimeError("physical backend failed")

            return chunks()

        async def decrease_serve_count(self):
            return None

    async def fake_require_model(*args, **kwargs):
        return FakeModel()

    monkeypatch.setattr(restful_api_module, "require_model", fake_require_model)
    monkeypatch.setattr(
        restful_api_module, "get_stream_outcome_reporter", lambda request: reporter
    )
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
    assert "event: error" in response.text
    assert "physical backend failed" in response.text
    assert reporter.outcome.state is StreamState.FAILED
    assert reporter.outcome.failure_origin is FailureOrigin.MODEL_GENERATOR


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["/v1/messages", "/v1/messages/count_tokens"])
async def test_model_access_is_checked_before_media_normalization(
    monkeypatch, endpoint
):
    api = make_rest_api()

    def denied(*args):
        raise HTTPException(403, "Model access denied")

    api._check_model_access = denied

    def parse(*args):
        pytest.fail("Denied requests must not fetch or render media")

    monkeypatch.setattr(restful_api_module, "parse_anthropic_request", parse)
    app = make_app(api)
    app.add_api_route(
        "/v1/messages/count_tokens", api.count_anthropic_tokens, methods=["POST"]
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            endpoint,
            headers={"anthropic-version": "2023-06-01"},
            json={
                "model": "denied",
                "max_tokens": 32,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "url",
                                    "url": "https://example.com/file.pdf",
                                },
                            }
                        ],
                    }
                ],
            },
        )
    assert response.status_code == 403
    assert response.json()["error"]["message"] == "Model access denied"


@pytest.mark.asyncio
async def test_anthropic_get_model_returns_404_for_unknown_model():
    api = RESTfulAPI.__new__(RESTfulAPI)

    class FakeSupervisor:
        async def list_models(self):
            return {"qwen": {"model_name": "qwen3", "model_type": "LLM"}}

    async def get_supervisor_ref(_self):
        return FakeSupervisor()

    api._get_supervisor_ref = MethodType(get_supervisor_ref, api)

    with pytest.raises(HTTPException) as exc_info:
        await api.anthropic_get_model("missing")
    assert exc_info.value.status_code == 404

    response = await api.anthropic_get_model("qwen")
    assert json.loads(response.body)["display_name"] == "qwen3"
