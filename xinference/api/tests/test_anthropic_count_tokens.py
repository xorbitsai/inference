import json

import httpx
import pytest

from xinference.api import restful_api as module

from .test_anthropic_messages import make_app, make_rest_api


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ["/v1/messages/count_tokens", "/anthropic/v1/messages/count_tokens"]
)
async def test_count_tokens_uses_backend_and_model_access(monkeypatch, path):
    api = make_rest_api()
    seen = []

    class Model:
        async def prepare_anthropic_request(self, messages, config):
            seen.append((messages, config))
            return json.dumps(
                {"messages": messages, "generate_config": config, "input_tokens": 23}
            ).encode()

        async def count_tokens(self, messages, config):
            pytest.fail("Preparation already counted the prompt")

    async def require_model(*args):
        return Model()

    monkeypatch.setattr(module, "require_model", require_model)
    app = make_app(api)
    app.add_api_route(path, api.count_anthropic_tokens, methods=["POST"])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            path,
            headers={"anthropic-version": "2023-06-01"},
            json={
                "model": "local",
                "system": "system",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
            },
        )
    assert response.status_code == 200
    assert response.json() == {"input_tokens": 23}
    assert seen[0][0][0] == {"role": "system", "content": "system"}
    assert seen[0][1]["tools"][0]["function"]["name"] == "Read"


@pytest.mark.asyncio
async def test_anthropic_endpoints_use_pd_decode_preparation(monkeypatch):
    from unittest.mock import AsyncMock

    from xinference.core.pd_model import PDModelActor

    actor = PDModelActor("local")
    decode = AsyncMock()
    decode.prepare_anthropic_request.return_value = (
        b'{"messages":[],"generate_config":{},"input_tokens":23}'
    )
    await actor.add_decode_actor("d", decode)
    actor._infer = AsyncMock(
        return_value=b'{"choices":[{"message":{"content":"answer"},"finish_reason":"stop"}],"usage":{}}'
    )
    monkeypatch.setattr(module, "require_model", AsyncMock(return_value=actor))
    api = make_rest_api()
    app = make_app(api)
    app.add_api_route(
        "/v1/messages/count_tokens", api.count_anthropic_tokens, methods=["POST"]
    )
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 32,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for path in ("/v1/messages", "/v1/messages/count_tokens"):
            response = await client.post(
                path, json=payload, headers={"anthropic-version": "2023-06-01"}
            )
            assert response.status_code == 200
            if path.endswith("count_tokens"):
                assert response.json() == {"input_tokens": 23}
            else:
                assert response.json()["content"] == [
                    {"type": "text", "text": "answer"}
                ]
    assert decode.prepare_anthropic_request.await_count == 2
    decode.count_tokens.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_prepared_count_preserves_unsupported_error(monkeypatch):
    from unittest.mock import AsyncMock

    model = AsyncMock()
    model.prepare_anthropic_request.return_value = {
        "messages": [],
        "generate_config": {},
        "input_tokens": 0,
    }
    model.count_tokens.side_effect = NotImplementedError("Tokenizer unavailable")
    monkeypatch.setattr(module, "require_model", AsyncMock(return_value=model))
    api = make_rest_api()
    app = make_app(api)
    app.add_api_route(
        "/v1/messages/count_tokens", api.count_anthropic_tokens, methods=["POST"]
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/messages/count_tokens",
            headers={"anthropic-version": "2023-06-01"},
            json={"model": "local", "messages": [{"role": "user", "content": "hello"}]},
        )
    assert response.status_code == 501
    assert response.json()["error"]["message"] == "Tokenizer unavailable"


@pytest.mark.asyncio
async def test_count_tokens_rejects_missing_version_and_unsupported_engine(monkeypatch):
    api = make_rest_api()

    class Model:
        async def prepare_anthropic_request(self, *args):
            raise NotImplementedError("Tokenizer unavailable")

    async def require_model(*args):
        return Model()

    monkeypatch.setattr(module, "require_model", require_model)
    app = make_app(api)
    app.add_api_route(
        "/v1/messages/count_tokens", api.count_anthropic_tokens, methods=["POST"]
    )
    payload = {"model": "local", "messages": [{"role": "user", "content": "hello"}]}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        missing_version = await client.post("/v1/messages/count_tokens", json=payload)
        unsupported = await client.post(
            "/v1/messages/count_tokens",
            json=payload,
            headers={"anthropic-version": "2023-06-01"},
        )
    assert missing_version.status_code == 400
    assert unsupported.status_code == 501
    assert unsupported.json()["error"]["message"] == "Tokenizer unavailable"


@pytest.mark.asyncio
async def test_count_tokens_preserves_rate_limit_errors(monkeypatch):
    api = make_rest_api()

    class Model:
        async def prepare_anthropic_request(self, *args):
            raise RuntimeError("Rate limit reached")

    async def require_model(*args):
        return Model()

    monkeypatch.setattr(module, "require_model", require_model)
    app = make_app(api)
    app.add_api_route(
        "/v1/messages/count_tokens", api.count_anthropic_tokens, methods=["POST"]
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/messages/count_tokens",
            headers={"anthropic-version": "2023-06-01"},
            json={"model": "local", "messages": [{"role": "user", "content": "hello"}]},
        )
    assert response.status_code == 429
