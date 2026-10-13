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
            return b"23"

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
