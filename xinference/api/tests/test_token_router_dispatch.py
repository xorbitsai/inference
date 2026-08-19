import json
from types import MethodType

import httpx
import pytest
from fastapi import FastAPI

from xinference.api import restful_api as restful_api_module


@pytest.fixture(autouse=True)
def enable_token_router(monkeypatch):
    monkeypatch.setattr(restful_api_module, "XINFERENCE_TOKEN_ROUTER_ENABLED", True)


from xinference.api.restful_api import RESTfulAPI
from xinference.core.router_config_store import RouterConfigStore
from xinference.core.router_registry import RouterRuntimeRegistry
from xinference.core.supervisor import SupervisorActor


def make_supervisor(tmp_path):
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._token_router_store = RouterConfigStore(str(tmp_path / "routers.db"))
    supervisor._token_router_registry = RouterRuntimeRegistry()
    supervisor._token_router_runtime_cursors = {}
    return supervisor


def create_enabled_router(supervisor):
    config = supervisor._token_router_store.create(
        "router-a",
        {
            "virtual_model_uid": "virtual-model",
            "model_type": "LLM",
        },
    )
    assert config["revision"] == 1
    return supervisor._token_router_store.set_enabled("router-a", True)


def register_ready_runtime(supervisor, instance_id, endpoint, revision):
    return supervisor._token_router_registry.register(
        "router-a",
        instance_id,
        {
            "endpoint": endpoint,
            "status": "ready",
            "acked_revision": revision,
            "config_error": "",
        },
    )


@pytest.mark.asyncio
async def test_resolve_virtual_model_runtime_and_round_robin(tmp_path):
    supervisor = make_supervisor(tmp_path)
    assert await supervisor.resolve_token_router_runtime("physical-model") is None

    disabled = supervisor._token_router_store.create(
        "router-a", {"virtual_model_uid": "virtual-model"}
    )
    resolution = await supervisor.resolve_token_router_runtime("virtual-model")
    assert resolution["matched"] is True
    assert resolution["available"] is False
    assert resolution["status"] == "disabled"

    config = supervisor._token_router_store.set_enabled("router-a", True)
    register_ready_runtime(
        supervisor, "instance-a", "http://router-a:10081/", config["revision"]
    )
    register_ready_runtime(
        supervisor, "instance-b", "http://router-b:10081", config["revision"]
    )

    first = await supervisor.resolve_token_router_runtime("virtual-model")
    second = await supervisor.resolve_token_router_runtime("virtual-model")
    third = await supervisor.resolve_token_router_runtime("virtual-model")
    assert [first["instance_id"], second["instance_id"], third["instance_id"]] == [
        "instance-a",
        "instance-b",
        "instance-a",
    ]
    assert first["endpoint"] == "http://router-a:10081"
    assert disabled["revision"] < first["revision"]


@pytest.mark.asyncio
async def test_resolve_rejects_stale_error_and_invalid_runtime(tmp_path):
    supervisor = make_supervisor(tmp_path)
    config = create_enabled_router(supervisor)
    revision = config["revision"]
    supervisor._token_router_registry.register(
        "router-a",
        "stale",
        {
            "endpoint": "http://stale:10081",
            "status": "ready",
            "acked_revision": revision - 1,
        },
    )
    supervisor._token_router_registry.register(
        "router-a",
        "error",
        {
            "endpoint": "http://error:10081",
            "status": "ready",
            "acked_revision": revision,
            "config_error": "bad config",
        },
    )
    supervisor._token_router_registry.register(
        "router-a",
        "invalid-endpoint",
        {
            "endpoint": "router:10081",
            "status": "ready",
            "acked_revision": revision,
        },
    )

    resolution = await supervisor.resolve_token_router_runtime("virtual-model")
    assert resolution["matched"] is True
    assert resolution["available"] is False


@pytest.mark.asyncio
async def test_resolve_rejects_offline_and_non_ready_runtime(tmp_path, monkeypatch):
    now = [100.0]
    monkeypatch.setattr("xinference.core.router_registry.time.time", lambda: now[0])
    supervisor = make_supervisor(tmp_path)
    supervisor._token_router_registry = RouterRuntimeRegistry(
        heartbeat_timeout_seconds=10
    )
    config = create_enabled_router(supervisor)
    supervisor._token_router_registry.register(
        "router-a",
        "starting",
        {
            "endpoint": "http://starting:10081",
            "status": "starting",
            "acked_revision": config["revision"],
        },
    )
    resolution = await supervisor.resolve_token_router_runtime("virtual-model")
    assert resolution["available"] is False

    supervisor._token_router_registry.heartbeat("starting", {"status": "ready"})
    now[0] = 111.0
    resolution = await supervisor.resolve_token_router_runtime("virtual-model")
    assert resolution["available"] is False
    assert resolution["status"] == "offline"


@pytest.mark.asyncio
async def test_list_virtual_models_only_exposes_enabled_sanitized_models(tmp_path):
    supervisor = make_supervisor(tmp_path)
    supervisor._token_router_store.create(
        "disabled-router", {"virtual_model_uid": "disabled-model"}
    )
    config = create_enabled_router(supervisor)
    register_ready_runtime(
        supervisor, "instance-a", "http://secret-router:10081", config["revision"]
    )

    models = await supervisor.list_virtual_models()
    assert list(models) == ["virtual-model"]
    assert models["virtual-model"] == {
        "model_name": "virtual-model",
        "model_type": "LLM",
        "model_engine": "token_router",
        "router_uid": "router-a",
        "router_status": "ready",
    }
    assert "endpoint" not in json.dumps(models)


class FakeSupervisor:
    def __init__(self, resolution, *, physical_models=None, virtual_models=None):
        self.resolution = resolution
        self.physical_models = physical_models or {}
        self.virtual_models = virtual_models or {}
        self.resolved_model_uids = []

    async def resolve_token_router_runtime(self, model_uid):
        self.resolved_model_uids.append(model_uid)
        return self.resolution

    async def list_models(self):
        return dict(self.physical_models)

    async def list_virtual_models(self):
        return dict(self.virtual_models)


class FakeAuthService:
    def __init__(self):
        self.checked = []

    def validate_model_access(self, token, model_uid, model_type):
        self.checked.append((token, model_uid, model_type))
        return True


def make_rest_api(supervisor, auth_service=None):
    api = RESTfulAPI.__new__(RESTfulAPI)
    api._advanced_auth_service = auth_service
    api._uid_to_model_name = {}

    async def get_supervisor_ref(_self):
        return supervisor

    api._get_supervisor_ref = MethodType(get_supervisor_ref, api)
    return api


def make_chat_app(api):
    app = FastAPI()
    app.add_api_route(
        "/v1/chat/completions", api.create_chat_completion, methods=["POST"]
    )
    return app


@pytest.mark.asyncio
async def test_virtual_chat_stream_is_proxied_with_auth_headers_and_done(monkeypatch):
    resolution = {
        "matched": True,
        "available": True,
        "virtual_model_uid": "virtual-model",
        "router_uid": "router-a",
        "instance_id": "instance-a",
        "endpoint": "http://router:10081",
    }
    supervisor = FakeSupervisor(resolution)
    auth_service = FakeAuthService()
    api = make_rest_api(supervisor, auth_service)
    app = make_chat_app(api)
    real_async_client = httpx.AsyncClient
    captured = {}

    class UpstreamStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices": []}\n\n'
            yield b"data: [DONE]\n\n"

        async def aclose(self):
            captured["stream_closed"] = True

    def upstream_handler(request):
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["traceparent"] = request.headers.get("traceparent")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "x-router-pool": "short",
                "transfer-encoding": "chunked",
            },
            stream=UpstreamStream(),
        )

    upstream_client = real_async_client(transport=httpx.MockTransport(upstream_handler))
    monkeypatch.setattr(
        restful_api_module.httpx, "AsyncClient", lambda **_: upstream_client
    )

    transport = httpx.ASGITransport(app=app)
    async with real_async_client(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer virtual-key",
                "traceparent": "00-test-trace",
                "x-request-id": "request-a",
            },
            json={
                "model": "virtual-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["x-router-pool"] == "short"
    assert response.headers["x-accel-buffering"] == "no"
    assert "transfer-encoding" not in response.headers
    assert response.content.endswith(b"data: [DONE]\n\n")
    assert captured == {
        "url": "http://router:10081/v1/chat/completions",
        "authorization": "Bearer virtual-key",
        "traceparent": "00-test-trace",
        "payload": {
            "model": "virtual-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
        "stream_closed": True,
    }
    assert auth_service.checked == [("virtual-key", "virtual-model", "LLM")]


@pytest.mark.asyncio
async def test_virtual_chat_non_stream_preserves_router_error(monkeypatch):
    resolution = {
        "matched": True,
        "available": True,
        "virtual_model_uid": "virtual-model",
        "router_uid": "router-a",
        "instance_id": "instance-a",
        "endpoint": "http://router:10081",
    }
    api = make_rest_api(FakeSupervisor(resolution))
    app = make_chat_app(api)
    real_async_client = httpx.AsyncClient

    class ErrorStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"error":{"type":"rate_limit_error"}}'

    def upstream_handler(request):
        return httpx.Response(
            429,
            headers={
                "content-type": "application/json",
                "retry-after": "5",
                "connection": "close",
            },
            stream=ErrorStream(),
        )

    upstream_client = real_async_client(transport=httpx.MockTransport(upstream_handler))
    monkeypatch.setattr(
        restful_api_module.httpx, "AsyncClient", lambda **_: upstream_client
    )
    transport = httpx.ASGITransport(app=app)
    async with real_async_client(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "virtual-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "5"
    assert "connection" not in response.headers
    assert response.json()["error"]["type"] == "rate_limit_error"


@pytest.mark.asyncio
async def test_virtual_chat_connect_error_returns_502_without_closing_shared_client(
    monkeypatch,
):
    resolution = {
        "matched": True,
        "available": True,
        "virtual_model_uid": "virtual-model",
        "router_uid": "router-a",
        "instance_id": "instance-a",
        "endpoint": "http://router:10081",
    }
    api = make_rest_api(FakeSupervisor(resolution))
    app = make_chat_app(api)
    real_async_client = httpx.AsyncClient

    def upstream_handler(request):
        raise httpx.ConnectError("router unavailable", request=request)

    upstream_client = real_async_client(transport=httpx.MockTransport(upstream_handler))
    monkeypatch.setattr(
        restful_api_module.httpx, "AsyncClient", lambda **_: upstream_client
    )
    transport = httpx.ASGITransport(app=app)
    async with real_async_client(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "virtual-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 502
    assert response.headers["retry-after"] == "1"
    assert response.json()["error"]["type"] == "router_unavailable"
    assert upstream_client.is_closed is False
    await api._close_token_router_client()
    assert upstream_client.is_closed is True


@pytest.mark.asyncio
async def test_virtual_chat_reuses_shared_proxy_client(monkeypatch):
    resolution = {
        "matched": True,
        "available": True,
        "virtual_model_uid": "virtual-model",
        "router_uid": "router-a",
        "instance_id": "instance-a",
        "endpoint": "http://router:10081",
    }
    api = make_rest_api(FakeSupervisor(resolution))
    app = make_chat_app(api)
    real_async_client = httpx.AsyncClient
    created_clients = []

    class UpstreamStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"id":"chatcmpl-router"}'

    def upstream_handler(_request):
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=UpstreamStream(),
        )

    upstream_client = real_async_client(transport=httpx.MockTransport(upstream_handler))

    def make_upstream_client(**_kwargs):
        created_clients.append(upstream_client)
        return upstream_client

    monkeypatch.setattr(restful_api_module.httpx, "AsyncClient", make_upstream_client)
    transport = httpx.ASGITransport(app=app)
    async with real_async_client(transport=transport, base_url="http://test") as client:
        for _ in range(2):
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "virtual-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
            )
            assert response.status_code == 200

    assert created_clients == [upstream_client]
    assert upstream_client.is_closed is False
    await api._close_token_router_client()
    assert upstream_client.is_closed is True


@pytest.mark.asyncio
async def test_virtual_chat_unavailable_returns_503_without_proxy(monkeypatch):
    supervisor = FakeSupervisor(
        {
            "matched": True,
            "available": False,
            "virtual_model_uid": "virtual-model",
            "router_uid": "router-a",
            "status": "offline",
        }
    )
    api = make_rest_api(supervisor)
    app = make_chat_app(api)

    def fail_client(**kwargs):
        raise AssertionError("proxy client must not be created")

    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(restful_api_module.httpx, "AsyncClient", fail_client)
    transport = httpx.ASGITransport(app=app)
    async with real_async_client(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "virtual-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )

    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
    assert response.json()["error"]["type"] == "router_unavailable"


@pytest.mark.asyncio
async def test_physical_chat_keeps_existing_path(monkeypatch):
    supervisor = FakeSupervisor(None)
    api = make_rest_api(supervisor)
    app = make_chat_app(api)

    class FakeModel:
        uid = "physical-model"

        async def chat(self, messages, kwargs, raw_params=None):
            return json.dumps(
                {
                    "id": "chatcmpl-physical",
                    "object": "chat.completion",
                    "choices": [],
                }
            )

        async def is_vllm_backend(self):
            return False

    async def fake_require_model(*args, **kwargs):
        return FakeModel()

    async def describe_model(model_uid):
        return {"model_family": "test-family"}

    supervisor.describe_model = describe_model
    monkeypatch.setattr(restful_api_module, "require_model", fake_require_model)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "physical-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )

    assert response.status_code == 200
    assert response.json()["id"] == "chatcmpl-physical"
    assert supervisor.resolved_model_uids == ["physical-model"]


@pytest.mark.asyncio
async def test_list_models_merges_sanitized_virtual_model(monkeypatch):
    supervisor = FakeSupervisor(
        None,
        physical_models={
            "physical-model": {"model_name": "physical", "model_type": "LLM"}
        },
        virtual_models={
            "virtual-model": {
                "model_name": "virtual-model",
                "model_type": "LLM",
                "model_engine": "token_router",
                "router_uid": "router-a",
                "router_status": "ready",
            }
        },
    )
    api = make_rest_api(supervisor)
    monkeypatch.setattr(
        "xinference.api.oauth2.advanced.audit.update_model_cache",
        lambda *args, **kwargs: None,
    )

    response = await api.list_models()
    payload = json.loads(response.body)
    by_id = {item["id"]: item for item in payload["data"]}
    assert set(by_id) == {"physical-model", "virtual-model"}
    assert by_id["virtual-model"]["model_engine"] == "token_router"
    assert by_id["virtual-model"]["router_status"] == "ready"
    assert "endpoint" not in by_id["virtual-model"]
