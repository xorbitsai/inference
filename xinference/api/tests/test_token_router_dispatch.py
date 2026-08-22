import json
from datetime import datetime, timedelta, timezone
from types import MethodType

import httpx
import pytest
from fastapi import FastAPI, Request

from xinference.api import restful_api as restful_api_module
from xinference.api.restful_api import RESTfulAPI
from xinference.core.router_config_store import RouterConfigStore
from xinference.core.router_orchestration import RouterOrchestrationController
from xinference.core.router_registry import RouterRuntimeRegistry
from xinference.core.supervisor import SupervisorActor


@pytest.mark.asyncio
async def test_restful_api_lifespan_closes_token_router_client():
    api = RESTfulAPI.__new__(RESTfulAPI)
    client = httpx.AsyncClient()
    api._token_router_client = client
    app = FastAPI(lifespan=api._lifespan)

    async with app.router.lifespan_context(app):
        assert client.is_closed is False

    assert client.is_closed is True
    assert api._token_router_client is None


def make_supervisor(tmp_path):
    supervisor = SupervisorActor.__new__(SupervisorActor)
    db_path = str(tmp_path / "routers.db")
    supervisor._token_router_store = RouterConfigStore(db_path)
    supervisor._token_router_registry = RouterRuntimeRegistry()
    supervisor._token_router_orchestration = RouterOrchestrationController(
        db_path, supervisor._token_router_store
    )
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


@pytest.mark.parametrize("revision", [None, "invalid"])
def test_runtime_health_rejects_invalid_config_revision(tmp_path, revision):
    supervisor = make_supervisor(tmp_path)
    register_ready_runtime(supervisor, "instance-a", "http://router-a:10081", 1)
    config = {
        "router_uid": "router-a",
        "virtual_model_uid": "virtual-model",
        "enabled": True,
        "revision": revision,
    }

    instances, effective, controllable = supervisor._token_router_runtime_health(config)
    model = supervisor._build_token_router_model_info(config)

    assert len(instances) == 1
    assert effective == []
    assert controllable == []
    assert model["router_status"] == "syncing"
    assert model["ready_instances"] == 0


def test_runtime_health_missing_router_uid_does_not_list_other_instances(tmp_path):
    supervisor = make_supervisor(tmp_path)
    register_ready_runtime(supervisor, "instance-a", "http://router-a:10081", 1)

    instances, effective, controllable = supervisor._token_router_runtime_health(
        {"revision": 1}
    )

    assert instances == []
    assert effective == []
    assert controllable == []


def test_build_virtual_model_info_tolerates_missing_uids(tmp_path):
    supervisor = make_supervisor(tmp_path)

    model = supervisor._build_token_router_model_info({"enabled": False, "revision": 1})

    assert model["model_name"] == ""
    assert model["router_uid"] == ""
    assert model["router_status"] == "disabled"
    assert model["runtime_instances"] == 0


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
    assert resolution["status"] == "unavailable"


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
    model = models["virtual-model"]
    assert model["model_name"] == "virtual-model"
    assert model["model_type"] == "LLM"
    assert model["model_engine"] == "token_router"
    assert model["model_ability"] == ["chat"]
    assert model["model_kind"] == "virtual"
    assert model["virtual_model_type"] == "token_router"
    assert model["router_uid"] == "router-a"
    assert model["router_status"] == "ready"
    assert model["runtime_instances"] == 1
    assert model["online_instances"] == 1
    assert model["ready_instances"] == 1
    assert model["backend_count"] == 0
    assert "endpoint" not in json.dumps(models)
    assert "api_key" not in json.dumps(models)


@pytest.mark.asyncio
async def test_describe_running_model_supports_virtual_and_physical_models(tmp_path):
    supervisor = make_supervisor(tmp_path)
    config = create_enabled_router(supervisor)
    register_ready_runtime(
        supervisor, "instance-a", "http://secret-router:10081", config["revision"]
    )

    async def describe_physical(_self, model_uid):
        return {"model_name": model_uid, "model_type": "LLM", "replica": 1}

    supervisor.describe_model = MethodType(describe_physical, supervisor)

    virtual = await supervisor.describe_running_model("virtual-model")
    physical = await supervisor.describe_running_model("physical-model")

    assert virtual["model_kind"] == "virtual"
    assert virtual["model_ability"] == ["chat"]
    assert virtual["router_uid"] == "router-a"
    assert "endpoint" not in json.dumps(virtual)
    assert physical == {
        "model_name": "physical-model",
        "model_type": "LLM",
        "replica": 1,
    }


@pytest.mark.asyncio
async def test_describe_running_model_honors_disabled_feature_gate(
    tmp_path, monkeypatch
):
    supervisor = make_supervisor(tmp_path)
    create_enabled_router(supervisor)
    fallback_calls = []

    async def describe_physical(_self, model_uid):
        fallback_calls.append(model_uid)
        return {"model_name": model_uid, "model_kind": "physical"}

    supervisor.describe_model = MethodType(describe_physical, supervisor)
    monkeypatch.setattr(
        "xinference.core.supervisor.XINFERENCE_TOKEN_ROUTER_ENABLED", False
    )

    model = await supervisor.describe_running_model("virtual-model")

    assert model == {"model_name": "virtual-model", "model_kind": "physical"}
    assert fallback_calls == ["virtual-model"]


@pytest.mark.asyncio
async def test_managed_runtime_remains_available_but_degraded_when_agent_is_offline(
    tmp_path,
):
    supervisor = make_supervisor(tmp_path)
    config = create_enabled_router(supervisor)
    orchestration = supervisor._token_router_orchestration
    orchestration.register_node(
        {
            "node_id": "node-a",
            "advertise_host": "127.0.0.1",
            "port_range_start": 12080,
            "port_range_end": 12089,
            "max_instances": 5,
        }
    )
    orchestration.update_deployment(
        "router-a", {"management_mode": "managed", "desired_replicas": 1}
    )
    orchestration.router_enabled("router-a", True)
    assignment = orchestration.list_assignments(router_uid="router-a")[0]
    supervisor._token_router_registry.register(
        "router-a",
        "instance-a",
        {
            "endpoint": assignment["public_endpoint"],
            "status": "ready",
            "acked_revision": config["revision"],
            "config_error": "",
            "assignment_id": assignment["assignment_id"],
            "assignment_generation": assignment["assignment_generation"],
            "node_id": "node-a",
        },
    )

    last_seen = (datetime.now(timezone.utc) - timedelta(seconds=46)).isoformat()
    with orchestration.nodes._connect() as conn:
        conn.execute(
            "UPDATE token_router_nodes SET last_seen_at = ? WHERE node_id = ?",
            (last_seen, "node-a"),
        )
    orchestration.sweep_nodes()

    resolution = await supervisor.resolve_token_router_runtime("virtual-model")
    status = supervisor._with_token_router_status(config)

    assert resolution["available"] is True
    assert resolution["status"] == "degraded"
    assert status["deployment"]["observed_ready_assignments"] == 0
    assert status["deployment"]["effective_ready_runtimes"] == 1
    assert status["deployment"]["controllable_ready_runtimes"] == 0
    assert status["deployment"]["ready_replicas"] == 1

    supervisor._token_router_registry.unregister("instance-a")
    unavailable = supervisor._with_token_router_status(config)
    assert unavailable["status"] == "unavailable"
    assert unavailable["deployment"]["ready_replicas"] == 0


@pytest.mark.asyncio
async def test_managed_stopped_deployment_is_disabled_and_not_dispatched(tmp_path):
    supervisor = make_supervisor(tmp_path)
    config = create_enabled_router(supervisor)
    orchestration = supervisor._token_router_orchestration
    orchestration.register_node(
        {
            "node_id": "node-a",
            "advertise_host": "127.0.0.1",
            "port_range_start": 12080,
            "port_range_end": 12089,
            "max_instances": 5,
        }
    )
    orchestration.update_deployment(
        "router-a", {"management_mode": "managed", "desired_replicas": 1}
    )
    orchestration.router_enabled("router-a", True)
    assignment = orchestration.list_assignments(router_uid="router-a")[0]
    supervisor._token_router_registry.register(
        "router-a",
        "instance-a",
        {
            "endpoint": assignment["public_endpoint"],
            "status": "ready",
            "acked_revision": config["revision"],
            "config_error": "",
            "assignment_id": assignment["assignment_id"],
            "assignment_generation": assignment["assignment_generation"],
            "node_id": "node-a",
        },
    )

    orchestration.update_deployment("router-a", {"desired_state": "stopped"})
    resolution = await supervisor.resolve_token_router_runtime("virtual-model")
    status = supervisor._with_token_router_status(config)

    assert status["status"] == "disabled"
    assert resolution["status"] == "disabled"
    assert resolution["available"] is False


class FakeSupervisor:
    def __init__(
        self,
        resolution,
        *,
        physical_models=None,
        virtual_models=None,
        running_model_details=None,
    ):
        self.resolution = resolution
        self.physical_models = physical_models or {}
        self.virtual_models = virtual_models or {}
        self.running_model_details = running_model_details or {}
        self.resolved_model_uids = []

    async def resolve_token_router_runtime(self, model_uid):
        self.resolved_model_uids.append(model_uid)
        return self.resolution

    async def list_models(self):
        return dict(self.physical_models)

    async def list_virtual_models(self):
        return dict(self.virtual_models)

    async def describe_running_model(self, model_uid):
        if model_uid not in self.running_model_details:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")
        return dict(self.running_model_details[model_uid])


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


def make_anthropic_app(api):
    app = FastAPI()
    app.add_api_route("/v1/messages", api.create_message, methods=["POST"])
    app.add_api_route("/anthropic/v1/messages", api.create_message, methods=["POST"])
    return app


@pytest.mark.asyncio
async def test_describe_virtual_model_returns_public_router_metadata():
    detail = {
        "model_name": "virtual-model",
        "model_type": "LLM",
        "model_engine": "token_router",
        "model_ability": ["chat"],
        "model_kind": "virtual",
        "virtual_model_type": "token_router",
        "router_uid": "router-a",
        "router_status": "ready",
    }
    supervisor = FakeSupervisor(None, running_model_details={"virtual-model": detail})
    api = make_rest_api(supervisor)

    response = await api.describe_model("virtual-model")

    assert response.status_code == 200
    assert json.loads(response.body) == detail


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
@pytest.mark.parametrize(
    "thinking_fields",
    [
        {"enable_thinking": True, "extra_body": {"enable_thinking": False}},
        {"extra_body": {"enable_thinking": True}},
    ],
)
async def test_virtual_chat_normalizes_router_estimation_fields(
    monkeypatch, thinking_fields
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
    captured = {}

    class UpstreamStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"id":"chatcmpl-router"}'

    def upstream_handler(request):
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=UpstreamStream(),
        )

    upstream_client = real_async_client(transport=httpx.MockTransport(upstream_handler))
    monkeypatch.setattr(
        restful_api_module.httpx, "AsyncClient", lambda **_: upstream_client
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with real_async_client(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "virtual-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                    "max_tokens": 8,
                    "max_completion_tokens": 16,
                    **thinking_fields,
                },
            )
    finally:
        await api._close_token_router_client()

    assert response.status_code == 200
    assert captured["payload"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "thinking": True,
    }
    assert captured["payload"]["max_tokens"] == 16


@pytest.mark.asyncio
async def test_stream_proxy_background_closes_uniterated_response_once():
    api = make_rest_api(FakeSupervisor(None))
    close_calls = 0

    class UpstreamStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: [DONE]\n\n"

        async def aclose(self):
            nonlocal close_calls
            close_calls += 1

    upstream_request = httpx.Request("POST", "http://router:10081/v1/chat/completions")
    upstream_response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=UpstreamStream(),
        request=upstream_request,
    )

    class FakeClient:
        def build_request(self, *args, **kwargs):
            return upstream_request

        async def send(self, request, *, stream):
            assert stream is True
            return upstream_response

    api._get_token_router_client = lambda: FakeClient()
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("test", 80),
            "client": ("127.0.0.1", 1234),
        }
    )

    response = await api._proxy_token_router_chat_completion(
        request,
        {"stream": True},
        {
            "endpoint": "http://router:10081",
            "virtual_model_uid": "virtual-model",
            "router_uid": "router-a",
            "instance_id": "instance-a",
        },
    )

    assert close_calls == 0
    assert response.background is not None
    await response.background()
    await response.background()
    assert close_calls == 1


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


@pytest.mark.asyncio
async def test_list_models_hides_persisted_virtual_models_when_router_disabled(
    monkeypatch,
):
    monkeypatch.setattr(restful_api_module, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
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
            }
        },
    )

    async def fail_list_virtual_models():
        raise AssertionError(
            "virtual models must not be loaded when Router is disabled"
        )

    supervisor.list_virtual_models = fail_list_virtual_models
    api = make_rest_api(supervisor)
    monkeypatch.setattr(
        "xinference.api.oauth2.advanced.audit.update_model_cache",
        lambda *args, **kwargs: None,
    )

    response = await api.list_models()
    payload = json.loads(response.body)

    assert [item["id"] for item in payload["data"]] == ["physical-model"]


@pytest.mark.asyncio
async def test_anthropic_virtual_non_stream_uses_openai_data_plane(monkeypatch):
    resolution = {
        "matched": True,
        "available": True,
        "virtual_model_uid": "virtual-model",
        "router_uid": "router-a",
        "instance_id": "instance-a",
        "endpoint": "http://router:10081",
    }
    auth_service = FakeAuthService()
    api = make_rest_api(FakeSupervisor(resolution), auth_service)
    app = make_anthropic_app(api)
    real_async_client = httpx.AsyncClient
    captured = {}
    monkeypatch.delenv("XINFERENCE_TOKEN_ROUTER_DATA_PLANE_TOKEN", raising=False)
    monkeypatch.delenv("XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN", raising=False)

    def upstream_handler(request):
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["x_api_key"] = request.headers.get("x-api-key")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "routed answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            },
        )

    upstream_client = real_async_client(transport=httpx.MockTransport(upstream_handler))
    monkeypatch.setattr(
        restful_api_module.httpx, "AsyncClient", lambda **_: upstream_client
    )
    transport = httpx.ASGITransport(app=app)
    async with real_async_client(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/messages",
            headers={
                "anthropic-version": "2023-06-01",
                "x-api-key": "external-anthropic-key",
                "x-request-id": "request-anthropic-a",
            },
            json={
                "model": "virtual-model",
                "system": "Be concise.",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 64,
                "stop_sequences": ["END"],
            },
        )

    assert response.status_code == 200
    assert response.headers["request-id"] == "request-anthropic-a"
    assert response.json()["model"] == "virtual-model"
    assert response.json()["stop_reason"] == "end_turn"
    assert response.json()["content"] == [{"type": "text", "text": "routed answer"}]
    assert captured == {
        "url": "http://router:10081/v1/chat/completions",
        "authorization": None,
        "x_api_key": None,
        "payload": {
            "model": "virtual-model",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"},
            ],
            "max_tokens": 64,
            "stream": False,
            "stop": ["END"],
        },
    }
    assert auth_service.checked == [("external-anthropic-key", "virtual-model", "LLM")]
    assert api._token_router_client is upstream_client
    assert not upstream_client.is_closed
    await api._close_token_router_client()
    assert upstream_client.is_closed


@pytest.mark.asyncio
async def test_anthropic_virtual_uses_internal_router_token(monkeypatch):
    resolution = {
        "matched": True,
        "available": True,
        "virtual_model_uid": "virtual-model",
        "router_uid": "router-a",
        "instance_id": "instance-a",
        "endpoint": "http://router:10081",
    }
    api = make_rest_api(FakeSupervisor(resolution))
    app = make_anthropic_app(api)
    real_async_client = httpx.AsyncClient
    captured = {}
    monkeypatch.setenv("XINFERENCE_TOKEN_ROUTER_DATA_PLANE_TOKEN", "internal-token")

    def upstream_handler(request):
        captured["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        )

    upstream_client = real_async_client(transport=httpx.MockTransport(upstream_handler))
    monkeypatch.setattr(
        restful_api_module.httpx, "AsyncClient", lambda **_: upstream_client
    )
    transport = httpx.ASGITransport(app=app)
    async with real_async_client(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/messages",
            headers={
                "anthropic-version": "2023-06-01",
                "Authorization": "Bearer external-token",
            },
            json={
                "model": "virtual-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
            },
        )

    assert response.status_code == 200
    assert captured["authorization"] == "Bearer internal-token"
    assert not upstream_client.is_closed
    await api._close_token_router_client()
    assert upstream_client.is_closed


@pytest.mark.asyncio
async def test_anthropic_stream_background_closes_uniterated_upstream_once():
    resolution = {
        "matched": True,
        "available": True,
        "virtual_model_uid": "virtual-model",
        "router_uid": "router-a",
        "instance_id": "instance-a",
        "endpoint": "http://router:10081",
    }
    api = make_rest_api(FakeSupervisor(resolution))
    close_calls = 0

    class UpstreamStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: [DONE]\n\n"

        async def aclose(self):
            nonlocal close_calls
            close_calls += 1

    upstream_request = httpx.Request("POST", "http://router:10081/v1/chat/completions")
    upstream_response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=UpstreamStream(),
        request=upstream_request,
    )

    class FakeClient:
        def build_request(self, *args, **kwargs):
            return upstream_request

        async def send(self, request, *, stream):
            assert stream is True
            return upstream_response

    api._get_token_router_client = lambda: FakeClient()
    body = json.dumps(
        {
            "model": "virtual-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32,
            "stream": True,
        }
    ).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages",
            "root_path": "",
            "headers": [(b"anthropic-version", b"2023-06-01")],
            "query_string": b"",
            "scheme": "http",
            "server": ("test", 80),
            "client": ("127.0.0.1", 1234),
        },
        receive,
    )

    response = await api.create_message(request)

    assert close_calls == 0
    assert response.background is not None
    await response.background()
    await response.background()
    assert close_calls == 1


@pytest.mark.asyncio
async def test_anthropic_virtual_stream_returns_native_event_sequence(monkeypatch):
    resolution = {
        "matched": True,
        "available": True,
        "virtual_model_uid": "virtual-model",
        "router_uid": "router-a",
        "instance_id": "instance-a",
        "endpoint": "http://router:10081",
    }
    api = make_rest_api(FakeSupervisor(resolution))
    app = make_anthropic_app(api)
    real_async_client = httpx.AsyncClient
    captured = {}

    class UpstreamStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1}}\n\n'
            yield b"data: [DONE]\n\n"

        async def aclose(self):
            captured["stream_closed"] = True

    def upstream_handler(request):
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=UpstreamStream(),
        )

    upstream_client = real_async_client(transport=httpx.MockTransport(upstream_handler))
    monkeypatch.setattr(
        restful_api_module.httpx, "AsyncClient", lambda **_: upstream_client
    )
    transport = httpx.ASGITransport(app=app)
    async with real_async_client(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/messages",
            headers={"anthropic-version": "2023-06-01"},
            json={
                "model": "virtual-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert captured["payload"]["stream_options"] == {"include_usage": True}
    event_names = [
        line.removeprefix("event: ")
        for line in response.text.splitlines()
        if line.startswith("event: ")
    ]
    assert event_names == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert '"model": "virtual-model"' in response.text
    assert '"stop_reason": "end_turn"' in response.text
    assert captured["stream_closed"] is True
    assert api._token_router_client is upstream_client
    assert not upstream_client.is_closed
    await api._close_token_router_client()
    assert upstream_client.is_closed


@pytest.mark.asyncio
async def test_anthropic_virtual_router_error_is_converted(monkeypatch):
    resolution = {
        "matched": True,
        "available": True,
        "virtual_model_uid": "virtual-model",
        "router_uid": "router-a",
        "instance_id": "instance-a",
        "endpoint": "http://router:10081",
    }
    api = make_rest_api(FakeSupervisor(resolution))
    app = make_anthropic_app(api)
    real_async_client = httpx.AsyncClient

    def upstream_handler(request):
        return httpx.Response(
            429,
            headers={"retry-after": "7"},
            json={"error": {"type": "rate_limit_error", "message": "busy"}},
        )

    upstream_client = real_async_client(transport=httpx.MockTransport(upstream_handler))
    monkeypatch.setattr(
        restful_api_module.httpx, "AsyncClient", lambda **_: upstream_client
    )
    transport = httpx.ASGITransport(app=app)
    async with real_async_client(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/messages",
            headers={"anthropic-version": "2023-06-01"},
            json={
                "model": "virtual-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
            },
        )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "7"
    assert response.json()["type"] == "error"
    assert response.json()["error"] == {
        "type": "rate_limit_error",
        "message": "busy",
    }
    assert response.json()["request_id"] == response.headers["request-id"]
    assert not upstream_client.is_closed
    await api._close_token_router_client()
    assert upstream_client.is_closed
