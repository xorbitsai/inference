from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI, Request, Response

from xinference.api.oauth2.advanced import audit as audit_module
from xinference.api.oauth2.advanced.audit import (
    classify_endpoint,
    should_skip_audit,
    should_skip_completed_audit,
)
from xinference.api.oauth2.advanced.crypto import sha256_hex
from xinference.api.restful_api import RESTfulAPI
from xinference.core import metrics as core_metrics


def test_internal_token_router_endpoints_skip_audit():
    assert should_skip_audit("/v1/internal/token-router/instances/register") is True
    assert (
        should_skip_audit("/v1/internal/token-router/instances/router-1/config-ack")
        is True
    )


def test_token_router_management_endpoints_are_admin_audited():
    endpoint = "/v1/token_routers/router-1/enable"
    assert should_skip_audit(endpoint) is False
    assert classify_endpoint(endpoint) == "admin"


@pytest.mark.parametrize(
    ("method", "endpoint", "status_code", "expected"),
    [
        ("GET", "/metrics", 200, True),
        ("GET", "/metrics", 204, True),
        ("GET", "/metrics", 302, True),
        ("GET", "/metrics", 399, True),
        ("GET", "/metrics", 400, False),
        ("GET", "/metrics", 401, False),
        ("GET", "/metrics", 403, False),
        ("GET", "/metrics", 404, False),
        ("GET", "/metrics", 500, False),
        ("POST", "/metrics", 200, False),
        ("HEAD", "/metrics", 200, False),
        ("GET", "/metrics/detail", 200, False),
        ("GET", "/metrics-admin", 200, False),
    ],
)
def test_completed_audit_skip_rules(method, endpoint, status_code, expected):
    assert should_skip_completed_audit(method, endpoint, status_code) is expected


@pytest.mark.asyncio
async def test_metrics_audit_skips_only_successful_get_requests():
    api = RESTfulAPI.__new__(RESTfulAPI)
    api._advanced_auth_service = object()
    recorded = []

    def record_admin_audit(
        self, request, status, latency_s=0.0, status_code=0, category=""
    ):
        recorded.append((request.method, request.url.path, status, status_code))

    api._record_admin_audit = MethodType(record_admin_audit, api)
    app = FastAPI()
    app.middleware("http")(api._audit_middleware)

    @app.api_route("/metrics", methods=["GET", "POST"])
    async def metrics(request: Request) -> Response:
        if request.query_params.get("raise") == "true":
            raise RuntimeError("metrics unavailable")
        if request.method != "GET":
            return Response(status_code=405)
        return Response(status_code=int(request.query_params.get("status", "200")))

    @app.get("/metrics/detail")
    async def metrics_detail() -> Response:
        return Response(status_code=200)

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        success_response = await client.get("/metrics")
        query_response = await client.get("/metrics?format=prometheus")
        failure_response = await client.get("/metrics?status=500")
        method_response = await client.post("/metrics")
        detail_response = await client.get("/metrics/detail")
        exception_response = await client.get("/metrics?raise=true")

    assert success_response.status_code == 200
    assert "x-request-id" in success_response.headers
    assert query_response.status_code == 200
    assert "x-request-id" in query_response.headers
    assert failure_response.status_code == 500
    assert method_response.status_code == 405
    assert detail_response.status_code == 200
    assert exception_response.status_code == 500
    assert recorded == [
        ("GET", "/metrics", "error", 500),
        ("POST", "/metrics", "error", 405),
        ("GET", "/metrics/detail", "success", 200),
        ("GET", "/metrics", "error", 500),
    ]


@pytest.mark.asyncio
async def test_token_router_management_request_records_final_audit_status():
    api = RESTfulAPI.__new__(RESTfulAPI)
    api._advanced_auth_service = object()
    recorded = []

    def record_admin_audit(
        self, request, status, latency_s=0.0, status_code=0, category=""
    ):
        recorded.append((request.url.path, status, latency_s, status_code, category))

    api._record_admin_audit = MethodType(record_admin_audit, api)
    app = FastAPI()
    app.middleware("http")(api._audit_middleware)

    @app.post("/v1/token_routers/{router_uid}/enable")
    async def enable_router(router_uid: str) -> Response:
        return Response(status_code=409)

    @app.post("/v1/internal/token-router/instances/register")
    async def register_runtime() -> Response:
        return Response(status_code=200)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        management_response = await client.post("/v1/token_routers/router-1/enable")
        internal_response = await client.post(
            "/v1/internal/token-router/instances/register"
        )

    assert management_response.status_code == 409
    assert internal_response.status_code == 200
    assert len(recorded) == 1
    endpoint, status, latency_s, status_code, category = recorded[0]
    assert endpoint == "/v1/token_routers/router-1/enable"
    assert status == "error"
    assert status_code == 409
    assert category == "admin"
    assert latency_s >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_access_allowed", "expected_status_code", "expected_audit_status"),
    [(True, 200, "success"), (False, 403, "denied")],
)
async def test_anthropic_x_api_key_records_inference_audit(
    monkeypatch,
    model_access_allowed,
    expected_status_code,
    expected_audit_status,
):
    token = "external-anthropic-key"
    entry = SimpleNamespace(
        user_id=7,
        name="anthropic-key",
        key_prefix="sk-test",
    )
    auth_service = SimpleNamespace(
        cache=MagicMock(),
        db=MagicMock(),
        validate_model_access=MagicMock(return_value=model_access_allowed),
    )
    auth_service.cache.get.return_value = entry
    auth_service.db.get_user_by_id.return_value = {"username": "anthropic-user"}

    recorded = []
    requests_total = MagicMock()
    request_duration = MagicMock()
    monkeypatch.setattr(
        audit_module, "record_audit_event", lambda **kwargs: recorded.append(kwargs)
    )
    monkeypatch.setattr(core_metrics, "api_key_requests_total", requests_total)
    monkeypatch.setattr(
        core_metrics, "api_key_request_duration_seconds", request_duration
    )

    api = RESTfulAPI.__new__(RESTfulAPI)
    api._advanced_auth_service = auth_service
    api._uid_to_model_name = {"virtual-model": "resolved-model"}
    app = FastAPI()
    app.middleware("http")(api._audit_middleware)

    @app.post("/v1/messages")
    async def create_message(request: Request) -> Response:
        api._check_model_access(request, "virtual-model", "LLM")
        return Response(status_code=200)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/messages", headers={"x-api-key": token})

    assert response.status_code == expected_status_code
    auth_service.validate_model_access.assert_called_once_with(
        token, "virtual-model", "LLM"
    )
    auth_service.cache.get.assert_called_once_with(sha256_hex(token))
    assert len(recorded) == 1
    assert recorded[0] == {
        "user": "anthropic-user",
        "api_key_name": "anthropic-key",
        "api_key_prefix": "sk-test",
        "model_id": "virtual-model",
        "model_name": "resolved-model",
        "model_type": "LLM",
        "endpoint": "/v1/messages",
        "status": expected_audit_status,
        "latency_ms": recorded[0]["latency_ms"],
        "client_ip": "127.0.0.1",
        "category": "inference",
        "auth_type": "api_key",
        "method": "POST",
        "status_code": expected_status_code,
        "request_id": response.headers["x-request-id"],
    }
    assert recorded[0]["latency_ms"] >= 0
    requests_total.inc.assert_called_once_with(
        {
            "user": "anthropic-user",
            "api_key_name": "anthropic-key",
            "model_id": "virtual-model",
            "model_name": "resolved-model",
            "model_type": "LLM",
            "status": expected_audit_status,
        }
    )
    request_duration.observe.assert_called_once()
    duration_labels, latency_s = request_duration.observe.call_args.args
    assert duration_labels == {
        "model_id": "virtual-model",
        "model_type": "LLM",
        "model_name": "resolved-model",
    }
    assert latency_s >= 0


@pytest.mark.asyncio
async def test_admin_request_records_one_final_audit_event(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        audit_module, "record_audit_event", lambda **kwargs: recorded.append(kwargs)
    )

    auth_service = SimpleNamespace(verify_access_token=MagicMock(return_value=None))
    api = RESTfulAPI.__new__(RESTfulAPI)
    api._advanced_auth_service = auth_service
    api._uid_to_model_name = {}
    app = FastAPI()
    app.middleware("http")(api._audit_middleware)

    @app.get("/v1/cluster/model-requests/{request_id}/body")
    async def request_body(request: Request, request_id: str) -> Response:
        request.state.audit_identity = {
            "user": "admin",
            "api_key_name": "",
            "api_key_prefix": "",
            "auth_type": "jwt",
        }
        return Response(status_code=404)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/v1/cluster/model-requests/xinf-old/body",
            headers={"x-request-id": "caller-request-id"},
        )

    assert response.status_code == 404
    assert response.headers["x-request-id"] == "caller-request-id"
    assert len(recorded) == 1
    assert recorded[0]["status"] == "error"
    assert recorded[0]["status_code"] == 404
    assert recorded[0]["method"] == "GET"
    assert recorded[0]["request_id"] == "caller-request-id"
