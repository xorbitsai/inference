from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI, Request, Response

from xinference.api.oauth2.advanced import audit as audit_module
from xinference.api.oauth2.advanced.audit import classify_endpoint, should_skip_audit
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


@pytest.mark.asyncio
async def test_token_router_management_request_records_final_audit_status():
    api = RESTfulAPI.__new__(RESTfulAPI)
    api._advanced_auth_service = object()
    recorded = []

    def record_admin_audit(self, request, status, latency_s=0.0):
        recorded.append((request.url.path, status, latency_s))

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
    endpoint, status, latency_s = recorded[0]
    assert endpoint == "/v1/token_routers/router-1/enable"
    assert status == "error"
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
