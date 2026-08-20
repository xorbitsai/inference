from types import MethodType

import httpx
import pytest
from fastapi import FastAPI, Response

from xinference.api.oauth2.advanced.audit import classify_endpoint, should_skip_audit
from xinference.api.restful_api import RESTfulAPI


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
