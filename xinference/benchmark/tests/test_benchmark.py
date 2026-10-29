# CI: linux
"""Exercise real SSE sockets and the production subprocess runner without a GPU."""

import asyncio
import json
from contextlib import nullcontext
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import pytest_asyncio
from aiohttp import web
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.security import SecurityScopes
from jose import jwt

from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService
from xinference.api.restful_api import RESTfulAPI, _request_credential
from xinference.api.routers import benchmarks
from xinference.benchmark.adapter import StageRunner
from xinference.benchmark.service import BenchmarkConfig, BenchmarkService


@pytest_asyncio.fixture
async def server(server_control):
    seen = []
    count_calls = []
    entered = asyncio.Event()

    async def count(request):
        body = await request.json()
        count_calls.append(body)
        if body["model"] in ("no-count", "missing-calibration-usage"):
            return web.json_response(
                {"error": "Token counting is not supported"}, status=501
            )
        if body["model"] == "forbidden-count":
            return web.json_response({"error": "Forbidden"}, status=403)
        return web.json_response(
            {"input_tokens": len(body["messages"][0]["content"].split()) * 2 + 8}
        )

    async def chat(request):
        body = await request.json()
        seen.append((body, request.headers.get("Authorization")))
        if not body.get("stream"):
            if body["model"] == "missing-calibration-usage":
                return web.json_response({"choices": [{"message": {"content": "x"}}]})
            return web.json_response(
                {
                    "usage": {
                        "prompt_tokens": len(body["messages"][0]["content"].split()) * 2
                        + 8,
                        "completion_tokens": 1,
                    }
                }
            )
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        entered.set()
        if body["model"] == "hang":
            await asyncio.sleep(30)
            return response

        async def event(data):
            raw = b"data: " + json.dumps(data).encode() + b"\n\n"
            # Split frames across TCP writes to exercise SSE framing.
            await response.write(raw[:7])
            await response.write(raw[7:])

        await event({"choices": [{"delta": {"role": "assistant", "content": ""}}]})
        await asyncio.sleep(0.03)
        await event({"choices": [{"delta": {"reasoning": "thinking"}}]})
        await asyncio.sleep(0.02)
        await event({"choices": [{"delta": {"content": "two tokens"}}]})
        await asyncio.sleep(0.04)
        if body["model"] != "no-usage":
            await event(
                {"choices": [], "usage": {"prompt_tokens": 24, "completion_tokens": 4}}
            )
        if body["model"] != "truncated":
            await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    @web.middleware
    async def authenticate(request, handler):
        validate = server_control.get("validate")
        if validate is not None:
            try:
                await validate(request)
            except HTTPException as exc:
                return web.json_response({"detail": exc.detail}, status=exc.status_code)
        return await handler(request)

    app = web.Application(middlewares=[authenticate])
    app.router.add_post("/v1/messages/count_tokens", count)
    app.router.add_post("/v1/chat/completions", chat)
    runner = web.AppRunner(app, shutdown_timeout=0.01)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}", seen, entered, count_calls
    await runner.cleanup()


@pytest.fixture
def server_control():
    return {}


@pytest.fixture
def benchmark_auth(tmp_path):
    auth = AdvancedAuthService(
        str(tmp_path / "auth.db"), "test-secret", "test-encryption"
    )
    auth._rate_limiter = None
    user_id = auth.db.create_user(
        username="benchmark-user",
        password_hash="unused",
        source="local",
        enabled=1,
        must_change_password=0,
        permissions=["models:read", "models:write"],
    )
    token = auth.create_access_token(
        user_id, "benchmark-user", ["models:read", "models:write"]
    )
    principal = auth.verify_access_token(token)
    return auth, token, principal


def auth_request(path="/v1/chat/completions", method="POST", body=None, token=""):
    raw = json.dumps(body or {"model": "model"}).encode()

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "headers": [
                (b"content-type", b"application/json"),
                (b"authorization", f"Bearer {token}".encode()),
            ],
            "client": ("127.0.0.1", 1),
        },
        receive,
    )


async def authenticate_benchmark(auth, request):
    body = await request.json()
    proxy = auth_request(
        request.path,
        request.method,
        body,
        request.headers.get("Authorization", "").removeprefix("Bearer "),
    )
    await auth(
        proxy, SecurityScopes(scopes=["models:read"]), _request_credential(proxy)
    )
    api = RESTfulAPI.__new__(RESTfulAPI)
    api._advanced_auth_service = auth
    api._check_model_access(proxy, body["model"], "LLM")


def expire_session(monkeypatch):
    class Later(datetime):
        @classmethod
        def utcnow(cls):
            return datetime.utcnow() + timedelta(days=1)

        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) + timedelta(days=1)

    monkeypatch.setattr(jwt, "datetime", Later)


@pytest.mark.asyncio
async def test_stream_metrics_ignore_role_usage_and_count_reasoning(server):
    endpoint, _, _, _ = server
    runner = StageRunner(
        endpoint + "/v1/chat/completions",
        "model",
        [("prompt", 10, 4)],
        True,
        1,
        warmup_requests=[],
        progress=lambda _: None,
    )
    await runner.run()
    result = runner.result()
    raw = result["raw_requests"][0]
    assert result["successful"] == 1
    assert raw["completion_tokens"] == 4  # two chunks are not two tokens
    assert raw["prompt_len"] == 24
    assert raw["ttft"] >= 0.025
    assert raw["last_token_latency"] < raw["latency"] - 0.025
    assert len(raw["itl"]) == 1
    assert result["tpot_ms"]["mean"] == pytest.approx(
        (raw["last_token_latency"] - raw["ttft"]) * 1000 / 3
    )
    assert result["output_throughput"] == pytest.approx(4 / result["duration_seconds"])


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["no-usage", "truncated"])
async def test_missing_measurements_and_incomplete_stream(server, model):
    endpoint, _, _, _ = server
    runner = StageRunner(
        endpoint + "/v1/chat/completions",
        model,
        [("prompt", 10, 4)],
        True,
        1,
        warmup_requests=[],
        progress=lambda _: None,
    )
    await runner.run()
    result = runner.result()
    assert result["output_throughput"] is None
    assert result["tpot_ms"] is None
    if model == "truncated":
        assert result["failed"] == 1
        assert result["ttft_ms"] is None
        assert "before [DONE]" in result["raw_requests"][0]["error"]
    else:
        assert result["successful"] == 1
        assert result["ttft_ms"] is not None


@pytest.mark.asyncio
async def test_subprocess_job_persistence_progress_and_no_saved_credentials(
    server, tmp_path
):
    endpoint, seen, _, _ = server
    service = BenchmarkService(tmp_path)
    config = BenchmarkConfig(
        model_uid="model",
        input_tokens=32,
        output_tokens=4,
        requests=3,
        concurrency=[1, 2],
        warmup=1,
    )
    job = service.create(config, {"model": "model"}, endpoint, "secret-token")
    await asyncio.wait_for(service.tasks[job["id"]], 30)
    result = service.get(job["id"], raw=True)
    assert result["state"] == "completed"
    assert result["progress"] == 100
    assert len(result["results"]) == 2
    assert [stage["successful"] for stage in result["results"]] == [3, 3]
    assert len(seen) == 8  # warmups run, but never enter measurements
    assert all(auth == "Bearer secret-token" for _, auth in seen)
    prompts = [body["messages"][0]["content"] for body, _ in seen]
    assert len(set(prompts)) == 8
    assert "raw_requests" not in service.list()[0]["results"][0]
    await service.close()
    reopened = BenchmarkService(tmp_path)
    assert reopened.get(job["id"])["state"] == "completed"
    assert "secret-token" not in "".join(
        path.read_text() for path in tmp_path.iterdir()
    )
    reopened.delete(job["id"])
    assert reopened.list() == []
    await reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("admin", [False, True])
async def test_job_credential_is_model_and_endpoint_scoped(benchmark_auth, admin):
    auth, session_token, principal = benchmark_auth
    if admin:
        auth.db.set_user_permissions(principal["user_id"], ["admin"])
    with auth.benchmark_credential(principal, "model", 60) as token:
        assert token != session_token
        assert auth.verify_access_token(token) is None
        assert auth.validate_api_key(token) is None
        for path in ("/v1/chat/completions", "/v1/messages/count_tokens"):
            user = await auth(
                auth_request(path), SecurityScopes(scopes=["models:read"]), token
            )
            assert user["id"] == principal["user_id"]
        assert auth.validate_model_access(token, "model", "LLM")
        assert not auth.validate_model_access(token, "another-model", "LLM")
        assert not auth.validate_model_access(token, "model", "embedding")
        for path, method, scopes in (
            ("/v1/models", "GET", ["models:read"]),
            ("/v1/benchmarks", "POST", ["models:read", "models:write"]),
            ("/v1/admin/users", "GET", []),
            ("/v1/chat/completions", "GET", []),
            ("/v1/chat/completions", "POST", ["models:write"]),
        ):
            with pytest.raises(HTTPException) as exc:
                await auth(
                    auth_request(path, method), SecurityScopes(scopes=scopes), token
                )
            assert exc.value.status_code == 401
    assert not auth._benchmark_credentials
    assert not auth.validate_model_access(token, "model", "LLM")
    with pytest.raises(HTTPException) as exc:
        await auth(auth_request(), SecurityScopes(scopes=["models:read"]), token)
    assert exc.value.status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize("revoke", ["password", "disabled", "permissions", "expired"])
async def test_job_credential_respects_live_revocation(benchmark_auth, revoke):
    auth, _, principal = benchmark_auth
    with auth.benchmark_credential(principal, "model", 60) as token:
        if revoke == "password":
            auth.db.update_password_and_revoke_tokens(principal["user_id"], "new-hash")
        elif revoke == "disabled":
            auth.disable_user(principal["user_id"])
        elif revoke == "permissions":
            auth.db.set_user_permissions(principal["user_id"], ["models:read"])
        else:
            next(iter(auth._benchmark_credentials.values()))["expires_at"] = 0
        assert not auth.validate_model_access(token, "model", "LLM")
        with pytest.raises(HTTPException) as exc:
            await auth(auth_request(), SecurityScopes(scopes=["models:read"]), token)
        assert exc.value.status_code == 401
    assert not auth._benchmark_credentials


@pytest.mark.asyncio
async def test_api_keys_still_cannot_authorize_benchmark_management(benchmark_auth):
    auth, _, principal = benchmark_auth
    key = auth.create_api_key_for_user(principal["user_id"], "inference-key")["key"]
    with pytest.raises(HTTPException) as exc:
        await auth(
            auth_request("/v1/benchmarks"),
            SecurityScopes(scopes=["models:read", "models:write"]),
            key,
        )
    assert exc.value.status_code == 403
    assert not auth._benchmark_credentials


@pytest.mark.asyncio
@pytest.mark.parametrize("expiry", ["queued", "between-stages"])
async def test_subprocess_outlives_session_expiry(
    server, server_control, benchmark_auth, tmp_path, monkeypatch, expiry
):
    endpoint, seen, _, _ = server
    auth, session_token, principal = benchmark_auth
    expired = False

    async def validate(request):
        nonlocal expired
        measured = [body for body, _ in seen if body.get("stream")]
        if expiry == "between-stages" and len(measured) >= 2 and not expired:
            expire_session(monkeypatch)
            expired = True
        await authenticate_benchmark(auth, request)

    server_control["validate"] = validate
    service = BenchmarkService(tmp_path / "jobs")
    config = BenchmarkConfig(
        model_uid="model",
        input_tokens=32,
        output_tokens=4,
        concurrency=[1, 2],
        requests=2,
        warmup=0,
    )
    await service._lock.acquire()
    job = service.create(
        config,
        {},
        endpoint,
        "",
        credential_factory=lambda: auth.benchmark_credential(principal, "model", 3660),
    )
    await asyncio.sleep(0)
    assert service.get(job["id"])["state"] == "queued"
    assert not auth._benchmark_credentials
    if expiry == "queued":
        expire_session(monkeypatch)
    service._lock.release()
    try:
        await asyncio.wait_for(service.tasks[job["id"]], 30)
        result = service.get(job["id"], raw=True)
        assert auth.verify_access_token(session_token) is None
        assert result["state"] == "completed", result["error"]
        assert [stage["successful"] for stage in result["results"]] == [2, 2]
        assert all(stage["failed"] == 0 for stage in result["results"])
        credentials = {header.removeprefix("Bearer ") for _, header in seen}
        assert len(credentials) == 1
        token = credentials.pop()
        assert token != session_token
        assert not auth._benchmark_credentials
        assert not auth.validate_model_access(token, "model", "LLM")
        saved = "".join(p.read_text() for p in service.directory.iterdir())
        assert token not in saved and session_token not in saved
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_password_reset_between_stages_aborts_job_and_revokes_lease(
    server, server_control, benchmark_auth, tmp_path
):
    endpoint, seen, _, _ = server
    auth, _, principal = benchmark_auth

    async def validate(request):
        measured = [body for body, _ in seen if body.get("stream")]
        if len(measured) >= 2:
            auth.db.update_password_and_revoke_tokens(principal["user_id"], "new-hash")
        await authenticate_benchmark(auth, request)

    server_control["validate"] = validate
    service = BenchmarkService(tmp_path / "jobs")
    config = BenchmarkConfig(
        model_uid="model", input_tokens=32, concurrency=[1, 2], requests=2, warmup=0
    )
    job = service.create(
        config,
        {},
        endpoint,
        "",
        credential_factory=lambda: auth.benchmark_credential(principal, "model", 3660),
    )
    try:
        await asyncio.wait_for(service.tasks[job["id"]], 30)
        result = service.get(job["id"])
        assert result["state"] == "failed"
        assert "authorization failed (HTTP 401)" in result["error"]
        assert [stage["concurrency"] for stage in result["results"]] == [1]
        assert not auth._benchmark_credentials
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_revoked_queued_job_never_gets_a_credential(
    server, benchmark_auth, tmp_path
):
    endpoint, seen, _, count_calls = server
    auth, _, principal = benchmark_auth
    service = BenchmarkService(tmp_path / "jobs")
    await service._lock.acquire()
    job = service.create(
        BenchmarkConfig(model_uid="model"),
        {},
        endpoint,
        "",
        credential_factory=lambda: auth.benchmark_credential(principal, "model", 3660),
    )
    auth.db.update_password_and_revoke_tokens(principal["user_id"], "new-hash")
    service._lock.release()
    try:
        await asyncio.wait_for(service.tasks[job["id"]], 5)
        result = service.get(job["id"])
        assert result["state"] == "failed"
        assert "revoked before execution" in result["error"]
        assert result["results"] == []
        assert not seen and not count_calls and not auth._benchmark_credentials
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_stop_revokes_running_job_credential(
    server, server_control, benchmark_auth, tmp_path
):
    endpoint, _, entered, _ = server
    auth, _, principal = benchmark_auth
    server_control["validate"] = lambda request: authenticate_benchmark(auth, request)
    service = BenchmarkService(tmp_path / "jobs")
    config = BenchmarkConfig(
        model_uid="hang", input_tokens=32, concurrency=[1], requests=1, warmup=0
    )
    job = service.create(
        config,
        {},
        endpoint,
        "",
        credential_factory=lambda: auth.benchmark_credential(principal, "hang", 3660),
    )
    try:
        await asyncio.wait_for(entered.wait(), 30)
        assert auth._benchmark_credentials
        assert (await service.stop(job["id"]))["state"] == "stopped"
        assert not auth._benchmark_credentials
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("phase", ["warmup", "measured"])
async def test_auth_failure_discards_stage_and_stops_requests(
    server, server_control, tmp_path, status, phase
):
    endpoint, seen, _, _ = server
    rejected = 0

    async def validate(request):
        nonlocal rejected
        body = await request.json()
        # Keep the first concurrency point, then fail during warmup or after
        # one successful measured request at the next point.
        if body.get("stream") and len(seen) >= 4:
            rejected += 1
            raise HTTPException(status, "Credential no longer valid")

    server_control["validate"] = validate
    service = BenchmarkService(tmp_path)
    config = BenchmarkConfig(
        model_uid="model",
        input_tokens=32,
        concurrency=[1, 2, 3],
        requests=3,
        warmup=1 if phase == "warmup" else 0,
    )
    job = service.create(config, {}, endpoint, "revoked-token")
    try:
        await asyncio.wait_for(service.tasks[job["id"]], 30)
        result = service.get(job["id"])
        assert result["state"] == "failed"
        assert f"authorization failed (HTTP {status})" in result["error"]
        assert [stage["concurrency"] for stage in result["results"]] == [1]
        assert result["results"][0]["successful"] == 3
        assert len(seen) == 4
        assert 1 <= rejected <= 2
        assert not any("concurrency 3" in line for line in result["logs"])
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_stop_running_and_queued_jobs_releases_process(server, tmp_path):
    endpoint, _, entered, _ = server
    service = BenchmarkService(tmp_path)
    config = BenchmarkConfig(
        model_uid="hang", input_tokens=32, concurrency=[1], requests=1, warmup=0
    )
    first = service.create(config, {}, endpoint, "")
    queued = service.create(config, {}, endpoint, "")
    await asyncio.wait_for(entered.wait(), 30)
    assert service.get(queued["id"])["state"] == "queued"
    with pytest.raises(ValueError, match="Stop"):
        service.delete(first["id"])
    assert (await service.stop(queued["id"]))["state"] == "stopped"
    assert (await asyncio.wait_for(service.stop(first["id"]), 5))["state"] == "stopped"
    assert not service.tasks
    await service.close()


@pytest.mark.asyncio
async def test_restart_marks_stale_jobs_interrupted(tmp_path):
    row = {"id": "abc", "state": "running", "results": [], "created_at": "2026-01-01"}
    (tmp_path / "abc.json").write_text(json.dumps(row))
    service = BenchmarkService(tmp_path)
    assert service.get("abc")["state"] == "interrupted"
    assert "restarted" in service.get("abc")["error"]
    await service.close()


@pytest.mark.parametrize(
    "body",
    [
        {"concurrency": [True]},
        {"concurrency": [1.5]},
        {"concurrency": [1, 1]},
        {"concurrency": [0]},
        {"concurrency": []},
        {"requests": True},
        {"requests": 1, "concurrency": [2]},
        {"runner": "untrusted-command"},
        {"input_tokens": 0},
        {"endpoint": "http://other"},
    ],
)
def test_invalid_config(body):
    with pytest.raises(ValueError):
        BenchmarkConfig(model_uid="model", **body)


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ["Authorization", "X-API-Key"])
async def test_api_permissions_validation_and_snapshot(tmp_path, header):
    async def auth(request: Request, security_scopes: SecurityScopes):
        scopes = {
            "reader": {"models:read"},
            "writer": {"models:read", "models:write"},
        }.get(_request_credential(request))
        if scopes is None:
            raise HTTPException(401)
        if not set(security_scopes.scopes).issubset(scopes):
            raise HTTPException(403)

    service = BenchmarkService(tmp_path)
    # Creation contract is tested separately from process execution above.
    create = Mock(return_value={"id": "job"})
    service.create = create
    supervisor = SimpleNamespace(
        describe_running_model=AsyncMock(
            return_value={
                "model_type": "LLM",
                "model_ability": ["chat"],
                "context_length": 4096,
                "model_name": "model",
                "env": {"SECRET": "private"},
            }
        ),
        get_model_reload_config=AsyncMock(
            return_value={
                "engine": "vllm",
                "engine_version": "0.1",
                "model_config": {"max_num_seqs": 64},
            }
        ),
        get_instance_info=AsyncMock(return_value=[]),
        get_cluster_device_info=AsyncMock(return_value=[]),
    )
    api = SimpleNamespace(
        _router=APIRouter(),
        _auth_service=auth,
        _advanced_auth_service=SimpleNamespace(
            verify_access_token=Mock(return_value={"user_id": 1, "token_version": 0}),
            benchmark_credential=Mock(return_value=nullcontext("job-token")),
        ),
        _benchmark_service=service,
        is_authenticated=lambda: True,
        _check_model_access=Mock(),
        _get_supervisor_ref=AsyncMock(return_value=supervisor),
        _host="0.0.0.0",
        _port=9997,
    )
    app = FastAPI()
    benchmarks.register_routes(api)
    app.include_router(api._router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://api"
    ) as client:
        for method, path in [
            ("GET", "/v1/benchmarks"),
            ("POST", "/v1/benchmarks"),
            ("GET", "/v1/benchmarks/unknown/export"),
            ("POST", "/v1/benchmarks/unknown/stop"),
            ("DELETE", "/v1/benchmarks/unknown"),
        ]:
            assert (await client.request(method, path)).status_code == 401
            assert (
                await client.request(
                    method, path, headers={"Authorization": "Bearer reader"}
                )
            ).status_code == 403
        headers = {header: "Bearer writer" if header == "Authorization" else "writer"}
        assert (
            await client.get("/v1/benchmarks/unknown", headers=headers)
        ).status_code == 404
        assert (
            await client.post(
                "/v1/benchmarks",
                json={"model_uid": "model", "concurrency": [0]},
                headers=headers,
            )
        ).status_code == 422
        response = await client.post(
            "/v1/benchmarks", json={"model_uid": "model"}, headers=headers
        )
        assert response.status_code == 201
        config, snapshot, endpoint, token = create.call_args.args
        assert "env" not in snapshot["model"]
        assert endpoint == "http://127.0.0.1:9997"
        assert token == ""
        api._advanced_auth_service.verify_access_token.assert_called_with("writer")
        factory = create.call_args.kwargs["credential_factory"]
        with factory() as credential:
            assert credential == "job-token"
        api._advanced_auth_service.benchmark_credential.assert_called_once_with(
            {"user_id": 1, "token_version": 0}, "model", 3660
        )
        assert config.runner == "xinference"
        api._check_model_access.assert_called_once()
        supervisor.describe_running_model.return_value["model_type"] = "embedding"
        assert (
            await client.post(
                "/v1/benchmarks", json={"model_uid": "model"}, headers=headers
            )
        ).status_code == 422
    await service.close()


@pytest.mark.asyncio
async def test_unsupported_token_count_uses_unmeasured_chat_usage(server, tmp_path):
    endpoint, seen, _, count_calls = server
    service = BenchmarkService(tmp_path)
    config = BenchmarkConfig(
        model_uid="no-count",
        input_tokens=32,
        output_tokens=4,
        requests=3,
        concurrency=[1, 2],
        warmup=0,
    )
    job = service.create(config, {}, endpoint, "token")
    await asyncio.wait_for(service.tasks[job["id"]], 30)
    result = service.get(job["id"], raw=True)
    assert result["state"] == "completed"
    assert len(count_calls) == 1
    probes = [body for body, _ in seen if not body.get("stream")]
    measured = [body for body, _ in seen if body.get("stream")]
    assert len(probes) > 1
    assert all(body["max_tokens"] == 1 for body in probes)
    assert len(measured) == 6
    assert all(
        stage["successful"] == 3 and len(stage["raw_requests"]) == 3
        for stage in result["results"]
    )
    calibration_prompts = {body["messages"][0]["content"] for body in probes}
    assert not calibration_prompts.intersection(
        body["messages"][0]["content"] for body in measured
    )
    assert any("untimed" in line for line in result["logs"])
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model, message",
    [
        ("forbidden-count", "HTTP 403"),
        ("missing-calibration-usage", "usage.prompt_tokens"),
    ],
)
async def test_calibration_failure_reports_actual_error_without_unsafe_fallback(
    server, tmp_path, model, message
):
    endpoint, seen, _, _ = server
    service = BenchmarkService(tmp_path)
    config = BenchmarkConfig(
        model_uid=model, input_tokens=32, requests=1, concurrency=[1], warmup=0
    )
    job = service.create(config, {}, endpoint, "")
    await asyncio.wait_for(service.tasks[job["id"]], 30)
    result = service.get(job["id"])
    assert result["state"] == "failed"
    assert message in result["error"]
    assert result["results"] == []
    if model == "forbidden-count":
        assert not seen
    await service.close()
