# CI: linux
"""Exercise real SSE sockets and the production subprocess runner without a GPU."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import pytest_asyncio
from aiohttp import web
from fastapi import APIRouter, FastAPI, Header, HTTPException
from fastapi.security import SecurityScopes

from xinference.api.routers import benchmarks
from xinference.benchmark.adapter import StageRunner
from xinference.benchmark.service import BenchmarkConfig, BenchmarkService


@pytest_asyncio.fixture
async def server():
    seen = []
    entered = asyncio.Event()

    async def count(request):
        body = await request.json()
        return web.json_response(
            {"input_tokens": len(body["messages"][0]["content"].split()) * 2 + 8}
        )

    async def chat(request):
        body = await request.json()
        seen.append((body, request.headers.get("Authorization")))
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

    app = web.Application()
    app.router.add_post("/v1/messages/count_tokens", count)
    app.router.add_post("/v1/chat/completions", chat)
    runner = web.AppRunner(app, shutdown_timeout=0.01)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}", seen, entered
    await runner.cleanup()


@pytest.mark.asyncio
async def test_stream_metrics_ignore_role_usage_and_count_reasoning(server):
    endpoint, _, _ = server
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
    endpoint, _, _ = server
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
    endpoint, seen, _ = server
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
async def test_stop_running_and_queued_jobs_releases_process(server, tmp_path):
    endpoint, _, entered = server
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
async def test_api_permissions_validation_and_snapshot(tmp_path):
    async def auth(
        security_scopes: SecurityScopes, authorization: str = Header(default="")
    ):
        scopes = {
            "Bearer reader": {"models:read"},
            "Bearer writer": {"models:read", "models:write"},
        }.get(authorization)
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
        headers = {"Authorization": "Bearer writer"}
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
        assert token == "writer"
        assert config.runner == "xinference"
        api._check_model_access.assert_called_once()
        supervisor.describe_running_model.return_value["model_type"] = "embedding"
        assert (
            await client.post(
                "/v1/benchmarks", json={"model_uid": "model"}, headers=headers
            )
        ).status_code == 422
    await service.close()
