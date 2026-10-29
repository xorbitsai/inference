"""Benchmark job API, using the same model-management permissions as deployment."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, Security

from ... import __version__
from ...benchmark.service import BenchmarkConfig, BenchmarkService
from ...constants import XINFERENCE_HOME
from ...core.exceptions import ModelNotReadyError
from ..responses import JSONResponse

if TYPE_CHECKING:
    from ..restful_api import RESTfulAPI


def service(api):
    if getattr(api, "_benchmark_service", None) is None:
        api._benchmark_service = BenchmarkService(Path(XINFERENCE_HOME) / "benchmarks")
    return api._benchmark_service


def register_routes(api: "RESTfulAPI") -> None:
    dependencies = (
        [Security(api._auth_service, scopes=["models:write", "models:read"])]
        if api.is_authenticated()
        else None
    )

    async def list_benchmarks() -> JSONResponse:
        return JSONResponse(content=service(api).list())

    async def create_benchmark(body: BenchmarkConfig, request: Request) -> JSONResponse:
        api._check_model_access(request, body.model_uid, "LLM")
        supervisor = await api._get_supervisor_ref()
        try:
            model = await supervisor.describe_running_model(body.model_uid)
        except ModelNotReadyError as exc:
            raise HTTPException(status_code=409, detail="Model is not ready") from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Model not found") from exc
        if model.get("model_type") != "LLM" or "chat" not in model.get(
            "model_ability", []
        ):
            raise HTTPException(
                status_code=422, detail="Benchmarks require a running chat LLM"
            )
        if body.input_tokens + body.output_tokens > model.get("context_length", 2**31):
            raise HTTPException(
                status_code=422,
                detail="Input and output exceed the model context length",
            )
        instances, hardware = await asyncio.gather(
            supervisor.get_instance_info(model_name=None, model_uid=body.model_uid),
            supervisor.get_cluster_device_info(detailed=True),
        )
        # Persist an explicit allowlist, never launch environments or credentials.
        fields = (
            "model_name",
            "model_type",
            "model_engine",
            "model_format",
            "model_size_in_billions",
            "quantization",
            "revision",
            "context_length",
            "replica",
            "address",
            "accelerators",
        )
        try:
            runtime = await supervisor.get_model_reload_config(body.model_uid)
            engine = {
                key: runtime.get(key)
                for key in ("engine", "engine_version", "model_config")
            }
        except (ValueError, NotImplementedError):
            # This runtime introspection is currently exposed by single-worker
            # vLLM/SGLang replicas. Other layouts retain model/placement metadata.
            engine = None
        snapshot = {
            "xinference_version": __version__,
            "engine": engine,
            "model": {k: model[k] for k in fields if k in model},
            "instances": [
                {
                    k: row[k]
                    for k in (
                        "model_uid",
                        "model_engine",
                        "replica",
                        "replica_statuses",
                        "n_worker",
                        "status",
                    )
                    if k in row
                }
                for row in instances
            ],
            "hardware": hardware,
            "target": "Xinference chat API (all replicas)",
            "input_length": "Approximate target calibrated with the model tokenizer; actual lengths from response usage",
            "cache_policy": "Independent prompts per stage; disjoint warmup prompts; server cache is not flushed",
        }
        host = api._host
        if host in ("0.0.0.0", "::", ""):
            host = "127.0.0.1"
        if ":" in host:
            host = f"[{host}]"
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        result = service(api).create(
            body, snapshot, f"http://{host}:{api._port}", token
        )
        return JSONResponse(content=result, status_code=201)

    def get_job(job_id, raw=False):
        try:
            return service(api).get(job_id, raw=raw)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Benchmark not found") from exc

    async def get_benchmark(job_id: str) -> JSONResponse:
        return JSONResponse(content=get_job(job_id))

    async def export_benchmark(job_id: str) -> JSONResponse:
        return JSONResponse(
            content=get_job(job_id, raw=True),
            headers={
                "Content-Disposition": f'attachment; filename="benchmark-{get_job(job_id)["id"]}.json"'
            },
        )

    async def stop_benchmark(job_id: str) -> JSONResponse:
        get_job(job_id)
        return JSONResponse(content=await service(api).stop(job_id))

    async def delete_benchmark(job_id: str) -> JSONResponse:
        get_job(job_id)
        try:
            service(api).delete(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(content={"status": "ok"})

    async def benchmark_logs(job_id: str) -> JSONResponse:
        get_job(job_id)
        return JSONResponse(content={"logs": service(api).logs(job_id)})

    for path, method, handler in (
        ("", "GET", list_benchmarks),
        ("", "POST", create_benchmark),
        ("/{job_id}", "GET", get_benchmark),
        ("/{job_id}", "DELETE", delete_benchmark),
        ("/{job_id}/stop", "POST", stop_benchmark),
        ("/{job_id}/export", "GET", export_benchmark),
        ("/{job_id}/logs", "GET", benchmark_logs),
    ):
        api._router.add_api_route(
            "/v1/benchmarks" + path,
            handler,
            methods=[method],
            dependencies=dependencies,
        )
