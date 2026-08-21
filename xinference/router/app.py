# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""OpenAI-compatible data plane for the Xinference Token-aware Router."""

from __future__ import annotations

import argparse
import asyncio
import hmac
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from .admission import AdmissionRejected
from .backend import request_headers, response_headers
from .classifier import (
    ContextLimitExceeded,
    RouteDecision,
    RouteRejected,
    ThinkingRejected,
)
from .config import RouterConfig, load_config
from .metrics import RouterMetrics
from .runtime import RouterDisabled, RouterRuntime, RuntimeSnapshot
from .tokenization import TokenizationWorkerUnavailable
from .tokenizer import TokenizationError

logger = logging.getLogger("xinference.router")


def _error(
    status: int,
    message: str,
    error_type: str,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        headers=headers,
        content={"error": {"message": message, "type": error_type, "code": status}},
    )


def _authorized(request: Request, config: RouterConfig) -> bool:
    if not config.require_auth:
        return True
    value = request.headers.get("authorization", "")
    prefix = "Bearer "
    if not value.startswith(prefix):
        return False
    return hmac.compare_digest(value[len(prefix) :], config.backend_api_key)


def _payload_thinking(payload: dict) -> bool:
    """Return whether the request asks for thinking mode.

    Mirrors the tokenizer normalization so capability enforcement rejects the
    request before expensive rendering when the asset does not support it.
    """
    value = payload.get("enable_thinking")
    if value is None:
        extra_body = payload.get("extra_body")
        if isinstance(extra_body, dict):
            value = extra_body.get("enable_thinking")
    if value is None:
        template_kwargs = payload.get("chat_template_kwargs")
        if isinstance(template_kwargs, dict):
            value = template_kwargs.get("enable_thinking")
    return bool(value)


def _router_headers(decision: RouteDecision, request_id: str) -> dict[str, str]:
    return {
        "x-request-id": request_id,
        "x-xinference-router-pool": decision.backend_id,
        "x-xinference-router-backend": decision.backend_id,
        "x-xinference-router-rule": decision.rule_id,
        "x-xinference-router-reason": decision.reason,
        "x-xinference-router-prompt-tokens": str(decision.budget.prompt_tokens),
        "x-xinference-router-total-budget": str(decision.budget.total_tokens),
    }


def create_app(config: RouterConfig) -> FastAPI:
    metrics = RouterMetrics()
    runtime = RouterRuntime(config, metrics)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        control_plane = getattr(app.state, "control_plane", None)
        control_stop = asyncio.Event()
        control_task = None
        try:
            await runtime.start()
            if control_plane is not None:
                # ACK only after the initial runtime snapshot is fully usable.
                await control_plane.ack(runtime.current.config.revision)
                control_task = asyncio.create_task(
                    control_plane.run(runtime, control_stop),
                    name="token-router-control-plane",
                )
            current = runtime.current.config
            logger.info(
                "Router started: router_uid=%s logical_model=%s backend=%s "
                "route_profile=%s backends=%s revision=%d enabled=%s",
                current.router_uid,
                current.logical_model,
                current.backend_url,
                current.route_profile,
                ",".join(backend.id for backend in current.backends),
                current.revision,
                current.enabled,
            )
            yield
        finally:
            control_stop.set()
            try:
                if control_task is not None:
                    # The control-plane task may be blocked in an HTTP call or
                    # while applying a new runtime snapshot. Setting the stop
                    # event alone cannot interrupt either operation, so cancel
                    # the task explicitly before waiting for shutdown.
                    control_task.cancel()
                    try:
                        await control_task
                    except asyncio.CancelledError:
                        pass
            finally:
                try:
                    if control_plane is not None:
                        await control_plane.unregister()
                finally:
                    await runtime.aclose()

    app = FastAPI(
        title="Xinference Token-aware Router",
        version="0.4.0",
        lifespan=lifespan,
    )

    def sync_state(snapshot: RuntimeSnapshot) -> None:
        # Keep these aliases for diagnostics and compatibility with the
        # standalone prototype tests. Request handling always uses runtime.
        app.state.config = snapshot.config
        app.state.tokenization = snapshot.tokenization
        app.state.policy = snapshot.policy
        app.state.gates = snapshot.gates
        app.state.client = snapshot.client

    runtime.set_on_swap(sync_state)
    app.state.runtime = runtime
    app.state.metrics = metrics

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        snapshot = runtime.current
        summary = await runtime.summary()
        status = "ok" if snapshot.config.enabled else "disabled"
        return JSONResponse(
            {
                "status": status,
                "router_uid": snapshot.config.router_uid,
                "logical_model": snapshot.config.logical_model,
                "backend_url": snapshot.config.backend_url,
                **summary,
            },
            status_code=200,
        )

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        snapshot = runtime.current
        if not snapshot.config.enabled:
            return JSONResponse({"status": "disabled"}, status_code=503)
        return JSONResponse(
            {
                "status": "ready",
                "revision": snapshot.config.revision,
                "router_uid": snapshot.config.router_uid,
                "tokenizer_asset_id": snapshot.config.tokenizer_asset_id,
                "tokenizer_asset_revision": snapshot.config.tokenizer_asset_revision,
            }
        )

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        return Response(await metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    async def models(request: Request) -> JSONResponse:
        current = runtime.current.config
        if not _authorized(request, current):
            return _error(
                401, "Invalid authentication credentials", "authentication_error"
            )
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": current.logical_model,
                        "object": "model",
                        "owned_by": "xinference-token-router",
                    }
                ],
            }
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        started = time.monotonic()
        request_id = request.headers.get("x-request-id") or f"router-{uuid.uuid4()}"
        try:
            snapshot = await runtime.acquire()
        except RouterDisabled:
            await metrics.increment("router_disabled", "none")
            return _error(
                503,
                "Token Router is disabled and is not accepting new requests",
                "router_disabled",
                headers={"retry-after": "1", "x-request-id": request_id},
            )

        runtime_release_owned = True
        config = snapshot.config

        try:
            if not _authorized(request, config):
                await metrics.increment("auth_rejected", "none")
                return _error(
                    401, "Invalid authentication credentials", "authentication_error"
                )

            try:
                body = await request.body()
                payload = await request.json()
            except Exception:
                await metrics.increment("invalid_json", "none")
                return _error(
                    400, "Request body must be valid JSON", "invalid_request_error"
                )
            if not isinstance(payload, dict):
                return _error(
                    400, "Request body must be a JSON object", "invalid_request_error"
                )

            accepted_models = {config.logical_model, *config.model_aliases}
            if payload.get("model") not in accepted_models:
                await metrics.increment("unknown_model", "none")
                return _error(404, "Unknown logical model", "model_not_found")
            capabilities = config.tokenizer_asset_capabilities
            if bool(payload.get("tools")) and "tools" not in capabilities:
                await metrics.increment("tools_not_allowed", "none")
                return _error(
                    400,
                    "Tool requests are not supported by this Tokenizer asset",
                    "tools_not_allowed",
                    headers={"x-request-id": request_id},
                )
            try:
                budget = await snapshot.tokenization.estimate(
                    payload, input_bytes=len(body)
                )
                decision = snapshot.policy.classify(
                    budget,
                    tools_present=bool(payload.get("tools")),
                    stream=bool(payload.get("stream", False)),
                )
                if budget.enable_thinking and "thinking" not in capabilities:
                    await metrics.increment("thinking_not_allowed", "none")
                    return _error(
                        400,
                        "Thinking-mode requests are not supported by this "
                        "Tokenizer asset",
                        "thinking_not_allowed",
                        headers={"x-request-id": request_id},
                    )
            except AdmissionRejected as exc:
                await metrics.increment(f"tokenization_admission_{exc.reason}", "none")
                return _error(
                    429,
                    "Router tokenization capacity is busy; retry later",
                    "rate_limit_error",
                    headers={
                        "retry-after": str(exc.retry_after_seconds),
                        "x-request-id": request_id,
                    },
                )
            except TokenizationWorkerUnavailable:
                await metrics.increment("tokenization_worker_unavailable", "none")
                logger.exception(
                    "request_id=%s tokenization worker unavailable", request_id
                )
                return _error(
                    503,
                    "Tokenization worker is unavailable; retry later",
                    "tokenization_worker_unavailable",
                    headers={"retry-after": "1", "x-request-id": request_id},
                )
            except ThinkingRejected as exc:
                await metrics.increment("thinking_rejected", "none")
                return _error(400, str(exc), "thinking_not_allowed")
            except ContextLimitExceeded as exc:
                await metrics.increment("context_limit_exceeded", "none")
                return _error(400, str(exc), "context_length_exceeded")
            except RouteRejected as exc:
                await metrics.increment(exc.reason, "none")
                return _error(400, str(exc), exc.reason)
            except TokenizationError as exc:
                await metrics.increment("tokenization_failed", "none")
                return _error(400, str(exc), "invalid_request_error")

            gate = snapshot.gates[decision.pool]
            try:
                await gate.acquire()
            except AdmissionRejected as exc:
                await metrics.increment(f"admission_{exc.reason}", decision.pool)
                return _error(
                    429,
                    f"{decision.backend_id} backend is busy; retry later",
                    "rate_limit_error",
                    headers={
                        "retry-after": str(exc.retry_after_seconds),
                        **_router_headers(decision, request_id),
                    },
                )

            gate_release_owned = True
            backend_payload = dict(payload)
            backend_payload["model"] = config.backend(decision.backend_id).model_uid
            backend_url = f"{config.backend_url}/v1/chat/completions"
            headers = request_headers(
                request.headers.raw,
                backend_api_key=config.backend_api_key,
                request_id=request_id,
            )
            router_headers = _router_headers(decision, request_id)
            stream = bool(payload.get("stream", False))
            try:
                if stream:
                    backend_request = snapshot.client.build_request(
                        "POST", backend_url, headers=headers, json=backend_payload
                    )
                    backend_response = await snapshot.client.send(
                        backend_request, stream=True
                    )
                    if backend_response.status_code >= 400:
                        try:
                            response_body = await backend_response.aread()
                        finally:
                            await backend_response.aclose()
                        await metrics.increment("backend_http_error", decision.pool)
                        return Response(
                            response_body,
                            status_code=backend_response.status_code,
                            headers={
                                **response_headers(backend_response),
                                **router_headers,
                            },
                        )

                    cleanup_task: asyncio.Task[None] | None = None

                    async def release_resources() -> None:
                        nonlocal cleanup_task

                        async def _release() -> None:
                            try:
                                await backend_response.aclose()
                            finally:
                                try:
                                    await gate.release()
                                finally:
                                    await runtime.release(snapshot)
                                    logger.info(
                                        "request_id=%s backend_id=%s "
                                        "elapsed_seconds=%.3f released=true",
                                        request_id,
                                        decision.pool,
                                        time.monotonic() - started,
                                    )

                        if cleanup_task is None:
                            cleanup_task = asyncio.create_task(_release())
                        await asyncio.shield(cleanup_task)

                    async def body_stream() -> AsyncIterator[bytes]:
                        disconnected = False
                        try:
                            async for chunk in backend_response.aiter_raw():
                                if await request.is_disconnected():
                                    disconnected = True
                                    await metrics.increment(
                                        "client_disconnected", decision.pool
                                    )
                                    break
                                yield chunk
                            if not disconnected:
                                await metrics.increment("completed", decision.pool)
                        except asyncio.CancelledError:
                            await metrics.increment("cancelled", decision.pool)
                            raise
                        except Exception:
                            await metrics.increment("stream_error", decision.pool)
                            logger.exception(
                                "request_id=%s backend stream failed", request_id
                            )
                            raise
                        finally:
                            await release_resources()

                    response = StreamingResponse(
                        body_stream(),
                        status_code=backend_response.status_code,
                        media_type=backend_response.headers.get(
                            "content-type", "text/event-stream"
                        ),
                        headers={
                            **response_headers(backend_response),
                            **router_headers,
                        },
                        background=BackgroundTask(release_resources),
                    )
                    gate_release_owned = False
                    runtime_release_owned = False
                    return response

                backend_response = await snapshot.client.post(
                    backend_url, headers=headers, json=backend_payload
                )
                await metrics.increment(
                    (
                        "completed"
                        if backend_response.status_code < 400
                        else "backend_http_error"
                    ),
                    decision.pool,
                )
                return Response(
                    backend_response.content,
                    status_code=backend_response.status_code,
                    headers={**response_headers(backend_response), **router_headers},
                )
            except httpx.TimeoutException:
                await metrics.increment("backend_timeout", decision.pool)
                return _error(
                    504,
                    f"{decision.pool} backend timed out",
                    "backend_timeout",
                    headers=router_headers,
                )
            except httpx.HTTPError:
                await metrics.increment("backend_unavailable", decision.pool)
                logger.exception("request_id=%s backend unavailable", request_id)
                return _error(
                    503,
                    f"{decision.pool} backend unavailable",
                    "backend_unavailable",
                    headers=router_headers,
                )
            finally:
                if gate_release_owned:
                    await gate.release()
                    logger.info(
                        "request_id=%s backend_id=%s elapsed_seconds=%.3f released=true",
                        request_id,
                        decision.pool,
                        time.monotonic() - started,
                    )
        finally:
            if runtime_release_owned:
                await runtime.release(snapshot)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Xinference Token-aware Router")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        create_app(config),
        host=config.listen_host,
        port=config.listen_port,
        log_level=config.log_level.lower(),
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
