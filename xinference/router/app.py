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
from typing import Any, AsyncIterator

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
from .logging_config import (
    configure_router_logging,
    normalize_log_level,
    router_access_log_enabled,
    router_log_extra,
    sanitize_log_url,
)
from .metrics import RouterMetrics
from .runtime import RouterDisabled, RouterRuntime, RuntimeSnapshot
from .tokenization import TokenizationWorkerUnavailable
from .tokenizer import TokenizationError

logger = logging.getLogger("xinference.router")


def _backend_mapping(config: RouterConfig) -> dict[str, str]:
    return {backend.id: backend.model_uid for backend in config.backends}


def _route_log_fields(
    config: RouterConfig,
    request_id: str,
    *,
    requested_model: str | None = None,
    decision: RouteDecision | None = None,
    stream: bool | None = None,
    **fields: Any,
) -> dict[str, dict[str, Any]]:
    values = {
        "request_id": request_id,
        "router_uid": config.router_uid,
        "requested_model": requested_model,
        "logical_model": config.logical_model,
        "route_profile": config.route_profile,
        "stream": stream,
        "revision": config.revision,
        **fields,
    }
    if decision is not None:
        backend = config.backend(decision.backend_id)
        values.update(
            {
                "rule_id": decision.rule_id,
                "route_reason": decision.reason,
                "backend_id": decision.backend_id,
                "backend_model_uid": backend.model_uid,
                "prompt_tokens": decision.budget.prompt_tokens,
                "output_tokens": decision.budget.output_tokens,
                "total_budget": decision.budget.total_tokens,
            }
        )
    return router_log_extra(**values)


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
                "Router started",
                extra=router_log_extra(
                    event="router_started",
                    router_uid=current.router_uid,
                    logical_model=current.logical_model,
                    route_profile=current.route_profile,
                    revision=current.revision,
                    enabled=current.enabled,
                    tokenizer_asset_id=current.tokenizer_asset_id,
                    listen_address=f"{current.listen_host}:{current.listen_port}",
                    backend_url=sanitize_log_url(current.backend_url),
                    backend_mapping=_backend_mapping(current),
                ),
            )
            yield
        finally:
            control_stop.set()
            try:
                if control_task is not None:
                    await control_task
            finally:
                try:
                    if control_plane is not None:
                        await control_plane.unregister()
                finally:
                    stopped = runtime.current.config
                    await runtime.aclose()
                    logger.info(
                        "Router stopped",
                        extra=router_log_extra(
                            event="router_stopped",
                            router_uid=stopped.router_uid,
                            logical_model=stopped.logical_model,
                            revision=stopped.revision,
                            outcome="completed",
                        ),
                    )

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
        requested_model: str | None = None
        stream = False
        try:
            snapshot = await runtime.acquire()
        except RouterDisabled:
            current = runtime.current.config
            await metrics.increment("router_disabled", "none")
            logger.warning(
                "Route rejected",
                extra=_route_log_fields(
                    current,
                    request_id,
                    event="route_rejected",
                    status_code=503,
                    outcome="router_disabled",
                    elapsed_seconds=round(time.monotonic() - started, 6),
                ),
            )
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
                logger.warning(
                    "Route rejected",
                    extra=_route_log_fields(
                        config,
                        request_id,
                        event="route_rejected",
                        status_code=401,
                        outcome="auth_rejected",
                        elapsed_seconds=round(time.monotonic() - started, 6),
                    ),
                )
                return _error(
                    401,
                    "Invalid authentication credentials",
                    "authentication_error",
                    headers={"x-request-id": request_id},
                )

            try:
                body = await request.body()
                payload = await request.json()
            except Exception:
                await metrics.increment("invalid_json", "none")
                logger.warning(
                    "Route rejected",
                    extra=_route_log_fields(
                        config,
                        request_id,
                        event="route_rejected",
                        status_code=400,
                        outcome="invalid_json",
                        elapsed_seconds=round(time.monotonic() - started, 6),
                    ),
                )
                return _error(
                    400,
                    "Request body must be valid JSON",
                    "invalid_request_error",
                    headers={"x-request-id": request_id},
                )
            if not isinstance(payload, dict):
                logger.warning(
                    "Route rejected",
                    extra=_route_log_fields(
                        config,
                        request_id,
                        event="route_rejected",
                        status_code=400,
                        outcome="invalid_json",
                        elapsed_seconds=round(time.monotonic() - started, 6),
                    ),
                )
                return _error(
                    400,
                    "Request body must be a JSON object",
                    "invalid_request_error",
                    headers={"x-request-id": request_id},
                )

            raw_model = payload.get("model")
            requested_model = raw_model if isinstance(raw_model, str) else None
            stream = bool(payload.get("stream", False))
            accepted_models = {config.logical_model, *config.model_aliases}
            if raw_model not in accepted_models:
                await metrics.increment("unknown_model", "none")
                logger.warning(
                    "Route rejected",
                    extra=_route_log_fields(
                        config,
                        request_id,
                        requested_model=requested_model,
                        stream=stream,
                        event="route_rejected",
                        status_code=404,
                        outcome="unknown_model",
                        elapsed_seconds=round(time.monotonic() - started, 6),
                    ),
                )
                return _error(
                    404,
                    "Unknown logical model",
                    "model_not_found",
                    headers={"x-request-id": request_id},
                )
            try:
                budget = await snapshot.tokenization.estimate(
                    payload, input_bytes=len(body)
                )
                decision = snapshot.policy.classify(
                    budget,
                    tools_present=bool(payload.get("tools")),
                    stream=stream,
                )
            except AdmissionRejected as exc:
                await metrics.increment(f"tokenization_admission_{exc.reason}", "none")
                logger.warning(
                    "Route rejected",
                    extra=_route_log_fields(
                        config,
                        request_id,
                        requested_model=requested_model,
                        stream=stream,
                        event="route_rejected",
                        status_code=429,
                        outcome=f"tokenization_capacity_{exc.reason}",
                        elapsed_seconds=round(time.monotonic() - started, 6),
                    ),
                )
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
                    "Route rejected because tokenization worker is unavailable",
                    extra=_route_log_fields(
                        config,
                        request_id,
                        requested_model=requested_model,
                        stream=stream,
                        event="route_rejected",
                        status_code=503,
                        outcome="tokenization_worker_unavailable",
                        elapsed_seconds=round(time.monotonic() - started, 6),
                    ),
                )
                return _error(
                    503,
                    "Tokenization worker is unavailable; retry later",
                    "tokenization_worker_unavailable",
                    headers={"retry-after": "1", "x-request-id": request_id},
                )
            except ThinkingRejected as exc:
                await metrics.increment("thinking_rejected", "none")
                outcome = "thinking_rejected"
                status_code = 400
                message = str(exc)
                error_type = "thinking_not_allowed"
            except ContextLimitExceeded as exc:
                await metrics.increment("context_limit_exceeded", "none")
                outcome = "context_limit_exceeded"
                status_code = 400
                message = str(exc)
                error_type = "context_length_exceeded"
            except RouteRejected as exc:
                await metrics.increment(exc.reason, "none")
                outcome = exc.reason
                status_code = 400
                message = str(exc)
                error_type = exc.reason
            except TokenizationError as exc:
                await metrics.increment("tokenization_failed", "none")
                outcome = "tokenization_failed"
                status_code = 400
                message = str(exc)
                error_type = "invalid_request_error"
            else:
                outcome = ""

            if outcome:
                logger.warning(
                    "Route rejected",
                    extra=_route_log_fields(
                        config,
                        request_id,
                        requested_model=requested_model,
                        stream=stream,
                        event="route_rejected",
                        status_code=status_code,
                        outcome=outcome,
                        elapsed_seconds=round(time.monotonic() - started, 6),
                    ),
                )
                return _error(
                    status_code,
                    message,
                    error_type,
                    headers={"x-request-id": request_id},
                )

            backend = config.backend(decision.backend_id)
            logger.info(
                "Route selected",
                extra=_route_log_fields(
                    config,
                    request_id,
                    requested_model=requested_model,
                    decision=decision,
                    stream=stream,
                    event="route_decision",
                ),
            )

            gate = snapshot.gates[decision.pool]
            try:
                await gate.acquire()
            except AdmissionRejected as exc:
                await metrics.increment(f"admission_{exc.reason}", decision.pool)
                logger.warning(
                    "Route rejected",
                    extra=_route_log_fields(
                        config,
                        request_id,
                        requested_model=requested_model,
                        decision=decision,
                        stream=stream,
                        event="route_rejected",
                        status_code=429,
                        outcome=f"backend_capacity_{exc.reason}",
                        elapsed_seconds=round(time.monotonic() - started, 6),
                    ),
                )
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
            backend_payload["model"] = backend.model_uid
            backend_url = f"{config.backend_url}/v1/chat/completions"
            headers = request_headers(
                request.headers.raw,
                backend_api_key=config.backend_api_key,
                request_id=request_id,
            )
            router_headers = _router_headers(decision, request_id)
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
                        logger.warning(
                            "Backend returned an HTTP error",
                            extra=_route_log_fields(
                                config,
                                request_id,
                                requested_model=requested_model,
                                decision=decision,
                                stream=stream,
                                event="backend_error",
                                status_code=backend_response.status_code,
                                outcome="backend_http_error",
                                elapsed_seconds=round(time.monotonic() - started, 6),
                            ),
                        )
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
                        final_outcome = "completed"
                        log_completion = True
                        try:
                            async for chunk in backend_response.aiter_raw():
                                if await request.is_disconnected():
                                    final_outcome = "client_disconnected"
                                    await metrics.increment(
                                        "client_disconnected", decision.pool
                                    )
                                    break
                                yield chunk
                            if final_outcome == "completed":
                                await metrics.increment("completed", decision.pool)
                        except asyncio.CancelledError:
                            final_outcome = "cancelled"
                            await metrics.increment("cancelled", decision.pool)
                            raise
                        except Exception:
                            final_outcome = "stream_error"
                            log_completion = False
                            await metrics.increment("stream_error", decision.pool)
                            logger.exception(
                                "Backend stream failed",
                                extra=_route_log_fields(
                                    config,
                                    request_id,
                                    requested_model=requested_model,
                                    decision=decision,
                                    stream=stream,
                                    event="backend_error",
                                    status_code=backend_response.status_code,
                                    outcome=final_outcome,
                                    elapsed_seconds=round(
                                        time.monotonic() - started, 6
                                    ),
                                ),
                            )
                            raise
                        finally:
                            await release_resources()
                            if log_completion:
                                log_method = (
                                    logger.info
                                    if final_outcome == "completed"
                                    else logger.warning
                                )
                                log_method(
                                    "Route completed",
                                    extra=_route_log_fields(
                                        config,
                                        request_id,
                                        requested_model=requested_model,
                                        decision=decision,
                                        stream=stream,
                                        event="route_completed",
                                        status_code=backend_response.status_code,
                                        outcome=final_outcome,
                                        elapsed_seconds=round(
                                            time.monotonic() - started, 6
                                        ),
                                    ),
                                )

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
                if backend_response.status_code < 400:
                    outcome = "completed"
                    await metrics.increment(outcome, decision.pool)
                    logger.info(
                        "Route completed",
                        extra=_route_log_fields(
                            config,
                            request_id,
                            requested_model=requested_model,
                            decision=decision,
                            stream=stream,
                            event="route_completed",
                            status_code=backend_response.status_code,
                            outcome=outcome,
                            elapsed_seconds=round(time.monotonic() - started, 6),
                        ),
                    )
                else:
                    outcome = "backend_http_error"
                    await metrics.increment(outcome, decision.pool)
                    logger.warning(
                        "Backend returned an HTTP error",
                        extra=_route_log_fields(
                            config,
                            request_id,
                            requested_model=requested_model,
                            decision=decision,
                            stream=stream,
                            event="backend_error",
                            status_code=backend_response.status_code,
                            outcome=outcome,
                            elapsed_seconds=round(time.monotonic() - started, 6),
                        ),
                    )
                return Response(
                    backend_response.content,
                    status_code=backend_response.status_code,
                    headers={**response_headers(backend_response), **router_headers},
                )
            except httpx.TimeoutException:
                await metrics.increment("backend_timeout", decision.pool)
                logger.exception(
                    "Backend timed out",
                    extra=_route_log_fields(
                        config,
                        request_id,
                        requested_model=requested_model,
                        decision=decision,
                        stream=stream,
                        event="backend_error",
                        status_code=504,
                        outcome="backend_timeout",
                        elapsed_seconds=round(time.monotonic() - started, 6),
                    ),
                )
                return _error(
                    504,
                    f"{decision.pool} backend timed out",
                    "backend_timeout",
                    headers=router_headers,
                )
            except httpx.HTTPError:
                await metrics.increment("backend_unavailable", decision.pool)
                logger.exception(
                    "Backend unavailable",
                    extra=_route_log_fields(
                        config,
                        request_id,
                        requested_model=requested_model,
                        decision=decision,
                        stream=stream,
                        event="backend_error",
                        status_code=503,
                        outcome="backend_unavailable",
                        elapsed_seconds=round(time.monotonic() - started, 6),
                    ),
                )
                return _error(
                    503,
                    f"{decision.pool} backend unavailable",
                    "backend_unavailable",
                    headers=router_headers,
                )
            finally:
                if gate_release_owned:
                    await gate.release()
        finally:
            if runtime_release_owned:
                await runtime.release(snapshot)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Xinference Token-aware Router")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    logging_conf = configure_router_logging(
        config.log_level, config.listen_host, config.listen_port
    )
    uvicorn.run(
        create_app(config),
        host=config.listen_host,
        port=config.listen_port,
        log_level=normalize_log_level(config.log_level).lower(),
        log_config=logging_conf,
        access_log=router_access_log_enabled(),
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
