import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import Request
from tokenizers import Tokenizer, models, pre_tokenizers

from xinference.router.admission import GateSnapshot
from xinference.router.app import create_app
from xinference.router.backend import request_headers
from xinference.router.config import (
    BackendConfig,
    RouteAction,
    RouterConfig,
    RoutingRule,
    RuleMatch,
    TokenizationConfig,
)
from xinference.router.tokenizer import TokenBudget


class FakeTokenizationService:
    def __init__(
        self,
        _tokenizer_path,
        _metrics,
        *,
        reserve_tokens: int,
        default_output_tokens: int,
        max_workers: int,
        max_active: int,
        max_queue: int,
        queue_timeout_seconds: float,
        retry_after_seconds: int,
        tokenizer_asset_files: tuple[str, ...] = (
            "tokenizer.json",
            "encoding/encoding_dsv4.py",
        ),
        expected_asset_fingerprint: str = "",
        expected_asset_revision: str = "",
    ) -> None:
        self._reserve_tokens = reserve_tokens
        self._default_output_tokens = default_output_tokens
        self.asset_fingerprint = "sha256:test-fingerprint"
        self.asset_revision = "0731"
        self._snapshot = GateSnapshot(
            active=0,
            waiting=0,
            max_active=max_active,
            max_queue=max_queue,
        )
        self.worker_pids = tuple(range(1000, 1000 + max_workers))

    async def start(self) -> None:
        return None

    async def estimate(self, payload, *, input_bytes: int) -> TokenBudget:
        thinking = (
            payload.get("enable_thinking")
            or (
                isinstance(payload.get("extra_body"), dict)
                and payload["extra_body"].get("enable_thinking")
            )
            or (
                isinstance(payload.get("chat_template_kwargs"), dict)
                and payload["chat_template_kwargs"].get("enable_thinking")
            )
        )
        return TokenBudget(
            prompt_tokens=1,
            output_tokens=int(payload.get("max_tokens", self._default_output_tokens)),
            reserve_tokens=self._reserve_tokens,
            total_tokens=1
            + int(payload.get("max_tokens", self._default_output_tokens))
            + self._reserve_tokens,
            enable_thinking=bool(thinking),
        )

    async def snapshot(self) -> GateSnapshot:
        return self._snapshot

    async def aclose(self) -> None:
        self.worker_pids = ()


@pytest.fixture(autouse=True)
def fake_tokenization_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "xinference.router.runtime.TokenizationService", FakeTokenizationService
    )


def event_fields(
    caplog: pytest.LogCaptureFixture, event: str
) -> list[dict[str, object]]:
    return [
        record.xinference_fields
        for record in caplog.records
        if getattr(record, "xinference_fields", {}).get("event") == event
    ]


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


def make_assets(tmp_path: Path) -> Path:
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    encoding = tmp_path / "encoding"
    encoding.mkdir()
    (encoding / "encoding_dsv4.py").write_text(
        "def encode_messages(messages, thinking_mode, reasoning_effort=None):\n"
        "    return ' '.join(str(m.get('content', '')) for m in messages)\n"
    )
    return tmp_path


def make_config(tmp_path: Path) -> RouterConfig:
    backends = (
        BackendConfig("short", "short-model", 200, 1, 0, 0, 1),
        BackendConfig("long", "long-model", 1000, 1, 0, 0, 1),
    )
    rules = (
        RoutingRule(
            "thinking-policy",
            100,
            RuleMatch(thinking=True),
            RouteAction(type="route", backend_id="long"),
        ),
        RoutingRule(
            "short-threshold",
            50,
            RuleMatch(total_tokens_lte=100),
            RouteAction(type="route", backend_id="short"),
        ),
        RoutingRule(
            "long-threshold",
            40,
            RuleMatch(total_tokens_gte=101),
            RouteAction(type="route", backend_id="long"),
        ),
    )
    return RouterConfig(
        listen_host="127.0.0.1",
        listen_port=10080,
        backend_url="http://backend",
        backend_api_key="secret",
        require_auth=True,
        logical_model="router-model",
        model_aliases=("router-alias",),
        tokenizer_path=make_assets(tmp_path),
        context_reserve_tokens=0,
        default_output_tokens=8,
        request_timeout_seconds=10,
        connect_timeout_seconds=1,
        tokenization=TokenizationConfig(2, 2, 2, 1, 1),
        backends=backends,
        rules=rules,
        default_action=RouteAction(type="reject", reason="context_length_exceeded"),
        log_level="INFO",
        config_version=1,
        strategy="token_budget",
        legacy_short_threshold_tokens=100,
        legacy_thinking_pool="long",
        tokenizer_asset_id="deepseek-v4-flash-0731",
        tokenizer_asset_revision="0731",
        tokenizer_asset_fingerprint="sha256:test-fingerprint",
    )


async def call_router(
    config: RouterConfig,
    backend_handler,
) -> tuple[httpx.Response, int, str]:
    app = create_app(config)
    async with app.router.lifespan_context(app):
        snapshot = app.state.runtime.current
        await snapshot.client.aclose()
        snapshot.client = httpx.AsyncClient(
            transport=httpx.MockTransport(backend_handler),
            timeout=10,
        )
        app.state.client = snapshot.client
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://router",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer secret"},
                json={
                    "model": "router-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 8,
                    "stream": True,
                },
            )
        active = (await app.state.gates["short"].snapshot()).active
        metrics = await app.state.metrics.render()
    return response, active, metrics


def assert_rejection_metrics(metrics: str, *, router_uid: str, result: str) -> None:
    assert (
        "xinference_token_router_route_requests_total"
        f'{{router_uid="{router_uid}",result="{result}",'
        'route_mode="non_stream",pool="none"} 1' in metrics
    )
    assert (
        f'xinference_token_router_requests_total{{event="{result}",pool="none"}} 1'
        in metrics
    )
    assert 'result="router_error"' not in metrics
    assert 'event="router_error",pool="none"' not in metrics
    assert (
        "xinference_token_router_requests_in_flight"
        f'{{router_uid="{router_uid}",pool="none"}} 0' in metrics
    )


def test_request_headers_promotes_backend_user_credential() -> None:
    headers = request_headers(
        [
            (b"authorization", b"Bearer internal-token"),
            (
                b"x-xinference-backend-authorization",
                b"Bearer external-user-token",
            ),
        ],
        backend_api_key="",
        request_id="request-a",
    )

    assert headers["authorization"] == "Bearer external-user-token"
    assert "x-xinference-backend-authorization" not in headers


@pytest.mark.parametrize(
    "env_name",
    [
        "XINFERENCE_TOKEN_ROUTER_DATA_PLANE_TOKEN",
        "XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN",
    ],
)
def test_request_headers_does_not_forward_internal_token_without_backend_auth(
    monkeypatch: pytest.MonkeyPatch, env_name: str
) -> None:
    monkeypatch.delenv("XINFERENCE_TOKEN_ROUTER_DATA_PLANE_TOKEN", raising=False)
    monkeypatch.delenv("XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN", raising=False)
    monkeypatch.setenv(env_name, "internal-token")

    headers = request_headers(
        [(b"authorization", b"Bearer internal-token")],
        backend_api_key="",
        request_id="request-a",
    )

    assert "authorization" not in headers


def test_request_headers_backend_api_key_has_highest_priority() -> None:
    headers = request_headers(
        [
            (b"authorization", b"Bearer internal-token"),
            (
                b"x-xinference-backend-authorization",
                b"Bearer external-user-token",
            ),
        ],
        backend_api_key="configured-backend-key",
        request_id="request-a",
    )

    assert headers["authorization"] == "Bearer configured-backend-key"
    assert "x-xinference-backend-authorization" not in headers


@pytest.mark.asyncio
async def test_stream_connect_error_releases_gate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    async def backend_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("backend unavailable", request=request)

    response, active, metrics = await call_router(
        make_config(tmp_path), backend_handler
    )

    assert response.status_code == 503
    assert active == 0
    assert 'event="backend_unavailable",pool="short"} 1' in metrics
    errors = event_fields(caplog, "backend_error")
    assert len(errors) == 1
    assert errors[0]["requested_model"] == "router-model"
    assert errors[0]["logical_model"] == "router-model"
    assert errors[0]["backend_id"] == "short"
    assert errors[0]["backend_model_uid"] == "short-model"
    assert errors[0]["outcome"] == "backend_unavailable"
    assert "secret" not in caplog.text
    assert "hello" not in caplog.text


@pytest.mark.asyncio
async def test_stream_backend_http_error_releases_gate(tmp_path: Path) -> None:
    async def backend_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "failed"})

    response, active, metrics = await call_router(
        make_config(tmp_path), backend_handler
    )

    assert response.status_code == 500
    assert active == 0
    assert 'event="backend_http_error",pool="short"} 1' in metrics


@pytest.mark.asyncio
async def test_completed_stream_releases_gate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="xinference.router")

    async def backend_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ChunkStream(
                b'data: {"choices":[]}\n\n',
                b"data: [DONE]\n\n",
            ),
        )

    response, active, metrics = await call_router(
        make_config(tmp_path), backend_handler
    )

    assert response.status_code == 200
    assert response.content.endswith(b"data: [DONE]\n\n")
    assert active == 0
    assert 'event="completed",pool="short"} 1' in metrics
    completed = event_fields(caplog, "route_completed")
    assert len(completed) == 1
    assert completed[0]["outcome"] == "completed"
    assert completed[0]["stream"] is True
    assert completed[0]["backend_model_uid"] == "short-model"


@pytest.mark.asyncio
async def test_client_disconnect_releases_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def disconnected(_request: Request) -> bool:
        return True

    monkeypatch.setattr(Request, "is_disconnected", disconnected)

    async def backend_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ChunkStream(
                b'data: {"choices":[]}\n\n',
                b"data: [DONE]\n\n",
            ),
        )

    response, active, metrics = await call_router(
        make_config(tmp_path), backend_handler
    )

    assert response.status_code == 200
    assert response.content == b""
    assert active == 0
    assert 'event="client_disconnected",pool="short"} 1' in metrics
    assert 'event="completed",pool="short"}' not in metrics


@pytest.mark.asyncio
async def test_disconnect_before_stream_starts_releases_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app(make_config(tmp_path))
    stream_started = False
    stream_closed = False

    class TrackingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            nonlocal stream_started
            stream_started = True
            yield b"data: [DONE]\n\n"

        async def aclose(self) -> None:
            nonlocal stream_closed
            stream_closed = True

    async def backend_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=TrackingStream(),
        )

    async with app.router.lifespan_context(app):
        snapshot = app.state.runtime.current
        await snapshot.client.aclose()
        snapshot.client = httpx.AsyncClient(
            transport=httpx.MockTransport(backend_handler), timeout=10
        )
        gate = snapshot.gates["short"]
        gate_release = AsyncMock(wraps=gate.release)
        runtime_release = AsyncMock(wraps=app.state.runtime.release)
        monkeypatch.setattr(gate, "release", gate_release)
        monkeypatch.setattr(app.state.runtime, "release", runtime_release)

        request_body = json.dumps(
            {
                "model": "router-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 8,
                "stream": True,
            }
        ).encode()
        receive_calls = 0

        async def receive():
            nonlocal receive_calls
            receive_calls += 1
            if receive_calls == 1:
                return {
                    "type": "http.request",
                    "body": request_body,
                    "more_body": False,
                }
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                await asyncio.sleep(3600)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"content-type", b"application/json"),
                (b"authorization", b"Bearer secret"),
            ],
            "client": ("test", 1234),
            "server": ("router", 10080),
        }
        await app(scope, receive, send)

        assert stream_started is False
        assert stream_closed is True
        assert gate_release.await_count == 1
        assert runtime_release.await_count == 1
        assert (await gate.snapshot()).active == 0
        assert snapshot.active_requests == 0


@pytest.mark.asyncio
async def test_health_exposes_process_tokenization_workers(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://router",
        ) as client:
            response = await client.get("/healthz")
            ready_response = await client.get("/readyz")

        assert response.status_code == 200
        body = response.json()
        assert set(body["pools"]) == {"short", "long"}
        assert body["tokenization"]["active"] == 0
        assert body["tokenization"]["waiting"] == 0
        assert body["tokenization"]["max_active"] == 2
        assert len(body["tokenization"]["worker_pids"]) == 2
        assert ready_response.status_code == 200


class FakeControlPlane:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def ack(self, revision: int) -> None:
        self.events.append(f"ack:{revision}")

    async def run(self, runtime, stop: asyncio.Event) -> None:
        self.events.append("run")
        await stop.wait()

    async def unregister(self) -> None:
        self.events.append("unregister")


class BlockingControlPlane(FakeControlPlane):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.started = asyncio.Event()
        self.cancelled = False
        self.run_task: asyncio.Task[None] | None = None

    async def run(self, runtime, stop: asyncio.Event) -> None:
        self.events.append("run")
        self.run_task = asyncio.current_task()
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.mark.asyncio
async def test_lifespan_acks_only_after_runtime_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app(make_config(tmp_path))
    runtime = app.state.runtime
    events: list[str] = []
    control_plane = FakeControlPlane(events)
    app.state.control_plane = control_plane

    async def start() -> None:
        events.append("start")

    monkeypatch.setattr(runtime, "start", start)

    async with app.router.lifespan_context(app):
        assert events[:2] == ["start", "ack:0"]

    assert events[-1] == "unregister"
    assert runtime.current.closed is True


@pytest.mark.asyncio
async def test_lifespan_cancels_control_plane_before_waiting_for_shutdown(
    tmp_path: Path,
) -> None:
    app = create_app(make_config(tmp_path))
    events: list[str] = []
    control_plane = BlockingControlPlane(events)
    app.state.control_plane = control_plane

    async def serve_and_shutdown() -> None:
        async with app.router.lifespan_context(app):
            await asyncio.wait_for(control_plane.started.wait(), 1)

    shutdown_task = asyncio.create_task(serve_and_shutdown())
    await asyncio.wait_for(control_plane.started.wait(), 1)
    await asyncio.sleep(0.05)

    # If lifespan only sets the stop event, it remains stuck awaiting the
    # deliberately blocked control-plane task. Cancel it as a bounded fallback
    # so this test fails without hanging the test process.
    forced_cancel = False
    if not shutdown_task.done():
        forced_cancel = True
        assert control_plane.run_task is not None
        control_plane.run_task.cancel()

    try:
        await shutdown_task
    except asyncio.CancelledError:
        # The forced fallback above cancels the old implementation's control
        # task, which propagates through its lifespan cleanup.
        pass

    assert forced_cancel is False
    assert control_plane.cancelled is True
    assert app.state.runtime.current.closed is True


@pytest.mark.asyncio
async def test_lifespan_start_failure_does_not_ack_and_still_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app(make_config(tmp_path))
    runtime = app.state.runtime
    events: list[str] = []
    app.state.control_plane = FakeControlPlane(events)

    async def fail_start() -> None:
        events.append("start")
        raise RuntimeError("tokenization startup failed")

    monkeypatch.setattr(runtime, "start", fail_start)

    with pytest.raises(RuntimeError, match="tokenization startup failed"):
        async with app.router.lifespan_context(app):
            pytest.fail("lifespan must not yield after startup failure")

    assert events == ["start", "unregister"]
    assert runtime.current.closed is True


@pytest.mark.asyncio
async def test_auth_rejection_releases_runtime_snapshot(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    app = create_app(make_config(tmp_path))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://router",
        ) as client:
            for _ in range(3):
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "router-model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 8,
                        "stream": True,
                    },
                )
                assert response.status_code == 401
                assert app.state.runtime.current.active_requests == 0

        metrics = await app.state.metrics.render()
        assert 'event="auth_rejected",pool="none"} 3' in metrics

    rejected = event_fields(caplog, "route_rejected")
    assert [item["outcome"] for item in rejected] == ["auth_rejected"] * 3
    assert "authorization" not in caplog.text.lower()
    assert "secret" not in caplog.text
    assert "hello" not in caplog.text


@pytest.mark.asyncio
async def test_cancelled_runtime_acquire_does_not_leak_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app(make_config(tmp_path))
    runtime = app.state.runtime
    acquire_started = asyncio.Event()
    original_acquire = runtime.acquire

    async def tracked_acquire():
        acquire_started.set()
        return await original_acquire()

    monkeypatch.setattr(runtime, "acquire", tracked_acquire)

    async with app.router.lifespan_context(app):
        await runtime._lock.acquire()
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://router"
            ) as client:
                request_task = asyncio.create_task(
                    client.post(
                        "/v1/chat/completions",
                        headers={"authorization": "Bearer secret"},
                        json={
                            "model": "router-model",
                            "messages": [{"role": "user", "content": "hello"}],
                            "max_tokens": 8,
                        },
                    )
                )
                await asyncio.wait_for(acquire_started.wait(), timeout=1)
                request_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request_task
        finally:
            runtime._lock.release()

        metrics = await app.state.metrics.render()
        in_flight = [
            float(line.rsplit(" ", 1)[1])
            for line in metrics.splitlines()
            if line.startswith("xinference_token_router_requests_in_flight{")
        ]
        assert all(value == 0 for value in in_flight)
        assert runtime.current.active_requests == 0


@pytest.mark.asyncio
async def test_v2_tools_rule_routes_to_dynamic_backend_and_sets_headers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from dataclasses import replace

    caplog.set_level(logging.INFO, logger="xinference.router")
    config = make_config(tmp_path)
    tools_backend = BackendConfig("tools", "tools-model", 400, 2, 1, 1, 1)
    config = replace(
        config,
        config_version=2,
        strategy="typed_rules",
        backends=(*config.backends, tools_backend),
        rules=(
            RoutingRule(
                "tools-route",
                200,
                RuleMatch(tools_present=True),
                RouteAction(type="route", backend_id="tools"),
            ),
            *config.rules,
        ),
    )
    received_payload = {}

    async def backend_handler(request: httpx.Request) -> httpx.Response:
        received_payload.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"id": "chatcmpl-test", "choices": []},
            headers={"content-type": "application/json"},
        )

    app = create_app(config)
    async with app.router.lifespan_context(app):
        snapshot = app.state.runtime.current
        await snapshot.client.aclose()
        snapshot.client = httpx.AsyncClient(
            transport=httpx.MockTransport(backend_handler), timeout=10
        )
        app.state.client = snapshot.client
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer secret"},
                json={
                    "model": "router-alias",
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": [{"type": "function", "function": {"name": "lookup"}}],
                    "max_tokens": 8,
                    "stream": False,
                },
            )

    assert response.status_code == 200
    assert received_payload["model"] == "tools-model"
    assert response.headers["x-xinference-router-backend"] == "tools"
    assert response.headers["x-xinference-router-rule"] == "tools-route"
    assert response.headers["x-xinference-router-pool"] == "tools"

    decisions = event_fields(caplog, "route_decision")
    assert len(decisions) == 1
    assert decisions[0]["requested_model"] == "router-alias"
    assert decisions[0]["logical_model"] == "router-model"
    assert decisions[0]["backend_id"] == "tools"
    assert decisions[0]["backend_model_uid"] == "tools-model"
    assert decisions[0]["rule_id"] == "tools-route"
    assert decisions[0]["prompt_tokens"] == 1
    assert decisions[0]["output_tokens"] == 8
    assert decisions[0]["total_budget"] == 9

    completed = event_fields(caplog, "route_completed")
    assert len(completed) == 1
    assert completed[0]["request_id"] == decisions[0]["request_id"]
    assert completed[0]["status_code"] == 200
    assert completed[0]["outcome"] == "completed"
    assert completed[0]["stream"] is False
    assert "secret" not in caplog.text
    assert "hello" not in caplog.text


@pytest.mark.asyncio
async def test_tools_request_rejected_when_asset_lacks_tools_capability(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    config = replace(
        make_config(tmp_path),
        tokenizer_asset_capabilities=("chat", "thinking"),
    )
    app = create_app(config)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer secret"},
                json={
                    "model": "router-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": [{"type": "function", "function": {"name": "lookup"}}],
                    "max_tokens": 8,
                },
            )
            metrics = (await client.get("/metrics")).text

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "tools_not_allowed"
    assert_rejection_metrics(
        metrics, router_uid=config.router_uid, result="tools_not_allowed"
    )


@pytest.mark.asyncio
async def test_thinking_request_is_rejected_before_tokenization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    config = replace(
        make_config(tmp_path),
        tokenizer_asset_capabilities=("chat", "tools"),
    )
    app = create_app(config)
    tokenization = app.state.runtime.current.tokenization
    estimate = AsyncMock(side_effect=AssertionError("tokenizer should not run"))
    monkeypatch.setattr(tokenization, "estimate", estimate)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer secret"},
                json={
                    "model": "router-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "chat_template_kwargs": json.dumps({"enable_thinking": True}),
                    "max_tokens": 8,
                },
            )
            metrics = (await client.get("/metrics")).text

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "thinking_not_allowed"
    estimate.assert_not_awaited()
    assert_rejection_metrics(
        metrics, router_uid=config.router_uid, result="thinking_not_allowed"
    )


@pytest.mark.asyncio
async def test_thinking_request_rejected_when_asset_lacks_thinking_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    config = replace(
        make_config(tmp_path),
        tokenizer_asset_capabilities=("chat", "tools"),
    )
    app = create_app(config)
    tokenization = app.state.runtime.current.tokenization
    estimate = AsyncMock(
        return_value=TokenBudget(
            prompt_tokens=1,
            output_tokens=8,
            reserve_tokens=0,
            total_tokens=9,
            enable_thinking=True,
        )
    )
    monkeypatch.setattr(tokenization, "estimate", estimate)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer secret"},
                json={
                    "model": "router-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 8,
                },
            )
            metrics = (await client.get("/metrics")).text

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "thinking_not_allowed"
    estimate.assert_awaited_once()
    assert_rejection_metrics(
        metrics, router_uid=config.router_uid, result="thinking_not_allowed"
    )
