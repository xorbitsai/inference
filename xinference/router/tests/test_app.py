import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import Request
from tokenizers import Tokenizer, models, pre_tokenizers

from xinference.router.admission import GateSnapshot
from xinference.router.app import create_app
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
    ) -> None:
        self._reserve_tokens = reserve_tokens
        self._default_output_tokens = default_output_tokens
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
        return TokenBudget(
            prompt_tokens=1,
            output_tokens=int(payload.get("max_tokens", self._default_output_tokens)),
            reserve_tokens=self._reserve_tokens,
            total_tokens=1
            + int(payload.get("max_tokens", self._default_output_tokens))
            + self._reserve_tokens,
            enable_thinking=False,
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


@pytest.mark.asyncio
async def test_stream_connect_error_releases_gate(tmp_path: Path) -> None:
    async def backend_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("backend unavailable", request=request)

    response, active, metrics = await call_router(
        make_config(tmp_path), backend_handler
    )

    assert response.status_code == 503
    assert active == 0
    assert 'event="backend_unavailable",pool="short"} 1' in metrics


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
async def test_completed_stream_releases_gate(tmp_path: Path) -> None:
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
async def test_auth_rejection_releases_runtime_snapshot(tmp_path: Path) -> None:
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


@pytest.mark.asyncio
async def test_v2_tools_rule_routes_to_dynamic_backend_and_sets_headers(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

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
                    "model": "router-model",
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
