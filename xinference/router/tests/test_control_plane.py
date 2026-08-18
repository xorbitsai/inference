from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from xinference.router import control_plane as control_plane_module
from xinference.router.control_plane import (
    RouterControlPlaneClient,
    RouterInstanceNotFound,
)


class FakeMetrics:
    async def summary(self) -> dict[str, int]:
        return {"requests": 3}


class FakeRuntime:
    def __init__(
        self, *, apply_error: Exception | None = None, revision: int = 2
    ) -> None:
        self.metrics = FakeMetrics()
        self.apply_error = apply_error
        self.revision = revision
        self.applied: list[Any] = []

    async def apply(self, config: Any) -> None:
        self.applied.append(config)
        if self.apply_error is not None:
            raise self.apply_error
        self.revision = config.revision

    async def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "revision": self.revision,
            "active_requests": 4,
            "tokenization": {"active": 1, "waiting": 2},
            "pools": {"short": {"active": 3}, "long": {"active": 1}},
        }


def make_client(*, revision: int = 1) -> RouterControlPlaneClient:
    client = RouterControlPlaneClient(
        "http://supervisor",
        "router-1",
        internal_token="internal-secret",
        endpoint="http://router:10080",
        instance_id="instance-1",
    )
    client.revision = revision
    return client


@pytest.mark.asyncio
async def test_ack_advances_revision_only_after_success() -> None:
    payloads: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(__import__("json").loads(request.content))
        return httpx.Response(200, json={"ok": True})

    client = make_client()
    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await client.ack(2, "invalid config")
        assert client.revision == 1
        await client.ack(2)
        assert client.revision == 2
    finally:
        await client._client.aclose()

    assert payloads == [
        {"router_uid": "router-1", "revision": 2, "error": "invalid config"},
        {"router_uid": "router-1", "revision": 2, "error": ""},
    ]


@pytest.mark.asyncio
async def test_instance_scoped_404_reports_lost_registration() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "not found"})

    client = make_client()
    await client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(RouterInstanceNotFound):
            await client.heartbeat(status="ready")
        with pytest.raises(RouterInstanceNotFound):
            await client.ack(2)
    finally:
        await client.aclose()

    assert client.revision == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403, 409, 422, 500])
async def test_heartbeat_non_404_preserves_http_error(status_code: int) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"detail": "failed"})

    client = make_client()
    await client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.heartbeat(status="ready")
    finally:
        await client.aclose()

    assert exc_info.value.response.status_code == status_code


@pytest.mark.asyncio
async def test_register_404_preserves_http_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "router not found"})

    client = make_client()
    await client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.register()
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_run_applies_revision_then_acks_and_heartbeats(monkeypatch) -> None:
    client = make_client()
    runtime: Any = FakeRuntime()
    stop = asyncio.Event()
    config = SimpleNamespace(revision=2)
    acked: list[tuple[int, str]] = []
    heartbeats: list[dict[str, Any]] = []

    async def get_config(after_revision: int = 0) -> dict[str, Any]:
        assert after_revision == 1
        return {"revision": 2}

    async def ack(revision: int, error: str = "") -> dict[str, Any]:
        acked.append((revision, error))
        if not error:
            client.revision = revision
        return {"ok": True}

    async def heartbeat(**kwargs: Any) -> dict[str, Any]:
        heartbeats.append(kwargs)
        stop.set()
        return {"ok": True}

    monkeypatch.setattr(client, "get_config", get_config)
    monkeypatch.setattr(client, "ack", ack)
    monkeypatch.setattr(client, "heartbeat", heartbeat)
    monkeypatch.setattr(
        control_plane_module,
        "config_from_control_plane",
        lambda data, **kwargs: config,
    )
    try:
        await client.run(
            runtime,
            stop,
            poll_interval_seconds=0.01,
            heartbeat_interval_seconds=30,
        )
    finally:
        await client._client.aclose()

    assert runtime.applied == [config]
    assert acked == [(2, "")]
    assert client.revision == 2
    assert heartbeats == [
        {
            "status": "ready",
            "metrics": {"requests": 3},
            "process": {
                "pid": __import__("os").getpid(),
                "revision": 2,
                "active_requests": 4,
                "tokenization": {"active": 1, "waiting": 2},
                "pools": {
                    "short": {"active": 3},
                    "long": {"active": 1},
                },
            },
        }
    ]


@pytest.mark.asyncio
async def test_run_reports_apply_error_without_advancing_revision(monkeypatch) -> None:
    client = make_client()
    runtime: Any = FakeRuntime(apply_error=RuntimeError("worker start failed"))
    stop = asyncio.Event()
    acked: list[tuple[int, str]] = []

    async def get_config(after_revision: int = 0) -> dict[str, Any]:
        assert after_revision == 1
        return {"revision": 2}

    async def ack(revision: int, error: str = "") -> dict[str, Any]:
        acked.append((revision, error))
        return {"ok": True}

    async def heartbeat(**kwargs: Any) -> dict[str, Any]:
        stop.set()
        return {"ok": True}

    monkeypatch.setattr(client, "get_config", get_config)
    monkeypatch.setattr(client, "ack", ack)
    monkeypatch.setattr(client, "heartbeat", heartbeat)
    monkeypatch.setattr(
        control_plane_module,
        "config_from_control_plane",
        lambda data, **kwargs: SimpleNamespace(revision=2),
    )
    try:
        await client.run(runtime, stop, poll_interval_seconds=0.01)
    finally:
        await client._client.aclose()

    assert len(runtime.applied) == 1
    assert acked == [(2, "worker start failed")]
    assert client.revision == 1


@pytest.mark.asyncio
async def test_run_ignores_already_acked_revision(monkeypatch) -> None:
    client = make_client(revision=2)
    runtime: Any = FakeRuntime()
    stop = asyncio.Event()
    acked: list[tuple[int, str]] = []

    async def get_config(after_revision: int = 0) -> dict[str, Any]:
        assert after_revision == 2
        return {"revision": 2}

    async def ack(revision: int, error: str = "") -> dict[str, Any]:
        acked.append((revision, error))
        return {"ok": True}

    async def heartbeat(**kwargs: Any) -> dict[str, Any]:
        stop.set()
        return {"ok": True}

    monkeypatch.setattr(client, "get_config", get_config)
    monkeypatch.setattr(client, "ack", ack)
    monkeypatch.setattr(client, "heartbeat", heartbeat)
    monkeypatch.setattr(
        control_plane_module,
        "config_from_control_plane",
        lambda data, **kwargs: pytest.fail("config must not be rebuilt"),
    )
    try:
        await client.run(runtime, stop, poll_interval_seconds=0.01)
    finally:
        await client._client.aclose()

    assert runtime.applied == []
    assert acked == []


@pytest.mark.asyncio
async def test_run_reregisters_after_heartbeat_404(monkeypatch) -> None:
    client = make_client(revision=2)
    runtime: Any = FakeRuntime(revision=2)
    stop = asyncio.Event()
    registered_revisions: list[int] = []
    heartbeat_attempts: list[dict[str, Any]] = []

    async def get_config(after_revision: int = 0) -> None:
        assert after_revision == 2
        return None

    async def register() -> dict[str, Any]:
        registered_revisions.append(client.revision)
        return {"ok": True}

    async def heartbeat(**kwargs: Any) -> dict[str, Any]:
        heartbeat_attempts.append(kwargs)
        if len(heartbeat_attempts) == 1:
            raise RouterInstanceNotFound("instance lost")
        stop.set()
        return {"ok": True}

    monkeypatch.setattr(client, "get_config", get_config)
    monkeypatch.setattr(client, "register", register)
    monkeypatch.setattr(client, "heartbeat", heartbeat)
    try:
        await client.run(runtime, stop, poll_interval_seconds=0.01)
    finally:
        await client.aclose()

    assert registered_revisions == [2]
    assert len(heartbeat_attempts) == 2
    assert heartbeat_attempts[1]["process"]["revision"] == 2
    assert runtime.applied == []


@pytest.mark.asyncio
async def test_run_reregisters_and_reacks_after_ack_404(monkeypatch) -> None:
    client = make_client(revision=1)
    runtime: Any = FakeRuntime(revision=1)
    stop = asyncio.Event()
    config = SimpleNamespace(revision=2)
    get_config_calls = 0
    registered_revisions: list[int] = []
    acked: list[tuple[int, str]] = []

    async def get_config(after_revision: int = 0) -> dict[str, Any]:
        nonlocal get_config_calls
        get_config_calls += 1
        assert after_revision == 1
        return {"revision": 2}

    async def register() -> dict[str, Any]:
        registered_revisions.append(client.revision)
        return {"ok": True}

    async def ack(revision: int, error: str = "") -> dict[str, Any]:
        acked.append((revision, error))
        if len(acked) == 1:
            raise RouterInstanceNotFound("instance lost")
        client.revision = revision
        return {"ok": True}

    async def heartbeat(**kwargs: Any) -> dict[str, Any]:
        stop.set()
        return {"ok": True}

    monkeypatch.setattr(client, "get_config", get_config)
    monkeypatch.setattr(client, "register", register)
    monkeypatch.setattr(client, "ack", ack)
    monkeypatch.setattr(client, "heartbeat", heartbeat)
    monkeypatch.setattr(
        control_plane_module,
        "config_from_control_plane",
        lambda data, **kwargs: config,
    )
    try:
        await client.run(runtime, stop, poll_interval_seconds=0.01)
    finally:
        await client.aclose()

    assert get_config_calls == 1
    assert runtime.applied == [config]
    assert registered_revisions == [1]
    assert acked == [(2, ""), (2, "")]
    assert client.revision == 2


@pytest.mark.asyncio
async def test_run_reregisters_old_revision_after_error_ack_404(monkeypatch) -> None:
    client = make_client(revision=1)
    runtime: Any = FakeRuntime(
        apply_error=RuntimeError("worker start failed"), revision=1
    )
    stop = asyncio.Event()
    registered_revisions: list[int] = []
    acked: list[tuple[int, str]] = []

    async def get_config(after_revision: int = 0) -> dict[str, Any]:
        assert after_revision == 1
        return {"revision": 2}

    async def register() -> dict[str, Any]:
        registered_revisions.append(client.revision)
        return {"ok": True}

    async def ack(revision: int, error: str = "") -> dict[str, Any]:
        acked.append((revision, error))
        if len(acked) == 1:
            raise RouterInstanceNotFound("instance lost")
        stop.set()
        return {"ok": True}

    async def heartbeat(**kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    monkeypatch.setattr(client, "get_config", get_config)
    monkeypatch.setattr(client, "register", register)
    monkeypatch.setattr(client, "ack", ack)
    monkeypatch.setattr(client, "heartbeat", heartbeat)
    monkeypatch.setattr(
        control_plane_module,
        "config_from_control_plane",
        lambda data, **kwargs: SimpleNamespace(revision=2),
    )
    try:
        await client.run(runtime, stop, poll_interval_seconds=0.01)
    finally:
        await client.aclose()

    assert len(runtime.applied) == 2
    assert registered_revisions == [1]
    assert acked == [
        (2, "worker start failed"),
        (2, "worker start failed"),
    ]
    assert client.revision == 1


@pytest.mark.asyncio
async def test_run_retries_failed_reregistration(monkeypatch) -> None:
    client = make_client(revision=2)
    runtime: Any = FakeRuntime(revision=2)
    stop = asyncio.Event()
    register_attempts = 0
    heartbeat_attempts = 0

    async def get_config(after_revision: int = 0) -> None:
        return None

    async def register() -> dict[str, Any]:
        nonlocal register_attempts
        register_attempts += 1
        if register_attempts == 1:
            raise RuntimeError("supervisor unavailable")
        return {"ok": True}

    async def heartbeat(**kwargs: Any) -> dict[str, Any]:
        nonlocal heartbeat_attempts
        heartbeat_attempts += 1
        if heartbeat_attempts == 1:
            raise RouterInstanceNotFound("instance lost")
        stop.set()
        return {"ok": True}

    monkeypatch.setattr(client, "get_config", get_config)
    monkeypatch.setattr(client, "register", register)
    monkeypatch.setattr(client, "heartbeat", heartbeat)
    try:
        await client.run(runtime, stop, poll_interval_seconds=0.01)
    finally:
        await client.aclose()

    assert register_attempts == 2
    assert heartbeat_attempts == 2


@pytest.mark.asyncio
async def test_run_stops_before_polling(monkeypatch) -> None:
    client = make_client()
    runtime: Any = FakeRuntime()
    stop = asyncio.Event()
    stop.set()

    async def unexpected_get_config(after_revision: int = 0) -> None:
        pytest.fail("stopped control plane must not poll")

    monkeypatch.setattr(client, "get_config", unexpected_get_config)
    try:
        await client.run(runtime, stop, poll_interval_seconds=0.01)
    finally:
        await client._client.aclose()


@pytest.mark.asyncio
async def test_unregister_404_is_idempotent() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "not found"})

    client = make_client()
    await client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    await client.unregister()

    assert client._client.is_closed


@pytest.mark.asyncio
async def test_register_reports_initial_revision_zero() -> None:
    payloads: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(__import__("json").loads(request.content))
        return httpx.Response(200, json={"ok": True})

    client = make_client(revision=0)
    await client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await client.register()
    finally:
        await client.aclose()

    assert payloads == [
        {
            "router_uid": "router-1",
            "instance_id": "instance-1",
            "endpoint": "http://router:10080",
            "version": "1",
            "acked_revision": 0,
        }
    ]
