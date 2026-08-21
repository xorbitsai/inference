from __future__ import annotations

import asyncio
from argparse import Namespace
from types import SimpleNamespace
from typing import Any

import pytest

from xinference.router import service as service_module


class FakeControlPlaneClient:
    instances: list["FakeControlPlaneClient"] = []
    config_data: dict[str, Any] | None = {"revision": 7}
    get_error: Exception | None = None
    register_error: Exception | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.revision = 0
        self.registered_revision: int | None = None
        self.get_loop: asyncio.AbstractEventLoop | None = None
        self.register_loop: asyncio.AbstractEventLoop | None = None
        self.closed = False
        self.__class__.instances.append(self)

    async def get_config(self) -> dict[str, Any] | None:
        self.get_loop = asyncio.get_running_loop()
        if self.__class__.get_error is not None:
            raise self.__class__.get_error
        return self.__class__.config_data

    async def register(self) -> dict[str, bool]:
        self.register_loop = asyncio.get_running_loop()
        self.registered_revision = self.revision
        if self.__class__.register_error is not None:
            raise self.__class__.register_error
        return {"ok": True}

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def reset_fake_client() -> None:
    FakeControlPlaneClient.instances = []
    FakeControlPlaneClient.config_data = {"revision": 7}
    FakeControlPlaneClient.get_error = None
    FakeControlPlaneClient.register_error = None


def make_args() -> Namespace:
    return Namespace(
        internal_token="internal-secret",
        public_endpoint="http://router:10080",
        host="127.0.0.1",
        port=10080,
        supervisor_url="http://supervisor",
        router_uid="router-1",
        log_level="INFO",
    )


@pytest.mark.asyncio
async def test_load_control_plane_config_registers_unacked_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = object()
    monkeypatch.setattr(
        service_module, "RouterControlPlaneClient", FakeControlPlaneClient
    )
    monkeypatch.setattr(
        service_module,
        "config_from_control_plane",
        lambda data, **kwargs: config,
    )

    loaded, client = await service_module._load_control_plane_config(make_args())

    assert loaded is config
    assert client.revision == 0
    assert client.registered_revision == 0
    assert client.closed is False
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["get", "config", "register", "missing"])
async def test_load_control_plane_config_closes_client_on_failure(
    failure_stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        service_module, "RouterControlPlaneClient", FakeControlPlaneClient
    )

    if failure_stage == "get":
        FakeControlPlaneClient.get_error = RuntimeError("get failed")
    elif failure_stage == "missing":
        FakeControlPlaneClient.config_data = None
    elif failure_stage == "register":
        FakeControlPlaneClient.register_error = RuntimeError("register failed")

    def build_config(data: dict[str, Any], **kwargs: Any) -> object:
        if failure_stage == "config":
            raise ValueError("invalid config")
        return object()

    monkeypatch.setattr(service_module, "config_from_control_plane", build_config)

    with pytest.raises((RuntimeError, ValueError)):
        await service_module._load_control_plane_config(make_args())

    assert len(FakeControlPlaneClient.instances) == 1
    assert FakeControlPlaneClient.instances[0].closed is True


@pytest.mark.asyncio
async def test_serve_uses_one_event_loop_for_control_plane_and_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        listen_host="127.0.0.1", listen_port=10081, log_level="INFO"
    )
    app = SimpleNamespace(state=SimpleNamespace())
    server_loops: list[asyncio.AbstractEventLoop] = []

    class FakeUvicornConfig:
        def __init__(self, configured_app: Any, **kwargs: Any) -> None:
            assert configured_app is app
            assert kwargs == {
                "host": "127.0.0.1",
                "port": 10081,
                "log_level": "info",
                "proxy_headers": True,
            }

    class FakeUvicornServer:
        def __init__(self, configured: FakeUvicornConfig) -> None:
            assert isinstance(configured, FakeUvicornConfig)

        async def serve(self) -> None:
            server_loops.append(asyncio.get_running_loop())

    monkeypatch.setattr(
        service_module, "RouterControlPlaneClient", FakeControlPlaneClient
    )
    monkeypatch.setattr(
        service_module, "config_from_control_plane", lambda data, **kwargs: config
    )
    monkeypatch.setattr(service_module, "create_app", lambda loaded: app)
    monkeypatch.setattr(service_module.uvicorn, "Config", FakeUvicornConfig)
    monkeypatch.setattr(service_module.uvicorn, "Server", FakeUvicornServer)

    await service_module._serve(make_args())

    client = FakeControlPlaneClient.instances[0]
    assert app.state.control_plane is client
    assert client.get_loop is server_loops[0]
    assert client.register_loop is server_loops[0]
    await client.aclose()


@pytest.mark.asyncio
async def test_serve_closes_control_plane_if_app_bootstrap_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        listen_host="127.0.0.1", listen_port=10081, log_level="INFO"
    )
    monkeypatch.setattr(
        service_module, "RouterControlPlaneClient", FakeControlPlaneClient
    )
    monkeypatch.setattr(
        service_module, "config_from_control_plane", lambda data, **kwargs: config
    )

    def fail_create_app(loaded: Any) -> None:
        assert loaded is config
        raise RuntimeError("app bootstrap failed")

    monkeypatch.setattr(service_module, "create_app", fail_create_app)

    with pytest.raises(RuntimeError, match="app bootstrap failed"):
        await service_module._serve(make_args())

    assert len(FakeControlPlaneClient.instances) == 1
    assert FakeControlPlaneClient.instances[0].closed is True
