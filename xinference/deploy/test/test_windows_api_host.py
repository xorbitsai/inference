# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import os
import socket
import sys
from pathlib import Path

import httpx
import pytest
import xoscar as xo
from xoscar.backends.allocate_strategy import RandomSubPool

from ...core.system_settings_store import SystemSettingsStore
from ..api_host import APIHost
from .test_api_host import EXTENSION, Backend, Supervisor, until


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "win32", reason="Native Windows socket sharing")
@pytest.mark.parametrize("control_lost", [False, True])
async def test_windows_api_cutover_preserves_stream_and_failed_startup(
    tmp_path, monkeypatch, control_lost
):
    (tmp_path / "acceptance_extension.py").write_text(EXTENSION)
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join([str(tmp_path), str(Path(__file__).resolve().parents[3])]),
    )
    monkeypatch.setenv("XINFERENCE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XINFERENCE_EXTENSIONS", "acceptance_extension:create")
    monkeypatch.setenv("XINFERENCE_AUTH_ADVANCED", "false")
    monkeypatch.setenv("XINFERENCE_DISABLE_METRICS", "1")
    monkeypatch.setenv("XINFERENCE_TOKEN_ROUTER_ENABLED", "0")
    monkeypatch.setenv("API_GATE_ROOT", str(tmp_path))
    monkeypatch.setenv("API_STREAM_RELEASE", str(tmp_path / "release"))
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    async with await xo.create_actor_pool("127.0.0.1", n_process=1) as pool:
        backend = await xo.create_actor(
            Backend, address=pool.external_address, allocate_strategy=RandomSubPool()
        )
        backend_pid = await backend.identity()
        supervisor = await xo.create_actor(
            Supervisor, backend, address=pool.external_address, uid="supervisor"
        )
        host = APIHost(
            listener,
            {
                "host": "127.0.0.1",
                "port": port,
                "supervisor_address": supervisor.address,
            },
            SystemSettingsStore(str(tmp_path / "settings.json")),
        )
        stopping = asyncio.Event()
        task = asyncio.create_task(host.run(stopping))
        try:

            async def ready():
                if task.done():
                    await task
                return host.active is not None and host.active.serving.is_set()

            await until(ready, timeout=180)
            source = host.active
            if control_lost:
                send = source.send

                async def fail_drain(command):
                    if command == "drain":
                        raise ConnectionError("Lost source control connection")
                    await send(command)

                monkeypatch.setattr(source, "send", fail_drain)
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}", timeout=180
            ) as client:
                async with client.stream("GET", "/acceptance/stream") as response:
                    lines = response.aiter_lines()
                    assert await lines.__anext__() == "data: start"
                    operation = host.create(request_id="windows-cutover")

                    async def draining():
                        record = host.operations[operation["operation_id"]]
                        assert record["status"] != "failed", record
                        return record["status"] == "draining"

                    await until(draining, timeout=180)
                    assert source.process.poll() is None
                    info = (await client.get("/acceptance/info")).json()
                    assert info["api_pid"] != source.process.pid
                    assert info["backend_pid"] == backend_pid
                    (tmp_path / "release").touch()
                    assert "data: finish" in [line async for line in lines]
                await asyncio.wait_for(asyncio.shield(host.operation_task), 30)
                assert (
                    host.operations[operation["operation_id"]]["status"] == "succeeded"
                )
                active = host.active
                active.environment["FAIL_API_STARTUP"] = "1"
                try:
                    operation = host.create(request_id="failed-startup")
                    await asyncio.wait_for(asyncio.shield(host.operation_task), 180)
                    assert (
                        host.operations[operation["operation_id"]]["status"] == "failed"
                    )
                    assert host.active is active
                    info = (await client.get("/acceptance/info")).json()
                    assert info == {
                        "api_pid": active.process.pid,
                        "backend_pid": backend_pid,
                    }
                finally:
                    active.environment.pop("FAIL_API_STARTUP", None)
                (tmp_path / "release").unlink()
                async with client.stream("GET", "/acceptance/stream") as response:
                    lines = response.aiter_lines()
                    assert await lines.__anext__() == "data: start"
                    stopping.set()

                    async def closing():
                        return active.writer is None or active.writer.is_closing()

                    await until(closing)
                    assert active.process.poll() is None
                    (tmp_path / "release").touch()
                    assert "data: finish" in [line async for line in lines]
        finally:
            (tmp_path / "release").touch()
            stopping.set()
            await asyncio.wait_for(task, 60)
