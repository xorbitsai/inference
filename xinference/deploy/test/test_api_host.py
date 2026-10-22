# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
import os
import shlex
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import xoscar as xo

from ..api_host import APIHost


class Backend(xo.StatelessActor):
    def identity(self):
        return os.getpid()


class Supervisor(xo.StatelessActor):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def get_supervisor_info(self):
        return {"pid": os.getpid()}

    def get_backend(self):
        return self.backend


EXTENSION = """
import asyncio, os
from pathlib import Path
from contextlib import asynccontextmanager
from starlette.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

class Extension:
    name = "api-host-acceptance"
    api_version = 1
    def configure_api(self, context):
        original = context.app.router.lifespan_context
        @asynccontextmanager
        async def lifespan(app):
            async with original(app) as state:
                if os.getenv("FAIL_API_STARTUP"):
                    raise RuntimeError("deliberate ASGI startup failure")
                yield state
        context.app.router.lifespan_context = lifespan
        async def info() -> JSONResponse:
            supervisor = await context.get_supervisor_ref()
            backend = await supervisor.get_backend()
            return JSONResponse({"api_pid": os.getpid(), "backend_pid": await backend.identity()})
        async def stream() -> EventSourceResponse:
            async def chunks():
                yield "start"
                while not Path(os.environ["API_STREAM_RELEASE"]).exists():
                    await asyncio.sleep(.01)
                yield "finish"
            return EventSourceResponse(chunks())
        context.router.add_api_route("/acceptance/info", info)
        context.router.add_api_route("/acceptance/stream", stream)

def create(): return Extension()
"""


async def until(function, timeout=30):
    async def wait():
        while True:
            result = await function()
            if result:
                return result
            await asyncio.sleep(0.02)

    return await asyncio.wait_for(wait(), timeout)


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="Inherited API listener requires POSIX")
async def test_api_cutover_preserves_stream_backend_and_failed_startup(
    tmp_path, monkeypatch
):
    from ...core.supervisor import SupervisorActor

    (tmp_path / "acceptance_extension.py").write_text(EXTENSION)
    release = tmp_path / "release"
    source_root = str(Path(__file__).resolve().parents[3])
    pythonpath = os.pathsep.join(
        [str(tmp_path), source_root, os.environ.get("PYTHONPATH", "")]
    )
    target_python = tmp_path / "target-python"
    target_python.write_text(
        "#!/bin/sh\nexport PYTHONPATH="
        + shlex.quote(pythonpath)
        + "\nexec "
        + shlex.quote(sys.executable)
        + ' "$@"\n'
    )
    target_python.chmod(0o700)
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.setenv("XINFERENCE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XINFERENCE_AUTH_ADVANCED", "false")
    monkeypatch.setenv("XINFERENCE_DISABLE_METRICS", "1")
    monkeypatch.setenv("XINFERENCE_TOKEN_ROUTER_ENABLED", "0")
    monkeypatch.setenv("XINFERENCE_EXTENSIONS", "acceptance_extension:create")
    monkeypatch.setenv("API_STREAM_RELEASE", str(release))
    listener = socket.create_server(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    async with await xo.create_actor_pool("127.0.0.1", n_process=1) as pool:
        from xoscar.backends.allocate_strategy import RandomSubPool

        backend = await xo.create_actor(
            Backend, address=pool.external_address, allocate_strategy=RandomSubPool()
        )
        backend_pid = await backend.identity()
        supervisor = await xo.create_actor(
            Supervisor,
            backend,
            address=pool.external_address,
            uid=SupervisorActor.default_uid(),
        )
        host = APIHost(
            listener,
            {
                "supervisor_address": supervisor.address,
                "host": "127.0.0.1",
                "port": port,
            },
        )
        stopping = asyncio.Event()
        task = asyncio.create_task(host.run(stopping))
        errors = []
        requests = []
        load = None
        try:

            async def ready():
                if task.done():
                    await task
                return host.active is not None and host.active.serving.is_set()

            await until(ready)
            first_pid = host.active.process.pid
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}", timeout=10
            ) as client:
                first = (await client.get("/acceptance/info")).json()
                assert first == {"api_pid": first_pid, "backend_pid": backend_pid}

                async def traffic():
                    while not stopping.is_set():
                        try:
                            response = await client.get("/acceptance/info")
                            response.raise_for_status()
                            result = response.json()
                            assert result["backend_pid"] == backend_pid
                            requests.append(result["api_pid"])
                        except Exception as error:
                            errors.append(error)
                        await asyncio.sleep(0.005)

                load = asyncio.create_task(traffic())
                async with client.stream("GET", "/acceptance/stream") as response:
                    chunks = (line async for line in response.aiter_lines() if line)
                    assert await chunks.__anext__() == "data: start"
                    response = await client.post(
                        "/v1/api/relaunch",
                        json={
                            "request_id": "first",
                            "python_executable": str(target_python),
                        },
                    )
                    assert response.status_code == 202, response.text
                    operation = response.json()["operation_id"]

                    async def switched():
                        result = (
                            await client.get(f"/v1/api-relaunches/{operation}")
                        ).json()
                        assert result["status"] != "failed", result
                        return result if result["status"] == "draining" else None

                    current = await until(switched)
                    # SSE shutdown watchers poll every 0.5s in newer versions.
                    await asyncio.sleep(0.7)
                    assert current["target_pid"] != first_pid
                    assert (
                        host.children[
                            next(
                                key
                                for key, child in host.children.items()
                                if child.process.pid == first_pid
                            )
                        ].process.poll()
                        is None
                    )
                    assert (await client.get("/acceptance/info")).json()[
                        "api_pid"
                    ] == current["target_pid"]
                    release.touch()
                    assert [line async for line in chunks] == ["data: finish"]

                async def completed():
                    result = (
                        await client.get(f"/v1/api-relaunches/{operation}")
                    ).json()
                    return result if result["status"] == "succeeded" else None

                await until(completed)
                replay = await client.post(
                    "/v1/api/relaunch",
                    json={
                        "request_id": "first",
                        "python_executable": str(target_python),
                    },
                )
                assert replay.json()["operation_id"] == operation
                assert (await supervisor.get_supervisor_info())["pid"] == os.getpid()
                assert await backend.identity() == backend_pid

                bad = tmp_path / "bad-python"
                bad.write_text(
                    "#!/bin/sh\nexport PYTHONPATH="
                    + shlex.quote(pythonpath)
                    + "\nexport FAIL_API_STARTUP=1\nexec "
                    + shlex.quote(sys.executable)
                    + ' "$@"\n'
                )
                bad.chmod(0o700)
                response = await client.post(
                    "/v1/api/relaunch", json={"python_executable": str(bad)}
                )
                failed_id = response.json()["operation_id"]

                async def failed():
                    result = (
                        await client.get(f"/v1/api-relaunches/{failed_id}")
                    ).json()
                    return result if result["status"] == "failed" else None

                result = await until(failed)
                assert "deliberate ASGI startup failure" in result["error"]
                assert (await client.get("/acceptance/info")).json() == {
                    "api_pid": current["target_pid"],
                    "backend_pid": backend_pid,
                }
                release.unlink()
                async with client.stream("GET", "/acceptance/stream") as response:
                    chunks = (line async for line in response.aiter_lines() if line)
                    assert await chunks.__anext__() == "data: start"
                    second = (await client.post("/v1/api/relaunch", json={})).json()
                    assert second["python_executable"] == str(target_python)

                    async def second_draining():
                        result = (
                            await client.get(
                                f'/v1/api-relaunches/{second["operation_id"]}'
                            )
                        ).json()
                        return result if result["status"] == "draining" else None

                    await until(second_draining)
                    # Stopping the service during drain must still reap every
                    # child, even when the replacement task is cancelled.
                    stopping.set()
                    await asyncio.sleep(0.1)
                    release.touch()
                    assert [line async for line in chunks] == ["data: finish"]
                assert requests and not errors, errors
        finally:
            stopping.set()
            if load is not None:
                await load
            await asyncio.wait_for(task, 40)
            assert all(
                child.process.poll() is not None for child in host.children.values()
            )


def test_operation_replay_conflict_and_one_inflight_upgrade():
    host = APIHost(MagicMock(), {})
    host.active = SimpleNamespace(
        executable="/python", version="version", process=MagicMock(pid=1)
    )
    host.operations["one"] = {
        "request_id": "request",
        "requested_executable": "/python",
    }
    assert host.create("/python", "request") == host.operations["one"]
    with pytest.raises(RuntimeError, match="different API relaunch"):
        host.create("/other", "request")
    host.operation_task = MagicMock(done=lambda: False)
    with pytest.raises(RuntimeError, match="already in progress"):
        host.create()
