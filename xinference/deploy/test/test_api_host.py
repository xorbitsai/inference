# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import io
import json
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import xoscar as xo

from ...core.system_settings_store import SystemSettingsStore
from ..api_host import (
    DIAGNOSTIC_LIMIT,
    APIDiagnostics,
    APIHost,
    _forward_output,
    _wait_process,
)


@pytest.mark.asyncio
async def test_exited_process_does_not_wait_for_descendant_pipe_holder(tmp_path):
    import psutil

    script = (
        "import subprocess,sys\nfrom pathlib import Path\n"
        "p=subprocess.Popen([sys.executable,'-S','-c','import time; time.sleep(60)'])\n"
        f"Path({str(tmp_path / 'holder')!r}).write_text(str(p.pid))\n"
        "print('last parent output',flush=True)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-S", "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    diagnostics = APIDiagnostics()
    try:
        assert await asyncio.wait_for(_wait_process(process, diagnostics), 5) == 0
        assert "last parent output" in diagnostics.read()
        assert process.stdout.closed and process.stderr.closed
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        if (tmp_path / "holder").exists():
            try:
                psutil.Process(int((tmp_path / "holder").read_text())).kill()
            except psutil.NoSuchProcess:
                pass


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

    def list_models(self):
        return {}

    def update_system_settings(self, settings):
        self.settings = settings

    def get_system_settings(self):
        return self.settings


EXTENSION = """
import asyncio, os, sys
from pathlib import Path
from contextlib import asynccontextmanager
from starlette.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

class Gate:
    def __init__(self, app):
        self.app = app
    async def __call__(self, scope, receive, send):
        gate = dict(scope.get("headers", [])).get(b"x-acceptance-gate")
        if gate:
            root = Path(os.environ["API_GATE_ROOT"])
            name = gate.decode()
            (root / (name + ".entered")).touch()
            while not (root / (name + ".release")).exists():
                await asyncio.sleep(.01)
        await self.app(scope, receive, send)

class Extension:
    name = "api-host-acceptance"
    api_version = 1
    def configure_api(self, context):
        context.app.add_middleware(Gate)
        original = context.app.router.lifespan_context
        @asynccontextmanager
        async def lifespan(app):
            async with original(app) as state:
                if os.getenv("FAIL_API_STARTUP"):
                    raise RuntimeError("deliberate ASGI startup failure")
                root = Path(os.environ["API_GATE_ROOT"])
                if (root / "prepare").exists():
                    (root / "prepared").touch()
                    while (root / "prepare").exists():
                        await asyncio.sleep(.01)
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
                # Exercise heartbeat comments without waiting for the ping timer.
                yield {"comment": "ping"}
                yield "finish"
            return EventSourceResponse(chunks())
        async def console() -> JSONResponse:
            from xinference.constants import XINFERENCE_HOME
            print(f"api-stdout-{os.getpid()}", flush=True)
            print(f"api-stderr-{os.getpid()}", file=sys.stderr, flush=True)
            return JSONResponse({"cwd": os.getcwd(), "home": str(Path(XINFERENCE_HOME).resolve())})
        context.router.add_api_route("/acceptance/info", info)
        context.router.add_api_route("/acceptance/stream", stream)
        context.router.add_api_route("/acceptance/console", console)

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
    tmp_path, monkeypatch, capfd
):
    from ...api.oauth2.advanced.auth_service import AdvancedAuthService
    from ...api.oauth2.advanced.crypto import get_password_hash
    from ...core.supervisor import SupervisorActor

    (tmp_path / "acceptance_extension.py").write_text(EXTENSION)
    monkeypatch.syspath_prepend(str(tmp_path))
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
    service_directory = tmp_path / "service"
    service_directory.mkdir()
    monkeypatch.chdir(service_directory)
    monkeypatch.setenv("XINFERENCE_HOME", "home")
    monkeypatch.setenv("XINFERENCE_AUTH_DB_PATH", "identities/auth.db")
    monkeypatch.setenv("XINFERENCE_AUTH_ADVANCED", "true")
    monkeypatch.setenv("XINFERENCE_AUTH_JWT_SECRET_KEY", "api-host-test-secret")
    monkeypatch.setenv("XINFERENCE_AUTH_ENCRYPTION_KEY", "api-host-test-encryption")
    monkeypatch.setenv("XINFERENCE_DISABLE_METRICS", "1")
    monkeypatch.setenv("XINFERENCE_TOKEN_ROUTER_ENABLED", "0")
    monkeypatch.setenv("XINFERENCE_EXTENSIONS", "acceptance_extension:create")
    monkeypatch.setenv("API_STREAM_RELEASE", str(release))
    monkeypatch.setenv("API_GATE_ROOT", str(tmp_path))
    monkeypatch.setenv("XINFERENCE_SYSTEM_SETTINGS_PATH", "home/system-settings.json")
    settings_store = SystemSettingsStore(
        str(service_directory / "home/system-settings.json"),
        environ={"XINFERENCE_MODEL_SRC": "huggingface", "HF_TOKEN": "hf_startup_token"},
    )
    saved = settings_store.get_public()
    saved.update(download_source="modelscope", hf_token="hf_saved_token")
    settings_store.save_public(saved)
    monkeypatch.setenv("XINFERENCE_MODEL_SRC", "modelscope")
    monkeypatch.setenv("HF_TOKEN", "hf_saved_token")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_saved_token")
    auth = AdvancedAuthService(
        str(service_directory / "identities/auth.db"),
        "api-host-test-secret",
        "api-host-test-encryption",
    )
    user_id = auth.db.create_user(
        "existing-admin",
        get_password_hash("password"),
        permissions=["admin", "models:list"],
    )
    api_key = auth.create_api_key_for_user(user_id, name="existing-key")["key"]
    keys = {
        name: auth.create_api_key_for_user(user_id, name=name)
        for name in ("deleted", "disabled", "restricted", "drained")
    }
    credentials = {"username": "existing-admin", "password": "password"}
    token = auth.login(**credentials)["access_token"]
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
        # The service cwd must remain the relative-path base without shadowing
        # the package selected by a replacement interpreter.
        shadow = service_directory / "xinference"
        shadow.mkdir()
        (shadow / "__init__.py").write_text(
            "raise RuntimeError('cwd package shadow')\n"
        )
        host = APIHost(
            listener,
            {
                "supervisor_address": supervisor.address,
                "host": "127.0.0.1",
                "port": port,
            },
            settings_store,
        )
        stopping = asyncio.Event()
        task = asyncio.create_task(host.run(stopping))
        errors = []
        requests = []
        load = None
        pending_requests = []
        try:

            async def ready():
                if task.done():
                    await task
                return host.active is not None and host.active.serving.is_set()

            await until(ready)
            first_pid = host.active.process.pid
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}",
                timeout=10,
                headers={"Authorization": "Bearer " + token},
            ) as client:

                async def check_identity_paths_and_console(pid):
                    login = await client.post("/token", json=credentials)
                    assert login.status_code == 200, login.text
                    models = await client.get(
                        "/v1/models", headers={"Authorization": "Bearer " + api_key}
                    )
                    assert models.status_code == 200, models.text
                    paths = (await client.get("/acceptance/console")).json()
                    assert paths == {
                        "cwd": str(service_directory),
                        "home": str(service_directory / "home"),
                    }
                    stdout, stderr = "", ""

                    async def visible():
                        nonlocal stdout, stderr
                        captured = capfd.readouterr()
                        stdout += captured.out
                        stderr += captured.err
                        return (
                            f"api-stdout-{pid}" in stdout
                            and f"api-stderr-{pid}" in stderr
                        )

                    await until(visible)

                first = (await client.get("/acceptance/info")).json()
                assert first == {"api_pid": first_pid, "backend_pid": backend_pid}
                await check_identity_paths_and_console(first_pid)

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
                    chunks = (
                        line
                        async for line in response.aiter_lines()
                        if line and not line.startswith(":")
                    )
                    assert await chunks.__anext__() == "data: start"
                    (tmp_path / "prepare").touch()
                    response = await client.post(
                        "/v1/api/relaunch",
                        json={
                            "request_id": "first",
                            "python_executable": str(target_python),
                        },
                    )
                    assert response.status_code == 202, response.text
                    operation = response.json()["operation_id"]

                    async def prepared():
                        return (tmp_path / "prepared").exists()

                    await until(prepared)
                    # These commits happen after the target has constructed
                    # its auth service and settings store, before activation.
                    for name, method, body in [
                        ("deleted", "DELETE", None),
                        ("disabled", "PUT", {"enabled": False}),
                        (
                            "restricted",
                            "PUT",
                            {
                                "model_permissions": [
                                    {
                                        "permission_type": "model_id",
                                        "permission_value": "allowed",
                                    }
                                ]
                            },
                        ),
                    ]:
                        result = await client.request(
                            method, f'/v1/admin/keys/{keys[name]["id"]}', json=body
                        )
                        assert result.status_code == 200, result.text
                    settings_url = "/v1/cluster/system_settings"
                    updated = (await client.get(settings_url)).json()
                    updated.update(
                        download_source="openmind_hub", hf_token="hf_prepare_token"
                    )
                    assert (
                        await client.put(settings_url, json=updated)
                    ).status_code == 200
                    late = dict(
                        updated, download_source="csghub", hf_token="hf_drain_token"
                    )
                    pending_requests = [
                        asyncio.create_task(
                            client.delete(
                                f'/v1/admin/keys/{keys["drained"]["id"]}',
                                headers={"x-acceptance-gate": "late-key"},
                            )
                        ),
                        asyncio.create_task(
                            client.put(
                                settings_url,
                                json=late,
                                headers={"x-acceptance-gate": "late-settings"},
                            )
                        ),
                    ]

                    async def admitted():
                        return all(
                            (tmp_path / (name + ".entered")).exists()
                            for name in ("late-key", "late-settings")
                        )

                    await until(admitted)
                    (tmp_path / "prepare").unlink()

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
                    await check_identity_paths_and_console(current["target_pid"])
                    for name in ("deleted", "disabled"):
                        result = await client.get(
                            "/v1/models",
                            headers={"Authorization": "Bearer " + keys[name]["key"]},
                        )
                        assert result.status_code == 401, result.text
                    result = await client.post(
                        "/v1/completions",
                        json={"model": "forbidden", "prompt": "test"},
                        headers={
                            "Authorization": "Bearer " + keys["restricted"]["key"]
                        },
                    )
                    assert result.status_code == 403, result.text
                    assert (await client.get(settings_url)).json()[
                        "download_source"
                    ] == "openmind_hub"
                    # Both old requests were admitted before the switch and
                    # commit only while the source is draining.
                    for name in ("late-key", "late-settings"):
                        (tmp_path / (name + ".release")).touch()
                    for result in await asyncio.gather(*pending_requests):
                        assert result.status_code == 200, result.text
                    result = await client.get(
                        "/v1/models",
                        headers={"Authorization": "Bearer " + keys["drained"]["key"]},
                    )
                    assert result.status_code == 401, result.text
                    public = (await client.get(settings_url)).json()
                    assert public["download_source"] == "csghub"
                    public["download_max_attempts"] += 1
                    assert (
                        await client.put(settings_url, json=public)
                    ).status_code == 200
                    assert settings_store.get().hf_token == "hf_drain_token"
                    assert (
                        await supervisor.get_system_settings()
                        == settings_store.get().to_dict()
                    )
                    result = await client.post(settings_url + "/reset")
                    assert result.json() == settings_store.public_dict(
                        settings_store.get_startup()
                    )
                    assert settings_store.get().download_source == "huggingface"
                    assert not Path(settings_store.path).exists()
                    assert not list(host.directory.glob("*.log"))
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
                    chunks = (
                        line
                        async for line in response.aiter_lines()
                        if line and not line.startswith(":")
                    )
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
            (tmp_path / "prepare").unlink(missing_ok=True)
            for name in ("late-key", "late-settings"):
                (tmp_path / (name + ".release")).touch()
            release.touch()
            stopping.set()
            await asyncio.gather(*pending_requests, return_exceptions=True)
            if load is not None:
                await load
            await asyncio.wait_for(task, 40)
            assert all(
                child.process.poll() is not None for child in host.children.values()
            )


def test_operation_replay_conflict_and_one_inflight_upgrade(tmp_path):
    executable = str(tmp_path / "python")
    host = APIHost(MagicMock(), {}, MagicMock())
    host.active = SimpleNamespace(
        executable=executable, version="version", process=MagicMock(pid=1)
    )
    host.operations["one"] = {
        "request_id": "request",
        "requested_executable": executable,
    }
    assert host.create(executable, "request") == host.operations["one"]
    with pytest.raises(RuntimeError, match="different API relaunch"):
        host.create(str(tmp_path / "other-python"), "request")
    host.operation_task = MagicMock(done=lambda: False)
    with pytest.raises(RuntimeError, match="already in progress"):
        host.create()


def test_diagnostics_are_bounded_without_truncating_service_output():
    diagnostics = APIDiagnostics()
    data = b"old output\n" * DIAGNOSTIC_LIMIT + b"latest failure\n"
    destination = io.StringIO()
    _forward_output(io.BytesIO(data), destination, diagnostics)
    assert destination.getvalue() == data.decode()
    assert diagnostics.read() == data[-DIAGNOSTIC_LIMIT:].decode()


def test_supervisor_cli_preserves_startup_baseline(tmp_path):
    path = tmp_path / "settings.json"
    startup = (
        SystemSettingsStore(
            str(path),
            environ={
                "XINFERENCE_MODEL_SRC": "huggingface",
                "HF_TOKEN": "hf_startup_token",
            },
        )
        .get_startup()
        .to_dict()
    )
    saved = dict(startup, download_source="modelscope", hf_token="hf_saved_token")
    path.write_text(json.dumps({"version": 1, "settings": saved}))
    source_root = str(Path(__file__).resolve().parents[3])
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([source_root, os.environ.get("PYTHONPATH", "")]),
        "XINFERENCE_HOME": str(tmp_path / "home"),
        "XINFERENCE_SYSTEM_SETTINGS_PATH": str(path),
        "XINFERENCE_MODEL_SRC": "huggingface",
        "HF_TOKEN": "hf_startup_token",
        "HUGGING_FACE_HUB_TOKEN": "hf_startup_token",
        "XINFERENCE_EXTENSIONS": "",
        "XINFERENCE_AUTH_ADVANCED": "false",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
from types import SimpleNamespace
from unittest.mock import MagicMock
import xoscar as xo
from xinference.deploy import api_host, cmdline, supervisor

updates = []
async def update(settings):
    updates.append(settings)
async def actor_ref(**kwargs):
    return SimpleNamespace(update_system_settings=update)
xo.actor_ref = actor_ref
supervisor.run_in_subprocess = lambda *args, **kwargs: MagicMock()
supervisor.health_check = lambda **kwargs: True

async def check_host(self, stopping):
    try:
        assert os.environ["XINFERENCE_MODEL_SRC"] == "modelscope"
        assert self.settings_store.get_startup().download_source == "huggingface"
        assert self.settings_store.get().download_source == "modelscope"
        result = await self.settings("reset")
        assert result["download_source"] == "huggingface"
        assert updates[-1]["hf_token"] == "hf_startup_token"
        print("baseline reset verified", flush=True)
    finally:
        self.listener.close()
api_host.APIHost.run = check_host
cmdline.supervisor.main(args=["--host", "127.0.0.1", "--port", "0"], standalone_mode=False)
""",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "baseline reset verified" in result.stdout
    assert not path.exists()
