# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import json
import os
import signal
import socket
import sys
import tempfile
import threading
import venv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import psutil
import pytest

from ..process_host import bind_listener, start_process
from ..worker_host import HOST_SOCKET, ProcessOwner, register_worker_host
from .test_worker_host import _source_environment


def test_local_health_check_remains_available():
    from ..local import health_check
    from ..utils import health_check as original

    assert health_check is original


@pytest.mark.skipif(
    sys.platform != "win32", reason="Windows exclusive listener semantics"
)
def test_windows_listener_rejects_duplicate_bind():
    with bind_listener("127.0.0.1", 0) as listener:
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(SystemExit):
            bind_listener("127.0.0.1", port)
        with socket.socket() as competitor:
            competitor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            with pytest.raises(OSError):
                competitor.bind(("127.0.0.1", port))


@pytest.mark.parametrize("default", [False, True])
def test_listener_conflict_retries_only_default_port(monkeypatch, default):
    from xoscar import utils

    from ...constants import XINFERENCE_DEFAULT_ENDPOINT_PORT
    from .. import api_host

    calls = []

    def bind(host, port):
        calls.append(port)
        if len(calls) == 1:
            raise SystemExit(1)
        return bind_listener(host, port)

    class Host(api_host.APIHost):
        async def run(self, stopping):
            self.listener.close()

    monkeypatch.setattr(api_host, "bind_listener", bind)
    monkeypatch.setattr(utils, "get_next_port", lambda: 0)
    configuration = {
        "host": "127.0.0.1",
        "port": XINFERENCE_DEFAULT_ENDPOINT_PORT if default else 0,
    }
    if default:
        api_host.run(configuration, host_class=Host)
        assert calls == [XINFERENCE_DEFAULT_ENDPOINT_PORT, 0]
    else:
        with pytest.raises(SystemExit):
            api_host.run(configuration, host_class=Host)
        assert calls == [0]


@pytest.mark.parametrize("force", [False, True])
def test_stop_continues_after_access_denied(monkeypatch, force):
    owner = ProcessOwner()
    denied, child = MagicMock(), MagicMock()
    denied.create_time.return_value = 100
    child.create_time.return_value = 200
    denied.terminate.side_effect = denied.kill.side_effect = psutil.AccessDenied(123)
    owner.descendants = {123: 100, 456: 200}
    monkeypatch.setattr(owner, "remember", lambda: None)
    monkeypatch.setattr(psutil, "Process", lambda pid: denied if pid == 123 else child)
    owner._stop_descendants(force)
    (child.kill if force else child.terminate).assert_called_once_with()


def test_process_scan_rechecks_user_only_for_new_identity(monkeypatch):
    owner = ProcessOwner()
    child = MagicMock(pid=123)
    child.create_time.return_value = 100
    check = MagicMock(return_value=True)
    monkeypatch.setattr(owner, "same_user", check)
    monkeypatch.setattr(owner.process, "children", lambda **kwargs: [child])
    owner.remember()
    owner.remember()
    assert check.call_count == 1
    child.create_time.return_value = 101
    owner.remember()
    assert check.call_count == 2
    assert owner.descendants == {123: 101}


@pytest.mark.parametrize("running", [False, True])
def test_windows_liveness_does_not_query_thread_status(monkeypatch, running):
    from .. import worker_host

    process = MagicMock()
    process.create_time.return_value = 100
    process.is_running.return_value = running
    monkeypatch.setattr(worker_host, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(psutil, "Process", lambda pid: process)
    assert ProcessOwner.alive(123, 100) is running
    process.status.assert_not_called()


@pytest.mark.asyncio
async def test_windows_environment_probe_does_not_block_event_loop(monkeypatch):
    from xoscar.backends.indigen import fate_sharing

    from .. import process_host

    entered, release = threading.Event(), threading.Event()
    process = MagicMock(pid=123)

    def probe(executable, environment):
        entered.set()
        assert release.wait(5), "Environment probe blocked the event loop"
        return executable, environment

    monkeypatch.setattr(process_host, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(process_host, "python_process_environment", probe)
    monkeypatch.setattr(
        process_host,
        "subprocess",
        SimpleNamespace(
            Popen=MagicMock(return_value=process), CREATE_NEW_PROCESS_GROUP=0x200
        ),
    )
    monkeypatch.setattr(psutil, "Process", MagicMock())
    monkeypatch.setattr(fate_sharing, "detect_fate_sharing_support_win32", lambda: True)
    monkeypatch.setattr(
        fate_sharing, "set_kill_child_on_death_win32", lambda process: None
    )
    task = asyncio.create_task(start_process(["python"]))
    try:
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 2), 3)
        assert not task.done()
    finally:
        release.set()
        result = await task
    assert result is process


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "created,status,accepted",
    [
        (100, psutil.STATUS_RUNNING, True),
        (101, psutil.STATUS_RUNNING, False),
        (100, psutil.STATUS_ZOMBIE, False),
    ],
)
async def test_orphan_replacement_requires_live_registered_ancestor(
    monkeypatch, created, status, accepted
):
    owner = ProcessOwner()
    owner.workers[123] = 100
    ancestor = SimpleNamespace(
        pid=123, create_time=lambda: created, status=lambda: status
    )
    child = SimpleNamespace(
        pid=456, parents=lambda: [ancestor], create_time=lambda: 200
    )
    monkeypatch.setattr(owner, "same_user", lambda process: True)
    monkeypatch.setattr(psutil, "Process", lambda pid: child)
    server, address = await owner.serve()
    connection = json.loads(address)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", connection["port"])
        writer.write(
            json.dumps({"version": 1, "pid": child.pid, "token": owner.token}).encode()
            + b"\n"
        )
        await writer.drain()
        assert await reader.readline() == (b"ok\n" if accepted else b"error\n")
        assert (child.pid in owner.workers) == accepted
        writer.close()
        await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


def test_process_scan_skips_inaccessible_descendant(monkeypatch):
    owner = ProcessOwner()
    denied, child = MagicMock(pid=123), MagicMock(pid=456)
    denied.username.side_effect = denied.uids.side_effect = psutil.AccessDenied(123)
    child.username.return_value = child.uids.return_value.real = owner.user
    child.create_time.return_value = 200
    monkeypatch.setattr(owner.process, "children", lambda **kwargs: [denied, child])
    monkeypatch.setattr(
        owner.process,
        "username",
        MagicMock(side_effect=AssertionError("host username must be cached")),
    )
    owner.remember()
    assert owner.descendants == {456: 200}


@pytest.mark.parametrize("inline", [False, True])
def test_only_inline_api_host_uses_api_event_loop(tmp_path, monkeypatch, inline):
    from .. import api_host
    from ..local_host import LocalHost

    selected = []

    def api_loop(coroutine):
        selected.append("selector")
        return asyncio.run(coroutine)

    async def run(host, stopping):
        host.listener.close()

    host_class = LocalHost if inline else api_host.APIHost
    monkeypatch.setattr(host_class, "run", run)
    monkeypatch.setattr(api_host, "run_api_loop", api_loop)
    api_host.run({"host": "127.0.0.1", "port": 0}, host_class=host_class)
    assert selected == (["selector"] if inline else [])


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console signal handlers")
@pytest.mark.parametrize("interrupt", ["SIGINT", "SIGBREAK"])
def test_windows_interrupt_drains_then_forces_and_restores_handlers(interrupt):
    from .. import api_host

    calls = []
    interrupt = getattr(signal, interrupt)
    previous = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGBREAK)
    }

    class Host(api_host.APIHost):
        async def run(self, stopping):
            try:
                signal.raise_signal(interrupt)
                await asyncio.wait_for(stopping.wait(), 1)
                assert not calls
                calls.append("draining")
                signal.raise_signal(interrupt)
                await asyncio.sleep(0)
                assert calls == ["draining", "forced"]
            finally:
                self.listener.close()

        def force_exit(self):
            calls.append("forced")

    api_host.run({"host": "127.0.0.1", "port": 0}, host_class=Host)
    assert all(signal.getsignal(sig) == value for sig, value in previous.items())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "use_venv",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                os.name != "nt", reason="Windows venv redirectors"
            ),
        ),
    ],
)
async def test_listener_survives_overlapping_process_generations(tmp_path, use_venv):
    executable = sys.executable
    if use_venv:
        directory = tmp_path / "venv"
        venv.EnvBuilder(with_pip=False, system_site_packages=True).create(directory)
        executable = str(directory / "Scripts/python.exe")
    script = tmp_path / "listener.py"
    script.write_text(
        "import json,os,sys,time\nfrom pathlib import Path\n"
        "from xinference.deploy.process_host import inherited_listener\n"
        "listener=inherited_listener(int(sys.argv[1]))\n"
        "client,_=listener.accept()\n"
        "client.sendall((json.dumps({'pid':os.getpid(),'prefix':sys.prefix,'executable':sys.executable})+'\\n').encode())\n"
        "while not Path(sys.argv[2]).exists(): time.sleep(.01)\n"
        "client.sendall(b'finished\\n')\nclient.close()\nlistener.close()\n"
    )
    processes, clients = [], []
    with bind_listener("127.0.0.1", 0) as listener:
        listener.listen()
        try:
            for generation in ("old", "new"):
                process = await start_process(
                    [
                        executable,
                        str(script),
                        str(listener.fileno()),
                        str(tmp_path / generation),
                    ],
                    listener=listener,
                    env=_source_environment(),
                )
                processes.append(process)
                client = socket.create_connection(listener.getsockname(), timeout=60)
                stream = client.makefile("rb")
                clients.append((client, stream))
                identity = json.loads(stream.readline())
                assert identity["pid"] == process.pid
                assert Path(identity["prefix"]).resolve() == (
                    directory.resolve() if use_venv else Path(sys.prefix).resolve()
                )
                assert (
                    Path(identity["executable"]).resolve() == Path(executable).resolve()
                )
            for generation, process, (_, stream) in zip(
                ("old", "new"), processes, clients
            ):
                (tmp_path / generation).touch()
                assert stream.readline() == b"finished\n"
                assert process.wait(timeout=10) == 0
        finally:
            for client, stream in clients:
                stream.close()
                client.close()
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)


@pytest.mark.asyncio
async def test_worker_registration_checks_capability_and_ancestry():
    owner = ProcessOwner()
    server, address = await owner.serve()
    connection = json.loads(address)
    try:
        for token, pid in [
            ("invalid", os.getpid()),
            *[
                (connection["token"], pid)
                for pid in (os.getpid(), "invalid", [], {}, 1.5)
            ],
        ]:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", connection["port"]
            )
            writer.write(
                json.dumps({"version": 1, "pid": pid, "token": token}).encode() + b"\n"
            )
            await writer.drain()
            assert await asyncio.wait_for(reader.readline(), 1) == b"error\n"
            writer.close()
            await writer.wait_closed()
        assert not owner.workers
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import asyncio\nfrom xinference.deploy.worker_host import register_worker_host\nasyncio.run(register_worker_host())",
            env={**_source_environment(), HOST_SOCKET: address},
        )
        try:
            assert await asyncio.wait_for(process.wait(), 60) == 0
            assert process.pid in owner.workers
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="Compatibility with an existing Unix host")
async def test_replacement_registers_with_legacy_unix_host(tmp_path, monkeypatch):
    received = []

    async def register(reader, writer):
        received.append(json.loads(await reader.readline()))
        writer.write(b"ok\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    with tempfile.TemporaryDirectory(prefix="xl-", dir="/tmp") as directory:
        address = str(Path(directory) / "worker.sock")
        server = await asyncio.start_unix_server(register, address)
        monkeypatch.setenv(HOST_SOCKET, address)
        try:
            await register_worker_host()
            assert received == [{"version": 1, "pid": os.getpid()}]
        finally:
            server.close()
            await server.wait_closed()
