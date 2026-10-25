# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import json
import os
import socket
import sys
import tempfile
import venv
from pathlib import Path

import pytest

from ..process_host import start_process
from ..worker_host import HOST_SOCKET, ProcessOwner, register_worker_host
from .test_worker_host import _source_environment


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
def test_listener_survives_overlapping_process_generations(tmp_path, use_venv):
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
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        try:
            for generation in ("old", "new"):
                process = start_process(
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
        for token in ("invalid", connection["token"]):
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", connection["port"]
            )
            writer.write(
                json.dumps({"version": 1, "pid": os.getpid(), "token": token}).encode()
                + b"\n"
            )
            await writer.drain()
            assert await reader.readline() == b"error\n"
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
