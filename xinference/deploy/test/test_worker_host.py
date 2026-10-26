# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil
import pytest


def _source_environment():
    return dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[3])
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
    )


@pytest.mark.parametrize("replacement", [False, True])
@pytest.mark.parametrize("parent_delay", [0, 0.6])
def test_worker_failure_reaches_service_manager(tmp_path, replacement, parent_delay):
    script = tmp_path / "failed_worker.py"
    script.write_text(
        """
import asyncio, subprocess, sys, time
from pathlib import Path
from xinference.deploy.worker_host import register_worker_host
ready = Path(sys.argv[1])
if sys.argv[2] == "parent":
    subprocess.Popen([sys.executable, __file__, str(ready), "replacement"])
    while not ready.exists(): time.sleep(.02)
    time.sleep(float(sys.argv[3]))
else:
    asyncio.run(register_worker_host())
    ready.write_text("ready")
    time.sleep(.3)
    sys.exit(17)
"""
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "xinference.deploy.worker_host",
            sys.executable,
            str(script),
            str(tmp_path / "ready"),
            "parent" if replacement else "replacement",
            str(parent_delay),
        ],
        env=_source_environment(),
        timeout=60 if sys.platform == "win32" else 20,
    )
    # Linux adopts grandchildren and preserves their exact exit status. macOS
    # cannot waitpid an orphan, but still reports it as a failed service.
    assert result.returncode == (
        1 if replacement and sys.platform in {"darwin", "win32"} else 17
    )


@pytest.mark.parametrize("relaunches", [1, 2])
def test_host_survives_original_worker_exit_and_stops_only_owned_descendants(
    tmp_path, relaunches
):
    script = tmp_path / "worker.py"
    script.write_text(
        """
import asyncio, os, subprocess, sys, time
from pathlib import Path
import psutil
from xinference.deploy.worker_host import register_worker_host
root = Path(sys.argv[1])
generation, relaunches = map(int, sys.argv[2:])
parent = psutil.Process(os.getppid())
if generation:
    asyncio.run(register_worker_host())
if generation < relaunches:
    if generation:
        (root / f"registered-{generation}").touch()
        # Relaunch again only after the original generation has retired.
        while True:
            try:
                if not parent.is_running() or parent.status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(.02)
    subprocess.Popen([sys.executable, __file__, str(root), str(generation + 1), str(relaunches)])
    while not (root / f"registered-{generation + 1}").exists(): time.sleep(.02)
    if not generation:
        (root / "original-exited").write_text(str(os.getpid()))
else:
    model = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (root / "model").write_text(str(model.pid))
    (root / "socket").write_text(os.environ["XINFERENCE_WORKER_HOST_SOCKET"])
    (root / "replacement").write_text(str(os.getpid()))
    (root / f"registered-{generation}").touch()
    time.sleep(60)
"""
    )
    host = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "xinference.deploy.worker_host",
            sys.executable,
            str(script),
            str(tmp_path),
            "0",
            str(relaunches),
        ],
        env=_source_environment(),
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    identities = {}
    try:
        deadline = (
            time.monotonic() + (60 if sys.platform == "win32" else 15) * relaunches
        )
        while not all(
            (tmp_path / name).exists() for name in ("original-exited", "replacement")
        ):
            assert host.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.02)
        for name in ("replacement", "model"):
            pid = int((tmp_path / name).read_text())
            identities[pid] = psutil.Process(pid).create_time()
        time.sleep(0.3)
        assert host.poll() is None

        async def outside_registration():
            connection = json.loads((tmp_path / "socket").read_text())
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", connection["port"]
            )
            writer.write(
                json.dumps(
                    {"version": 1, "pid": os.getpid(), "token": connection["token"]}
                ).encode()
                + b"\n"
            )
            await writer.drain()
            assert await reader.readline() == b"error\n"
            writer.close()
            await writer.wait_closed()

        asyncio.run(outside_registration())
        host.send_signal(signal.SIGTERM)
        host.wait(timeout=10)
        assert unrelated.poll() is None
        for pid, created in identities.items():
            try:
                proc = psutil.Process(pid)
                if os.name == "nt" and proc.create_time() == created:
                    # Closing the host's job starts asynchronous child cleanup.
                    proc.wait(timeout=5)
                assert (
                    proc.create_time() != created
                    or proc.status() == psutil.STATUS_ZOMBIE
                )
            except psutil.NoSuchProcess:
                pass

    finally:
        if host.poll() is None:
            host.kill()
            host.wait()
        unrelated.terminate()
        unrelated.wait()
        for pid, created in identities.items():
            try:
                proc = psutil.Process(pid)
                if proc.create_time() == created:
                    proc.kill()
            except psutil.NoSuchProcess:
                pass


@pytest.mark.skipif(
    os.getenv("XINFERENCE_TEST_WORKER_HOST_DOCKER") != "1",
    reason="Explicit Docker lifecycle acceptance",
)
def test_docker_pid1_adopts_replacement_and_survives_original_exit(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(
        """
import asyncio, os, signal, subprocess, sys, time
from pathlib import Path
from xinference.deploy.worker_host import register_worker_host
root = Path("/evidence")
if len(sys.argv) == 1:
    (root / "initial-parent").write_text(str(os.getppid()))
    subprocess.Popen([sys.executable, __file__, "replacement"])
    while not (root / "ready").exists(): time.sleep(.02)
    (root / "original-exited").write_text("done")
else:
    asyncio.run(register_worker_host())
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    (root / "ready").write_text("ready")
    stopping = False
    def stop(*args):
        global stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    while not stopping:
        (root / "replacement-parent").write_text(str(os.getppid()))
        (root / "heartbeat.tmp").write_text(str(time.time()))
        (root / "heartbeat.tmp").replace(root / "heartbeat")
        time.sleep(.1)
    (root / "stopped").write_text("done")
"""
    )
    name = "xinference-worker-host-" + uuid.uuid4().hex[:12]
    source = Path(__file__).resolve().parents[3]
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "-v",
                f"{source}:/source:ro",
                "-v",
                f"{tmp_path}:/evidence",
                "-e",
                "PYTHONPATH=/source",
                "python:3.12-slim",
                "sh",
                "-c",
                "pip install --disable-pip-version-check psutil 'xoscar>=0.12.2' && exec python -m xinference.deploy.worker_host python /evidence/worker.py",
            ],
            check=True,
            timeout=300,
        )
        deadline = time.monotonic() + 120
        while (
            not (tmp_path / "original-exited").exists()
            or not (tmp_path / "replacement-parent").exists()
            or (tmp_path / "replacement-parent").read_text() != "1"
            or not (tmp_path / "heartbeat").exists()
        ):
            assert time.monotonic() < deadline, subprocess.run(
                ["docker", "logs", name], capture_output=True, text=True
            ).stdout
            time.sleep(0.1)
        assert (tmp_path / "initial-parent").read_text() == "1"
        before = float((tmp_path / "heartbeat").read_text())
        time.sleep(0.3)
        assert float((tmp_path / "heartbeat").read_text()) > before
        subprocess.run(["docker", "stop", "-t", "10", name], check=True, timeout=20)
        assert (tmp_path / "stopped").read_text() == "done"
        result = subprocess.check_output(
            ["docker", "inspect", "-f", "{{.State.ExitCode}}", name], text=True
        )
        assert result.strip() == "0"
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
