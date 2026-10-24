# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""A stable service/container process owns successive Worker processes.

Usage: python -m xinference.deploy.worker_host xinference-worker --endpoint ...
The service manager stops this process only on an explicit service stop or when
all registered Workers have exited, rather than when the original Worker retires.
"""

import asyncio
import contextlib
import json
import os
import signal
import sys
import tempfile
from pathlib import Path

import psutil

HOST_SOCKET = "XINFERENCE_WORKER_HOST_SOCKET"
HOST_PID = "XINFERENCE_WORKER_HOST_PID"


async def register_worker_host() -> None:
    address = os.environ.get(HOST_SOCKET)
    if not address:
        return
    reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(address), 5)
    try:
        writer.write(json.dumps({"version": 1, "pid": os.getpid()}).encode() + b"\n")
        await writer.drain()
        if await asyncio.wait_for(reader.readline(), 5) != b"ok\n":
            raise RuntimeError("Worker service host refused registration")
    finally:
        writer.close()
        await writer.wait_closed()


def _enable_subreaper() -> None:
    if sys.platform == "linux":
        import ctypes

        if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0):
            raise OSError(
                ctypes.get_errno(), "Cannot supervise retained model processes"
            )


class ProcessOwner:
    """Track and stop only descendants belonging to this service."""

    def __init__(self):
        self.process = psutil.Process()
        self.workers = {}
        self.descendants = {}

    @staticmethod
    def alive(pid, created):
        try:
            proc = psutil.Process(pid)
            return (
                proc.create_time() == created and proc.status() != psutil.STATUS_ZOMBIE
            )
        except psutil.NoSuchProcess:
            return False

    def remember(self):
        roots = [self.process]
        for pid, created in self.workers.items():
            if self.alive(pid, created):
                with contextlib.suppress(psutil.NoSuchProcess):
                    roots.append(psutil.Process(pid))
        for root in roots:
            with contextlib.suppress(psutil.NoSuchProcess):
                for proc in root.children(recursive=True):
                    if proc.uids().real == os.getuid():
                        self.descendants[proc.pid] = proc.create_time()

    def reap(self, watched=()):
        """Reap adopted children without racing their explicit process watchers."""
        for pid, created in tuple(self.descendants.items()):
            if pid not in watched and not self.alive(pid, created):
                with contextlib.suppress(ChildProcessError, ProcessLookupError):
                    os.waitpid(pid, os.WNOHANG)
                self.descendants.pop(pid, None)

    async def register(self, reader, writer):
        try:
            payload = json.loads(await asyncio.wait_for(reader.readline(), 5))
            proc = psutil.Process(payload["pid"])
            if (
                payload.get("version") != 1
                or proc.uids().real != os.getuid()
                or self.process.pid not in {parent.pid for parent in proc.parents()}
            ):
                raise ValueError("Worker is outside this service")
            self.workers[proc.pid] = proc.create_time()
            self.descendants[proc.pid] = self.workers[proc.pid]
            writer.write(b"ok\n")
            await writer.drain()
        except (ValueError, KeyError, asyncio.TimeoutError, psutil.Error):
            writer.write(b"error\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def stop(self):
        self.remember()
        # Identities come only from this host's children/registered Workers;
        # neither shutdown nor a failed upgrade scans other GPU services.
        for pid, created in self.descendants.items():
            if self.alive(pid, created):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGTERM)
        deadline = asyncio.get_running_loop().time() + 30
        while asyncio.get_running_loop().time() < deadline and any(
            self.alive(pid, created) for pid, created in self.descendants.items()
        ):
            await asyncio.sleep(0.1)
        for pid, created in self.descendants.items():
            if self.alive(pid, created):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)


async def run_worker_host(command):
    if os.name != "posix":
        raise RuntimeError("Worker service hosting requires Linux or macOS")
    if not command:
        raise ValueError("Expected a Worker command")
    _enable_subreaper()
    owner = ProcessOwner()
    own = owner.process
    workers, descendants = owner.workers, owner.descendants
    exits = {}
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stopping.set)

    alive, remember, register = owner.alive, owner.remember, owner.register

    # A private short namespace also works when this process is Docker PID 1.
    with tempfile.TemporaryDirectory(prefix="xinf-host-", dir="/tmp") as directory:
        address = str(Path(directory) / "worker.sock")
        server = await asyncio.start_unix_server(register, address, limit=4096)
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                env={**os.environ, HOST_SOCKET: address, HOST_PID: str(own.pid)},
            )
            workers[process.pid] = psutil.Process(process.pid).create_time()
            while not stopping.is_set():
                remember()
                for pid, created in tuple(workers.items()):
                    if pid not in exits and not alive(pid, created):
                        if pid == process.pid:
                            exits[pid] = await process.wait()
                        else:
                            if sys.platform == "linux":
                                # A replacement can exit before its parent
                                # retires. Wait until the subreaper adopts it.
                                with contextlib.suppress(psutil.NoSuchProcess):
                                    if psutil.Process(pid).ppid() != own.pid:
                                        continue
                            try:
                                waited, status = os.waitpid(pid, os.WNOHANG)
                                if waited:
                                    exits[pid] = os.waitstatus_to_exitcode(status)
                            except ChildProcessError:
                                # macOS reparents grandchildren to launchd.
                                # Unknown termination is a service failure.
                                exits[pid] = 1
                if all(pid in exits for pid in workers):
                    break
                # Reap adopted grandchildren without racing asyncio's watcher
                # for the initial command.
                for pid, created in tuple(descendants.items()):
                    if pid not in workers and not alive(pid, created):
                        with contextlib.suppress(ChildProcessError, ProcessLookupError):
                            os.waitpid(pid, os.WNOHANG)
                    if not alive(pid, created):
                        descendants.pop(pid, None)
                try:
                    await asyncio.wait_for(stopping.wait(), 0.1)
                except asyncio.TimeoutError:
                    pass
            return 0 if stopping.is_set() else exits.get(next(reversed(workers)), 1)
        finally:
            server.close()
            await server.wait_closed()
            await owner.stop()
            if process is not None:
                await process.wait()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.remove_signal_handler(sig)


def main():
    raise SystemExit(asyncio.run(run_worker_host(sys.argv[1:])))


async def run_worker_configuration(config):
    with tempfile.TemporaryDirectory(prefix="xinf-worker-", dir="/tmp") as directory:
        path = Path(directory) / "worker.json"
        path.write_text(json.dumps(config))
        path.chmod(0o600)
        return await run_worker_host(
            [sys.executable, "-m", "xinference.deploy.worker_relaunch", str(path)]
        )


if __name__ == "__main__":
    main()
