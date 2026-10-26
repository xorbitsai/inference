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
import secrets
import sys
import tempfile
from pathlib import Path

import psutil

from .process_host import shutdown_signals, start_process

HOST_SOCKET = "XINFERENCE_WORKER_HOST_SOCKET"
HOST_PID = "XINFERENCE_WORKER_HOST_PID"


async def register_worker_host() -> None:
    address = os.environ.get(HOST_SOCKET)
    if not address:
        return
    # Keep accepting the Unix endpoint inherited from an older POSIX host.
    connection = json.loads(address) if address.startswith("{") else None
    if connection is not None:
        opening = asyncio.open_connection("127.0.0.1", connection["port"])
    elif sys.platform != "win32":
        opening = asyncio.open_unix_connection(address)
    else:
        raise RuntimeError("Windows Worker hosting requires a TCP endpoint")
    reader, writer = await asyncio.wait_for(opening, 5)
    try:
        payload = {"version": 1, "pid": os.getpid()}
        if connection is not None:
            payload["token"] = connection["token"]
        writer.write(json.dumps(payload).encode() + b"\n")
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
        self.user = self.process.username() if os.name == "nt" else os.getuid()
        self.workers = {}
        self.descendants = {}
        self.token = secrets.token_urlsafe(32)

    async def serve(self):
        server = await asyncio.start_server(self.register, "127.0.0.1", 0, limit=4096)
        return server, json.dumps(
            {"port": server.sockets[0].getsockname()[1], "token": self.token}
        )

    def same_user(self, process):
        if os.name == "nt":
            return process.username() == self.user
        return process.uids().real == self.user

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
            with contextlib.suppress(psutil.Error):
                for proc in root.children(recursive=True):
                    with contextlib.suppress(psutil.Error):
                        if self.same_user(proc):
                            self.descendants[proc.pid] = proc.create_time()

    def reap(self, watched=()):
        """Reap adopted children without racing their explicit process watchers."""
        for pid, created in tuple(self.descendants.items()):
            if not self.alive(pid, created):
                if pid not in watched and os.name != "nt":
                    with contextlib.suppress(ChildProcessError, ProcessLookupError):
                        os.waitpid(pid, os.WNOHANG)
                self.descendants.pop(pid, None)

    async def register(self, reader, writer):
        try:
            payload = json.loads(await asyncio.wait_for(reader.readline(), 5))
            if not isinstance(payload, dict) or not secrets.compare_digest(
                str(payload.get("token", "")), self.token
            ):
                raise ValueError("Invalid Worker host capability")
            proc = psutil.Process(payload["pid"])
            if (
                payload.get("version") != 1
                or not self.same_user(proc)
                or not any(
                    parent.pid == self.process.pid
                    or (
                        self.workers.get(parent.pid) == parent.create_time()
                        and parent.status() != psutil.STATUS_ZOMBIE
                    )
                    for parent in proc.parents()
                )
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
        self._stop_descendants(force=False)
        deadline = asyncio.get_running_loop().time() + 30
        while asyncio.get_running_loop().time() < deadline and any(
            self.alive(pid, created) for pid, created in self.descendants.items()
        ):
            await asyncio.sleep(0.1)
        self.kill()

    def kill(self):
        self._stop_descendants(force=True)

    def _stop_descendants(self, force):
        self.remember()
        # Identities come only from this host's children/registered Workers.
        for pid, created in self.descendants.items():
            with contextlib.suppress(psutil.NoSuchProcess):
                process = psutil.Process(pid)
                if process.create_time() == created:
                    process.kill() if force else process.terminate()


async def run_worker_host(command):
    if not command:
        raise ValueError("Expected a Worker command")
    _enable_subreaper()
    owner = ProcessOwner()
    exits = {}
    stopping = asyncio.Event()
    with shutdown_signals(stopping.set):
        server, address = await owner.serve()
        process = None
        try:
            process = await start_process(
                command,
                env={
                    **os.environ,
                    HOST_SOCKET: address,
                    HOST_PID: str(owner.process.pid),
                },
            )
            owner.workers[process.pid] = psutil.Process(process.pid).create_time()
            while not stopping.is_set():
                owner.remember()
                for pid, created in tuple(owner.workers.items()):
                    if pid not in exits and not owner.alive(pid, created):
                        if pid == process.pid:
                            exits[pid] = await asyncio.to_thread(process.wait)
                        else:
                            if os.name == "nt":
                                # An orphan's exact exit status is unavailable.
                                exits[pid] = 1
                                continue
                            if sys.platform == "linux":
                                # A replacement can exit before its parent
                                # retires. Wait until the subreaper adopts it.
                                with contextlib.suppress(psutil.NoSuchProcess):
                                    if psutil.Process(pid).ppid() != owner.process.pid:
                                        continue
                            try:
                                waited, status = os.waitpid(pid, os.WNOHANG)
                                if waited:
                                    exits[pid] = os.waitstatus_to_exitcode(status)
                            except ChildProcessError:
                                # macOS reparents grandchildren to launchd.
                                # Unknown termination is a service failure.
                                exits[pid] = 1
                if all(pid in exits for pid in owner.workers):
                    break
                # The initial command is reaped through its Popen handle.
                owner.reap(owner.workers)
                try:
                    await asyncio.wait_for(stopping.wait(), 0.1)
                except asyncio.TimeoutError:
                    pass
            return (
                0 if stopping.is_set() else exits.get(next(reversed(owner.workers)), 1)
            )
        finally:
            server.close()
            await server.wait_closed()
            await owner.stop()
            if process is not None:
                await asyncio.to_thread(process.wait)


def main():
    raise SystemExit(asyncio.run(run_worker_host(sys.argv[1:])))


async def run_worker_configuration(config):
    with tempfile.TemporaryDirectory(prefix="xinf-worker-") as directory:
        path = Path(directory) / "worker.json"
        path.write_text(json.dumps(config))
        path.chmod(0o600)
        return await run_worker_host(
            [sys.executable, "-m", "xinference.deploy.worker_relaunch", str(path)]
        )


if __name__ == "__main__":
    main()
