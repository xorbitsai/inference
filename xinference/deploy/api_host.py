# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Own the API listener independently of each serving interpreter."""

import asyncio
import contextlib
import copy
import json
import os
import secrets
import selectors
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol

from ..core.system_settings_store import SystemSettingsStore
from ..core.worker_relaunch import validate_relaunch_options
from .process_host import (
    pipe_bytes_available,
    run_api_loop,
    shutdown_signals,
    start_process,
)

if TYPE_CHECKING:
    from .supervisor_host import SupervisorHost

API_HOST_ENV = "XINFERENCE_API_HOST"
STARTUP_TIMEOUT = 120
DIAGNOSTIC_LIMIT = 8192


class APIDiagnostics:
    def __init__(self):
        self._tail = b""
        self._lock = threading.Lock()

    def append(self, data: bytes) -> None:
        with self._lock:
            self._tail = (self._tail + data)[-DIAGNOSTIC_LIMIT:]

    def read(self) -> str:
        with self._lock:
            return self._tail.decode(errors="replace")


def _forward_output(
    source,
    destination,
    diagnostics: APIDiagnostics,
    stopping: Optional[threading.Event] = None,
) -> None:
    with source, contextlib.ExitStack() as stack:
        reader = (
            stack.enter_context(selectors.DefaultSelector())
            if stopping is not None and os.name != "nt"
            else None
        )
        if reader is not None:
            reader.register(source, selectors.EVENT_READ)
        while stopping is None or not stopping.is_set():
            if reader is not None:
                if not reader.select(0.1):
                    continue
                data = os.read(source.fileno(), 65536)
            elif stopping is not None:
                available = pipe_bytes_available(source)
                if available < 0:
                    break
                if not available:
                    stopping.wait(0.1)
                    continue
                data = os.read(source.fileno(), min(available, 65536))
            else:
                data = source.read1(65536)
            if not data:
                break
            diagnostics.append(data)
            try:
                stream = getattr(destination, "buffer", None)
                if stream is not None:
                    stream.write(data)
                else:
                    destination.write(data.decode(errors="replace"))
                destination.flush()
            except (OSError, ValueError):
                # Retain diagnostics even if the service collector is closed.
                pass


async def _wait_process(process: subprocess.Popen, diagnostics: APIDiagnostics) -> int:
    stopping = threading.Event()
    waiting = asyncio.create_task(asyncio.to_thread(process.wait))
    forwarding = asyncio.gather(
        asyncio.to_thread(
            _forward_output, process.stdout, sys.stdout, diagnostics, stopping
        ),
        asyncio.to_thread(
            _forward_output, process.stderr, sys.stderr, diagnostics, stopping
        ),
    )
    try:
        exit_code = await waiting
        try:
            await asyncio.wait_for(asyncio.shield(forwarding), 1)
        except asyncio.TimeoutError:
            stopping.set()
            await forwarding
        return exit_code
    finally:
        stopping.set()


async def host_request(command: str, **options: Any) -> Any:
    address = os.environ.get(API_HOST_ENV)
    if not address:
        component = "Supervisor" if command.startswith("supervisor-") else "API"
        raise RuntimeError(
            f"{component} relaunch requires a hosted xinference-local or "
            "xinference-supervisor service"
        )
    connection = json.loads(address)
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection("127.0.0.1", connection["port"], limit=1024 * 1024), 5
    )
    try:
        writer.write(
            json.dumps(
                {"token": connection["token"], "command": command, **options}
            ).encode()
            + b"\n"
        )
        await writer.drain()
        result = json.loads(
            await asyncio.wait_for(
                reader.readline(), None if command == "settings" else 5
            )
        )
        if "error" in result:
            error_type = {"ValueError": ValueError, "KeyError": KeyError}.get(
                result.get("type"), RuntimeError
            )
            raise error_type(result["error"])
        return result["result"]
    finally:
        writer.close()
        await writer.wait_closed()


class ProcessHandle(Protocol):
    pid: int

    def poll(self) -> Optional[int]: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


@dataclass
class APIProcess:
    identity: str
    process: ProcessHandle
    executable: str
    environment: Dict[str, str]
    diagnostics: APIDiagnostics
    exited: asyncio.Task
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    serving: asyncio.Event = field(default_factory=asyncio.Event)
    draining: asyncio.Event = field(default_factory=asyncio.Event)
    writer: Optional[asyncio.StreamWriter] = None
    version: Optional[str] = None

    async def send(self, command: str) -> None:
        if self.writer is None or self.writer.is_closing():
            raise RuntimeError("API process control connection is unavailable")
        self.writer.write(json.dumps({"command": command}).encode() + b"\n")
        await self.writer.drain()

    async def wait(self, event: asyncio.Event) -> None:
        pending = asyncio.create_task(event.wait())
        try:
            done, _ = await asyncio.wait(
                {pending, self.exited},
                timeout=STARTUP_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if pending in done and self.process.poll() is None:
                return
            details = self.diagnostics.read()[-2000:]
            raise RuntimeError("API process failed to become ready: " + details)
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


class APIHost:
    serves_api_inline = False

    def __init__(
        self,
        listener: socket.socket,
        configuration: Dict[str, Any],
        settings_store: SystemSettingsStore,
    ):
        self.listener = listener
        self.configuration = configuration
        self.children: Dict[str, APIProcess] = {}
        self.operations: Dict[str, Dict[str, Any]] = {}
        self.active: Optional[APIProcess] = None
        self.operation_task: Optional[asyncio.Task] = None
        self.token = secrets.token_urlsafe(32)
        self.connection: Dict[str, Any] = {}
        self.directory = Path()
        self.working_directory = Path.cwd()
        self.settings_store = settings_store
        self.settings_lock = asyncio.Lock()
        self.supervisor_host: Optional["SupervisorHost"] = None

    async def settings(self, operation: str, data=None) -> Dict[str, Any]:
        if operation == "baseline":
            return self.settings_store.get_startup().to_dict()
        async with self.settings_lock:
            if operation == "get":
                return self.settings_store.get_public()
            if operation == "save":
                result = self.settings_store.save_public(data)
            elif operation == "reset":
                result = self.settings_store.reset()
            else:
                raise ValueError("Unknown system settings operation")
            import xoscar as xo

            from ..core.supervisor import SupervisorActor

            supervisor = await xo.actor_ref(
                address=self.configuration["supervisor_address"],
                uid=SupervisorActor.default_uid(),
            )
            await supervisor.update_system_settings(self.settings_store.get().to_dict())
            return result

    async def _connection(self, reader, writer):
        child = None
        try:
            request = json.loads(await asyncio.wait_for(reader.readline(), 10))
            if not secrets.compare_digest(str(request.get("token", "")), self.token):
                raise ValueError("Invalid API host capability")
            if request.get("command") == "register":
                child = self.children[request["identity"]]
                if request.get("pid") != child.process.pid or child.writer is not None:
                    raise ValueError("Unexpected API process registration")
                child.writer = writer
                while line := await reader.readline():
                    event = json.loads(line)
                    if event.get("state") == "ready" and event.get("protocol") == 1:
                        child.version = event["version"]
                        child.ready.set()
                    elif event.get("state") == "serving":
                        child.serving.set()
                    elif event.get("state") == "draining":
                        child.draining.set()
                return
            command = request.get("command")
            if command in {"supervisor-relaunch", "supervisor-get", "supervisor-list"}:
                if self.supervisor_host is None:
                    raise RuntimeError(
                        "Supervisor relaunch requires a hosted Supervisor"
                    )
                result = self.supervisor_host.handle(command, request)
            elif command == "relaunch":
                result = self.create(
                    request.get("python_executable"), request.get("request_id")
                )
            elif command == "get":
                result = copy.deepcopy(self.operations[request["operation_id"]])
            elif command == "list":
                result = copy.deepcopy(list(self.operations.values()))
            elif command == "settings":
                try:
                    result = await self.settings(
                        request["operation"], request.get("data")
                    )
                except (ValueError, KeyError):
                    raise
                except Exception as error:
                    raise RuntimeError(str(error)) from error
            else:
                raise ValueError("Unknown API host command")
            writer.write(json.dumps({"result": result}).encode() + b"\n")
            await writer.drain()
        except (ValueError, KeyError, RuntimeError) as error:
            writer.write(
                json.dumps({"error": str(error), "type": type(error).__name__}).encode()
                + b"\n"
            )
            with contextlib.suppress(ConnectionError):
                await writer.drain()
        finally:
            if child is not None and child.writer is writer:
                child.writer = None
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()

    async def spawn(self, executable: str, environment: Dict[str, str]) -> APIProcess:
        identity = uuid.uuid4().hex
        path = self.directory / (identity + ".json")
        path.write_text(
            json.dumps(
                {
                    **self.configuration,
                    "identity": identity,
                    "listener_fd": self.listener.fileno(),
                    "control": self.connection,
                }
            )
        )
        path.chmod(0o600)
        diagnostics = APIDiagnostics()
        process = await start_process(
            [executable, "-u", str(self.directory / "start_api.py"), str(path)],
            listener=self.listener,
            env={**environment, API_HOST_ENV: json.dumps(self.connection)},
            cwd=self.working_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        child = APIProcess(
            identity,
            process,
            executable,
            environment,
            diagnostics,
            asyncio.create_task(_wait_process(process, diagnostics)),
        )
        self.children[identity] = child
        return child

    def create(self, executable=None, request_id=None) -> Dict[str, Any]:
        validate_relaunch_options(executable, request_id)
        if request_id is not None:
            for operation in self.operations.values():
                if operation["request_id"] == request_id:
                    if operation["requested_executable"] != executable:
                        raise RuntimeError(
                            "request_id already names a different API relaunch"
                        )
                    return copy.deepcopy(operation)
        if self.active is None:
            raise RuntimeError("API is not ready")
        if self.operation_task is not None and not self.operation_task.done():
            raise RuntimeError("An API relaunch is already in progress")
        if any(
            child is not self.active and child.process.poll() is None
            for child in self.children.values()
        ):
            raise RuntimeError("A previous API process is still draining")
        identity = uuid.uuid4().hex
        operation = self.operations[identity] = {
            "operation_id": identity,
            "request_id": request_id,
            "requested_executable": executable,
            "python_executable": executable or self.active.executable,
            "source_pid": self.active.process.pid,
            "source_version": self.active.version,
            "status": "starting",
            "created_at": time.time(),
        }
        self.operation_task = asyncio.create_task(self.replace(operation))
        return copy.deepcopy(operation)

    async def replace(self, operation: Dict[str, Any]) -> None:
        source = self.active
        assert source is not None
        target = None
        activated = False
        try:
            environment = source.environment
            if operation["requested_executable"] is not None:
                from ..core.utils import build_subpool_envs_for_python

                environment = build_subpool_envs_for_python(
                    environment, operation["python_executable"]
                )
            target = await self.spawn(operation["python_executable"], environment)
            operation["target_pid"] = target.process.pid
            await target.wait(target.ready)
            operation.update(status="switching", target_version=target.version)
            activated = True
            await target.send("activate")
            await target.wait(target.serving)
            self.active = target
            operation["status"] = "draining"
            try:
                await source.send("drain")
                await source.wait(source.draining)
            except (ConnectionError, RuntimeError):
                # Control EOF also starts draining, including on Windows where
                # Popen.terminate() would immediately cancel active requests.
                if source.writer is not None:
                    source.writer.close()
            await asyncio.shield(source.exited)
            operation.update(status="succeeded", finished_at=time.time())
        except Exception as error:
            operation.update(status="failed", error=str(error), finished_at=time.time())
            if target is not None and self.active is not target:
                if not activated and target.process.poll() is None:
                    target.process.terminate()
                if target.writer is not None:
                    target.writer.close()
                if not activated:
                    try:
                        await asyncio.wait_for(asyncio.shield(target.exited), 5)
                    except asyncio.TimeoutError:
                        target.process.kill()
                        await target.exited
                # Once activation was sent, control EOF drains any admitted
                # requests. A failed preparation has admitted none.

    async def start_components(self, environment: Dict[str, str]) -> None:
        if self.configuration.get("host_supervisor"):
            from .supervisor_host import SupervisorHost

            self.supervisor_host = SupervisorHost(
                self.configuration, self.directory, self.settings_store
            )
            await self.supervisor_host.start(environment)
        self.active = await self.spawn(sys.executable, environment)
        await self.active.wait(self.active.ready)
        await self.active.send("activate")
        await self.active.wait(self.active.serving)

    def check(self) -> None:
        if self.supervisor_host is not None:
            self.supervisor_host.check()
        if self.active is not None and self.active.process.poll() is not None:
            raise RuntimeError("Serving API process exited unexpectedly")

    def force_exit(self) -> None:
        processes = [child.process for child in self.children.values()]
        if self.supervisor_host is not None:
            processes.extend(child.process for child in self.supervisor_host.children)
        for process in processes:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()

    async def run(self, stopping: asyncio.Event) -> None:
        with tempfile.TemporaryDirectory(prefix="xinf-api-") as directory:
            self.directory = Path(directory)
            # A private script keeps the service cwd available to relative
            # configuration paths without putting that cwd ahead of the
            # selected interpreter's installed Xinference package on sys.path.
            (self.directory / "start_api.py").write_text(
                "from xinference.deploy.api_worker import main\nmain()\n"
            )
            server = await asyncio.start_server(self._connection, "127.0.0.1", 0)
            self.connection = {
                "port": server.sockets[0].getsockname()[1],
                "token": self.token,
            }
            try:
                environment = dict(os.environ)
                source_root = str(Path(__file__).resolve().parents[2])
                python_paths = [
                    os.path.abspath(path)
                    for path in environment.get("PYTHONPATH", "").split(os.pathsep)
                    if path
                ]
                environment["PYTHONPATH"] = os.pathsep.join(
                    [source_root, *python_paths]
                )
                await self.start_components(environment)
                while not stopping.is_set():
                    self.check()
                    try:
                        await asyncio.wait_for(stopping.wait(), 0.1)
                    except asyncio.TimeoutError:
                        pass
            finally:
                if self.operation_task is not None:
                    self.operation_task.cancel()
                    await asyncio.gather(self.operation_task, return_exceptions=True)
                server.close()
                for child in self.children.values():
                    if child.process.poll() is None:
                        if child.writer is not None:
                            child.writer.close()
                        else:
                            child.process.terminate()
                for child in self.children.values():
                    try:
                        await asyncio.wait_for(asyncio.shield(child.exited), 30)
                    except asyncio.TimeoutError:
                        child.process.kill()
                        await child.exited
                await server.wait_closed()
                if self.supervisor_host is not None:
                    await self.supervisor_host.stop()
                self.listener.close()


def run(configuration: Dict[str, Any], host_class=APIHost) -> None:
    from uvicorn import Config
    from xoscar.utils import get_next_port

    from ..constants import (
        XINFERENCE_DEFAULT_ENDPOINT_PORT,
        XINFERENCE_SYSTEM_SETTINGS_PATH,
    )
    from ..core.system_settings_store import get_system_settings_store

    try:
        listener = Config(
            None, host=configuration["host"], port=configuration["port"]
        ).bind_socket()
    except SystemExit:
        if configuration["port"] is not XINFERENCE_DEFAULT_ENDPOINT_PORT:
            raise
        configuration = {**configuration, "port": get_next_port()}
        listener = Config(
            None, host=configuration["host"], port=configuration["port"]
        ).bind_socket()
    configuration = {**configuration, "port": listener.getsockname()[1]}
    listener.listen(2048)

    async def serve():
        stopping = asyncio.Event()
        host = host_class(
            listener,
            configuration,
            get_system_settings_store(XINFERENCE_SYSTEM_SETTINGS_PATH),
        )

        def stop():
            if stopping.is_set():
                host.force_exit()
            stopping.set()

        with shutdown_signals(stop):
            await host.run(stopping)

    if host_class.serves_api_inline:
        run_api_loop(serve())
    else:
        asyncio.run(serve())
