# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Retain the Supervisor actor endpoint while replacing its control process."""

import asyncio
import contextlib
import copy
import json
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import xoscar as xo
from xoscar.utils import get_next_port

from ..core.supervisor_upgrade import PROTOCOL_VERSION, SupervisorGateway
from ..core.worker_relaunch import validate_relaunch_options
from .api_host import STARTUP_TIMEOUT, APIDiagnostics, _wait_process


@dataclass
class SupervisorProcess:
    process: subprocess.Popen
    ref: Any
    executable: str
    environment: Dict[str, str]
    diagnostics: APIDiagnostics
    exited: asyncio.Task
    version: Optional[str] = None
    shared: bool = False


class SupervisorHost:
    def __init__(self, configuration, directory: Path, settings_store):
        self.configuration = configuration
        self.address = configuration["supervisor_address"]
        self.directory = directory
        self.settings_store = settings_store
        self.pool: Any = None
        self.gateway: Any = None
        self.active: Optional[SupervisorProcess] = None
        self.children: list[SupervisorProcess] = []
        self.operations: Dict[str, Dict[str, Any]] = {}
        self.operation_task: Optional[asyncio.Task] = None

    async def start(self, environment):
        from ..core.cache_tracker import CacheTrackerActor
        from ..core.event import EventCollectorActor
        from ..core.progress_tracker import ProgressTrackerActor
        from ..core.status_guard import StatusGuardActor

        self.pool = await xo.create_actor_pool(address=self.address, n_process=0)
        self.gateway = await xo.create_actor(
            SupervisorGateway, address=self.address, uid="supervisor"
        )
        for cls in (
            CacheTrackerActor,
            EventCollectorActor,
            ProgressTrackerActor,
            StatusGuardActor,
        ):
            await xo.create_actor(cls, address=self.address, uid=cls.default_uid())
        (self.directory / "start_supervisor.py").write_text(
            "from xinference.deploy.supervisor_worker import main\nmain()\n"
        )
        source = self.spawn(sys.executable, environment)
        await self.ready(source)
        await source.ref.activate_supervisor()
        await self.gateway.select_supervisor(source.ref)
        await self.gateway.release_supervisor()
        self.active = source

    def spawn(self, executable, environment):
        identity = uuid.uuid4().hex
        address = f"127.0.0.1:{get_next_port()}"
        path = self.directory / ("supervisor-" + identity + ".json")
        path.write_text(
            json.dumps(
                {
                    "address": address,
                    "service_address": self.address,
                    "settings": self.settings_store.get().to_dict(),
                    "logging_conf": self.configuration.get("logging_conf"),
                    "local_deployment": self.configuration.get("local_deployment"),
                }
            )
        )
        path.chmod(0o600)
        diagnostics = APIDiagnostics()
        process = subprocess.Popen(
            [executable, "-u", str(self.directory / "start_supervisor.py"), str(path)],
            env=environment,
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        child = SupervisorProcess(
            process,
            xo.create_actor_ref(address, "supervisor"),
            executable,
            environment,
            diagnostics,
            asyncio.create_task(_wait_process(process, diagnostics)),
        )
        self.children.append(child)
        return child

    async def ready(self, child):
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while child.process.poll() is None and time.monotonic() < deadline:
            try:
                info = await xo.wait_for(child.ref.supervisor_upgrade_info(), 1)
                if (
                    info["protocol"] != PROTOCOL_VERSION
                    or info["pid"] != child.process.pid
                ):
                    raise ValueError("Unsupported Supervisor handoff protocol")
                if not info.get("ready"):
                    await asyncio.sleep(0.05)
                    continue
                child.version = info["version"]
                return
            except (ConnectionError, OSError, xo.ActorNotExist, asyncio.TimeoutError):
                await asyncio.sleep(0.05)
        raise RuntimeError(
            "Supervisor failed to become ready: " + child.diagnostics.read()[-2000:]
        )

    def create(self, executable=None, request_id=None):
        validate_relaunch_options(executable, request_id)
        for operation in self.operations.values():
            if request_id is not None and operation["request_id"] == request_id:
                if operation["requested_executable"] != executable:
                    raise RuntimeError(
                        "request_id already names a different Supervisor relaunch"
                    )
                return copy.deepcopy(operation)
        if self.active is None:
            raise RuntimeError("Supervisor is not ready")
        if self.operation_task is not None and not self.operation_task.done():
            raise RuntimeError("A Supervisor relaunch is already in progress")
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

    async def stop_child(self, child):
        if child.shared:
            with contextlib.suppress(xo.ActorNotExist):
                await xo.destroy_actor(child.ref)
            return
        if child.process.poll() is None:
            child.process.terminate()
        try:
            await asyncio.wait_for(asyncio.shield(child.exited), 10)
        except asyncio.TimeoutError:
            child.process.kill()
            await child.exited

    async def replace(self, operation):
        source = self.active
        assert source is not None
        target = None
        committed = False
        try:
            environment = source.environment
            if operation["requested_executable"] is not None:
                from ..core.utils import build_subpool_envs_for_python

                environment = build_subpool_envs_for_python(
                    environment, operation["python_executable"]
                )
            target = self.spawn(operation["python_executable"], environment)
            operation["target_pid"] = target.process.pid
            await self.ready(target)
            operation.update(status="preparing", target_version=target.version)
            await xo.wait_for(
                self.gateway.prepare_supervisor_switch(final=False), STARTUP_TIMEOUT
            )
            await xo.wait_for(source.ref.pause_supervisor(), STARTUP_TIMEOUT)
            await xo.wait_for(self.gateway.prepare_supervisor_switch(), STARTUP_TIMEOUT)
            snapshot = await xo.wait_for(
                source.ref.get_supervisor_snapshot(), STARTUP_TIMEOUT
            )
            await xo.wait_for(target.ref.restore_supervisor(snapshot), STARTUP_TIMEOUT)
            await xo.wait_for(
                target.ref.prepare_supervisor_activation(), STARTUP_TIMEOUT
            )
            operation["status"] = "switching"
            await self.gateway.select_supervisor(target.ref)
            self.active = target
            committed = True
            await target.ref.activate_supervisor()
            await self.gateway.release_supervisor()
            operation["status"] = "draining"
            await self.gateway.drain_supervisor_reads(source.ref.address)
            await self.stop_child(source)
            operation.update(status="succeeded", finished_at=time.time())
        except Exception as error:
            if not committed:
                if target is not None:
                    await self.stop_child(target)
                await source.ref.resume_supervisor()
            operation.update(status="failed", error=str(error), finished_at=time.time())
        finally:
            await self.gateway.release_supervisor()

    def handle(self, command, request):
        if command == "supervisor-relaunch":
            return self.create(
                request.get("python_executable"), request.get("request_id")
            )
        if command == "supervisor-get":
            return copy.deepcopy(self.operations[request["operation_id"]])
        if command == "supervisor-list":
            return copy.deepcopy(list(self.operations.values()))
        raise ValueError("Unknown Supervisor host command")

    def check(self):
        if self.active is not None and self.active.process.poll() is not None:
            raise RuntimeError("Serving Supervisor process exited unexpectedly")

    async def stop(self):
        if self.operation_task is not None:
            self.operation_task.cancel()
            await asyncio.gather(self.operation_task, return_exceptions=True)
        for child in self.children:
            await self.stop_child(child)
        if self.pool is not None:
            await self.pool.stop()
