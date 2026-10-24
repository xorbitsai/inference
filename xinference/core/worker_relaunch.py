# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Relaunch jobs outlive the Worker that accepted them."""

import asyncio
import contextlib
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

import xoscar as xo
from xoscar.utils import get_next_port

from .worker_state import WorkerJournalError, read_worker_journal, write_worker_journal

TERMINAL = {"succeeded", "failed"}
STATUSES = {"starting", "transferring", "pending", *TERMINAL}


def validate_relaunch_options(python_executable=None, request_id=None):
    if python_executable is not None and (
        not isinstance(python_executable, str)
        or not os.path.isabs(python_executable)
        or "\x00" in python_executable
    ):
        raise ValueError("python_executable must be an absolute executable path")
    if request_id is not None and (
        not isinstance(request_id, str) or not 1 <= len(request_id) <= 128
    ):
        raise ValueError("request_id must contain between 1 and 128 characters")


class WorkerRelaunchManager:
    def __init__(self, supervisor, path: Optional[Path] = None):
        from ..constants import XINFERENCE_HOME

        self.supervisor = supervisor
        self.path = path or Path(XINFERENCE_HOME) / "worker-relaunches.json"
        payload = read_worker_journal(self.path, {"version": 1, "operations": {}})
        if (
            not isinstance(payload, dict)
            or payload.get("version") != 1
            or not isinstance(payload.get("operations"), dict)
        ):
            raise WorkerJournalError("Invalid or unsupported Worker relaunch journal")
        self.records = payload["operations"]
        required = {
            "operation_id",
            "status",
            "source_worker",
            "source_incarnation",
            "target_worker",
            "created_at",
            "python_executable",
        }
        if any(
            not isinstance(record, dict)
            or not required <= record.keys()
            or record["status"] not in STATUSES
            for record in self.records.values()
        ):
            raise WorkerJournalError("Invalid Worker relaunch journal records")
        self.tasks: Dict[str, asyncio.Task] = {}

    def _commit(self, records):
        write_worker_journal(self.path, {"version": 1, "operations": records})
        self.records = records

    def update(self, operation_id, **fields):
        current = self.records[operation_id]
        if current["status"] in TERMINAL:
            return copy.deepcopy(current)
        if all(current.get(key) == value for key, value in fields.items()):
            return copy.deepcopy(current)
        records = dict(self.records)
        record = records[operation_id] = dict(current, **fields, updated_at=time.time())
        if record["status"] in TERMINAL:
            record["finished_at"] = record["updated_at"]
        self._commit(records)
        return copy.deepcopy(record)

    def get(self, operation_id):
        if operation_id not in self.records:
            raise KeyError("Worker relaunch operation not found")
        return copy.deepcopy(self.records[operation_id])

    def list(self):
        return sorted(
            copy.deepcopy(list(self.records.values())),
            key=lambda r: r["created_at"],
            reverse=True,
        )

    def resume(self):
        for operation_id, record in self.records.items():
            if record["status"] not in TERMINAL and operation_id not in self.tasks:
                self.tasks[operation_id] = asyncio.create_task(
                    self._monitor(operation_id)
                )

    async def create(self, address, python_executable=None, request_id=None):
        validate_relaunch_options(python_executable, request_id)
        registry = self.supervisor._ownership_registry()
        async with registry.lock(address):
            if request_id:
                for record in self.records.values():
                    if record.get("request_id") == request_id:
                        if (record["source_worker"], record["python_executable"]) != (
                            address,
                            python_executable,
                        ):
                            raise RuntimeError(
                                "request_id already belongs to another relaunch"
                            )
                        return self.get(record["operation_id"])
            worker = self.supervisor._worker_address_to_worker.get(address)
            incarnation = registry.incarnations.get(address)
            if worker is not None:
                from .worker_handoff import PROTOCOL_VERSION

                try:
                    protocol = await xo.wait_for(worker.worker_handoff_protocol(), 5)
                    if protocol != PROTOCOL_VERSION:
                        raise ValueError("Unsupported Worker handoff protocol")
                    if incarnation is None:
                        incarnation = await xo.wait_for(worker.worker_incarnation(), 5)
                except AttributeError as error:
                    raise ValueError(
                        "Worker does not support automatic relaunch"
                    ) from error
                except asyncio.TimeoutError as error:
                    raise ValueError(
                        "Timed out checking Worker relaunch support"
                    ) from error
            for record in reversed(list(self.records.values())):
                if record["source_worker"] != address:
                    continue
                if (
                    incarnation is not None
                    and record["source_incarnation"] != incarnation
                ):
                    continue
                if record["status"] not in TERMINAL:
                    if record["python_executable"] != python_executable:
                        raise RuntimeError("Worker already has a relaunch in progress")
                    return self.get(record["operation_id"])
                if worker is None and record["status"] == "succeeded":
                    return self.get(record["operation_id"])
            if worker is None:
                raise ValueError("Worker is not registered")
            if registry.protected(address):
                raise RuntimeError("Worker ownership is already being transferred")
            operation_id = uuid.uuid4().hex
            record = dict(
                operation_id=operation_id,
                request_id=request_id,
                source_worker=address,
                source_incarnation=incarnation,
                target_worker=None,
                target_incarnation=None,
                target_pid=None,
                python_executable=python_executable,
                status="starting",
                error="",
                created_at=time.time(),
                updated_at=time.time(),
                finished_at=None,
            )
            records = dict(self.records)
            # Keep active jobs and the last 100 completed jobs.
            completed = [
                key for key, value in records.items() if value["status"] in TERMINAL
            ]
            for key in completed[:-100]:
                records.pop(key)
            records[operation_id] = record
            self._commit(records)
            self.resume()
            return copy.deepcopy(record)

    def plan(self, operation_id, source, incarnation, target):
        record = self.records[operation_id]
        if record["status"] in TERMINAL:
            raise RuntimeError("Worker relaunch operation has already finished")
        if (record["source_worker"], record["source_incarnation"]) != (
            source,
            incarnation,
        ):
            raise RuntimeError("Worker relaunch source changed")
        if record["target_worker"] not in (None, target):
            raise RuntimeError("Worker relaunch target changed")
        return self.update(operation_id, target_worker=target)

    async def report(self, operation_id, target, incarnation, status, error=""):
        if status not in STATUSES:
            raise ValueError("Invalid worker relaunch status")
        record = self.records[operation_id]
        if record["target_worker"] != target:
            raise RuntimeError("Worker relaunch target changed")
        if record["target_incarnation"] not in (None, incarnation):
            raise RuntimeError("Worker relaunch incarnation changed")
        pid = record.get("target_pid")
        if incarnation is not None and record["target_incarnation"] is None:
            worker = await self._worker(target)
            if await xo.wait_for(worker.worker_incarnation(), 5) != incarnation:
                raise RuntimeError("Worker relaunch incarnation changed")
            pid = (await xo.wait_for(worker.worker_relaunch_state(operation_id), 5))[
                "pid"
            ]
        return self.update(
            operation_id,
            target_incarnation=incarnation,
            target_pid=pid,
            status=status,
            error=error,
        )

    async def _worker(self, address):
        from .worker import WorkerActor

        return await xo.actor_ref(address=address, uid=WorkerActor.default_uid())

    async def _monitor(self, operation_id):
        try:
            while self.records[operation_id]["status"] not in TERMINAL:
                record = self.records[operation_id]
                try:
                    if record["target_worker"]:
                        try:
                            target = await self._worker(record["target_worker"])
                            state = await xo.wait_for(
                                target.worker_relaunch_state(operation_id), 5
                            )
                            if state["status"] == "succeeded":
                                await self.report(
                                    operation_id,
                                    record["target_worker"],
                                    state["incarnation"],
                                    "succeeded",
                                )
                                continue
                        except Exception:
                            pass
                    source = await self._worker(record["source_worker"])
                    if (
                        await xo.wait_for(source.worker_incarnation(), 5)
                        != record["source_incarnation"]
                    ):
                        if not self.records[operation_id]["target_worker"]:
                            self.update(
                                operation_id,
                                status="failed",
                                error="Original Worker incarnation changed before launch",
                            )
                            continue
                        raise RuntimeError("Original Worker is no longer available")
                    if not record["target_worker"]:
                        state = await xo.wait_for(
                            source.launch_worker_replacement(
                                operation_id, record["python_executable"]
                            ),
                            15,
                        )
                        if state.get("error"):
                            self.update(
                                operation_id, status="failed", error=state["error"]
                            )
                    else:
                        state = await xo.wait_for(
                            source.worker_relaunch_process(operation_id), 5
                        )
                        if (
                            state["exit_code"] is not None
                            and state["transfer"] == "idle"
                        ):
                            self.update(
                                operation_id,
                                status="failed",
                                error=state.get("error")
                                or f"Replacement exited with code {state['exit_code']}",
                            )
                except Exception as error:
                    # Losing a control connection is not proof that transfer failed.
                    self.update(
                        operation_id,
                        status=(
                            "failed"
                            if isinstance(error, xo.ActorNotExist)
                            and not self.records[operation_id]["target_worker"]
                            else "pending"
                        ),
                        error=str(error) or type(error).__name__,
                    )
                await asyncio.sleep(0.5)
        finally:
            self.tasks.pop(operation_id, None)

    async def close(self):
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class WorkerRelaunchLauncher:
    def __init__(self, worker):
        self.worker = worker
        self.lock = asyncio.Lock()
        self.operation_id = None
        self.process = None
        self._reaper = None
        self.target = None
        self.errors = {}

    async def launch(self, operation_id, python_executable):
        from ..constants import XINFERENCE_HOME
        from ..deploy.utils import get_log_file
        from ..device_utils import get_available_device_env_name
        from .lifecycle import WorkerPhase

        validate_relaunch_options(python_executable)
        async with self.lock:
            worker = self.worker
            if self.operation_id == operation_id and self.process is not None:
                return self.state(operation_id)
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("Worker already has a replacement process")
            if (
                worker._lifecycle.phase != WorkerPhase.ACTIVE
                or not worker._services_started
            ):
                raise RuntimeError("Worker is not available for relaunch")
            supervisor = await worker.check_worker_handoff_support()
            executable = python_executable or sys.executable
            if not os.path.isfile(executable) or not os.access(executable, os.X_OK):
                raise ValueError("Python executable is not available on the Worker")
            host = worker.address.rsplit(":", 1)[0]
            target = f"{host}:{get_next_port()}"
            config = dict(
                address=target,
                supervisor_address=worker._supervisor_address,
                supervisor_endpoint=worker._supervisor_endpoint,
                metrics_exporter_host=worker._metrics_exporter_config[0],
                metrics_exporter_port=worker._metrics_exporter_config[1],
                logging_conf=getattr(worker, "_worker_logging_conf", None),
                replace_worker=worker.address,
                relaunch_operation_id=operation_id,
            )
            # Journal the target before creating it. A replacement cannot commit
            # a handoff whose destination would be lost on Supervisor restart.
            await supervisor.plan_worker_relaunch(
                operation_id, worker.address, worker._incarnation, target
            )
            if worker._lifecycle.phase != WorkerPhase.ACTIVE:
                raise RuntimeError("Worker ownership is already being transferred")
            os.makedirs(XINFERENCE_HOME, exist_ok=True)
            fd, path = tempfile.mkstemp(
                prefix="worker-relaunch-", suffix=".json", dir=XINFERENCE_HOME
            )
            env = dict(os.environ)
            if python_executable is not None:
                env.pop("PYTHONPATH", None)
                env.pop("PYTHONHOME", None)
                for name in ("VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
                    env.pop(name, None)
            env["PATH"] = os.path.dirname(executable) + os.pathsep + env.get("PATH", "")
            device_env = get_available_device_env_name() or "CUDA_VISIBLE_DEVICES"
            env[device_env] = ",".join(map(str, worker._total_gpu_devices)) or "-1"
            self.operation_id, self.target = operation_id, target
            try:
                with os.fdopen(fd, "w") as stream:
                    json.dump(config, stream)
                with open(get_log_file(f"worker-relaunch-{operation_id}"), "ab") as log:
                    self.process = subprocess.Popen(
                        [executable, "-m", "xinference.deploy.worker_relaunch", path],
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                self._reaper = asyncio.create_task(self._reap(self.process))
            except BaseException:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(path)
                raise
            return self.state(operation_id)

    @staticmethod
    async def _reap(process):
        # Local mode retains the original shared process after Worker handoff.
        # Keep reaping its child even after this Worker's pool has retired.
        while process.poll() is None:
            await asyncio.sleep(0.1)

    def state(self, operation_id):
        if operation_id in self.errors:
            return dict(exit_code=-1, transfer="idle", error=self.errors[operation_id])
        if self.operation_id != operation_id or self.process is None:
            raise ValueError("Worker relaunch process is not available")
        worker = self.worker
        outgoing = worker._outgoing_handoff
        return dict(
            target_worker=self.target,
            pid=self.process.pid,
            exit_code=self.process.poll(),
            transfer=(
                worker.worker_handoff_status(outgoing.ticket) if outgoing else "idle"
            ),
        )


class SupervisorRelaunchMixin:
    def _relaunch_manager(self):
        manager = getattr(self, "_worker_relaunch_manager", None)
        if manager is None:
            self._worker_relaunch_manager = manager = WorkerRelaunchManager(self)
            manager.resume()
        return manager

    async def relaunch_worker(self, address, python_executable=None, request_id=None):
        return await self._relaunch_manager().create(
            address, python_executable, request_id
        )

    def get_worker_relaunch(self, operation_id):
        return self._relaunch_manager().get(operation_id)

    def list_worker_relaunches(self):
        return self._relaunch_manager().list()

    def plan_worker_relaunch(self, operation_id, source, incarnation, target):
        return self._relaunch_manager().plan(operation_id, source, incarnation, target)

    async def report_worker_relaunch(
        self, operation_id, target, incarnation, status, error=""
    ):
        return await self._relaunch_manager().report(
            operation_id, target, incarnation, status, error
        )
