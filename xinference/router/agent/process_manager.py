# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Operating-system process reconciliation for Router Runtime assignments."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ..logging_config import router_log_extra
from .control_plane import RouterAgentControlPlaneClient

logger = logging.getLogger(__name__)

_ALLOWED_ENVIRONMENT = (
    "PATH",
    "VIRTUAL_ENV",
    "LANG",
    "LC_ALL",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "XINFERENCE_HOME",
    "HF_HOME",
    "HF_ENDPOINT",
    "MODELSCOPE_CACHE",
    "XINFERENCE_LOG_ROTATION",
    "XINFERENCE_LOG_RETENTION_DAYS",
    "XINFERENCE_LOG_MAX_BYTES",
    "XINFERENCE_LOG_BACKUP_COUNT",
    "TOKENIZERS_PARALLELISM",
    "PYTHONUNBUFFERED",
    "PYTHONFAULTHANDLER",
)


@dataclass
class ManagedRuntimeProcess:
    assignment: Dict[str, Any]
    process: Optional[asyncio.subprocess.Process] = None
    monitor_task: Optional[asyncio.Task[None]] = None
    restart_task: Optional[asyncio.Task[None]] = None
    restart_times: list[float] = field(default_factory=list)
    next_restart_at: float = 0.0
    stopping: bool = False

    @property
    def assignment_id(self) -> str:
        return str(self.assignment["assignment_id"])

    @property
    def generation(self) -> int:
        return int(self.assignment["assignment_generation"])


class RouterRuntimeProcessManager:
    """Reconcile a full Supervisor Assignment snapshot with local processes."""

    def __init__(
        self,
        *,
        node_id: str,
        supervisor_url: str,
        internal_token: str,
        runtime_executable: str,
        runtime_log_root: str,
        log_level: str,
        drain_timeout_seconds: float,
        max_restart_backoff_seconds: float,
        control_plane: RouterAgentControlPlaneClient,
    ) -> None:
        self.node_id = node_id
        self.supervisor_url = supervisor_url
        self.internal_token = internal_token
        self.runtime_executable = runtime_executable
        self.runtime_log_root = runtime_log_root
        self.log_level = log_level
        self.drain_timeout_seconds = drain_timeout_seconds
        self.max_restart_backoff_seconds = max_restart_backoff_seconds
        self.control_plane = control_plane
        self._processes: Dict[str, ManagedRuntimeProcess] = {}
        self._lock = asyncio.Lock()
        self._stopping = False

    @property
    def running_count(self) -> int:
        return sum(
            item.process is not None and item.process.returncode is None
            for item in self._processes.values()
        )

    def observed_assignments(self) -> list[Dict[str, Any]]:
        result = []
        for managed in self._processes.values():
            process = managed.process
            result.append(
                {
                    "assignment_id": managed.assignment_id,
                    "assignment_generation": managed.generation,
                    "router_uid": managed.assignment.get("router_uid"),
                    "pid": (
                        process.pid if process and process.returncode is None else None
                    ),
                    "running": bool(process and process.returncode is None),
                    "listen_port": managed.assignment.get("listen_port"),
                }
            )
        return result

    @staticmethod
    def _port_available(host: str, port: int) -> bool:
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    def _child_environment(self, assignment: Dict[str, Any]) -> Dict[str, str]:
        env = {
            key: value
            for key in _ALLOWED_ENVIRONMENT
            if (value := os.environ.get(key)) is not None
        }
        asset = assignment.get("tokenizer_asset", {})
        env.update(
            {
                "XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN": self.internal_token,
                "XINFERENCE_TOKEN_ROUTER_ASSIGNMENT_ID": str(
                    assignment["assignment_id"]
                ),
                "XINFERENCE_TOKEN_ROUTER_ASSIGNMENT_GENERATION": str(
                    assignment["assignment_generation"]
                ),
                "XINFERENCE_TOKEN_ROUTER_NODE_ID": self.node_id,
                "XINFERENCE_TOKEN_ROUTER_ACCESS_LOG": os.getenv(
                    "XINFERENCE_TOKEN_ROUTER_ACCESS_LOG", "false"
                ),
                "XINFERENCE_LOG_FORMAT": os.getenv("XINFERENCE_LOG_FORMAT", "json"),
                "XINFERENCE_LOG_CONSOLE": os.getenv("XINFERENCE_LOG_CONSOLE", "false"),
                "XINFERENCE_LOG_DIR": str(
                    Path(self.runtime_log_root) / str(assignment["assignment_id"])
                ),
                "PYTHONUNBUFFERED": "1",
                "PYTHONFAULTHANDLER": "1",
                "TOKENIZERS_PARALLELISM": os.getenv("TOKENIZERS_PARALLELISM", "false"),
            }
        )
        if asset:
            env.update(
                {
                    "ROUTER_TOKENIZER_PATH": str(asset.get("local_path") or ""),
                    "XINFERENCE_TOKEN_ROUTER_TOKENIZER_ASSET_ID": str(
                        asset.get("asset_id") or ""
                    ),
                    "XINFERENCE_TOKEN_ROUTER_TOKENIZER_ASSET_REVISION": str(
                        asset.get("revision") or ""
                    ),
                    "XINFERENCE_TOKEN_ROUTER_TOKENIZER_ASSET_FINGERPRINT": str(
                        asset.get("fingerprint") or ""
                    ),
                    "XINFERENCE_TOKEN_ROUTER_TOKENIZER_BINDING_GENERATION": str(
                        asset.get("binding_generation") or ""
                    ),
                }
            )
        return env

    async def reconcile(self, assignments: Iterable[Dict[str, Any]]) -> None:
        desired = {str(item["assignment_id"]): dict(item) for item in assignments}
        async with self._lock:
            for assignment_id, assignment in desired.items():
                managed = self._processes.get(assignment_id)
                if assignment.get("desired_state") != "running":
                    if managed is not None:
                        await self._stop_locked(managed, report=True)
                        self._processes.pop(assignment_id, None)
                    continue
                if self._stopping:
                    continue
                if managed is None:
                    managed = ManagedRuntimeProcess(assignment=assignment)
                    self._processes[assignment_id] = managed
                elif managed.generation != int(assignment["assignment_generation"]):
                    await self._stop_locked(managed, report=False)
                    managed = ManagedRuntimeProcess(assignment=assignment)
                    self._processes[assignment_id] = managed
                else:
                    managed.assignment = assignment
                if managed.process is None or managed.process.returncode is not None:
                    await self._start_locked(managed)

            for assignment_id in list(self._processes):
                if assignment_id not in desired:
                    managed = self._processes.pop(assignment_id)
                    await self._stop_locked(managed, report=False)

    async def _start_locked(self, managed: ManagedRuntimeProcess) -> None:
        now = time.monotonic()
        if managed.next_restart_at > now:
            return
        assignment = managed.assignment
        asset = assignment.get("tokenizer_asset", {})
        if asset and (
            asset.get("observed_state") != "ready"
            or not str(asset.get("local_path") or "")
            or not Path(str(asset.get("local_path") or "")).is_dir()
        ):
            await self.control_plane.report_assignment_status(
                managed.assignment_id,
                node_id=self.node_id,
                assignment_generation=managed.generation,
                observed_state="failed",
                last_error="Tokenizer Asset Binding is not ready on this Router Agent",
            )
            return
        host = str(assignment["listen_host"])
        port = int(assignment["listen_port"])
        if not self._port_available(host, port):
            logger.warning(
                "Router Runtime port conflict",
                extra=router_log_extra(
                    event="port_conflict",
                    node_id=self.node_id,
                    router_uid=assignment["router_uid"],
                    assignment_id=managed.assignment_id,
                    assignment_generation=managed.generation,
                    listen_port=port,
                    outcome="reassign_requested",
                ),
            )
            await self.control_plane.report_assignment_status(
                managed.assignment_id,
                node_id=self.node_id,
                assignment_generation=managed.generation,
                observed_state="port_conflict",
                listen_port=port,
                last_error=f"Router Runtime port is already in use: {host}:{port}",
            )
            return

        Path(self.runtime_log_root, managed.assignment_id).mkdir(
            parents=True, exist_ok=True
        )
        await self.control_plane.report_assignment_status(
            managed.assignment_id,
            node_id=self.node_id,
            assignment_generation=managed.generation,
            observed_state="starting",
            listen_port=port,
        )
        logger.info(
            "Starting Router Runtime",
            extra=router_log_extra(
                event="runtime_starting",
                node_id=self.node_id,
                router_uid=assignment["router_uid"],
                assignment_id=managed.assignment_id,
                assignment_generation=managed.generation,
                listen_port=port,
            ),
        )
        command = (
            self.runtime_executable,
            "--supervisor-url",
            self.supervisor_url,
            "--router-uid",
            str(assignment["router_uid"]),
            "--host",
            host,
            "--port",
            str(port),
            "--public-endpoint",
            str(assignment["public_endpoint"]),
            "--log-level",
            self.log_level,
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                env=self._child_environment(assignment),
            )
        except Exception as exc:
            await self._record_start_failure(managed, str(exc))
            return
        managed.process = process
        managed.stopping = False
        managed.monitor_task = asyncio.create_task(
            self._monitor(managed, process),
            name=f"router-runtime-{managed.assignment_id}",
        )
        await self.control_plane.report_assignment_status(
            managed.assignment_id,
            node_id=self.node_id,
            assignment_generation=managed.generation,
            observed_state="starting",
            pid=process.pid,
            listen_port=port,
        )
        logger.info(
            "Started Router Runtime",
            extra=router_log_extra(
                event="runtime_started",
                node_id=self.node_id,
                router_uid=assignment["router_uid"],
                assignment_id=managed.assignment_id,
                assignment_generation=managed.generation,
                listen_port=port,
                outcome="completed",
            ),
        )

    async def _record_start_failure(
        self, managed: ManagedRuntimeProcess, error: str
    ) -> None:
        state = self._schedule_restart(managed)
        self._ensure_restart_task(managed)
        await self.control_plane.report_assignment_status(
            managed.assignment_id,
            node_id=self.node_id,
            assignment_generation=managed.generation,
            observed_state=state,
            last_error=error,
        )

    def _schedule_restart(self, managed: ManagedRuntimeProcess) -> str:
        now = time.monotonic()
        managed.restart_times = [
            value for value in managed.restart_times if now - value <= 600.0
        ]
        managed.restart_times.append(now)
        if len(managed.restart_times) >= 10:
            managed.next_restart_at = now + 600.0
            return "crash_loop"
        delay = min(
            5.0 * (2 ** max(0, len(managed.restart_times) - 1)),
            self.max_restart_backoff_seconds,
        )
        managed.next_restart_at = now + delay
        return "failed"

    def _ensure_restart_task(self, managed: ManagedRuntimeProcess) -> None:
        existing = managed.restart_task
        current = asyncio.current_task()
        if existing is not None and not existing.done() and existing is not current:
            return
        managed.restart_task = asyncio.create_task(
            self._restart_when_due(managed),
            name=f"router-runtime-restart-{managed.assignment_id}",
        )

    async def _restart_when_due(self, managed: ManagedRuntimeProcess) -> None:
        current_task = asyncio.current_task()
        try:
            await asyncio.sleep(max(0.0, managed.next_restart_at - time.monotonic()))
            async with self._lock:
                if (
                    self._stopping
                    or managed.stopping
                    or self._processes.get(managed.assignment_id) is not managed
                    or managed.assignment.get("desired_state") != "running"
                ):
                    return
                process = managed.process
                if process is None or process.returncode is not None:
                    await self._start_locked(managed)
        finally:
            if managed.restart_task is current_task:
                managed.restart_task = None

    async def _monitor(
        self,
        managed: ManagedRuntimeProcess,
        process: asyncio.subprocess.Process,
    ) -> None:
        returncode = await process.wait()
        if managed.process is not process or managed.stopping:
            return
        state = self._schedule_restart(managed)
        self._ensure_restart_task(managed)
        logger.error(
            "Router Runtime exited unexpectedly",
            extra=router_log_extra(
                event=(
                    "runtime_crash_loop" if state == "crash_loop" else "runtime_crashed"
                ),
                node_id=self.node_id,
                router_uid=managed.assignment["router_uid"],
                assignment_id=managed.assignment_id,
                assignment_generation=managed.generation,
                listen_port=managed.assignment["listen_port"],
                outcome=state,
            ),
        )
        try:
            await self.control_plane.report_assignment_status(
                managed.assignment_id,
                node_id=self.node_id,
                assignment_generation=managed.generation,
                observed_state=state,
                pid=process.pid,
                listen_port=int(managed.assignment["listen_port"]),
                last_error=f"Router Runtime exited with code {returncode}",
                observed={"returncode": returncode},
            )
        except Exception:
            logger.exception(
                "Failed to report Router Runtime exit assignment=%s",
                managed.assignment_id,
            )

    async def _report_stop_status(
        self,
        managed: ManagedRuntimeProcess,
        observed_state: str,
        *,
        pid: Optional[int],
    ) -> None:
        try:
            await self.control_plane.report_assignment_status(
                managed.assignment_id,
                node_id=self.node_id,
                assignment_generation=managed.generation,
                observed_state=observed_state,
                pid=pid,
            )
        except Exception:
            logger.exception(
                "Failed to report Router Runtime %s status assignment=%s",
                observed_state,
                managed.assignment_id,
            )

    async def _stop_locked(
        self, managed: ManagedRuntimeProcess, *, report: bool
    ) -> None:
        process = managed.process
        managed.stopping = True
        restart_task = managed.restart_task
        if restart_task is not None and restart_task is not asyncio.current_task():
            restart_task.cancel()
            await asyncio.gather(restart_task, return_exceptions=True)
        managed.restart_task = None
        if process is not None and process.returncode is None:
            if report:
                await self._report_stop_status(managed, "draining", pid=process.pid)
            logger.info(
                "Stopping Router Runtime",
                extra=router_log_extra(
                    event="runtime_stopping",
                    node_id=self.node_id,
                    router_uid=managed.assignment["router_uid"],
                    assignment_id=managed.assignment_id,
                    assignment_generation=managed.generation,
                    listen_port=managed.assignment["listen_port"],
                ),
            )
            process.terminate()
            try:
                await asyncio.wait_for(
                    process.wait(), timeout=self.drain_timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        if managed.monitor_task is not None and not managed.monitor_task.done():
            managed.monitor_task.cancel()
            await asyncio.gather(managed.monitor_task, return_exceptions=True)
        if report:
            await self._report_stop_status(
                managed,
                "stopped",
                pid=process.pid if process is not None else None,
            )
        logger.info(
            "Router Runtime stopped",
            extra=router_log_extra(
                event="runtime_stopped",
                node_id=self.node_id,
                router_uid=managed.assignment["router_uid"],
                assignment_id=managed.assignment_id,
                assignment_generation=managed.generation,
                listen_port=managed.assignment["listen_port"],
                outcome="completed",
            ),
        )
        managed.process = None

    async def shutdown(self) -> None:
        self._stopping = True
        async with self._lock:
            runtimes = list(self._processes.values())
            self._processes.clear()
            await asyncio.gather(
                *(self._stop_locked(item, report=True) for item in runtimes),
                return_exceptions=True,
            )
