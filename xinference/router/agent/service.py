# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Router Agent service bootstrap and reconciliation loop."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import signal
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

import httpx
import psutil

from xinference import __version__

from ..control_plane import _software_revision
from ..logging_config import configure_router_logging, normalize_log_level
from .asset_manager import RouterAgentAssetManager, asset_binding_snapshot
from .control_plane import RouterAgentControlPlaneClient, assignment_snapshot
from .process_manager import RouterRuntimeProcessManager

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_RETRYABLE_BOOTSTRAP_STATUS_CODES = frozenset({429, 502, 503, 504})


class _RouterAgentStopRequested(Exception):
    """Internal control flow used to stop bootstrap without an error."""


class _RouterAgentBootstrapTimeout(TimeoutError):
    """Raised when Router Agent bootstrap exceeds its configured deadline."""


def _is_retryable_bootstrap_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_BOOTSTRAP_STATUS_CODES
    return isinstance(exc, httpx.TransportError)


def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    if exc.response.status_code != 429:
        return None
    value = exc.response.headers.get("Retry-After")
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, seconds)


def _gather_host_resources() -> Dict[str, Any]:
    """Collect Router Agent host CPU and memory resources for heartbeats."""

    try:
        mem_info = psutil.virtual_memory()
        cpu_usage = max(0.0, min(1.0, psutil.cpu_percent() / 100.0))
        return {
            "cpu": {
                "usage": cpu_usage,
                "total": psutil.cpu_count() or 0,
            },
            "memory": {
                "used": mem_info.used,
                "available": mem_info.available,
                "total": mem_info.total,
            },
        }
    except Exception:
        # Host metrics are diagnostic data.  Do not turn a sampling failure
        # into a Router Agent heartbeat failure.
        logger.warning("Failed to collect Router Agent host resources", exc_info=True)
        return {}


@dataclass(frozen=True)
class RouterAgentConfig:
    supervisor_url: str
    node_id: str
    node_host: str
    port_range_start: int
    port_range_end: int
    max_instances: int
    runtime_executable: str
    runtime_log_root: str
    internal_token: str
    heartbeat_seconds: float = 15.0
    watch_seconds: float = 30.0
    max_restart_backoff_seconds: float = 60.0
    drain_timeout_seconds: float = 7200.0
    startup_retry_initial_seconds: float = 1.0
    startup_retry_max_seconds: float = 15.0
    startup_retry_timeout_seconds: float = 120.0
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, *, log_level: Optional[str] = None) -> "RouterAgentConfig":
        def required(name: str) -> str:
            value = os.getenv(name, "").strip()
            if not value:
                raise ValueError(
                    f"Required Router Agent environment variable is missing: {name}"
                )
            return value

        config = cls(
            supervisor_url=required("XINFERENCE_TOKEN_ROUTER_SUPERVISOR_URL"),
            node_id=required("XINFERENCE_TOKEN_ROUTER_NODE_ID"),
            node_host=required("XINFERENCE_TOKEN_ROUTER_NODE_HOST"),
            port_range_start=int(required("XINFERENCE_TOKEN_ROUTER_PORT_RANGE_START")),
            port_range_end=int(required("XINFERENCE_TOKEN_ROUTER_PORT_RANGE_END")),
            max_instances=int(required("XINFERENCE_TOKEN_ROUTER_MAX_INSTANCES")),
            runtime_executable=required("XINFERENCE_TOKEN_ROUTER_RUNTIME_EXECUTABLE"),
            runtime_log_root=required("XINFERENCE_TOKEN_ROUTER_RUNTIME_LOG_ROOT"),
            internal_token=required("XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN"),
            heartbeat_seconds=float(
                os.getenv("XINFERENCE_TOKEN_ROUTER_AGENT_HEARTBEAT_SECONDS", "15")
            ),
            watch_seconds=float(
                os.getenv("XINFERENCE_TOKEN_ROUTER_AGENT_WATCH_SECONDS", "30")
            ),
            max_restart_backoff_seconds=float(
                os.getenv(
                    "XINFERENCE_TOKEN_ROUTER_AGENT_MAX_RESTART_BACKOFF_SECONDS",
                    "60",
                )
            ),
            drain_timeout_seconds=float(
                os.getenv("XINFERENCE_TOKEN_ROUTER_AGENT_DRAIN_TIMEOUT_SECONDS", "7200")
            ),
            startup_retry_initial_seconds=float(
                os.getenv(
                    "XINFERENCE_TOKEN_ROUTER_AGENT_STARTUP_RETRY_INITIAL_SECONDS",
                    "1",
                )
            ),
            startup_retry_max_seconds=float(
                os.getenv(
                    "XINFERENCE_TOKEN_ROUTER_AGENT_STARTUP_RETRY_MAX_SECONDS",
                    "15",
                )
            ),
            startup_retry_timeout_seconds=float(
                os.getenv(
                    "XINFERENCE_TOKEN_ROUTER_AGENT_STARTUP_RETRY_TIMEOUT_SECONDS",
                    "120",
                )
            ),
            log_level=normalize_log_level(
                log_level
                or os.getenv("XINFERENCE_TOKEN_ROUTER_LOG_LEVEL", "INFO")
                or "INFO"
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.node_id or any(ch.isspace() for ch in self.node_id):
            raise ValueError(
                "Router Agent node_id must be non-empty without whitespace"
            )
        if not 1024 <= self.port_range_start <= self.port_range_end <= 65535:
            raise ValueError("Router Agent port range must be within 1024..65535")
        if self.max_instances <= 0:
            raise ValueError("Router Agent max_instances must be greater than zero")
        if self.max_instances > self.port_range_end - self.port_range_start + 1:
            raise ValueError("Router Agent max_instances exceeds its port range")
        if self.heartbeat_seconds <= 0 or self.watch_seconds < 0:
            raise ValueError("Router Agent heartbeat/watch intervals are invalid")
        if self.max_restart_backoff_seconds <= 0 or self.drain_timeout_seconds <= 0:
            raise ValueError("Router Agent restart/drain timeouts must be positive")
        if self.startup_retry_initial_seconds <= 0:
            raise ValueError(
                "Router Agent startup retry initial delay must be positive"
            )
        if self.startup_retry_max_seconds < self.startup_retry_initial_seconds:
            raise ValueError(
                "Router Agent startup retry maximum must not be less than initial"
            )
        if self.startup_retry_timeout_seconds <= 0:
            raise ValueError("Router Agent startup retry timeout must be positive")
        runtime = Path(self.runtime_executable)
        if not runtime.is_file() or not os.access(runtime, os.X_OK):
            raise ValueError(
                f"Router Runtime executable is missing or not executable: {runtime}"
            )
        Path(self.runtime_log_root).mkdir(parents=True, exist_ok=True)


class RouterAgent:
    def __init__(
        self,
        config: RouterAgentConfig,
        *,
        control_plane: Optional[RouterAgentControlPlaneClient] = None,
        process_manager: Optional[RouterRuntimeProcessManager] = None,
        asset_manager: Optional[RouterAgentAssetManager] = None,
    ) -> None:
        self.config = config
        self.control_plane = control_plane or RouterAgentControlPlaneClient(
            config.supervisor_url, config.internal_token
        )
        self.process_manager = process_manager or RouterRuntimeProcessManager(
            node_id=config.node_id,
            supervisor_url=config.supervisor_url,
            internal_token=config.internal_token,
            runtime_executable=config.runtime_executable,
            runtime_log_root=config.runtime_log_root,
            log_level=config.log_level,
            drain_timeout_seconds=config.drain_timeout_seconds,
            max_restart_backoff_seconds=config.max_restart_backoff_seconds,
            control_plane=self.control_plane,
        )
        self.asset_manager = asset_manager or RouterAgentAssetManager(
            config.node_id,
            self.control_plane,
            inventory_path=str(
                Path(config.runtime_log_root).parent / "router-agent-assets.json"
            ),
        )
        self._stop_event = asyncio.Event()
        self._cursor = ""
        self._asset_cursor = ""
        self._assignments: list[Dict[str, Any]] = []
        self._asset_bindings: list[Dict[str, Any]] = []

    def request_stop(self) -> None:
        self._stop_event.set()

    def _node_registration(self) -> Dict[str, Any]:
        reported_labels: Dict[str, str] = {
            "system.hostname": socket.gethostname(),
        }
        return {
            "node_id": self.config.node_id,
            "advertise_host": self.config.node_host,
            "port_range_start": self.config.port_range_start,
            "port_range_end": self.config.port_range_end,
            "max_instances": self.config.max_instances,
            "software_version": __version__,
            "software_revision": _software_revision(),
            "reported_labels": reported_labels,
            "capabilities": {
                "hostname": socket.gethostname(),
            },
        }

    def _heartbeat_payload(self, status: str) -> Dict[str, Any]:
        running = self.process_manager.running_count
        return {
            "status": status,
            "running_instances": running,
            "available_slots": (
                0
                if status == "draining"
                else max(0, self.config.max_instances - running)
            ),
            "assignments": self.process_manager.observed_assignments(),
            "resources": _gather_host_resources(),
        }

    async def _heartbeat_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.process_manager.reconcile(self._assignments)
                await self.control_plane.heartbeat_node(
                    self.config.node_id,
                    self._heartbeat_payload("ready"),
                )
            except Exception:
                logger.exception("Router Agent heartbeat/reconcile failed")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.config.heartbeat_seconds
                )
            except asyncio.TimeoutError:
                pass

    async def _watch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = await self.control_plane.watch_assignments(
                    self.config.node_id,
                    after_cursor=self._cursor,
                    wait_seconds=self.config.watch_seconds,
                )
                if payload is None:
                    continue
                self._cursor = str(payload.get("cursor", ""))
                self._assignments = assignment_snapshot(payload)
                await self.process_manager.reconcile(self._assignments)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Router Agent Assignment watch failed")
                await asyncio.sleep(min(5.0, self.config.heartbeat_seconds))

    async def _asset_watch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = await self.control_plane.watch_asset_bindings(
                    self.config.node_id,
                    after_cursor=self._asset_cursor,
                    wait_seconds=self.config.watch_seconds,
                )
                if payload is None:
                    continue
                self._asset_cursor = str(payload.get("cursor", ""))
                self._asset_bindings = asset_binding_snapshot(payload)
                await self.asset_manager.reconcile(self._asset_bindings)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Router Agent Asset Binding watch failed")
                await asyncio.sleep(min(5.0, self.config.heartbeat_seconds))

    async def _wait_for_startup_retry(self, delay: float) -> None:
        if self._stop_event.is_set():
            raise _RouterAgentStopRequested
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
        except asyncio.TimeoutError:
            return
        raise _RouterAgentStopRequested

    async def _run_startup_operation(
        self,
        operation_name: str,
        operation: Callable[[], Awaitable[_T]],
        *,
        timeout: float,
    ) -> _T:
        operation_task = asyncio.ensure_future(operation())
        stop_task = asyncio.create_task(
            self._stop_event.wait(),
            name=f"router-agent-bootstrap-stop-{operation_name}",
        )
        try:
            done, _ = await asyncio.wait(
                {operation_task, stop_task},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise asyncio.TimeoutError
            if operation_task in done:
                return await operation_task
            raise _RouterAgentStopRequested
        finally:
            for task in (operation_task, stop_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(operation_task, stop_task, return_exceptions=True)

    async def _run_startup_step(
        self,
        operation_name: str,
        operation: Callable[[], Awaitable[_T]],
        *,
        deadline: float,
    ) -> _T:
        delay = self.config.startup_retry_initial_seconds
        attempt = 0
        step_started = time.monotonic()
        while True:
            if self._stop_event.is_set():
                raise _RouterAgentStopRequested
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _RouterAgentBootstrapTimeout(
                    f"Router Agent bootstrap timed out during {operation_name}"
                )
            attempt += 1
            try:
                result = await self._run_startup_operation(
                    operation_name,
                    operation,
                    timeout=remaining,
                )
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError as exc:
                raise _RouterAgentBootstrapTimeout(
                    f"Router Agent bootstrap timed out during {operation_name}"
                ) from exc
            except Exception as exc:
                if not _is_retryable_bootstrap_error(exc):
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _RouterAgentBootstrapTimeout(
                        f"Router Agent bootstrap timed out during {operation_name}"
                    ) from exc
                retry_after = _retry_after_seconds(exc)
                if retry_after is not None:
                    retry_delay = min(retry_after, remaining)
                else:
                    retry_delay = min(
                        delay * random.uniform(0.9, 1.1),
                        self.config.startup_retry_max_seconds,
                        remaining,
                    )
                log_retry = (
                    logger.warning if attempt == 1 or attempt % 5 == 0 else logger.info
                )
                log_retry(
                    "Router Agent bootstrap step failed; operation=%s "
                    "attempt=%d retry_in_seconds=%.3f error=%s",
                    operation_name,
                    attempt,
                    retry_delay,
                    type(exc).__name__,
                )
                await self._wait_for_startup_retry(retry_delay)
                delay = min(
                    delay * 2,
                    self.config.startup_retry_max_seconds,
                )
                continue
            if attempt > 1:
                logger.info(
                    "Router Agent bootstrap step recovered; operation=%s "
                    "attempts=%d elapsed_seconds=%.3f",
                    operation_name,
                    attempt,
                    time.monotonic() - step_started,
                )
            return result

    async def _bootstrap(self) -> bool:
        started = time.monotonic()
        deadline = started + self.config.startup_retry_timeout_seconds
        try:
            await self._run_startup_step(
                "register_node",
                lambda: self.control_plane.register_node(self._node_registration()),
                deadline=deadline,
            )
            initial_assets = await self._run_startup_step(
                "watch_asset_bindings",
                lambda: self.control_plane.watch_asset_bindings(
                    self.config.node_id, wait_seconds=0
                ),
                deadline=deadline,
            )
            if initial_assets is not None:
                self._asset_cursor = str(initial_assets.get("cursor", ""))
                self._asset_bindings = asset_binding_snapshot(initial_assets)
            await self._run_startup_step(
                "reconcile_asset_bindings",
                lambda: self.asset_manager.reconcile(self._asset_bindings),
                deadline=deadline,
            )

            initial = await self._run_startup_step(
                "watch_assignments",
                lambda: self.control_plane.watch_assignments(
                    self.config.node_id, wait_seconds=0
                ),
                deadline=deadline,
            )
            if initial is not None:
                self._cursor = str(initial.get("cursor", ""))
                self._assignments = assignment_snapshot(initial)
            await self._run_startup_step(
                "reconcile_assignments",
                lambda: self.process_manager.reconcile(self._assignments),
                deadline=deadline,
            )
        except _RouterAgentStopRequested:
            logger.info("Router Agent bootstrap stopped before completion")
            return False
        logger.info(
            "Router Agent bootstrap succeeded; elapsed_seconds=%.3f",
            time.monotonic() - started,
        )
        return True

    async def run(self) -> None:
        tasks: list[asyncio.Task] = []
        bootstrapped = False
        try:
            bootstrapped = await self._bootstrap()
            if not bootstrapped:
                return
            tasks = [
                asyncio.create_task(
                    self._heartbeat_loop(), name="router-agent-heartbeat"
                ),
                asyncio.create_task(
                    self._watch_loop(), name="router-agent-assignment-watch"
                ),
                asyncio.create_task(
                    self._asset_watch_loop(), name="router-agent-asset-watch"
                ),
            ]
            await self._stop_event.wait()
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if bootstrapped:
                try:
                    await self.control_plane.heartbeat_node(
                        self.config.node_id,
                        self._heartbeat_payload("draining"),
                    )
                except Exception:
                    logger.warning(
                        "Failed to publish final Router Agent heartbeat", exc_info=True
                    )
            try:
                await self.process_manager.shutdown()
            finally:
                await self.control_plane.aclose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Xinference Token Router Agent")
    parser.add_argument("--log-level", default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    config = RouterAgentConfig.from_env(log_level=args.log_level)
    configure_router_logging(config.log_level, config.node_id)

    async def serve() -> None:
        agent = RouterAgent(config)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, agent.request_stop)
            except NotImplementedError:  # pragma: no cover - Windows event loops
                pass
        await agent.run()

    asyncio.run(serve())
