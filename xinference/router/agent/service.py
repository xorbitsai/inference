# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Router Agent service bootstrap and reconciliation loop."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import psutil

from xinference import __version__

from ..control_plane import _software_revision
from ..logging_config import configure_router_logging, normalize_log_level
from .asset_manager import RouterAgentAssetManager, asset_binding_snapshot
from .control_plane import RouterAgentControlPlaneClient, assignment_snapshot
from .process_manager import RouterRuntimeProcessManager

logger = logging.getLogger(__name__)


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
        legacy_assets = [
            value.strip()
            for value in os.getenv(
                "XINFERENCE_TOKEN_ROUTER_TOKENIZER_ASSETS", ""
            ).split(",")
            if value.strip()
        ]
        legacy_labels: Dict[str, str] = {}
        for item in os.getenv("XINFERENCE_TOKEN_ROUTER_NODE_LABELS", "").split(","):
            if "=" in item:
                key, value = item.split("=", 1)
                if key.strip():
                    legacy_labels[key.strip()] = value.strip()
        if legacy_assets:
            logger.warning(
                "XINFERENCE_TOKEN_ROUTER_TOKENIZER_ASSETS is deprecated; "
                "migrate Assets to persistent Asset-Agent Bindings"
            )
        if legacy_labels:
            logger.warning(
                "XINFERENCE_TOKEN_ROUTER_NODE_LABELS is deprecated; "
                "manage Router Agent labels through the control plane"
            )
        reported_labels: Dict[str, str] = {
            "system.hostname": socket.gethostname(),
        }
        reported_labels.update(legacy_labels)
        return {
            "node_id": self.config.node_id,
            "advertise_host": self.config.node_host,
            "port_range_start": self.config.port_range_start,
            "port_range_end": self.config.port_range_end,
            "max_instances": self.config.max_instances,
            "software_version": __version__,
            "software_revision": _software_revision(),
            "labels": reported_labels,
            "reported_labels": reported_labels,
            "capabilities": {
                # Compatibility for one migration release only.
                "tokenizer_assets": legacy_assets,
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

    async def run(self) -> None:
        await self.control_plane.register_node(self._node_registration())
        initial_assets = await self.control_plane.watch_asset_bindings(
            self.config.node_id, wait_seconds=0
        )
        if initial_assets is not None:
            self._asset_cursor = str(initial_assets.get("cursor", ""))
            self._asset_bindings = asset_binding_snapshot(initial_assets)
        await self.asset_manager.reconcile(self._asset_bindings)

        initial = await self.control_plane.watch_assignments(
            self.config.node_id, wait_seconds=0
        )
        if initial is not None:
            self._cursor = str(initial.get("cursor", ""))
            self._assignments = assignment_snapshot(initial)
        await self.process_manager.reconcile(self._assignments)
        heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name="router-agent-heartbeat"
        )
        watch_task = asyncio.create_task(
            self._watch_loop(), name="router-agent-assignment-watch"
        )
        asset_watch_task = asyncio.create_task(
            self._asset_watch_loop(), name="router-agent-asset-watch"
        )
        try:
            await self._stop_event.wait()
        finally:
            heartbeat_task.cancel()
            watch_task.cancel()
            asset_watch_task.cancel()
            await asyncio.gather(
                heartbeat_task, watch_task, asset_watch_task, return_exceptions=True
            )
            try:
                await self.control_plane.heartbeat_node(
                    self.config.node_id,
                    self._heartbeat_payload("draining"),
                )
            except Exception:
                logger.warning(
                    "Failed to publish final Router Agent heartbeat", exc_info=True
                )
            await self.process_manager.shutdown()
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
