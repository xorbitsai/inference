# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Supervisor control-plane client for independent Router runtimes."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import socket
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional
from urllib.parse import urlsplit

import httpx

from xinference import __version__

from .config import config_from_control_plane
from .logging_config import router_log_extra, set_router_log_identity
from .process_metrics import ProcessMetricsCollector

if TYPE_CHECKING:
    from .runtime import RouterRuntime

logger = logging.getLogger(__name__)


def _software_revision() -> Optional[str]:
    try:
        from xinference._commit import full_revisionid
    except ImportError:
        try:
            from xinference._version import commit_id
        except ImportError:
            commit_id = None
        return commit_id.lstrip("g") if commit_id else None
    return full_revisionid or None


def _default_instance_id_prefix(listen_host: str, endpoint: str) -> str:
    host = str(listen_host or "").strip()
    if host not in {"", "*", "0.0.0.0", "::", "[::]"}:
        return host
    endpoint_host = urlsplit(str(endpoint or "")).hostname
    return endpoint_host or socket.gethostname()


def _config_log_fields(config: Any) -> Dict[str, Any]:
    backends = getattr(config, "backends", ())
    return {
        "logical_model": getattr(config, "logical_model", None),
        "route_profile": getattr(config, "route_profile", None),
        "enabled": getattr(config, "enabled", None),
        "backend_mapping": {
            backend.id: backend.model_uid
            for backend in backends
            if getattr(backend, "id", None) and getattr(backend, "model_uid", None)
        },
    }


class RouterInstanceNotFound(RuntimeError):
    """The Supervisor no longer has this Router runtime registration."""


class RouterControlPlaneClient:
    def __init__(
        self,
        supervisor_url: str,
        router_uid: str,
        *,
        internal_token: str,
        endpoint: str,
        instance_id: Optional[str] = None,
        timeout_seconds: float = 10.0,
        listen_host: str = "127.0.0.1",
        listen_port: int = 10080,
        log_level: str = "INFO",
        assignment_id: Optional[str] = None,
        assignment_generation: Optional[int] = None,
        node_id: Optional[str] = None,
    ) -> None:
        if not internal_token:
            raise ValueError("Token Router internal token is required")
        self.supervisor_url = supervisor_url.rstrip("/")
        self.router_uid = router_uid
        self.internal_token = internal_token
        self.endpoint = endpoint
        self.instance_id = instance_id or (
            f"{_default_instance_id_prefix(listen_host, endpoint)}-{uuid.uuid4()}"
        )
        self._revision = 0
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.log_level = log_level
        self.assignment_id = assignment_id
        self.assignment_generation = assignment_generation
        self.node_id = node_id
        set_router_log_identity(
            router_uid=router_uid,
            node_id=node_id,
            assignment_id=assignment_id,
            assignment_generation=assignment_generation,
            instance_id=self.instance_id,
            listen_port=listen_port,
            config_revision=0,
        )
        self._client = httpx.AsyncClient(timeout=timeout_seconds)
        self._process_metrics: Any = ProcessMetricsCollector()
        self._last_process_metrics_warning = 0.0
        self._runtime_metadata: Dict[str, Any] = {
            "hostname": socket.gethostname(),
            "python_version": platform.python_version(),
            "platform": f"{platform.system()}-{platform.machine()}",
            "assignment_id": assignment_id,
            "assignment_generation": assignment_generation,
            "node_id": node_id,
            "software_version": __version__,
            "software_revision": _software_revision() or "",
        }
        if self._process_metrics.started_at is not None:
            self._runtime_metadata["started_at"] = self._process_metrics.started_at

    @property
    def revision(self) -> int:
        return self._revision

    @revision.setter
    def revision(self, value: int) -> None:
        self._revision = int(value)
        set_router_log_identity(config_revision=self._revision)

    @property
    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.internal_token}"}

    async def get_config(self, after_revision: int = 0) -> Optional[Dict[str, Any]]:
        response = await self._client.get(
            f"{self.supervisor_url}/v1/internal/token-router/configs/{self.router_uid}",
            params={"after_revision": after_revision},
            headers=self._headers,
        )
        if response.status_code == 204:
            return None
        response.raise_for_status()
        data = response.json()
        return data

    async def register(self) -> Dict[str, Any]:
        response = await self._client.post(
            f"{self.supervisor_url}/v1/internal/token-router/instances/register",
            headers=self._headers,
            json={
                "router_uid": self.router_uid,
                "instance_id": self.instance_id,
                "endpoint": self.endpoint,
                "assignment_id": self.assignment_id,
                "assignment_generation": self.assignment_generation,
                "node_id": self.node_id,
                # Keep the legacy field for older Supervisors and clients.
                "version": "1",
                "protocol_version": "1",
                "software_version": __version__,
                "software_revision": _software_revision(),
                "acked_revision": self.revision,
                "metadata": self._runtime_metadata,
            },
        )
        response.raise_for_status()
        return response.json()

    async def heartbeat(
        self,
        *,
        status: str,
        metrics: Optional[Dict[str, Any]] = None,
        backend_health: Optional[Dict[str, Any]] = None,
        process: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        response = await self._client.post(
            f"{self.supervisor_url}/v1/internal/token-router/instances/"
            f"{self.instance_id}/heartbeat",
            headers=self._headers,
            json={
                "status": status,
                "metrics": metrics or {},
                "backend_health": backend_health or {},
                "process": process or {"pid": os.getpid()},
            },
        )
        if response.status_code == 404:
            raise RouterInstanceNotFound(
                f"Token Router instance {self.instance_id} is not registered"
            )
        response.raise_for_status()
        return response.json()

    async def ack(self, revision: int, error: str = "") -> Dict[str, Any]:
        response = await self._client.post(
            f"{self.supervisor_url}/v1/internal/token-router/instances/"
            f"{self.instance_id}/config-ack",
            headers=self._headers,
            json={
                "router_uid": self.router_uid,
                "revision": revision,
                "error": error,
            },
        )
        if response.status_code == 404:
            raise RouterInstanceNotFound(
                f"Token Router instance {self.instance_id} is not registered"
            )
        response.raise_for_status()
        if not error:
            self.revision = revision
            set_router_log_identity(config_revision=revision)
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def unregister(self) -> None:
        try:
            response = await self._client.post(
                f"{self.supervisor_url}/v1/internal/token-router/instances/"
                f"{self.instance_id}/unregister",
                headers=self._headers,
            )
            if response.status_code not in (200, 404):
                response.raise_for_status()
        finally:
            await self.aclose()

    async def _collect_process_resources(self) -> Dict[str, Any]:
        try:
            return await asyncio.to_thread(self._process_metrics.collect)
        except Exception:
            now = asyncio.get_running_loop().time()
            if now - self._last_process_metrics_warning >= 300.0:
                self._last_process_metrics_warning = now
                logger.warning(
                    "Failed to collect Token Router process metrics", exc_info=True
                )
            return {}

    async def _publish_runtime_state(
        self,
        runtime: "RouterRuntime",
        summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        summary = summary or await runtime.summary()
        status = "ready" if summary["enabled"] else "disabled"
        await self.heartbeat(
            status=status,
            metrics=await runtime.metrics.summary(),
            process={
                "pid": os.getpid(),
                "revision": summary["revision"],
                "active_requests": summary["active_requests"],
                "resources": await self._collect_process_resources(),
                "tokenization": summary["tokenization"],
                "pools": summary["pools"],
                "tokenizer_asset": summary.get("tokenizer_asset", {}),
            },
        )
        return summary

    async def _reregister(self, runtime: "RouterRuntime") -> None:
        summary = await runtime.summary()
        runtime_revision = int(summary["revision"])
        if runtime_revision < self.revision:
            raise RuntimeError(
                "Token Router runtime revision is older than the last ACKed "
                f"revision: runtime={runtime_revision}, acked={self.revision}"
            )

        await self.register()
        if runtime_revision > self.revision:
            await self.ack(runtime_revision)
        await self._publish_runtime_state(runtime, summary)
        logger.info(
            "Token Router instance re-registered: instance_id=%s revision=%s",
            self.instance_id,
            runtime_revision,
        )

    async def run(
        self,
        runtime: "RouterRuntime",
        stop: asyncio.Event,
        *,
        poll_interval_seconds: float = 5.0,
        heartbeat_interval_seconds: float = 30.0,
    ) -> None:
        """Poll revisions, atomically apply them, and publish runtime state."""
        next_heartbeat = 0.0
        registration_lost = False
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            if registration_lost:
                try:
                    await self._reregister(runtime)
                except Exception:
                    logger.exception("Token Router instance re-registration failed")
                else:
                    registration_lost = False
                    next_heartbeat = loop.time() + heartbeat_interval_seconds
            else:
                try:
                    data = await self.get_config(after_revision=self.revision)
                    if data is not None:
                        revision = int(data["revision"])
                        if revision <= self.revision:
                            logger.debug(
                                "Ignoring already applied Token Router revision %s",
                                revision,
                            )
                        else:
                            previous_revision = self.revision
                            try:
                                config = config_from_control_plane(
                                    data,
                                    listen_host=self.listen_host,
                                    listen_port=self.listen_port,
                                    log_level=self.log_level,
                                )
                                await runtime.apply(config)
                            except Exception as exc:
                                logger.exception(
                                    "Failed to apply Token Router configuration",
                                    extra=router_log_extra(
                                        event="config_apply_failed",
                                        router_uid=self.router_uid,
                                        current_revision=previous_revision,
                                        target_revision=revision,
                                        outcome="config_apply_failed",
                                    ),
                                )
                                try:
                                    await self.ack(revision, str(exc))
                                except Exception:
                                    logger.exception(
                                        "Failed to acknowledge rejected Token Router configuration",
                                        extra=router_log_extra(
                                            event="config_apply_failed",
                                            router_uid=self.router_uid,
                                            current_revision=previous_revision,
                                            target_revision=revision,
                                            outcome="config_error_ack_failed",
                                        ),
                                    )
                                    raise
                            else:
                                try:
                                    await self.ack(revision)
                                except Exception:
                                    logger.exception(
                                        "Failed to acknowledge applied Token Router configuration",
                                        extra=router_log_extra(
                                            event="config_apply_failed",
                                            router_uid=self.router_uid,
                                            current_revision=previous_revision,
                                            target_revision=revision,
                                            outcome="config_ack_failed",
                                        ),
                                    )
                                    raise
                                logger.info(
                                    "Token Router configuration applied",
                                    extra=router_log_extra(
                                        event="config_applied",
                                        router_uid=self.router_uid,
                                        previous_revision=previous_revision,
                                        revision=revision,
                                        outcome="completed",
                                        **_config_log_fields(config),
                                    ),
                                )

                    now = loop.time()
                    if now >= next_heartbeat:
                        await self._publish_runtime_state(runtime)
                        next_heartbeat = now + heartbeat_interval_seconds
                except RouterInstanceNotFound:
                    registration_lost = True
                    logger.warning(
                        "Token Router instance registration was lost; "
                        "re-registering: instance_id=%s",
                        self.instance_id,
                    )
                except Exception:
                    logger.exception(
                        "Token Router control-plane synchronization failed"
                    )

            try:
                await asyncio.wait_for(stop.wait(), timeout=poll_interval_seconds)
            except asyncio.TimeoutError:
                pass
