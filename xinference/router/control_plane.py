# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Supervisor control-plane client for independent Router runtimes."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional

import httpx

from .config import config_from_control_plane

if TYPE_CHECKING:
    from .runtime import RouterRuntime

logger = logging.getLogger(__name__)


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
    ) -> None:
        if not internal_token:
            raise ValueError("Token Router internal token is required")
        self.supervisor_url = supervisor_url.rstrip("/")
        self.router_uid = router_uid
        self.internal_token = internal_token
        self.endpoint = endpoint
        self.instance_id = instance_id or f"{socket.gethostname()}-{uuid.uuid4()}"
        self.revision = 0
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.log_level = log_level
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

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
                "version": "1",
                "acked_revision": self.revision,
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
                "tokenization": summary["tokenization"],
                "pools": summary["pools"],
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
                                    "Failed to apply Token Router revision %s",
                                    revision,
                                )
                                await self.ack(revision, str(exc))
                            else:
                                await self.ack(revision)

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
            except TimeoutError:
                pass
