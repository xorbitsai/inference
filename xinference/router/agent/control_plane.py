# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Supervisor control-plane client used by Router Agents."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx


class RouterAgentControlPlaneClient:
    """Small authenticated client for the Router Agent internal API."""

    def __init__(
        self,
        supervisor_url: str,
        internal_token: str,
        *,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not supervisor_url:
            raise ValueError("Router Agent supervisor URL is required")
        if not internal_token:
            raise ValueError("Router Agent internal token is required")
        self.supervisor_url = supervisor_url.rstrip("/")
        self.internal_token = internal_token
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    @property
    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.internal_token}"}

    async def register_node(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = await self._client.post(
            f"{self.supervisor_url}/v1/internal/token-router/nodes/register",
            headers=self._headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def heartbeat_node(
        self, node_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        response = await self._client.post(
            f"{self.supervisor_url}/v1/internal/token-router/nodes/{node_id}/heartbeat",
            headers=self._headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def watch_assignments(
        self,
        node_id: str,
        *,
        after_cursor: str = "",
        wait_seconds: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        # Long polling needs a timeout longer than the server-side wait.
        response = await self._client.get(
            f"{self.supervisor_url}/v1/internal/token-router/nodes/{node_id}/assignments",
            headers=self._headers,
            params={
                "after_cursor": after_cursor,
                "wait_seconds": wait_seconds,
            },
            timeout=max(10.0, wait_seconds + 10.0),
        )
        if response.status_code == 204:
            return None
        response.raise_for_status()
        return response.json()

    async def watch_asset_bindings(
        self,
        node_id: str,
        *,
        after_cursor: str = "",
        wait_seconds: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        response = await self._client.get(
            f"{self.supervisor_url}/v1/internal/token-router/nodes/"
            f"{node_id}/asset-bindings",
            headers=self._headers,
            params={"after_cursor": after_cursor, "wait_seconds": wait_seconds},
            timeout=max(10.0, wait_seconds + 10.0),
        )
        if response.status_code == 204:
            return None
        response.raise_for_status()
        return response.json()

    async def report_asset_binding_status(
        self,
        node_id: str,
        *,
        asset_id: str,
        generation: int,
        observed_state: str,
        observed_revision: str = "",
        observed_fingerprint: str = "",
        local_path: str = "",
        last_error_code: str = "",
        last_error: str = "",
    ) -> Dict[str, Any]:
        response = await self._client.post(
            f"{self.supervisor_url}/v1/internal/token-router/nodes/"
            f"{node_id}/asset-bindings/status",
            headers=self._headers,
            json={
                "asset_id": asset_id,
                "generation": generation,
                "observed_state": observed_state,
                "observed_revision": observed_revision,
                "observed_fingerprint": observed_fingerprint,
                "local_path": local_path,
                "last_error_code": last_error_code,
                "last_error": last_error,
            },
        )
        response.raise_for_status()
        return response.json()

    async def report_assignment_status(
        self,
        assignment_id: str,
        *,
        node_id: str,
        assignment_generation: int,
        observed_state: str,
        pid: Optional[int] = None,
        instance_id: Optional[str] = None,
        listen_port: Optional[int] = None,
        last_error: str = "",
        observed: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "node_id": node_id,
            "assignment_generation": assignment_generation,
            "observed_state": observed_state,
            "last_error": last_error,
            "observed": observed or {},
        }
        if pid is not None:
            payload["pid"] = pid
        if instance_id:
            payload["instance_id"] = instance_id
        if listen_port is not None:
            payload["listen_port"] = listen_port
        response = await self._client.put(
            f"{self.supervisor_url}/v1/internal/token-router/assignments/"
            f"{assignment_id}/status",
            headers=self._headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()


def assignment_snapshot(payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not payload:
        return []
    assignments = payload.get("assignments", [])
    if not isinstance(assignments, list):
        raise ValueError("Supervisor Assignment snapshot must be a list")
    return [dict(item) for item in assignments]
