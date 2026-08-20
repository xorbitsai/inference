# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""In-memory registry for independent Token Router runtime instances."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional


class RouterRuntimeRegistry:
    def __init__(self, heartbeat_timeout_seconds: float = 90.0) -> None:
        self._heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self._instances: Dict[str, Dict[str, Any]] = {}

    def register(
        self, router_uid: str, instance_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        existing = self._instances.get(instance_id)
        if existing is not None and existing["router_uid"] != router_uid:
            raise ValueError(
                f"Router instance {instance_id} is already registered to "
                f"{existing['router_uid']}"
            )
        now = time.time()
        instance = {
            **data,
            "router_uid": router_uid,
            "instance_id": instance_id,
            "registered_at": now,
            "last_heartbeat": now,
            "acked_revision": int(data.get("acked_revision", 0)),
        }
        self._instances[instance_id] = instance
        return self._render(instance, now)

    def heartbeat(self, instance_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        instance = self._instances.get(instance_id)
        if instance is None:
            raise KeyError(instance_id)
        now = time.time()
        immutable = {"router_uid", "instance_id", "registered_at"}
        instance.update({k: v for k, v in data.items() if k not in immutable})
        instance["last_heartbeat"] = now
        return self._render(instance, now)

    def ack(self, instance_id: str, revision: int, error: str = "") -> Dict[str, Any]:
        instance = self._instances.get(instance_id)
        if instance is None:
            raise KeyError(instance_id)
        current_revision = int(instance.get("acked_revision", 0))
        if revision < current_revision:
            raise ValueError(
                f"Stale Token Router ACK: revision {revision} is older than "
                f"acked revision {current_revision}"
            )
        instance["acked_revision"] = revision
        instance["config_error"] = error
        instance["last_heartbeat"] = time.time()
        return self._render(instance)

    def unregister(self, instance_id: str) -> bool:
        return self._instances.pop(instance_id, None) is not None

    def get(self, instance_id: str) -> Optional[Dict[str, Any]]:
        instance = self._instances.get(instance_id)
        return self._render(instance) if instance is not None else None

    def list(self, router_uid: Optional[str] = None) -> List[Dict[str, Any]]:
        now = time.time()
        values: Iterable[Dict[str, Any]] = self._instances.values()
        if router_uid is not None:
            values = (v for v in values if v["router_uid"] == router_uid)
        return sorted(
            (self._render(v, now) for v in values),
            key=lambda item: item["instance_id"],
        )

    def remove_router(self, router_uid: str) -> None:
        for instance_id in [
            key
            for key, value in self._instances.items()
            if value["router_uid"] == router_uid
        ]:
            self._instances.pop(instance_id, None)

    def _render(
        self, instance: Dict[str, Any], now: Optional[float] = None
    ) -> Dict[str, Any]:
        now = time.time() if now is None else now
        result = dict(instance)
        age = max(0.0, now - float(instance["last_heartbeat"]))
        result["heartbeat_age_seconds"] = age
        result["online"] = age <= self._heartbeat_timeout_seconds
        return result
