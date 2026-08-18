# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Supervisor-side orchestration controller for managed Token Routers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .router_assignment_store import RouterAssignmentStore
from .router_config_store import RouterConfigStore
from .router_deployment_store import RouterDeploymentStore
from .router_node_store import RouterNodeStore
from .router_scheduler import RouterScheduler
from .tokenizer_asset_store import TokenizerAssetStore

logger = logging.getLogger(__name__)


class RouterOrchestrationController:
    """Own persistent desired state while Agents own operating-system processes."""

    def __init__(
        self,
        db_path: str,
        config_store: RouterConfigStore,
        *,
        node_suspect_seconds: float = 30.0,
        node_offline_seconds: float = 45.0,
        node_timeout_seconds: Optional[float] = None,
    ) -> None:
        # ``node_timeout_seconds`` is retained as a compatibility alias for
        # callers that previously configured the single Router Agent timeout.
        if node_timeout_seconds is not None:
            node_suspect_seconds = node_timeout_seconds
            node_offline_seconds = node_timeout_seconds
        if node_suspect_seconds <= 0:
            raise ValueError("node_suspect_seconds must be greater than zero")
        if node_offline_seconds < node_suspect_seconds:
            raise ValueError(
                "node_offline_seconds must be greater than or equal to "
                "node_suspect_seconds"
            )
        self.deployments = RouterDeploymentStore(db_path)
        self.nodes = RouterNodeStore(db_path)
        self.assignments = RouterAssignmentStore(db_path)
        self.tokenizer_assets = TokenizerAssetStore(db_path)
        self.scheduler = RouterScheduler(
            config_store,
            self.deployments,
            self.nodes,
            self.assignments,
            self.tokenizer_assets,
            node_suspect_seconds=node_suspect_seconds,
            node_offline_seconds=node_offline_seconds,
        )
        self._config_store = config_store
        self._node_suspect_seconds = node_suspect_seconds
        self._node_offline_seconds = node_offline_seconds
        self._watch_epoch = uuid.uuid4().hex
        self.deployments.ensure_many(
            item["router_uid"] for item in self._config_store.list()
        )

    @staticmethod
    def _parse_time(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _node_age_seconds(
        self, node: Dict[str, Any], *, now: Optional[datetime] = None
    ) -> float:
        seen = self._parse_time(node.get("last_seen_at"))
        if seen is None:
            return float("inf")
        current = now or datetime.now(timezone.utc)
        return max(0.0, (current - seen).total_seconds())

    def _node_connectivity_status(
        self, node: Dict[str, Any], *, now: Optional[datetime] = None
    ) -> str:
        age = self._node_age_seconds(node, now=now)
        if age >= self._node_offline_seconds:
            return "offline"
        if age >= self._node_suspect_seconds:
            return "suspected"
        return "online"

    @staticmethod
    def _assignment_public(item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value for key, value in item.items() if key not in {"observed"}
        } | {"observed": item.get("observed", {})}

    def ensure_router(self, router_uid: str) -> Dict[str, Any]:
        return self.deployments.ensure(router_uid)

    def router_created(self, router_uid: str) -> None:
        self.deployments.ensure(router_uid)

    def router_config_updated(self, router_uid: str) -> None:
        self.scheduler.reconcile_router(router_uid)

    def router_enabled(self, router_uid: str, enabled: bool) -> None:
        deployment = self.deployments.ensure(router_uid)
        if deployment["management_mode"] == "managed":
            self.deployments.update(
                router_uid, desired_state="running" if enabled else "stopped"
            )
        self.scheduler.reconcile_router(router_uid)

    def router_delete_allowed(self, router_uid: str) -> bool:
        assignments = self.assignments.list(router_uid=router_uid)
        return all(item["observed_state"] == "stopped" for item in assignments)

    def router_deleted(self, router_uid: str) -> None:
        self.assignments.delete_router(router_uid)
        self.deployments.delete(router_uid)

    def get_deployment(self, router_uid: str) -> Optional[Dict[str, Any]]:
        if self._config_store.get(router_uid) is None:
            return None
        return self.deployments.ensure(router_uid)

    def update_deployment(
        self, router_uid: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self._config_store.get(router_uid) is None:
            raise KeyError(router_uid)
        deployment = self.deployments.update(
            router_uid,
            management_mode=data.get("management_mode"),
            desired_replicas=data.get("desired_replicas"),
            desired_state=data.get("desired_state"),
            placement=data.get("placement"),
            rollout=data.get("rollout"),
            expected_generation=data.get("deployment_generation"),
        )
        config = self._config_store.get(router_uid)
        assert config is not None
        if deployment["management_mode"] == "managed" and config["enabled"]:
            if "desired_state" not in data:
                deployment = self.deployments.update(
                    router_uid, desired_state="running"
                )
        self.scheduler.reconcile_router(router_uid)
        return self.deployments.ensure(router_uid)

    def validate_managed_deployment(self, router_uid: str) -> List[str]:
        deployment = self.deployments.ensure(router_uid)
        if deployment["management_mode"] != "managed":
            return []
        if deployment["desired_replicas"] > 0 and not self.scheduler.eligible_nodes(
            router_uid
        ):
            return [
                "No online active Router Agent node satisfies placement, capacity, "
                "and tokenizer asset requirements"
            ]
        return []

    def register_node(self, data: Dict[str, Any]) -> Dict[str, Any]:
        node = self.nodes.register(data)
        # Compatibility migration: turn the old static capability into persisted
        # legacy Bindings when the Asset is already present in the Catalog.
        for value in data.get("capabilities", {}).get("tokenizer_assets", []):
            asset_id = value if isinstance(value, str) else value.get("asset_id", "")
            asset = self.tokenizer_assets.get_asset(str(asset_id))
            if asset is None:
                continue
            binding = self.tokenizer_assets.upsert_binding(
                str(asset_id),
                node["node_id"],
                desired_state="present",
                binding_mode="legacy",
                username="legacy-capability-import",
            )
            try:
                self.tokenizer_assets.report_binding_status(
                    str(asset_id),
                    node["node_id"],
                    binding["generation"],
                    "ready",
                    observed_revision=asset["revision"],
                    observed_fingerprint=asset["fingerprint"],
                    local_path=str(asset.get("source", {}).get("path") or ""),
                )
            except ValueError:
                # A structured legacy capability may advertise a different
                # revision/fingerprint. It stays pending until Agent reconcile.
                pass
        self.scheduler.reconcile_all()
        return self.render_node(node)

    def heartbeat_node(self, node_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        node = self.nodes.heartbeat(node_id, data)
        self.scheduler.reconcile_all()
        return self.render_node(node)

    def sweep_nodes(self) -> List[Dict[str, Any]]:
        """Persist Router Agent lifecycle transitions and reconcile dependents."""

        now = datetime.now(timezone.utc)
        transitions: List[Dict[str, Any]] = []
        for node in self.nodes.list():
            previous = str(node.get("connectivity_status") or "offline")
            current = self._node_connectivity_status(node, now=now)
            if previous == current:
                continue
            updated = self.nodes.set_connectivity_status(
                node["node_id"],
                current,
                expected_last_seen_at=node.get("last_seen_at"),
            )
            if updated.get("connectivity_status") != current:
                # A concurrent heartbeat or re-registration won the race.  Do
                # not let an old sweep overwrite the recovered Agent state.
                continue
            if current == "offline":
                self.tokenizer_assets.mark_node_bindings_stale(node["node_id"])
            transition = {
                "node_id": node["node_id"],
                "previous_status": previous,
                "connectivity_status": current,
                "last_seen_at": updated.get("last_seen_at"),
                "heartbeat_age_seconds": self._node_age_seconds(updated, now=now),
            }
            transitions.append(transition)
            logger.info(
                "Router Agent lifecycle transition. node_id=%s previous=%s current=%s age=%.3f",
                node["node_id"],
                previous,
                current,
                transition["heartbeat_age_seconds"],
            )
        if transitions:
            self.scheduler.reconcile_all()
        return transitions

    def list_nodes(self, *, include_offline: bool = True) -> List[Dict[str, Any]]:
        rendered = [self.render_node(item) for item in self.nodes.list()]
        if not include_offline:
            rendered = [
                item for item in rendered if item["connectivity_status"] != "offline"
            ]
        return rendered

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        node = self.nodes.get(node_id)
        return self.render_node(node) if node is not None else None

    def render_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(node)
        heartbeat_age = self._node_age_seconds(node)
        connectivity_status = self._node_connectivity_status(node)
        result["connectivity_status"] = connectivity_status
        result["management_state"] = node["desired_state"]
        result["heartbeat_age_seconds"] = heartbeat_age
        result["online"] = connectivity_status == "online"
        result["can_schedule"] = (
            connectivity_status == "online" and node["desired_state"] == "active"
        )
        result["failure_reason"] = (
            "router_agent_heartbeat_timeout"
            if connectivity_status == "offline"
            else (
                "router_agent_heartbeat_delayed"
                if connectivity_status == "suspected"
                else ""
            )
        )
        assignments = self.assignments.list(node_id=node["node_id"])
        result["assignments"] = len(assignments)
        result["used_ports"] = sorted(item["listen_port"] for item in assignments)
        result["available_slots"] = max(0, node["max_instances"] - len(assignments))
        result["tokenizer_asset_bindings"] = self.tokenizer_assets.list_bindings(
            node_id=node["node_id"]
        )
        return result

    def set_node_state(self, node_id: str, state: str) -> Dict[str, Any]:
        node = self.nodes.set_desired_state(node_id, state)
        self.scheduler.reconcile_all()
        return self.render_node(node)

    def set_node_labels(self, node_id: str, labels: Dict[str, Any]) -> Dict[str, Any]:
        node = self.nodes.set_managed_labels(node_id, labels)
        self.scheduler.reconcile_all()
        return self.render_node(node)

    def list_assignments(
        self, *, router_uid: Optional[str] = None, node_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for item in self.assignments.list(router_uid=router_uid, node_id=node_id):
            public = self._assignment_public(item)
            node = self.nodes.get(item["node_id"])
            connectivity_status = (
                self._node_connectivity_status(node) if node is not None else "offline"
            )
            public["management_state"] = (
                "node_lost"
                if connectivity_status == "offline"
                else (
                    "node_suspected"
                    if connectivity_status == "suspected"
                    else "manageable"
                )
            )
            public["failure_reason"] = (
                "router_agent_offline"
                if connectivity_status == "offline"
                else (
                    "router_agent_suspected"
                    if connectivity_status == "suspected"
                    else ""
                )
            )
            config = self._config_store.get(item["router_uid"])
            asset_id = str((config or {}).get("tokenizer_asset_id") or "")
            if asset_id:
                binding = self.tokenizer_assets.get_binding(asset_id, item["node_id"])
                if binding is not None:
                    public["tokenizer_asset"] = {
                        "asset_id": asset_id,
                        "revision": binding["desired_revision"],
                        "fingerprint": binding["desired_fingerprint"],
                        "binding_generation": binding["generation"],
                        "observed_state": binding["observed_state"],
                        "local_path": binding["local_path"],
                    }
            result.append(public)
        return result

    def _snapshot_cursor(self, node_id: str, assignments: List[Dict[str, Any]]) -> str:
        stable = [
            {
                key: item.get(key)
                for key in (
                    "assignment_id",
                    "router_uid",
                    "replica_index",
                    "node_id",
                    "listen_host",
                    "listen_port",
                    "public_endpoint",
                    "desired_state",
                    "assignment_generation",
                    "config_revision",
                    "updated_at",
                )
            }
            for item in assignments
        ]
        digest = hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:20]
        return f"{self._watch_epoch}:{node_id}:{digest}"

    async def watch_assignments(
        self, node_id: str, after_cursor: str = "", wait_seconds: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        if self.nodes.get(node_id) is None:
            raise KeyError(node_id)
        deadline = time.monotonic() + min(max(wait_seconds, 0.0), 60.0)
        while True:
            assignments = self.list_assignments(node_id=node_id)
            cursor = self._snapshot_cursor(node_id, assignments)
            if cursor != after_cursor:
                return {
                    "cursor": cursor,
                    "full_snapshot": True,
                    "assignments": assignments,
                }
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            await asyncio.sleep(min(0.5, remaining))

    def report_assignment_status(
        self, assignment_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        assignment = self.assignments.get(assignment_id)
        if assignment is None:
            raise KeyError(assignment_id)
        if assignment["node_id"] != data["node_id"]:
            raise ValueError("Assignment does not belong to node_id")
        state = str(data["observed_state"])
        generation = int(data["assignment_generation"])
        result = self.assignments.report_status(
            assignment_id,
            generation,
            state,
            pid=data.get("pid"),
            instance_id=data.get("instance_id"),
            last_error=str(data.get("last_error") or ""),
            observed=data.get("observed", {}),
        )
        if state == "port_conflict":
            return self.scheduler.reassign_port(assignment_id)
        if state == "stopped" and result["desired_state"] == "stopped":
            router_uid = result["router_uid"]
            self.assignments.delete(assignment_id)
            self.scheduler.reconcile_router(router_uid)
            return {**result, "released": True}
        return result

    def _render_tokenizer_asset(self, asset: Dict[str, Any]) -> Dict[str, Any]:
        bindings = self.tokenizer_assets.list_bindings(asset_id=asset["asset_id"])
        metadata = asset.get("metadata", {})
        references = sum(
            config.get("tokenizer_asset_id") == asset["asset_id"]
            for config in self._config_store.list()
        )
        return {
            **asset,
            "status": "available" if asset["enabled"] else "disabled",
            "valid": bool(asset["enabled"]),
            "compatible_models": metadata.get("compatible_models", []),
            "bindings": len(bindings),
            "ready_bindings": sum(
                item["desired_state"] == "present" and item["observed_state"] == "ready"
                for item in bindings
            ),
            "binding_states": {
                state: sum(item["observed_state"] == state for item in bindings)
                for state in sorted({item["observed_state"] for item in bindings})
            },
            "router_references": references,
        }

    def list_tokenizer_assets(self) -> List[Dict[str, Any]]:
        return [
            self._render_tokenizer_asset(asset)
            for asset in self.tokenizer_assets.list_assets()
        ]

    def get_tokenizer_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        asset = self.tokenizer_assets.get_asset(asset_id)
        if asset is None:
            return None
        return {
            **self._render_tokenizer_asset(asset),
            "binding_items": self.tokenizer_assets.list_bindings(asset_id=asset_id),
        }

    def create_tokenizer_asset(
        self, data: Dict[str, Any], username: str = ""
    ) -> Dict[str, Any]:
        return self.tokenizer_assets.create_asset(data, username)

    def update_tokenizer_asset(
        self, asset_id: str, data: Dict[str, Any], username: str = ""
    ) -> Dict[str, Any]:
        result = self.tokenizer_assets.update_asset(asset_id, data, username)
        self.scheduler.reconcile_all()
        return result

    def delete_tokenizer_asset(self, asset_id: str) -> bool:
        for config in self._config_store.list():
            if config.get("tokenizer_asset_id") == asset_id:
                raise ValueError("Tokenizer Asset is referenced by a Router")
        return self.tokenizer_assets.delete_asset(asset_id)

    def list_tokenizer_asset_bindings(
        self, *, asset_id: Optional[str] = None, node_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return self.tokenizer_assets.list_bindings(asset_id=asset_id, node_id=node_id)

    def upsert_tokenizer_asset_bindings(
        self, asset_id: str, data: Dict[str, Any], username: str = ""
    ) -> List[Dict[str, Any]]:
        node_ids = [str(value) for value in data.get("node_ids", [])]
        selector = data.get("selector", {})
        if selector:
            node_ids.extend(
                node["node_id"]
                for node in self.nodes.list()
                if all(
                    node.get("labels", {}).get(key) == value
                    for key, value in selector.items()
                )
            )
        node_ids = sorted(set(node_ids))
        if not node_ids:
            raise ValueError("At least one Router Agent node must be selected")
        results = []
        for node_id in node_ids:
            if self.nodes.get(node_id) is None:
                raise KeyError(node_id)
            results.append(
                self.tokenizer_assets.upsert_binding(
                    asset_id,
                    node_id,
                    desired_state=str(data.get("desired_state") or "present"),
                    binding_mode=str(data.get("binding_mode") or "manual"),
                    owner_type=str(data.get("owner_type") or ""),
                    owner_id=str(data.get("owner_id") or ""),
                    username=username,
                )
            )
        self.scheduler.reconcile_all()
        return results

    def update_tokenizer_asset_binding(
        self, asset_id: str, node_id: str, data: Dict[str, Any], username: str = ""
    ) -> Dict[str, Any]:
        current = self.tokenizer_assets.get_binding(asset_id, node_id)
        if current is None:
            raise KeyError((asset_id, node_id))
        result = self.tokenizer_assets.upsert_binding(
            asset_id,
            node_id,
            desired_state=str(data.get("desired_state") or current["desired_state"]),
            binding_mode=str(data.get("binding_mode") or current["binding_mode"]),
            owner_type=str(data.get("owner_type", current["owner_type"])),
            owner_id=str(data.get("owner_id", current["owner_id"])),
            username=username,
        )
        self.scheduler.reconcile_all()
        return result

    def revalidate_tokenizer_asset_binding(
        self, asset_id: str, node_id: str, username: str = ""
    ) -> Dict[str, Any]:
        result = self.tokenizer_assets.revalidate_binding(asset_id, node_id, username)
        self.scheduler.reconcile_all()
        return result

    def delete_tokenizer_asset_binding(
        self, asset_id: str, node_id: str, *, force: bool = False
    ) -> bool:
        if any(
            assignment["node_id"] == node_id
            and (self._config_store.get(assignment["router_uid"]) or {}).get(
                "tokenizer_asset_id"
            )
            == asset_id
            for assignment in self.assignments.list()
        ):
            raise ValueError("Tokenizer Asset Binding is used by a Runtime Assignment")
        return self.tokenizer_assets.delete_binding(asset_id, node_id, force=force)

    def _binding_cursor(self, node_id: str, bindings: List[Dict[str, Any]]) -> str:
        stable = [
            {
                key: item.get(key)
                for key in (
                    "asset_id",
                    "node_id",
                    "desired_state",
                    "desired_revision",
                    "desired_fingerprint",
                    "binding_mode",
                    "generation",
                    "updated_at",
                )
            }
            for item in bindings
        ]
        digest = hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:20]
        return f"{self._watch_epoch}:asset:{node_id}:{digest}"

    async def watch_tokenizer_asset_bindings(
        self, node_id: str, after_cursor: str = "", wait_seconds: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        if self.nodes.get(node_id) is None:
            raise KeyError(node_id)
        deadline = time.monotonic() + min(max(wait_seconds, 0.0), 60.0)
        while True:
            bindings = self.tokenizer_assets.list_bindings(node_id=node_id)
            payload = []
            for binding in bindings:
                asset = self.tokenizer_assets.get_asset(binding["asset_id"])
                if asset is not None:
                    payload.append({**binding, "asset": asset})
            cursor = self._binding_cursor(node_id, payload)
            if cursor != after_cursor:
                return {"cursor": cursor, "full_snapshot": True, "bindings": payload}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            await asyncio.sleep(min(0.5, remaining))

    def report_tokenizer_asset_binding_status(
        self, asset_id: str, node_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        result = self.tokenizer_assets.report_binding_status(
            asset_id,
            node_id,
            int(data["generation"]),
            str(data["observed_state"]),
            observed_revision=str(data.get("observed_revision") or ""),
            observed_fingerprint=str(data.get("observed_fingerprint") or ""),
            local_path=str(data.get("local_path") or ""),
            last_error_code=str(data.get("last_error_code") or ""),
            last_error=str(data.get("last_error") or ""),
        )
        self.scheduler.reconcile_all()
        return result

    def validate_runtime_registration(
        self, router_uid: str, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        deployment = self.deployments.ensure(router_uid)
        if deployment["management_mode"] != "managed":
            return None
        required = ("assignment_id", "assignment_generation", "node_id")
        missing = [name for name in required if data.get(name) in (None, "")]
        if missing:
            raise ValueError(
                "Managed Router runtime is missing Assignment fields: "
                + ", ".join(missing)
            )
        assignment = self.assignments.get(str(data["assignment_id"]))
        if assignment is None:
            raise ValueError("Router Runtime Assignment does not exist")
        if assignment["router_uid"] != router_uid:
            raise ValueError("Assignment does not belong to router_uid")
        if assignment["node_id"] != data["node_id"]:
            raise ValueError("Assignment does not belong to node_id")
        if assignment["desired_state"] != "running":
            raise ValueError("Assignment is not in running desired state")
        if assignment["assignment_generation"] != int(data["assignment_generation"]):
            raise ValueError("Stale Router Runtime Assignment generation")
        if assignment["public_endpoint"].rstrip("/") != str(
            data.get("endpoint", "")
        ).rstrip("/"):
            raise ValueError("Router Runtime endpoint does not match Assignment")
        return assignment

    def validate_registered_instance(self, instance: Dict[str, Any]) -> None:
        deployment = self.deployments.ensure(instance["router_uid"])
        if deployment["management_mode"] != "managed":
            return
        self.validate_runtime_registration(instance["router_uid"], instance)

    def runtime_is_current(self, instance: Dict[str, Any]) -> bool:
        """Return whether a registered Runtime may serve the current deployment.

        External Runtime registrations remain backward compatible. Managed
        registrations must still match the live Assignment contract so an old
        process cannot receive traffic during generation replacement or a
        switch from external to managed operation.
        """

        try:
            self.validate_registered_instance(instance)
        except (KeyError, TypeError, ValueError):
            return False
        return True

    def runtime_is_controllable(self, instance: Dict[str, Any]) -> bool:
        """Return whether the control plane can currently manage a Runtime."""

        deployment = self.deployments.ensure(instance["router_uid"])
        if deployment["management_mode"] != "managed":
            return True
        node_id = str(instance.get("node_id") or "")
        node = self.nodes.get(node_id) if node_id else None
        return (
            node is not None
            and self._node_connectivity_status(node) == "online"
            and node["desired_state"] not in {"draining", "disabled"}
        )

    def runtime_registered(self, instance: Dict[str, Any]) -> None:
        assignment_id = instance.get("assignment_id")
        if not assignment_id:
            return
        self.assignments.report_status(
            str(assignment_id),
            int(instance["assignment_generation"]),
            "starting",
            instance_id=str(instance["instance_id"]),
        )

    def runtime_heartbeat(
        self, instance: Dict[str, Any], heartbeat: Dict[str, Any]
    ) -> None:
        assignment_id = instance.get("assignment_id")
        if not assignment_id:
            return
        status = str(heartbeat.get("status") or "starting")
        observed_state = "ready" if status in {"ready", "disabled"} else "starting"
        process = heartbeat.get("process", {})
        self.assignments.report_status(
            str(assignment_id),
            int(instance["assignment_generation"]),
            observed_state,
            pid=process.get("pid"),
            instance_id=str(instance["instance_id"]),
            observed={"runtime_status": status},
        )

    def runtime_acked(self, instance: Dict[str, Any], error: str = "") -> None:
        assignment_id = instance.get("assignment_id")
        if not assignment_id:
            return
        self.assignments.report_status(
            str(assignment_id),
            int(instance["assignment_generation"]),
            "failed" if error else "ready",
            instance_id=str(instance["instance_id"]),
            last_error=error,
        )

    def deployment_summary(
        self,
        router_uid: str,
        *,
        effective_ready_runtimes: Optional[List[Dict[str, Any]]] = None,
        controllable_ready_runtimes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        deployment = self.deployments.ensure(router_uid)
        assignments = self.assignments.list(router_uid=router_uid)
        observed_ready = sum(
            item["desired_state"] == "running" and item["observed_state"] == "ready"
            for item in assignments
        )
        pending = sum(
            item["desired_state"] == "running"
            and item["observed_state"] not in {"ready", "failed", "crash_loop"}
            for item in assignments
        )
        effective_ready = (
            observed_ready
            if effective_ready_runtimes is None
            else len(effective_ready_runtimes)
        )
        controllable_ready = (
            effective_ready
            if controllable_ready_runtimes is None
            else len(controllable_ready_runtimes)
        )
        result = {
            **deployment,
            "observed_ready_assignments": observed_ready,
            "effective_ready_runtimes": effective_ready,
            "controllable_ready_runtimes": controllable_ready,
            "ready_replicas": effective_ready,
            "pending_replicas": pending,
            "assignments": len(assignments),
        }
        if effective_ready == 0 and deployment["desired_replicas"] > 0:
            result["failure_reason"] = "no_effective_ready_runtime"
            result["recovery_state"] = "recovering" if pending > 0 else "blocked"
        elif controllable_ready < effective_ready:
            result["failure_reason"] = "router_agent_unavailable"
            result["recovery_state"] = "degraded"
        else:
            result["failure_reason"] = ""
            result["recovery_state"] = "stable"
        return result
