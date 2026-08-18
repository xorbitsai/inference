# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Desired-state scheduler for managed Token Router runtimes."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set

from .router_assignment_store import RouterAssignmentStore
from .router_config_store import RouterConfigStore
from .router_deployment_store import RouterDeploymentStore
from .router_node_store import RouterNodeStore
from .tokenizer_asset_store import TokenizerAssetStore

logger = logging.getLogger(__name__)


class RouterScheduler:
    def __init__(
        self,
        config_store: RouterConfigStore,
        deployment_store: RouterDeploymentStore,
        node_store: RouterNodeStore,
        assignment_store: RouterAssignmentStore,
        asset_store: TokenizerAssetStore,
        *,
        node_suspect_seconds: float = 30.0,
        node_offline_seconds: float = 45.0,
    ) -> None:
        if node_suspect_seconds <= 0:
            raise ValueError("node_suspect_seconds must be greater than zero")
        if node_offline_seconds < node_suspect_seconds:
            raise ValueError(
                "node_offline_seconds must be greater than or equal to "
                "node_suspect_seconds"
            )
        self._config_store = config_store
        self._deployment_store = deployment_store
        self._node_store = node_store
        self._assignment_store = assignment_store
        self._asset_store = asset_store
        self._node_suspect_seconds = node_suspect_seconds
        self._node_offline_seconds = node_offline_seconds

    @staticmethod
    def _parse_time(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _node_age_seconds(self, node: Dict[str, Any]) -> float:
        last_seen = self._parse_time(node.get("last_seen_at"))
        if last_seen is None:
            return float("inf")
        return max(0.0, (datetime.now(timezone.utc) - last_seen).total_seconds())

    def _node_schedulable(self, node: Dict[str, Any]) -> bool:
        return (
            node.get("connectivity_status", "online") == "online"
            and self._node_age_seconds(node) < self._node_suspect_seconds
        )

    def _node_offline(self, node: Dict[str, Any]) -> bool:
        return (
            node.get("connectivity_status") == "offline"
            or self._node_age_seconds(node) >= self._node_offline_seconds
        )

    def _node_has_asset(self, node: Dict[str, Any], config: Dict[str, Any]) -> bool:
        asset_id = str(config.get("tokenizer_asset_id") or "")
        if not asset_id:
            return True
        binding = self._asset_store.get_binding(asset_id, node["node_id"])
        if binding is not None:
            return (
                binding["desired_state"] == "present"
                and binding["observed_state"] == "ready"
                and binding["observed_revision"]
                == str(config.get("tokenizer_asset_revision") or "")
                and binding["observed_fingerprint"].lower()
                == str(config.get("tokenizer_asset_fingerprint") or "").lower()
            )

        # Compatibility window only: old Agents may still publish a static
        # tokenizer_assets capability. Registration imports it into a legacy
        # Binding whenever the Catalog contains the Asset.
        assets = node.get("capabilities", {}).get("tokenizer_assets", [])
        if assets:
            logger.warning(
                "Using deprecated Router Agent tokenizer_assets capability for node %s",
                node["node_id"],
            )
        for asset in assets:
            if isinstance(asset, str) and asset == asset_id:
                return True
            if (
                isinstance(asset, dict)
                and asset.get("asset_id") == asset_id
                and str(asset.get("revision") or "")
                == str(config.get("tokenizer_asset_revision") or "")
                and str(asset.get("fingerprint") or "").lower()
                == str(config.get("tokenizer_asset_fingerprint") or "").lower()
            ):
                return True
        return False

    @staticmethod
    def _placement_matches(node: Dict[str, Any], placement: Dict[str, Any]) -> bool:
        required = placement.get("labels", {})
        if not isinstance(required, dict):
            return False
        labels = node.get("labels", {})
        if not all(labels.get(key) == value for key, value in required.items()):
            return False

        configured_node_ids = placement.get("node_ids")
        if configured_node_ids is None and placement.get("node_id"):
            configured_node_ids = [placement["node_id"]]
        if configured_node_ids is None:
            return True
        if not isinstance(configured_node_ids, list):
            return False
        allowed_node_ids = {
            str(node_id).strip()
            for node_id in configured_node_ids
            if str(node_id).strip()
        }
        return bool(allowed_node_ids) and node["node_id"] in allowed_node_ids

    def _node_matches_deployment(
        self,
        node: Dict[str, Any],
        config: Dict[str, Any],
        deployment: Dict[str, Any],
    ) -> bool:
        return self._placement_matches(
            node, deployment.get("placement", {})
        ) and self._node_has_asset(node, config)

    def _ensure_on_demand_bindings(
        self, config: Dict[str, Any], deployment: Dict[str, Any]
    ) -> None:
        asset_id = str(config.get("tokenizer_asset_id") or "")
        if not asset_id or self._asset_store.get_asset(asset_id) is None:
            return
        for node in self._node_store.list():
            if node["desired_state"] != "active" or not self._node_schedulable(node):
                continue
            if not self._placement_matches(node, deployment.get("placement", {})):
                continue
            if self._asset_store.get_binding(asset_id, node["node_id"]) is None:
                self._asset_store.upsert_binding(
                    asset_id,
                    node["node_id"],
                    desired_state="present",
                    binding_mode="on_demand",
                    owner_type="router",
                    owner_id=str(config["router_uid"]),
                    username="scheduler",
                )

    def _candidate_nodes(
        self,
        config: Dict[str, Any],
        deployment: Dict[str, Any],
        assignments: Iterable[Dict[str, Any]],
        *,
        exclude_nodes: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        exclude_nodes = exclude_nodes or set()
        counts: Dict[str, int] = {}
        for assignment in assignments:
            # A stopped Assignment remains persisted until its Agent confirms
            # that the process has exited. Count every persisted Assignment so
            # a draining Runtime continues to reserve both capacity and port.
            counts[assignment["node_id"]] = counts.get(assignment["node_id"], 0) + 1
        candidates = []
        for node in self._node_store.list():
            if node["node_id"] in exclude_nodes:
                continue
            if node["desired_state"] != "active" or not self._node_schedulable(node):
                continue
            if counts.get(node["node_id"], 0) >= node["max_instances"]:
                continue
            if not self._node_matches_deployment(node, config, deployment):
                continue
            node = dict(node)
            node["_load"] = counts.get(node["node_id"], 0) / node["max_instances"]
            candidates.append(node)
        return sorted(candidates, key=lambda item: (item["_load"], item["node_id"]))

    def eligible_nodes(self, router_uid: str) -> List[Dict[str, Any]]:
        """Return online active nodes satisfying this Router's placement/assets."""
        config = self._config_store.get(router_uid)
        if config is None:
            return []
        deployment = self._deployment_store.ensure(router_uid)
        return self._candidate_nodes(config, deployment, self._assignment_store.list())

    @staticmethod
    def _free_port(
        node: Dict[str, Any], assignments: Iterable[Dict[str, Any]]
    ) -> Optional[int]:
        used = {
            int(item["listen_port"])
            for item in assignments
            if item["node_id"] == node["node_id"]
        }
        for port in range(node["port_range_start"], node["port_range_end"] + 1):
            if port not in used:
                return port
        return None

    @staticmethod
    def _assignment_id(router_uid: str, replica_index: int) -> str:
        return f"{router_uid}-{replica_index}"

    def reconcile_router(self, router_uid: str) -> List[Dict[str, Any]]:
        config = self._config_store.get(router_uid)
        if config is None:
            return []
        deployment = self._deployment_store.ensure(router_uid)
        assignments = self._assignment_store.list(router_uid=router_uid)
        should_run = (
            deployment["management_mode"] == "managed"
            and deployment["desired_state"] == "running"
            and bool(config["enabled"])
        )
        desired_replicas = deployment["desired_replicas"] if should_run else 0
        if desired_replicas:
            self._ensure_on_demand_bindings(config, deployment)

        # Stop surplus or disabled assignments. They remain persisted until the
        # Agent reports stopped, so the port cannot be reused prematurely.
        for assignment in assignments:
            if assignment["replica_index"] >= desired_replicas:
                if assignment["desired_state"] != "stopped":
                    self._assignment_store.update_desired(
                        assignment["assignment_id"], desired_state="stopped"
                    )

        assignments = self._assignment_store.list()
        router_assignments = [
            item for item in assignments if item["router_uid"] == router_uid
        ]
        used_nodes: Set[str] = {
            item["node_id"]
            for item in router_assignments
            if item["replica_index"] < desired_replicas
            and item["desired_state"] == "running"
        }
        auto_failover = bool(deployment.get("rollout", {}).get("auto_failover", False))

        for replica_index in range(desired_replicas):
            current = next(
                (
                    item
                    for item in router_assignments
                    if item["replica_index"] == replica_index
                ),
                None,
            )
            if current is not None:
                node = self._node_store.get(current["node_id"])
                node_draining = node is not None and node["desired_state"] in {
                    "draining",
                    "disabled",
                }
                node_lost = node is None or self._node_offline(node)
                placement_mismatch = (
                    node is not None
                    and not node_lost
                    and not self._placement_matches(
                        node, deployment.get("placement", {})
                    )
                )
                asset_unavailable = (
                    node is not None
                    and not node_lost
                    and not placement_mismatch
                    and not self._node_has_asset(node, config)
                )
                should_replace = (
                    node_draining
                    or placement_mismatch
                    or asset_unavailable
                    or (node_lost and auto_failover)
                )
                if should_replace:
                    previous_node_id = current["node_id"]
                    candidates = self._candidate_nodes(
                        config,
                        deployment,
                        assignments,
                        exclude_nodes={current["node_id"]} | used_nodes,
                    ) or self._candidate_nodes(
                        config,
                        deployment,
                        assignments,
                        exclude_nodes={current["node_id"]},
                    )
                    if candidates:
                        target = candidates[0]
                        port = self._free_port(target, assignments)
                        if port is not None:
                            current = self._assignment_store.update_desired(
                                current["assignment_id"],
                                node_id=target["node_id"],
                                listen_host=target["advertise_host"],
                                listen_port=port,
                                public_endpoint=f"http://{target['advertise_host']}:{port}",
                                desired_state="running",
                                config_revision=config["revision"],
                                bump_generation=True,
                            )
                            assignments = [
                                (
                                    current
                                    if item["assignment_id"] == current["assignment_id"]
                                    else item
                                )
                                for item in assignments
                            ]
                            router_assignments = [
                                (
                                    current
                                    if item["assignment_id"] == current["assignment_id"]
                                    else item
                                )
                                for item in router_assignments
                            ]
                            used_nodes.discard(previous_node_id)
                            used_nodes.add(target["node_id"])
                    elif (
                        not node_lost
                        and not asset_unavailable
                        and current["desired_state"] != "stopped"
                    ):
                        # Administrative drain/disable and an explicit placement
                        # mismatch are desired-state changes.  When an Agent is
                        # lost, or its Tokenizer Asset Binding is temporarily stale
                        # during recovery, fencing the old generation before a
                        # replacement exists would unnecessarily interrupt a
                        # Runtime whose data plane may still be healthy.
                        current = self._assignment_store.update_desired(
                            current["assignment_id"],
                            desired_state="stopped",
                            config_revision=config["revision"],
                            bump_generation=True,
                        )
                        assignments = [
                            (
                                current
                                if item["assignment_id"] == current["assignment_id"]
                                else item
                            )
                            for item in assignments
                        ]
                        router_assignments = [
                            (
                                current
                                if item["assignment_id"] == current["assignment_id"]
                                else item
                            )
                            for item in router_assignments
                        ]
                        used_nodes.discard(previous_node_id)
                else:
                    self._assignment_store.update_desired(
                        current["assignment_id"],
                        desired_state="running",
                        config_revision=config["revision"],
                    )
                    used_nodes.add(current["node_id"])
                continue

            candidates = self._candidate_nodes(
                config, deployment, assignments, exclude_nodes=used_nodes
            ) or self._candidate_nodes(config, deployment, assignments)
            for target in candidates:
                port = self._free_port(target, assignments)
                if port is None:
                    continue
                created = self._assignment_store.create(
                    {
                        "assignment_id": self._assignment_id(router_uid, replica_index),
                        "router_uid": router_uid,
                        "replica_index": replica_index,
                        "node_id": target["node_id"],
                        "listen_host": target["advertise_host"],
                        "listen_port": port,
                        "public_endpoint": f"http://{target['advertise_host']}:{port}",
                        "desired_state": "running",
                        "assignment_generation": 1,
                        "config_revision": config["revision"],
                    }
                )
                assignments.append(created)
                router_assignments.append(created)
                used_nodes.add(target["node_id"])
                break

        return self._assignment_store.list(router_uid=router_uid)

    def reconcile_all(self) -> None:
        for config in self._config_store.list():
            self.reconcile_router(config["router_uid"])

    def reassign_port(self, assignment_id: str) -> Dict[str, Any]:
        assignment = self._assignment_store.get(assignment_id)
        if assignment is None:
            raise KeyError(assignment_id)
        node = self._node_store.get(assignment["node_id"])
        if node is None:
            raise KeyError(assignment["node_id"])
        assignments = self._assignment_store.list()
        used = {
            int(item["listen_port"])
            for item in assignments
            if item["node_id"] == node["node_id"]
            and item["assignment_id"] != assignment_id
        }
        port = next(
            (
                candidate
                for candidate in range(
                    node["port_range_start"], node["port_range_end"] + 1
                )
                if candidate not in used and candidate != assignment["listen_port"]
            ),
            None,
        )
        if port is None:
            raise RuntimeError(f"No free Router port on node {node['node_id']}")
        return self._assignment_store.update_desired(
            assignment_id,
            listen_port=port,
            public_endpoint=f"http://{node['advertise_host']}:{port}",
            bump_generation=True,
        )
