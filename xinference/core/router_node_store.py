# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Persistent Router Agent node inventory."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class RouterNodeStore:
    _STATES = {"active", "cordoned", "draining", "disabled"}

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _columns(conn: sqlite3.Connection) -> set[str]:
        return {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(token_router_nodes)").fetchall()
        }

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS token_router_nodes (
                    node_id TEXT PRIMARY KEY,
                    advertise_host TEXT NOT NULL,
                    port_range_start INTEGER NOT NULL,
                    port_range_end INTEGER NOT NULL,
                    max_instances INTEGER NOT NULL,
                    labels_json TEXT NOT NULL DEFAULT '{}',
                    reported_labels_json TEXT NOT NULL DEFAULT '{}',
                    managed_labels_json TEXT NOT NULL DEFAULT '{}',
                    capabilities_json TEXT NOT NULL DEFAULT '{}',
                    software_version TEXT NOT NULL DEFAULT '',
                    software_revision TEXT,
                    desired_state TEXT NOT NULL DEFAULT 'active',
                    connectivity_status TEXT NOT NULL DEFAULT 'offline',
                    suspected_at TEXT,
                    offline_at TEXT,
                    observed_json TEXT NOT NULL DEFAULT '{}',
                    last_seen_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    CHECK (desired_state IN ('active', 'cordoned', 'draining', 'disabled')),
                    CHECK (port_range_start > 0),
                    CHECK (port_range_end >= port_range_start),
                    CHECK (max_instances > 0)
                )"""
            )
            columns = self._columns(conn)
            if "reported_labels_json" not in columns:
                conn.execute(
                    "ALTER TABLE token_router_nodes ADD COLUMN "
                    "reported_labels_json TEXT NOT NULL DEFAULT '{}'"
                )
                conn.execute(
                    "UPDATE token_router_nodes SET reported_labels_json = labels_json"
                )
            if "managed_labels_json" not in columns:
                conn.execute(
                    "ALTER TABLE token_router_nodes ADD COLUMN "
                    "managed_labels_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "connectivity_status" not in columns:
                conn.execute(
                    "ALTER TABLE token_router_nodes ADD COLUMN "
                    "connectivity_status TEXT NOT NULL DEFAULT 'offline'"
                )
            if "suspected_at" not in columns:
                conn.execute(
                    "ALTER TABLE token_router_nodes ADD COLUMN suspected_at TEXT"
                )
            if "offline_at" not in columns:
                conn.execute(
                    "ALTER TABLE token_router_nodes ADD COLUMN offline_at TEXT"
                )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _labels(row: sqlite3.Row) -> tuple[Dict[str, Any], Dict[str, Any]]:
        keys = set(row.keys())
        reported_raw = (
            row["reported_labels_json"]
            if "reported_labels_json" in keys
            else row["labels_json"]
        )
        managed_raw = (
            row["managed_labels_json"] if "managed_labels_json" in keys else "{}"
        )
        reported = json.loads(reported_raw or "{}")
        managed = json.loads(managed_raw or "{}")
        return reported, managed

    @classmethod
    def _row_to_dict(cls, row: sqlite3.Row) -> Dict[str, Any]:
        reported, managed = cls._labels(row)
        result = {
            "node_id": row["node_id"],
            "advertise_host": row["advertise_host"],
            "port_range_start": int(row["port_range_start"]),
            "port_range_end": int(row["port_range_end"]),
            "max_instances": int(row["max_instances"]),
            "reported_labels": reported,
            "managed_labels": managed,
            # Backward-compatible merged placement view; managed values win.
            "labels": {**reported, **managed},
            "capabilities": json.loads(row["capabilities_json"] or "{}"),
            "software_version": row["software_version"],
            "software_revision": row["software_revision"],
            "desired_state": row["desired_state"],
            "connectivity_status": row["connectivity_status"],
            "suspected_at": row["suspected_at"],
            "offline_at": row["offline_at"],
            "last_seen_at": row["last_seen_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        result.update(json.loads(row["observed_json"] or "{}"))
        return result

    @staticmethod
    def _validate(data: Dict[str, Any]) -> None:
        if not str(data.get("node_id") or "").strip():
            raise ValueError("node_id is required")
        if not str(data.get("advertise_host") or "").strip():
            raise ValueError("advertise_host is required")
        start = int(data["port_range_start"])
        end = int(data["port_range_end"])
        maximum = int(data["max_instances"])
        if not 1024 <= start <= 65535 or not start <= end <= 65535:
            raise ValueError("Router node port range must be between 1024 and 65535")
        if maximum <= 0:
            raise ValueError("max_instances must be greater than zero")
        if maximum > end - start + 1:
            raise ValueError("max_instances exceeds the Router node port range")
        labels = data.get("reported_labels")
        if labels is None:
            labels = data.get("labels", {})
        if not isinstance(labels, dict):
            raise ValueError("reported_labels must be an object")
        if not isinstance(data.get("capabilities", {}), dict):
            raise ValueError("capabilities must be an object")

    def register(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self._validate(data)
        node_id = str(data["node_id"])
        current = self.get(node_id)
        now = self._now()
        desired_state = current["desired_state"] if current else "active"
        created_at = current["created_at"] if current else now
        reported_labels_value = data.get("reported_labels")
        if reported_labels_value is None:
            reported_labels_value = data.get("labels", {})
        reported_labels = dict(reported_labels_value)
        managed_labels = current.get("managed_labels", {}) if current else {}
        merged_labels = {**reported_labels, **managed_labels}
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO token_router_nodes
                   (node_id, advertise_host, port_range_start, port_range_end,
                    max_instances, labels_json, reported_labels_json,
                    managed_labels_json, capabilities_json, software_version,
                    software_revision, desired_state, connectivity_status,
                    suspected_at, offline_at, observed_json, last_seen_at,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'online', NULL,
                           NULL, '{}', ?, ?, ?)
                   ON CONFLICT(node_id) DO UPDATE SET
                    advertise_host = excluded.advertise_host,
                    port_range_start = excluded.port_range_start,
                    port_range_end = excluded.port_range_end,
                    max_instances = excluded.max_instances,
                    labels_json = excluded.labels_json,
                    reported_labels_json = excluded.reported_labels_json,
                    capabilities_json = excluded.capabilities_json,
                    software_version = excluded.software_version,
                    software_revision = excluded.software_revision,
                    connectivity_status = 'online',
                    suspected_at = NULL,
                    offline_at = NULL,
                    last_seen_at = excluded.last_seen_at,
                    updated_at = excluded.updated_at""",
                (
                    node_id,
                    str(data["advertise_host"]),
                    int(data["port_range_start"]),
                    int(data["port_range_end"]),
                    int(data["max_instances"]),
                    json.dumps(merged_labels, ensure_ascii=False),
                    json.dumps(reported_labels, ensure_ascii=False),
                    json.dumps(managed_labels, ensure_ascii=False),
                    json.dumps(data.get("capabilities", {}), ensure_ascii=False),
                    str(data.get("software_version") or ""),
                    data.get("software_revision"),
                    desired_state,
                    now,
                    created_at,
                    now,
                ),
            )
        result = self.get(node_id)
        assert result is not None
        return result

    def heartbeat(self, node_id: str, observed: Dict[str, Any]) -> Dict[str, Any]:
        if self.get(node_id) is None:
            raise KeyError(node_id)
        now = self._now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE token_router_nodes
                   SET observed_json = ?, connectivity_status = 'online',
                       suspected_at = NULL, offline_at = NULL,
                       last_seen_at = ?, updated_at = ?
                   WHERE node_id = ?""",
                (json.dumps(observed, ensure_ascii=False), now, now, node_id),
            )
        result = self.get(node_id)
        assert result is not None
        return result

    def set_connectivity_status(
        self,
        node_id: str,
        status: str,
        *,
        expected_last_seen_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        if status not in {"online", "suspected", "offline"}:
            raise ValueError(f"Unsupported Router node connectivity status: {status}")
        current = self.get(node_id)
        if current is None:
            raise KeyError(node_id)
        if current.get("connectivity_status") == status:
            return current
        now = self._now()
        suspected_at = now if status == "suspected" else None
        offline_at = now if status == "offline" else None
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE token_router_nodes
                   SET connectivity_status = ?,
                       suspected_at = CASE
                           WHEN ? = 'suspected' THEN COALESCE(suspected_at, ?)
                           WHEN ? = 'online' THEN NULL
                           ELSE suspected_at
                       END,
                       offline_at = CASE
                           WHEN ? = 'offline' THEN COALESCE(offline_at, ?)
                           WHEN ? = 'online' THEN NULL
                           ELSE offline_at
                       END,
                       updated_at = ?
                   WHERE node_id = ?
                     AND (? IS NULL OR last_seen_at = ?)""",
                (
                    status,
                    status,
                    suspected_at,
                    status,
                    status,
                    offline_at,
                    status,
                    now,
                    node_id,
                    expected_last_seen_at,
                    expected_last_seen_at,
                ),
            )
        result = self.get(node_id)
        assert result is not None
        return result

    def set_desired_state(self, node_id: str, desired_state: str) -> Dict[str, Any]:
        if desired_state not in self._STATES:
            raise ValueError(f"Unsupported Router node state: {desired_state}")
        if self.get(node_id) is None:
            raise KeyError(node_id)
        now = self._now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE token_router_nodes SET desired_state = ?, updated_at = ? WHERE node_id = ?",
                (desired_state, now, node_id),
            )
        result = self.get(node_id)
        assert result is not None
        return result

    def set_managed_labels(
        self, node_id: str, labels: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not isinstance(labels, dict):
            raise ValueError("managed labels must be an object")
        current = self.get(node_id)
        if current is None:
            raise KeyError(node_id)
        reserved = [key for key in labels if str(key).startswith("system.")]
        if reserved:
            raise ValueError("managed labels cannot use the system.* namespace")
        merged = {**current.get("reported_labels", {}), **labels}
        now = self._now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE token_router_nodes SET managed_labels_json = ?,
                     labels_json = ?, updated_at = ? WHERE node_id = ?""",
                (
                    json.dumps(labels, ensure_ascii=False),
                    json.dumps(merged, ensure_ascii=False),
                    now,
                    node_id,
                ),
            )
        result = self.get(node_id)
        assert result is not None
        return result

    def get(self, node_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM token_router_nodes WHERE node_id = ?", (node_id,)
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def list(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM token_router_nodes ORDER BY node_id"
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]
