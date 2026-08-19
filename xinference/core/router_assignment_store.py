# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Persistent Router Runtime assignments and limited observed state."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class RouterAssignmentStore:
    _DESIRED_STATES = {"running", "stopped"}

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

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS token_router_assignments (
                    assignment_id TEXT PRIMARY KEY,
                    router_uid TEXT NOT NULL,
                    replica_index INTEGER NOT NULL,
                    node_id TEXT NOT NULL,
                    listen_host TEXT NOT NULL,
                    listen_port INTEGER NOT NULL,
                    public_endpoint TEXT NOT NULL,
                    desired_state TEXT NOT NULL,
                    assignment_generation INTEGER NOT NULL,
                    config_revision INTEGER NOT NULL,
                    observed_state TEXT NOT NULL DEFAULT 'pending',
                    pid INTEGER,
                    instance_id TEXT,
                    last_error TEXT NOT NULL DEFAULT '',
                    observed_json TEXT NOT NULL DEFAULT '{}',
                    last_seen_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(router_uid, replica_index),
                    UNIQUE(node_id, listen_port),
                    CHECK (desired_state IN ('running', 'stopped')),
                    CHECK (replica_index >= 0),
                    CHECK (assignment_generation >= 1),
                    CHECK (listen_port > 0 AND listen_port <= 65535)
                )"""
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        result = dict(row)
        result["replica_index"] = int(result["replica_index"])
        result["listen_port"] = int(result["listen_port"])
        result["assignment_generation"] = int(result["assignment_generation"])
        result["config_revision"] = int(result["config_revision"])
        result["observed"] = json.loads(result.pop("observed_json") or "{}")
        return result

    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if data["desired_state"] not in self._DESIRED_STATES:
            raise ValueError("Invalid Assignment desired_state")
        now = self._now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO token_router_assignments
                   (assignment_id, router_uid, replica_index, node_id,
                    listen_host, listen_port, public_endpoint, desired_state,
                    assignment_generation, config_revision, observed_state,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
                (
                    data["assignment_id"],
                    data["router_uid"],
                    int(data["replica_index"]),
                    data["node_id"],
                    data["listen_host"],
                    int(data["listen_port"]),
                    data["public_endpoint"],
                    data["desired_state"],
                    int(data.get("assignment_generation", 1)),
                    int(data["config_revision"]),
                    now,
                    now,
                ),
            )
        result = self.get(data["assignment_id"])
        assert result is not None
        return result

    def update_desired(
        self,
        assignment_id: str,
        *,
        node_id: Optional[str] = None,
        listen_host: Optional[str] = None,
        listen_port: Optional[int] = None,
        public_endpoint: Optional[str] = None,
        desired_state: Optional[str] = None,
        config_revision: Optional[int] = None,
        bump_generation: bool = False,
    ) -> Dict[str, Any]:
        current = self.get(assignment_id)
        if current is None:
            raise KeyError(assignment_id)
        next_state = desired_state or current["desired_state"]
        if next_state not in self._DESIRED_STATES:
            raise ValueError("Invalid Assignment desired_state")
        next_values = {
            "node_id": node_id or current["node_id"],
            "listen_host": listen_host or current["listen_host"],
            "listen_port": int(listen_port or current["listen_port"]),
            "public_endpoint": public_endpoint or current["public_endpoint"],
            "desired_state": next_state,
            "config_revision": int(
                config_revision
                if config_revision is not None
                else current["config_revision"]
            ),
        }
        contract_changed = any(
            next_values[key] != current[key]
            for key in ("node_id", "listen_host", "listen_port", "public_endpoint")
        ) or (current["desired_state"] == "stopped" and next_state == "running")
        generation = current["assignment_generation"] + int(
            bump_generation or contract_changed
        )
        observed_state = (
            "pending"
            if generation != current["assignment_generation"]
            else current["observed_state"]
        )
        now = self._now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE token_router_assignments
                   SET node_id = ?, listen_host = ?, listen_port = ?,
                       public_endpoint = ?, desired_state = ?,
                       assignment_generation = ?, config_revision = ?,
                       observed_state = ?, updated_at = ?
                   WHERE assignment_id = ?""",
                (
                    next_values["node_id"],
                    next_values["listen_host"],
                    next_values["listen_port"],
                    next_values["public_endpoint"],
                    next_values["desired_state"],
                    generation,
                    next_values["config_revision"],
                    observed_state,
                    now,
                    assignment_id,
                ),
            )
        result = self.get(assignment_id)
        assert result is not None
        return result

    def report_status(
        self,
        assignment_id: str,
        assignment_generation: int,
        observed_state: str,
        *,
        pid: Optional[int] = None,
        instance_id: Optional[str] = None,
        last_error: str = "",
        observed: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        current = self.get(assignment_id)
        if current is None:
            raise KeyError(assignment_id)
        if current["assignment_generation"] != assignment_generation:
            raise ValueError(
                f"Stale Assignment generation {assignment_generation}; current is "
                f"{current['assignment_generation']}"
            )
        now = self._now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE token_router_assignments
                   SET observed_state = ?, pid = ?, instance_id = ?, last_error = ?,
                       observed_json = ?, last_seen_at = ?, updated_at = ?
                   WHERE assignment_id = ?""",
                (
                    observed_state,
                    pid,
                    instance_id,
                    last_error,
                    json.dumps(
                        current["observed"] if observed is None else observed,
                        ensure_ascii=False,
                    ),
                    now,
                    now,
                    assignment_id,
                ),
            )
        result = self.get(assignment_id)
        assert result is not None
        return result

    def get(self, assignment_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM token_router_assignments WHERE assignment_id = ?",
                (assignment_id,),
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def get_by_replica(
        self, router_uid: str, replica_index: int
    ) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM token_router_assignments
                   WHERE router_uid = ? AND replica_index = ?""",
                (router_uid, replica_index),
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def list(
        self, *, router_uid: Optional[str] = None, node_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        values: List[Any] = []
        if router_uid is not None:
            clauses.append("router_uid = ?")
            values.append(router_uid)
        if node_id is not None:
            clauses.append("node_id = ?")
            values.append(node_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM token_router_assignments"
                + where
                + " ORDER BY router_uid, replica_index",
                values,
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def delete(self, assignment_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM token_router_assignments WHERE assignment_id = ?",
                (assignment_id,),
            )
        return cursor.rowcount > 0

    def delete_router(self, router_uid: str) -> int:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM token_router_assignments WHERE router_uid = ?",
                (router_uid,),
            )
        return cursor.rowcount
