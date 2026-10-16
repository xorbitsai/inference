# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Persistent desired deployment state for Token Router runtimes."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


class RouterDeploymentStore:
    """SQLite-backed Router deployment desired-state store."""

    _MODES = {"external", "managed"}
    _STATES = {"running", "stopped"}

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
                """CREATE TABLE IF NOT EXISTS token_router_deployments (
                    router_uid TEXT PRIMARY KEY,
                    management_mode TEXT NOT NULL DEFAULT 'external',
                    desired_replicas INTEGER NOT NULL DEFAULT 1,
                    desired_state TEXT NOT NULL DEFAULT 'stopped',
                    placement_json TEXT NOT NULL DEFAULT '{}',
                    rollout_json TEXT NOT NULL DEFAULT '{}',
                    deployment_generation INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    CHECK (management_mode IN ('external', 'managed')),
                    CHECK (desired_state IN ('running', 'stopped')),
                    CHECK (desired_replicas >= 0)
                )"""
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "router_uid": row["router_uid"],
            "management_mode": row["management_mode"],
            "desired_replicas": int(row["desired_replicas"]),
            "desired_state": row["desired_state"],
            "placement": json.loads(row["placement_json"] or "{}"),
            "rollout": json.loads(row["rollout_json"] or "{}"),
            "deployment_generation": int(row["deployment_generation"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @classmethod
    def _validate(
        cls,
        management_mode: str,
        desired_replicas: int,
        desired_state: str,
        placement: Dict[str, Any],
        rollout: Dict[str, Any],
    ) -> None:
        if management_mode not in cls._MODES:
            raise ValueError("management_mode must be external or managed")
        if desired_state not in cls._STATES:
            raise ValueError("desired_state must be running or stopped")
        if desired_replicas < 0:
            raise ValueError("desired_replicas must be greater than or equal to zero")
        if not isinstance(placement, dict):
            raise ValueError("placement must be an object")
        if not isinstance(rollout, dict):
            raise ValueError("rollout must be an object")

    def ensure(self, router_uid: str) -> Dict[str, Any]:
        existing = self.get(router_uid)
        if existing is not None:
            return existing
        now = self._now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO token_router_deployments
                   (router_uid, management_mode, desired_replicas, desired_state,
                    placement_json, rollout_json, deployment_generation,
                    created_at, updated_at)
                   VALUES (?, 'external', 1, 'stopped', '{}', '{}', 1, ?, ?)""",
                (router_uid, now, now),
            )
        result = self.get(router_uid)
        assert result is not None
        return result

    def ensure_many(self, router_uids: Iterable[str]) -> None:
        for router_uid in router_uids:
            self.ensure(router_uid)

    def get(self, router_uid: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM token_router_deployments WHERE router_uid = ?",
                (router_uid,),
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def list(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM token_router_deployments ORDER BY router_uid"
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def update(
        self,
        router_uid: str,
        *,
        management_mode: Optional[str] = None,
        desired_replicas: Optional[int] = None,
        desired_state: Optional[str] = None,
        placement: Optional[Dict[str, Any]] = None,
        rollout: Optional[Dict[str, Any]] = None,
        expected_generation: Optional[int] = None,
    ) -> Dict[str, Any]:
        current = self.ensure(router_uid)
        next_value: Dict[str, Any] = {
            "management_mode": (
                management_mode
                if management_mode is not None
                else current["management_mode"]
            ),
            "desired_replicas": (
                desired_replicas
                if desired_replicas is not None
                else current["desired_replicas"]
            ),
            "desired_state": (
                desired_state if desired_state is not None else current["desired_state"]
            ),
            "placement": placement if placement is not None else current["placement"],
            "rollout": rollout if rollout is not None else current["rollout"],
        }
        self._validate(
            management_mode=str(next_value["management_mode"]),
            desired_replicas=int(next_value["desired_replicas"]),
            desired_state=str(next_value["desired_state"]),
            placement=dict(next_value["placement"]),
            rollout=dict(next_value["rollout"]),
        )
        if (
            expected_generation is not None
            and current["deployment_generation"] != expected_generation
        ):
            raise RuntimeError(
                "Token Router deployment generation conflict: "
                f"expected {expected_generation}, "
                f"current {current['deployment_generation']}"
            )
        changed = any(current[key] != value for key, value in next_value.items())
        if not changed:
            return current
        now = self._now()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """UPDATE token_router_deployments
                   SET management_mode = ?, desired_replicas = ?, desired_state = ?,
                       placement_json = ?, rollout_json = ?,
                       deployment_generation = deployment_generation + 1,
                       updated_at = ?
                   WHERE router_uid = ? AND deployment_generation = ?""",
                (
                    next_value["management_mode"],
                    next_value["desired_replicas"],
                    next_value["desired_state"],
                    json.dumps(next_value["placement"], ensure_ascii=False),
                    json.dumps(next_value["rollout"], ensure_ascii=False),
                    now,
                    router_uid,
                    current["deployment_generation"],
                ),
            )
            if cursor.rowcount == 0:
                raise RuntimeError("Token Router deployment generation conflict")
        result = self.get(router_uid)
        assert result is not None
        return result

    def delete(self, router_uid: str) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM token_router_deployments WHERE router_uid = ?",
                (router_uid,),
            )
        return cursor.rowcount > 0
