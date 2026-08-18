# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
"""Persistent Token Router configuration store."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class RouterConfigStore:
    """SQLite-backed store with monotonic per-router configuration revisions."""

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
                """CREATE TABLE IF NOT EXISTS token_router_configs (
                    router_uid TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    revision INTEGER NOT NULL DEFAULT 1,
                    created_by TEXT NOT NULL DEFAULT '',
                    updated_by TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        config = json.loads(row["config_json"])
        config.update(
            {
                "router_uid": row["router_uid"],
                "enabled": bool(row["enabled"]),
                "revision": row["revision"],
                "created_by": row["created_by"],
                "updated_by": row["updated_by"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
        return config

    def list(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM token_router_configs ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get(self, router_uid: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM token_router_configs WHERE router_uid = ?",
                (router_uid,),
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def get_by_virtual_model_uid(
        self, virtual_model_uid: str
    ) -> Optional[Dict[str, Any]]:
        """Return the Router config that owns an exact Virtual Model UID."""
        for config in self.list():
            if config.get("virtual_model_uid") == virtual_model_uid:
                return config
        return None

    def create(
        self, router_uid: str, config: Dict[str, Any], username: str = ""
    ) -> Dict[str, Any]:
        payload = self._sanitize(config)
        existing = self.get(router_uid)
        if existing is not None:
            existing_payload = self._sanitize(existing)
            if existing_payload == payload:
                return existing
            raise ValueError(f"Token Router already exists: {router_uid}")
        now = self._now()
        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute(
                        """INSERT INTO token_router_configs
                           (router_uid, config_json, enabled, revision,
                            created_by, updated_by, created_at, updated_at)
                           VALUES (?, ?, 0, 1, ?, ?, ?, ?)""",
                        (
                            router_uid,
                            json.dumps(payload, ensure_ascii=False),
                            username,
                            username,
                            now,
                            now,
                        ),
                    )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Token Router already exists: {router_uid}") from exc
        result = self.get(router_uid)
        assert result is not None
        return result

    @staticmethod
    def _sanitize(config: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(config)
        for key in (
            "router_uid",
            "enabled",
            "revision",
            "created_by",
            "updated_by",
            "created_at",
            "updated_at",
            "status",
            "runtime_instances",
            "online_instances",
        ):
            payload.pop(key, None)
        return payload

    def update(
        self,
        router_uid: str,
        config: Dict[str, Any],
        username: str = "",
        expected_revision: Optional[int] = None,
    ) -> Dict[str, Any]:
        current = self.get(router_uid)
        if current is None:
            raise KeyError(router_uid)
        if expected_revision is not None and current["revision"] != expected_revision:
            raise RuntimeError(
                f"Token Router revision conflict: expected {expected_revision}, "
                f"current {current['revision']}"
            )
        payload = self._sanitize(config)
        now = self._now()
        with self._lock, self._connect() as conn:
            if expected_revision is None:
                cursor = conn.execute(
                    """UPDATE token_router_configs
                       SET config_json = ?, revision = revision + 1,
                           updated_by = ?, updated_at = ?
                       WHERE router_uid = ?""",
                    (
                        json.dumps(payload, ensure_ascii=False),
                        username,
                        now,
                        router_uid,
                    ),
                )
            else:
                cursor = conn.execute(
                    """UPDATE token_router_configs
                       SET config_json = ?, revision = revision + 1,
                           updated_by = ?, updated_at = ?
                       WHERE router_uid = ? AND revision = ?""",
                    (
                        json.dumps(payload, ensure_ascii=False),
                        username,
                        now,
                        router_uid,
                        expected_revision,
                    ),
                )
            if cursor.rowcount == 0:
                raise RuntimeError("Token Router revision conflict")
        result = self.get(router_uid)
        assert result is not None
        return result

    def set_enabled(
        self, router_uid: str, enabled: bool, username: str = ""
    ) -> Dict[str, Any]:
        current = self.get(router_uid)
        if current is None:
            raise KeyError(router_uid)
        if current["enabled"] == enabled:
            return current
        now = self._now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE token_router_configs
                   SET enabled = ?, revision = revision + 1,
                       updated_by = ?, updated_at = ?
                   WHERE router_uid = ?""",
                (1 if enabled else 0, username, now, router_uid),
            )
        result = self.get(router_uid)
        assert result is not None
        return result

    def delete(self, router_uid: str) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM token_router_configs WHERE router_uid = ?", (router_uid,)
            )
            return cursor.rowcount > 0
