# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Persistent Tokenizer Asset catalog and Router Agent desired bindings."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_ASSET_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$")
_FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-fA-F]{64}$")


class TokenizerAssetStore:
    """Store catalog metadata and per-node desired/observed Asset state."""

    _ORIGINS = {"builtin", "artifact", "shared_fs", "local", "external"}
    _DESIRED_STATES = {"present", "absent"}
    _OBSERVED_STATES = {
        "pending",
        "preparing",
        "validating",
        "ready",
        "unavailable",
        "error",
        "removing",
        "absent",
        "stale",
    }
    _BINDING_MODES = {"manual", "on_demand"}

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
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS tokenizer_assets (
                    asset_id TEXT PRIMARY KEY,
                    origin TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    source_json TEXT NOT NULL DEFAULT '{}',
                    capabilities_json TEXT NOT NULL DEFAULT '{}',
                    display_name TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_by TEXT NOT NULL DEFAULT '',
                    updated_by TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    CHECK (enabled IN (0, 1))
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS tokenizer_asset_bindings (
                    asset_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    desired_state TEXT NOT NULL,
                    observed_state TEXT NOT NULL DEFAULT 'pending',
                    desired_revision TEXT NOT NULL,
                    desired_fingerprint TEXT NOT NULL,
                    observed_revision TEXT NOT NULL DEFAULT '',
                    observed_fingerprint TEXT NOT NULL DEFAULT '',
                    local_path TEXT NOT NULL DEFAULT '',
                    binding_mode TEXT NOT NULL DEFAULT 'manual',
                    owner_type TEXT NOT NULL DEFAULT '',
                    owner_id TEXT NOT NULL DEFAULT '',
                    generation INTEGER NOT NULL DEFAULT 1,
                    last_error_code TEXT NOT NULL DEFAULT '',
                    last_error TEXT NOT NULL DEFAULT '',
                    last_seen_at TEXT,
                    created_by TEXT NOT NULL DEFAULT '',
                    updated_by TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(asset_id, node_id),
                    FOREIGN KEY(asset_id) REFERENCES tokenizer_assets(asset_id),
                    FOREIGN KEY(node_id) REFERENCES token_router_nodes(node_id),
                    CHECK (desired_state IN ('present', 'absent')),
                    CHECK (binding_mode IN ('manual', 'on_demand')),
                    CHECK (generation >= 1)
                )"""
            )
            conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_tokenizer_assets_enabled
                   ON tokenizer_assets(enabled)"""
            )
            conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_tokenizer_bindings_node_state
                   ON tokenizer_asset_bindings(node_id, desired_state, observed_state)"""
            )
            conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_tokenizer_bindings_asset_state
                   ON tokenizer_asset_bindings(asset_id, desired_state, observed_state)"""
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _json(value: Any, name: str) -> str:
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be an object")
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    @classmethod
    def _validate_asset(cls, data: Dict[str, Any]) -> None:
        asset_id = str(data.get("asset_id") or "").strip()
        if not _ASSET_ID_RE.fullmatch(asset_id):
            raise ValueError("Invalid Tokenizer asset_id")
        origin = str(data.get("origin") or "").strip()
        if origin not in cls._ORIGINS:
            raise ValueError("Invalid Tokenizer asset origin")
        if not str(data.get("revision") or "").strip():
            raise ValueError("Tokenizer asset revision is required")
        fingerprint = str(data.get("fingerprint") or "").strip()
        if not _FINGERPRINT_RE.fullmatch(fingerprint):
            raise ValueError("Tokenizer asset fingerprint must be sha256:<64 hex>")
        cls._json(data.get("source", {}), "source")
        cls._json(data.get("capabilities", {}), "capabilities")
        cls._json(data.get("metadata", {}), "metadata")

    @staticmethod
    def _asset_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "asset_id": row["asset_id"],
            "origin": row["origin"],
            "revision": row["revision"],
            "fingerprint": row["fingerprint"],
            "source": json.loads(row["source_json"] or "{}"),
            "capabilities": json.loads(row["capabilities_json"] or "{}"),
            "display_name": row["display_name"] or row["asset_id"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "enabled": bool(row["enabled"]),
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _binding_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "asset_id": row["asset_id"],
            "node_id": row["node_id"],
            "desired_state": row["desired_state"],
            "observed_state": row["observed_state"],
            "desired_revision": row["desired_revision"],
            "desired_fingerprint": row["desired_fingerprint"],
            "observed_revision": row["observed_revision"],
            "observed_fingerprint": row["observed_fingerprint"],
            "local_path": row["local_path"],
            "binding_mode": row["binding_mode"],
            "owner_type": row["owner_type"],
            "owner_id": row["owner_id"],
            "generation": int(row["generation"]),
            "last_error_code": row["last_error_code"],
            "last_error": row["last_error"],
            "last_seen_at": row["last_seen_at"],
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def create_asset(self, data: Dict[str, Any], username: str = "") -> Dict[str, Any]:
        self._validate_asset(data)
        now = self._now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO tokenizer_assets
                   (asset_id, origin, revision, fingerprint, source_json,
                    capabilities_json, display_name, metadata_json, enabled,
                    created_by, updated_by, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(data["asset_id"]).strip(),
                    str(data["origin"]).strip(),
                    str(data["revision"]).strip(),
                    str(data["fingerprint"]).strip().lower(),
                    self._json(data.get("source", {}), "source"),
                    self._json(data.get("capabilities", {}), "capabilities"),
                    str(data.get("display_name") or data["asset_id"]),
                    self._json(data.get("metadata", {}), "metadata"),
                    1 if data.get("enabled", True) else 0,
                    username,
                    username,
                    now,
                    now,
                ),
            )
        result = self.get_asset(str(data["asset_id"]))
        assert result is not None
        return result

    def import_asset(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Idempotently import Registry metadata without overwriting user state."""
        self._validate_asset(data)
        current = self.get_asset(str(data["asset_id"]))
        if current is None:
            return self.create_asset(data, username="registry-import")
        imported = {
            "origin": str(data["origin"]).strip(),
            "revision": str(data["revision"]).strip(),
            "fingerprint": str(data["fingerprint"]).strip().lower(),
            "source": data.get("source", {}),
            "capabilities": data.get("capabilities", {}),
            "display_name": str(data.get("display_name") or data["asset_id"]),
            "metadata": data.get("metadata", {}),
        }
        if all(current[key] == value for key, value in imported.items()):
            return current
        content_changed = any(
            current[key] != imported[key]
            for key in ("revision", "fingerprint", "source")
        )
        return self.update_asset(
            str(data["asset_id"]),
            imported,
            username="registry-import",
            force_binding_generation=content_changed,
        )

    def update_asset(
        self,
        asset_id: str,
        data: Dict[str, Any],
        username: str = "",
        *,
        force_binding_generation: bool = False,
    ) -> Dict[str, Any]:
        current = self.get_asset(asset_id)
        if current is None:
            raise KeyError(asset_id)
        merged = {**current, **data, "asset_id": asset_id}
        merged["origin"] = str(merged["origin"]).strip()
        merged["revision"] = str(merged["revision"]).strip()
        merged["fingerprint"] = str(merged["fingerprint"]).strip().lower()
        merged["display_name"] = str(merged.get("display_name") or asset_id)
        self._validate_asset(merged)
        comparable_keys = (
            "origin",
            "revision",
            "fingerprint",
            "source",
            "capabilities",
            "display_name",
            "metadata",
            "enabled",
        )
        if not force_binding_generation and all(
            merged[key] == current[key] for key in comparable_keys
        ):
            return current
        content_changed = force_binding_generation or any(
            merged[key] != current[key] for key in ("revision", "fingerprint", "source")
        )
        now = self._now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE tokenizer_assets SET
                     origin = ?, revision = ?, fingerprint = ?, source_json = ?,
                     capabilities_json = ?, display_name = ?, metadata_json = ?,
                     enabled = ?, updated_by = ?, updated_at = ?
                   WHERE asset_id = ?""",
                (
                    merged["origin"],
                    merged["revision"],
                    str(merged["fingerprint"]).lower(),
                    self._json(merged.get("source", {}), "source"),
                    self._json(merged.get("capabilities", {}), "capabilities"),
                    str(merged.get("display_name") or asset_id),
                    self._json(merged.get("metadata", {}), "metadata"),
                    1 if merged.get("enabled", True) else 0,
                    username,
                    now,
                    asset_id,
                ),
            )
            if content_changed:
                conn.execute(
                    """UPDATE tokenizer_asset_bindings SET
                         desired_revision = ?, desired_fingerprint = ?,
                         observed_state = 'pending', generation = generation + 1,
                         last_error_code = '', last_error = '', updated_by = ?,
                         updated_at = ?
                       WHERE asset_id = ? AND desired_state = 'present'""",
                    (
                        merged["revision"],
                        str(merged["fingerprint"]).lower(),
                        username,
                        now,
                        asset_id,
                    ),
                )
        result = self.get_asset(asset_id)
        assert result is not None
        return result

    def get_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tokenizer_assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
        return self._asset_row(row) if row is not None else None

    def list_assets(self, *, enabled: Optional[bool] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM tokenizer_assets"
        params: List[Any] = []
        if enabled is not None:
            sql += " WHERE enabled = ?"
            params.append(1 if enabled else 0)
        sql += " ORDER BY asset_id"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._asset_row(row) for row in rows]

    def delete_asset(self, asset_id: str) -> bool:
        with self._lock, self._connect() as conn:
            binding = conn.execute(
                "SELECT 1 FROM tokenizer_asset_bindings WHERE asset_id = ? LIMIT 1",
                (asset_id,),
            ).fetchone()
            if binding is not None:
                raise ValueError(
                    "Delete Tokenizer Asset bindings before deleting the Asset"
                )
            cursor = conn.execute(
                "DELETE FROM tokenizer_assets WHERE asset_id = ?", (asset_id,)
            )
            return cursor.rowcount > 0

    def upsert_binding(
        self,
        asset_id: str,
        node_id: str,
        *,
        desired_state: str = "present",
        binding_mode: str = "manual",
        owner_type: str = "",
        owner_id: str = "",
        username: str = "",
    ) -> Dict[str, Any]:
        if desired_state not in self._DESIRED_STATES:
            raise ValueError("Invalid Tokenizer Asset Binding desired_state")
        if binding_mode not in self._BINDING_MODES:
            raise ValueError("Invalid Tokenizer Asset Binding mode")
        asset = self.get_asset(asset_id)
        if asset is None:
            raise KeyError(asset_id)
        if desired_state == "present" and not asset["enabled"]:
            raise ValueError("Disabled Tokenizer Asset cannot be bound")
        now = self._now()
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                """SELECT * FROM tokenizer_asset_bindings
                   WHERE asset_id = ? AND node_id = ?""",
                (asset_id, node_id),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """INSERT INTO tokenizer_asset_bindings
                       (asset_id, node_id, desired_state, observed_state,
                        desired_revision, desired_fingerprint, binding_mode,
                        owner_type, owner_id, generation, created_by, updated_by,
                        created_at, updated_at)
                       VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)""",
                    (
                        asset_id,
                        node_id,
                        desired_state,
                        asset["revision"],
                        asset["fingerprint"],
                        binding_mode,
                        owner_type,
                        owner_id,
                        username,
                        username,
                        now,
                        now,
                    ),
                )
            else:
                changed = any(
                    (
                        existing["desired_state"] != desired_state,
                        existing["desired_revision"] != asset["revision"],
                        existing["desired_fingerprint"] != asset["fingerprint"],
                        existing["binding_mode"] != binding_mode,
                        existing["owner_type"] != owner_type,
                        existing["owner_id"] != owner_id,
                    )
                )
                conn.execute(
                    """UPDATE tokenizer_asset_bindings SET
                         desired_state = ?, desired_revision = ?,
                         desired_fingerprint = ?, binding_mode = ?, owner_type = ?,
                         owner_id = ?, observed_state = CASE WHEN ? THEN 'pending'
                                                            ELSE observed_state END,
                         generation = generation + CASE WHEN ? THEN 1 ELSE 0 END,
                         updated_by = ?, updated_at = ?
                       WHERE asset_id = ? AND node_id = ?""",
                    (
                        desired_state,
                        asset["revision"],
                        asset["fingerprint"],
                        binding_mode,
                        owner_type,
                        owner_id,
                        1 if changed else 0,
                        1 if changed else 0,
                        username,
                        now,
                        asset_id,
                        node_id,
                    ),
                )
        result = self.get_binding(asset_id, node_id)
        assert result is not None
        return result

    def get_binding(self, asset_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM tokenizer_asset_bindings
                   WHERE asset_id = ? AND node_id = ?""",
                (asset_id, node_id),
            ).fetchone()
        return self._binding_row(row) if row is not None else None

    def list_bindings(
        self, *, asset_id: Optional[str] = None, node_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if asset_id is not None:
            clauses.append("asset_id = ?")
            params.append(asset_id)
        if node_id is not None:
            clauses.append("node_id = ?")
            params.append(node_id)
        sql = "SELECT * FROM tokenizer_asset_bindings"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY asset_id, node_id"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._binding_row(row) for row in rows]

    def report_binding_status(
        self,
        asset_id: str,
        node_id: str,
        generation: int,
        observed_state: str,
        *,
        observed_revision: str = "",
        observed_fingerprint: str = "",
        local_path: str = "",
        last_error_code: str = "",
        last_error: str = "",
    ) -> Dict[str, Any]:
        if observed_state not in self._OBSERVED_STATES:
            raise ValueError("Invalid Tokenizer Asset Binding observed_state")
        current = self.get_binding(asset_id, node_id)
        if current is None:
            raise KeyError((asset_id, node_id))
        if int(generation) != current["generation"]:
            raise ValueError("Stale Tokenizer Asset Binding generation")
        if observed_state == "ready" and (
            observed_revision != current["desired_revision"]
            or observed_fingerprint.lower() != current["desired_fingerprint"].lower()
        ):
            raise ValueError(
                "Observed Tokenizer Asset revision or fingerprint mismatch"
            )
        now = self._now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE tokenizer_asset_bindings SET
                     observed_state = ?, observed_revision = ?,
                     observed_fingerprint = ?, local_path = ?,
                     last_error_code = ?, last_error = ?, last_seen_at = ?,
                     updated_at = ?
                   WHERE asset_id = ? AND node_id = ? AND generation = ?""",
                (
                    observed_state,
                    observed_revision,
                    observed_fingerprint.lower(),
                    local_path,
                    last_error_code,
                    last_error,
                    now,
                    now,
                    asset_id,
                    node_id,
                    int(generation),
                ),
            )
        result = self.get_binding(asset_id, node_id)
        assert result is not None
        return result

    def revalidate_binding(
        self, asset_id: str, node_id: str, username: str = ""
    ) -> Dict[str, Any]:
        current = self.get_binding(asset_id, node_id)
        if current is None:
            raise KeyError((asset_id, node_id))
        if current["desired_state"] != "present":
            raise ValueError("Only present Tokenizer Asset Bindings can be revalidated")
        now = self._now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE tokenizer_asset_bindings SET
                     observed_state = 'pending', generation = generation + 1,
                     last_error_code = '', last_error = '', updated_by = ?,
                     updated_at = ?
                   WHERE asset_id = ? AND node_id = ?""",
                (username, now, asset_id, node_id),
            )
        result = self.get_binding(asset_id, node_id)
        assert result is not None
        return result

    def mark_node_bindings_stale(self, node_id: str) -> None:
        now = self._now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE tokenizer_asset_bindings SET observed_state = 'stale',
                     updated_at = ?
                   WHERE node_id = ? AND observed_state = 'ready'""",
                (now, node_id),
            )

    def delete_binding(
        self, asset_id: str, node_id: str, *, force: bool = False
    ) -> bool:
        current = self.get_binding(asset_id, node_id)
        if current is None:
            return False
        if not force and not (
            current["desired_state"] == "absent"
            and current["observed_state"] == "absent"
        ):
            raise ValueError("Binding must be absent before deletion")
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """DELETE FROM tokenizer_asset_bindings
                   WHERE asset_id = ? AND node_id = ?""",
                (asset_id, node_id),
            )
            return cursor.rowcount > 0
