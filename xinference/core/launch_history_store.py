# Copyright 2022-2026 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from .autostart import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_PRIORITY,
    DEFAULT_RETRY_INTERVAL_SECONDS,
)

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS launch_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_uid TEXT NOT NULL DEFAULT '',
    data TEXT NOT NULL,
    autostart_enabled INTEGER NOT NULL DEFAULT 0,
    autostart_priority INTEGER NOT NULL DEFAULT 100,
    autostart_max_retries INTEGER NOT NULL DEFAULT 3,
    autostart_retry_interval_seconds INTEGER NOT NULL DEFAULT 30,
    created_by TEXT DEFAULT '',
    updated_by TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_launch_history_model
    ON launch_history(model_name, model_uid, created_by);
"""

AUTOSTART_COLUMNS = {
    "autostart_enabled": "INTEGER NOT NULL DEFAULT 0",
    "autostart_priority": f"INTEGER NOT NULL DEFAULT {DEFAULT_PRIORITY}",
    "autostart_max_retries": f"INTEGER NOT NULL DEFAULT {DEFAULT_MAX_RETRIES}",
    "autostart_retry_interval_seconds": (
        f"INTEGER NOT NULL DEFAULT {DEFAULT_RETRY_INTERVAL_SECONDS}"
    ),
}

# Fields that are safe to expose to other users when sharing launch history.
# Anything not listed here (env vars, arbitrary custom kwargs, filesystem
# paths, infra hints) is stripped from another user's entries on read so that
# secrets such as tokens injected via `envs` or custom parameters never leak
# across users. The owner of an entry always receives the full payload.
SHAREABLE_KEYS = frozenset(
    {
        "model_uid",
        "model_name",
        "model_type",
        "model_engine",
        "model_format",
        "model_size_in_billions",
        "quantization",
        "n_worker",
        "n_gpu",
        "n_gpu_layers",
        "replica",
        "request_limits",
        "gpu_idx",
        "download_hub",
        "reasoning_content",
        "gguf_quantization",
        "lightning_version",
        "cpu_offload",
        "enable_thinking",
        "multimodal_projector",
        "enable_virtual_env",
        "virtual_env_packages",
    }
)


class LaunchHistoryStore:
    """SQLite-backed store for model launch configuration history.

    The store is independent of the auth database so it works in any deployment
    mode. ``created_by`` / ``updated_by`` are populated from the authenticated
    username when authentication is enabled, and left empty otherwise.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._get_conn() as conn:
            conn.executescript(SCHEMA_SQL)
            self._migrate_db(conn)

    @staticmethod
    def _migrate_db(conn):
        existing_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(launch_history)")
        }
        for column, definition in AUTOSTART_COLUMNS.items():
            if column not in existing_columns:
                conn.execute(
                    f"ALTER TABLE launch_history ADD COLUMN {column} {definition}"
                )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_launch_history_autostart_model_uid "
            "ON launch_history(model_uid) WHERE autostart_enabled = 1"
        )

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def list(
        self,
        model_name: Optional[str] = None,
        username: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            if model_name:
                rows = conn.execute(
                    "SELECT * FROM launch_history WHERE model_name = ? "
                    "ORDER BY updated_at DESC",
                    (model_name,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM launch_history ORDER BY updated_at DESC"
                ).fetchall()
        return [self._row_to_item(row, username) for row in rows]

    @staticmethod
    def _format_timestamp(value: Any) -> Any:
        if value:
            return str(value).replace(" ", "T") + "Z"
        return value

    def _row_to_item(
        self, row: sqlite3.Row, username: Optional[str] = None
    ) -> Dict[str, Any]:
        item = dict(row)
        try:
            item["data"] = json.loads(item["data"])
        except (json.JSONDecodeError, TypeError):
            pass
        # Redact sensitive fields from entries owned by other users so that
        # secrets (envs, custom kwargs, ...) never leak across users.
        # ``username is None`` means an unscoped/trusted call (no redaction).
        if username is not None and item.get("created_by") != username:
            data = item.get("data")
            if isinstance(data, dict):
                item["data"] = {k: v for k, v in data.items() if k in SHAREABLE_KEYS}
        if "autostart_enabled" in item:
            item["autostart_enabled"] = bool(item["autostart_enabled"])
        # SQLite CURRENT_TIMESTAMP yields naive UTC "YYYY-MM-DD HH:MM:SS";
        # emit ISO-8601 with explicit Z so clients parse it as UTC.
        for key in ("created_at", "updated_at"):
            item[key] = self._format_timestamp(item.get(key))
        return item

    def list_autostart(self, username: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM launch_history WHERE autostart_enabled = 1 "
                "ORDER BY autostart_priority ASC, updated_at DESC"
            ).fetchall()

        result = []
        for row in rows:
            item = self._row_to_item(row, username=username)
            data = item.get("data")
            if not isinstance(data, dict):
                logger.warning(
                    "Skip autostart entry with invalid launch payload: %s",
                    item.get("model_uid"),
                )
                continue
            launch = dict(data)
            launch["model_name"] = launch.get("model_name") or item["model_name"]
            launch["model_uid"] = (
                launch.get("model_uid") or item.get("model_uid") or item["model_name"]
            )
            result.append(
                {
                    "enabled": True,
                    "priority": item.get("autostart_priority", DEFAULT_PRIORITY),
                    "max_retries": item.get(
                        "autostart_max_retries", DEFAULT_MAX_RETRIES
                    ),
                    "retry_interval_seconds": item.get(
                        "autostart_retry_interval_seconds",
                        DEFAULT_RETRY_INTERVAL_SECONDS,
                    ),
                    "launch": launch,
                    "created_by": item.get("created_by", ""),
                    "updated_at": item.get("updated_at"),
                }
            )
        return result

    def upsert(
        self,
        model_name: str,
        model_uid: str,
        data: Dict[str, Any],
        username: str = "",
    ) -> None:
        # Each user owns a separate row per (model_name, model_uid); a later
        # upsert by the same user updates only their own row.
        data_json = json.dumps(data, ensure_ascii=False)
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO launch_history
                           (model_name, model_uid, data, created_by, updated_by)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(model_name, model_uid, created_by) DO UPDATE SET
                           data = excluded.data,
                           updated_by = excluded.updated_by,
                           updated_at = CURRENT_TIMESTAMP""",
                    (model_name, model_uid, data_json, username, username),
                )

    def upsert_autostart(self, entry: Dict[str, Any], username: str = "") -> None:
        launch = entry["launch"]
        model_name = launch["model_name"]
        model_uid = launch["model_uid"]
        data_json = json.dumps(launch, ensure_ascii=False)
        autostart_enabled = 1 if entry.get("enabled", True) else 0

        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "UPDATE launch_history SET autostart_enabled = 0, "
                    "updated_by = ?, updated_at = CURRENT_TIMESTAMP "
                    "WHERE model_uid = ? AND autostart_enabled = 1",
                    (username, model_uid),
                )
                conn.execute(
                    """INSERT INTO launch_history
                           (model_name, model_uid, data, created_by, updated_by,
                            autostart_enabled, autostart_priority,
                            autostart_max_retries,
                            autostart_retry_interval_seconds)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(model_name, model_uid, created_by) DO UPDATE SET
                           data = excluded.data,
                           updated_by = excluded.updated_by,
                           autostart_enabled = excluded.autostart_enabled,
                           autostart_priority = excluded.autostart_priority,
                           autostart_max_retries = excluded.autostart_max_retries,
                           autostart_retry_interval_seconds =
                               excluded.autostart_retry_interval_seconds,
                           updated_at = CURRENT_TIMESTAMP""",
                    (
                        model_name,
                        model_uid,
                        data_json,
                        username,
                        username,
                        autostart_enabled,
                        entry.get("priority", DEFAULT_PRIORITY),
                        entry.get("max_retries", DEFAULT_MAX_RETRIES),
                        entry.get(
                            "retry_interval_seconds",
                            DEFAULT_RETRY_INTERVAL_SECONDS,
                        ),
                    ),
                )

    def remove_autostart(self, model_uid: str) -> bool:
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "UPDATE launch_history SET autostart_enabled = 0, "
                    "updated_at = CURRENT_TIMESTAMP "
                    "WHERE model_uid = ? AND autostart_enabled = 1",
                    (model_uid,),
                )
                return cursor.rowcount > 0

    def delete(self, model_name: str, model_uid: str, username: str = "") -> bool:
        # A user may only delete their own row.
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "DELETE FROM launch_history "
                    "WHERE model_name = ? AND model_uid = ? AND created_by = ?",
                    (model_name, model_uid, username),
                )
                return cursor.rowcount > 0
