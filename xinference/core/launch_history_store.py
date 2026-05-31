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

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS launch_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_uid TEXT NOT NULL DEFAULT '',
    data TEXT NOT NULL,
    created_by TEXT DEFAULT '',
    updated_by TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_launch_history_model
    ON launch_history(model_name, model_uid, created_by);
"""

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
        result = []
        for row in rows:
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
                    item["data"] = {
                        k: v for k, v in data.items() if k in SHAREABLE_KEYS
                    }
            # SQLite CURRENT_TIMESTAMP yields naive UTC "YYYY-MM-DD HH:MM:SS";
            # emit ISO-8601 with explicit Z so clients parse it as UTC.
            for key in ("created_at", "updated_at"):
                value = item.get(key)
                if value:
                    item[key] = str(value).replace(" ", "T") + "Z"
            result.append(item)
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
