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
    ON launch_history(model_name, model_uid);
"""


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

    def list(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
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
        # ON CONFLICT intentionally does not update created_by so the original
        # creator is preserved across later updates.
        data_json = json.dumps(data, ensure_ascii=False)
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO launch_history
                           (model_name, model_uid, data, created_by, updated_by)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(model_name, model_uid) DO UPDATE SET
                           data = excluded.data,
                           updated_by = excluded.updated_by,
                           updated_at = CURRENT_TIMESTAMP""",
                    (model_name, model_uid, data_json, username, username),
                )

    def delete(self, model_name: str, model_uid: str) -> bool:
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "DELETE FROM launch_history "
                    "WHERE model_name = ? AND model_uid = ?",
                    (model_name, model_uid),
                )
                return cursor.rowcount > 0
