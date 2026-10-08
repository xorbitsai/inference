# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

ACTIVE_DOWNLOAD_STATUSES = ("pending", "resuming", "downloading", "pausing")
RESUMABLE_DOWNLOAD_STATUSES = ("paused", "interrupted", "failed")
TERMINAL_DOWNLOAD_STATUSES = ("completed", "cancelled")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS download_tasks (
    cache_uid TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    model_engine TEXT,
    model_version TEXT,
    worker_address TEXT,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    payload TEXT NOT NULL,
    download_files TEXT NOT NULL DEFAULT '[]',
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_download_tasks_status_updated
    ON download_tasks(status, updated_at DESC);
"""

_UPDATEABLE_COLUMNS = {
    "model_name",
    "model_type",
    "model_engine",
    "model_version",
    "worker_address",
    "status",
    "progress",
    "payload",
    "download_files",
    "error",
}
_JSON_COLUMNS = {"payload", "download_files"}


class DownloadTaskStore:
    """SQLite-backed state for resumable cache-only downloads."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
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

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> Dict[str, Any]:
        item = dict(row)
        for key, fallback in (("payload", {}), ("download_files", [])):
            try:
                item[key] = json.loads(item[key])
            except (json.JSONDecodeError, TypeError):
                item[key] = fallback
        return item

    def upsert(self, task: Dict[str, Any]) -> None:
        now = time.time()
        payload = json.dumps(task.get("payload") or {}, ensure_ascii=False, default=str)
        download_files = json.dumps(
            task.get("download_files") or [], ensure_ascii=False, default=str
        )
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO download_tasks (
                           cache_uid, model_name, model_type, model_engine,
                           model_version, worker_address, status, progress,
                           payload, download_files, error, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(cache_uid) DO UPDATE SET
                           model_name = excluded.model_name,
                           model_type = excluded.model_type,
                           model_engine = excluded.model_engine,
                           model_version = excluded.model_version,
                           worker_address = excluded.worker_address,
                           status = excluded.status,
                           progress = excluded.progress,
                           payload = excluded.payload,
                           download_files = excluded.download_files,
                           error = excluded.error,
                           updated_at = excluded.updated_at""",
                    (
                        task["cache_uid"],
                        task["model_name"],
                        task.get("model_type") or "LLM",
                        task.get("model_engine"),
                        task.get("model_version"),
                        task.get("worker_address"),
                        task.get("status") or "pending",
                        float(task.get("progress") or 0),
                        payload,
                        download_files,
                        task.get("error"),
                        float(task.get("created_at") or now),
                        now,
                    ),
                )

    def update(self, cache_uid: str, **changes: Any) -> bool:
        invalid = set(changes) - _UPDATEABLE_COLUMNS
        if invalid:
            raise ValueError(f"Unsupported download task fields: {sorted(invalid)}")
        if not changes:
            return self.get(cache_uid) is not None

        values: List[Any] = []
        assignments = []
        for key, value in changes.items():
            assignments.append(f"{key} = ?")
            values.append(
                json.dumps(value, ensure_ascii=False, default=str)
                if key in _JSON_COLUMNS
                else value
            )
        assignments.append("updated_at = ?")
        values.extend((time.time(), cache_uid))

        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    f"UPDATE download_tasks SET {', '.join(assignments)} "
                    "WHERE cache_uid = ?",
                    values,
                )
                return cursor.rowcount > 0

    def get(self, cache_uid: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM download_tasks WHERE cache_uid = ?", (cache_uid,)
            ).fetchone()
        return self._row_to_item(row) if row is not None else None

    def list_unfinished(self) -> List[Dict[str, Any]]:
        placeholders = ", ".join("?" for _ in TERMINAL_DOWNLOAD_STATUSES)
        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM download_tasks WHERE status NOT IN ({placeholders}) "
                "ORDER BY updated_at DESC",
                TERMINAL_DOWNLOAD_STATUSES,
            ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def mark_active_interrupted(self) -> int:
        placeholders = ", ".join("?" for _ in ACTIVE_DOWNLOAD_STATUSES)
        now = time.time()
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "UPDATE download_tasks SET status = 'interrupted', "
                    "error = COALESCE(error, 'Download interrupted by service restart'), "
                    f"updated_at = ? WHERE status IN ({placeholders})",
                    (now, *ACTIVE_DOWNLOAD_STATUSES),
                )
                return cursor.rowcount

    def delete(self, cache_uid: str) -> bool:
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "DELETE FROM download_tasks WHERE cache_uid = ?", (cache_uid,)
                )
                return cursor.rowcount > 0
