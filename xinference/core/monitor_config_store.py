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
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS monitor_config (
    key           TEXT PRIMARY KEY,
    value         TEXT NOT NULL,
    description   TEXT DEFAULT '',
    updated_by    TEXT DEFAULT '',
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

DEFAULT_KEYS = {
    "grafana_url": "Grafana base URL",
    "grafana_datasource": "Prometheus datasource name",
    "grafana_alert_datasource": "Alert datasource name",
    "cluster_name": "Cluster name (for Grafana var-cluster)",
    "dashboard_overview": "Overview dashboard UID",
    "dashboard_model_load": "Model load dashboard UID",
    "dashboard_llm_slo": "LLM SLO dashboard UID",
    "dashboard_gpu": "GPU resources dashboard UID",
    "dashboard_host": "Host resources dashboard UID",
    "dashboard_security": "Security audit dashboard UID",
}

ENV_MAPPING = {
    "grafana_url": "XINFERENCE_GRAFANA_URL",
    "grafana_datasource": "XINFERENCE_GRAFANA_DATASOURCE",
    "grafana_alert_datasource": "XINFERENCE_GRAFANA_ALERT_DATASOURCE",
    "cluster_name": "XINFERENCE_CLUSTER_NAME",
}

DEFAULT_UIDS = {
    "dashboard_overview": "xinference-overview",
    "dashboard_model_load": "xinference-model-load",
    "dashboard_llm_slo": "xinference-llm-slo",
    "dashboard_gpu": "xinference-gpu-resources",
    "dashboard_host": "xinference-host-resources",
    "dashboard_security": "xinference-security-audit",
}


class MonitorConfigStore:
    """SQLite-backed store for monitoring dashboard configuration.

    The store persists Grafana URL, datasource names, and dashboard UIDs.
    Values follow a fallback chain: DB value (non-empty) → environment variable → code default.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._get_conn() as conn:
            conn.executescript(SCHEMA_SQL)
            for key, desc in DEFAULT_KEYS.items():
                conn.execute(
                    "INSERT OR IGNORE INTO monitor_config (key, value, description) VALUES (?, ?, ?)",
                    (key, "", desc),
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

    def _get_fallback(self, key: str) -> str:
        """Get value from environment variable or default."""
        if key in ENV_MAPPING:
            env_val = os.environ.get(ENV_MAPPING[key], "")
            if env_val:
                return env_val
        if key in DEFAULT_UIDS:
            return DEFAULT_UIDS[key]
        return ""

    def get_all(self) -> Dict[str, str]:
        """Return all config values with fallback chain applied."""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT key, value FROM monitor_config").fetchall()
            db_values = {row["key"]: row["value"] for row in rows}
            result = {}
            for key in DEFAULT_KEYS.keys():
                db_val = db_values.get(key, "")
                if db_val:
                    result[key] = db_val
                else:
                    result[key] = self._get_fallback(key)

            # Special fallback: grafana_alert_datasource → grafana_datasource
            if not result.get("grafana_alert_datasource"):
                result["grafana_alert_datasource"] = result.get(
                    "grafana_datasource", ""
                )

            return result

    def get_dashboards(self) -> Dict[str, str]:
        """Return dashboard UIDs as a dict: {"overview": "uid", ...}."""
        all_cfg = self.get_all()
        dashboards = {}
        for key in DEFAULT_UIDS.keys():
            dashboard_name = key.replace("dashboard_", "")
            dashboards[dashboard_name] = all_cfg.get(key, "")
        return dashboards

    def get_sources(self) -> Dict[str, str]:
        """Return value source for each key: 'db', 'env', or 'default'."""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT key, value FROM monitor_config").fetchall()
            db_values = {row["key"]: row["value"] for row in rows}
            sources = {}
            for key in DEFAULT_KEYS.keys():
                db_val = db_values.get(key, "")
                if db_val:
                    sources[key] = "db"
                elif key in ENV_MAPPING and os.environ.get(ENV_MAPPING[key]):
                    sources[key] = "env"
                else:
                    sources[key] = "default"
            return sources

    def update(self, updates: Dict[str, str], username: str = "") -> None:
        """Batch update config values."""
        with self._lock:
            with self._get_conn() as conn:
                for key, value in updates.items():
                    conn.execute(
                        """INSERT INTO monitor_config (key, value, updated_by)
                           VALUES (?, ?, ?)
                           ON CONFLICT(key) DO UPDATE SET
                               value = excluded.value,
                               updated_by = excluded.updated_by,
                               updated_at = CURRENT_TIMESTAMP""",
                        (key, value, username),
                    )

    def reset(self) -> None:
        """Delete all rows. Subsequent get_all() falls back to env/default."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM monitor_config")
