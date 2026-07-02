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
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL,
    password_hash TEXT,
    source TEXT NOT NULL DEFAULT 'local',
    oidc_sub TEXT,
    enabled INTEGER DEFAULT 1,
    must_change_password INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_users_source_username ON users(source, username);

CREATE TABLE IF NOT EXISTS user_permissions (
    user_id INTEGER NOT NULL,
    permission TEXT NOT NULL,
    PRIMARY KEY (user_id, permission),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    key_hash TEXT UNIQUE NOT NULL,
    key_encrypted TEXT NOT NULL,
    key_prefix TEXT NOT NULL,
    name TEXT,
    description TEXT,
    enabled INTEGER DEFAULT 1,
    expires_at TIMESTAMP,
    encryption_version INTEGER DEFAULT 1,
    rate_limit_max_failures INTEGER,
    rate_limit_window_seconds INTEGER,
    rate_limit_ban_seconds INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS api_key_model_permissions (
    id INTEGER PRIMARY KEY,
    api_key_id INTEGER NOT NULL,
    permission_type TEXT NOT NULL,
    permission_value TEXT,
    UNIQUE(api_key_id, permission_type, permission_value),
    FOREIGN KEY (api_key_id) REFERENCES api_keys(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS refresh_tokens (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    token_hash TEXT UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS system_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Database:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._get_conn() as conn:
            conn.executescript(SCHEMA_SQL)
            # Schema migrations for existing databases
            cursor = conn.execute("PRAGMA table_info(api_keys)")
            columns = {row[1] for row in cursor.fetchall()}
            if "description" not in columns:
                conn.execute("ALTER TABLE api_keys ADD COLUMN description TEXT")

            cursor = conn.execute("PRAGMA table_info(users)")
            user_columns = {row[1] for row in cursor.fetchall()}
            if "oidc_sub" not in user_columns:
                conn.execute("ALTER TABLE users ADD COLUMN oidc_sub TEXT")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_oidc_sub "
                "ON users(oidc_sub) WHERE oidc_sub IS NOT NULL"
            )

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # --- Users ---

    def create_user(
        self,
        username: str,
        password_hash: Optional[str],
        source: str = "local",
        oidc_sub: Optional[str] = None,
        enabled: int = 1,
        must_change_password: int = 0,
        permissions: Optional[List[str]] = None,
    ) -> int:
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "INSERT INTO users (username, password_hash, source, oidc_sub, enabled, must_change_password) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        username,
                        password_hash,
                        source,
                        oidc_sub,
                        enabled,
                        must_change_password,
                    ),
                )
                user_id = cursor.lastrowid
                if permissions:
                    for perm in permissions:
                        conn.execute(
                            "INSERT INTO user_permissions (user_id, permission) VALUES (?, ?)",
                            (user_id, perm),
                        )
                return user_id

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if not row:
                return None
            user = dict(row)
            perms = conn.execute(
                "SELECT permission FROM user_permissions WHERE user_id = ?", (user_id,)
            ).fetchall()
            user["permissions"] = [p["permission"] for p in perms]
            return user

    def get_user_by_username(
        self, username: str, source: str = "local"
    ) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ? AND source = ?",
                (username, source),
            ).fetchone()
            if not row:
                return None
            user = dict(row)
            perms = conn.execute(
                "SELECT permission FROM user_permissions WHERE user_id = ?",
                (user["id"],),
            ).fetchall()
            user["permissions"] = [p["permission"] for p in perms]
            return user

    def get_user_by_oidc_sub(self, oidc_sub: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE oidc_sub = ?", (oidc_sub,)
            ).fetchone()
            if not row:
                return None
            user = dict(row)
            perms = conn.execute(
                "SELECT permission FROM user_permissions WHERE user_id = ?",
                (user["id"],),
            ).fetchall()
            user["permissions"] = [p["permission"] for p in perms]
            return user

    def list_users(self, source: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            if source:
                rows = conn.execute(
                    "SELECT * FROM users WHERE source = ?", (source,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM users").fetchall()
            users = [dict(row) for row in rows]
            if not users:
                return users
            user_ids = [u["id"] for u in users]
            placeholders = ",".join("?" * len(user_ids))
            perms_rows = conn.execute(
                f"SELECT user_id, permission FROM user_permissions "
                f"WHERE user_id IN ({placeholders})",
                user_ids,
            ).fetchall()
            perms_map: Dict[int, List[str]] = {}
            for p in perms_rows:
                perms_map.setdefault(p["user_id"], []).append(p["permission"])
            for user in users:
                user["permissions"] = perms_map.get(user["id"], [])
            return users

    def update_user(self, user_id: int, **kwargs) -> bool:
        allowed = {
            "username",
            "password_hash",
            "enabled",
            "must_change_password",
            "oidc_sub",
        }
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return False
        with self._lock:
            with self._get_conn() as conn:
                set_clause = ", ".join(f"{k} = ?" for k in fields)
                values = list(fields.values()) + [user_id]
                conn.execute(f"UPDATE users SET {set_clause} WHERE id = ?", values)
                return True

    def delete_user(self, user_id: int) -> bool:
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
                return True

    def set_user_permissions(self, user_id: int, permissions: List[str]):
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "DELETE FROM user_permissions WHERE user_id = ?", (user_id,)
                )
                for perm in permissions:
                    conn.execute(
                        "INSERT INTO user_permissions (user_id, permission) VALUES (?, ?)",
                        (user_id, perm),
                    )

    def user_count(self) -> int:
        with self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM users").fetchone()
            return row["cnt"]

    # --- API Keys ---

    def create_api_key(
        self,
        user_id: int,
        key_hash: str,
        key_encrypted: str,
        key_prefix: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        expires_at: Optional[str] = None,
        rate_limit_max_failures: Optional[int] = None,
        rate_limit_window_seconds: Optional[int] = None,
        rate_limit_ban_seconds: Optional[int] = None,
        model_permissions: Optional[List[Dict[str, Optional[str]]]] = None,
    ) -> int:
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    """INSERT INTO api_keys (user_id, key_hash, key_encrypted, key_prefix, name, description, expires_at,
                       rate_limit_max_failures, rate_limit_window_seconds, rate_limit_ban_seconds)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        user_id,
                        key_hash,
                        key_encrypted,
                        key_prefix,
                        name,
                        description,
                        expires_at,
                        rate_limit_max_failures,
                        rate_limit_window_seconds,
                        rate_limit_ban_seconds,
                    ),
                )
                key_id = cursor.lastrowid
                if model_permissions:
                    for mp in model_permissions:
                        conn.execute(
                            "INSERT INTO api_key_model_permissions (api_key_id, permission_type, permission_value) VALUES (?, ?, ?)",
                            (key_id, mp["permission_type"], mp.get("permission_value")),
                        )
                return key_id

    def get_api_key_by_id(self, key_id: int) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM api_keys WHERE id = ?", (key_id,)
            ).fetchone()
            if not row:
                return None
            key = dict(row)
            perms = conn.execute(
                "SELECT * FROM api_key_model_permissions WHERE api_key_id = ?",
                (key_id,),
            ).fetchall()
            key["model_permissions"] = [dict(p) for p in perms]
            return key

    def get_api_key_by_hash(self, key_hash: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)
            ).fetchone()
            if not row:
                return None
            key = dict(row)
            perms = conn.execute(
                "SELECT * FROM api_key_model_permissions WHERE api_key_id = ?",
                (key["id"],),
            ).fetchall()
            key["model_permissions"] = [dict(p) for p in perms]
            return key

    def list_api_keys(self, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            if user_id is not None:
                rows = conn.execute(
                    "SELECT * FROM api_keys WHERE user_id = ?", (user_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM api_keys").fetchall()
            keys = []
            for row in rows:
                key = dict(row)
                perms = conn.execute(
                    "SELECT * FROM api_key_model_permissions WHERE api_key_id = ?",
                    (key["id"],),
                ).fetchall()
                key["model_permissions"] = [dict(p) for p in perms]
                keys.append(key)
            return keys

    def update_api_key(self, key_id: int, **kwargs) -> bool:
        allowed = {
            "name",
            "description",
            "enabled",
            "expires_at",
            "rate_limit_max_failures",
            "rate_limit_window_seconds",
            "rate_limit_ban_seconds",
        }
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return False
        with self._lock:
            with self._get_conn() as conn:
                set_clause = ", ".join(f"{k} = ?" for k in fields)
                values = list(fields.values()) + [key_id]
                conn.execute(f"UPDATE api_keys SET {set_clause} WHERE id = ?", values)
                return True

    def delete_api_key(self, key_id: int) -> bool:
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
                return True

    def set_api_key_model_permissions(
        self, key_id: int, permissions: List[Dict[str, Optional[str]]]
    ):
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "DELETE FROM api_key_model_permissions WHERE api_key_id = ?",
                    (key_id,),
                )
                for mp in permissions:
                    conn.execute(
                        "INSERT INTO api_key_model_permissions (api_key_id, permission_type, permission_value) VALUES (?, ?, ?)",
                        (key_id, mp["permission_type"], mp.get("permission_value")),
                    )

    def get_all_api_keys_with_users(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT ak.*, u.username, u.enabled as user_enabled
                   FROM api_keys ak JOIN users u ON ak.user_id = u.id"""
            ).fetchall()
            keys = []
            for row in rows:
                key = dict(row)
                perms = conn.execute(
                    "SELECT * FROM api_key_model_permissions WHERE api_key_id = ?",
                    (key["id"],),
                ).fetchall()
                key["model_permissions"] = [dict(p) for p in perms]
                keys.append(key)
            return keys

    # --- Refresh Tokens ---

    def create_refresh_token(
        self, user_id: int, token_hash: str, expires_at: str
    ) -> int:
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES (?, ?, ?)",
                    (user_id, token_hash, expires_at),
                )
                return cursor.lastrowid

    def get_refresh_token(self, token_hash: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM refresh_tokens WHERE token_hash = ?", (token_hash,)
            ).fetchone()
            return dict(row) if row else None

    def delete_refresh_token(self, token_hash: str) -> bool:
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "DELETE FROM refresh_tokens WHERE token_hash = ?", (token_hash,)
                )
                return True

    def delete_user_refresh_tokens(self, user_id: int):
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM refresh_tokens WHERE user_id = ?", (user_id,))

    def cleanup_expired_refresh_tokens(self):
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "DELETE FROM refresh_tokens WHERE expires_at < datetime('now')"
                )

    # --- System Config ---

    def get_config(self, key: str) -> Optional[str]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT value FROM system_config WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None

    def set_config(self, key: str, value: str):
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO system_config (key, value) VALUES (?, ?)",
                    (key, value),
                )
