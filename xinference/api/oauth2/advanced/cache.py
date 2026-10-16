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
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .database import Database

logger = logging.getLogger(__name__)


@dataclass
class ModelPermission:
    permission_type: str
    permission_value: Optional[str] = None


@dataclass
class ApiKeyCacheEntry:
    key_id: int
    user_id: int
    key_hash: str
    key_prefix: str
    name: Optional[str]
    enabled: bool
    user_enabled: bool
    expires_at: Optional[datetime]
    model_permissions: List[ModelPermission] = field(default_factory=list)

    def is_valid(self) -> bool:
        if not self.enabled or not self.user_enabled:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    def has_model_access(
        self, model_uid: str, model_type: Optional[str] = None
    ) -> bool:
        if not self.model_permissions:
            return False
        for mp in self.model_permissions:
            if mp.permission_type == "all":
                return True
            if mp.permission_type == "model_id" and mp.permission_value == model_uid:
                return True
            if (
                mp.permission_type == "model_type"
                and model_type
                and mp.permission_value == model_type
            ):
                return True
        return False


class ApiKeyCache:
    def __init__(self, db: Database):
        self._db = db
        self._cache: Dict[str, ApiKeyCacheEntry] = {}
        self._lock = threading.RLock()
        self._load_all()

    def _load_all(self):
        keys = self._db.get_all_api_keys_with_users()
        with self._lock:
            self._cache.clear()
            for k in keys:
                entry = self._build_entry(k)
                self._cache[entry.key_hash] = entry
        logger.info("Loaded %d API keys into cache", len(self._cache))

    def _build_entry(self, k: dict) -> ApiKeyCacheEntry:
        expires_at = None
        if k.get("expires_at"):
            try:
                expires_at = datetime.fromisoformat(k["expires_at"])
            except (ValueError, TypeError):
                pass
        model_perms = [
            ModelPermission(
                permission_type=mp["permission_type"],
                permission_value=mp.get("permission_value"),
            )
            for mp in k.get("model_permissions", [])
        ]
        return ApiKeyCacheEntry(
            key_id=k["id"],
            user_id=k["user_id"],
            key_hash=k["key_hash"],
            key_prefix=k["key_prefix"],
            name=k.get("name"),
            enabled=bool(k.get("enabled", 1)),
            user_enabled=bool(k.get("user_enabled", 1)),
            expires_at=expires_at,
            model_permissions=model_perms,
        )

    def get(self, key_hash: str) -> Optional[ApiKeyCacheEntry]:
        with self._lock:
            return self._cache.get(key_hash)

    def add(self, key_data: dict):
        entry = self._build_entry(key_data)
        with self._lock:
            self._cache[entry.key_hash] = entry

    def remove(self, key_hash: str):
        with self._lock:
            self._cache.pop(key_hash, None)

    def invalidate_user_keys(self, user_id: int, enabled: bool):
        with self._lock:
            for entry in self._cache.values():
                if entry.user_id == user_id:
                    entry.user_enabled = enabled

    def update_key(self, key_hash: str, **kwargs):
        with self._lock:
            entry = self._cache.get(key_hash)
            if entry:
                if "enabled" in kwargs:
                    entry.enabled = bool(kwargs["enabled"])
                if "model_permissions" in kwargs:
                    entry.model_permissions = kwargs["model_permissions"]

    def reload(self):
        self._load_all()
