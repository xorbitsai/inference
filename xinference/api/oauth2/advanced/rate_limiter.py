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
"""Dual-layer brute-force protection: IP-level and (IP, Key)-level banning."""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ....constants import (
    XINFERENCE_RATE_LIMIT_IP_BAN_SECONDS,
    XINFERENCE_RATE_LIMIT_IP_MAX_FAILURES,
    XINFERENCE_RATE_LIMIT_IP_WINDOW_SECONDS,
    XINFERENCE_RATE_LIMIT_KEY_BAN_SECONDS,
    XINFERENCE_RATE_LIMIT_KEY_MAX_FAILURES,
    XINFERENCE_RATE_LIMIT_KEY_WINDOW_SECONDS,
)


@dataclass
class RateLimitConfig:
    max_failures: int = 10
    window_seconds: int = 300
    ban_seconds: int = 600


@dataclass
class FailureRecord:
    timestamps: List[float] = field(default_factory=list)
    banned_until: float = 0.0


class RateLimiter:
    def __init__(self):
        self._lock = threading.Lock()
        self._ip_records: Dict[str, FailureRecord] = {}
        self._key_records: Dict[Tuple[str, int], FailureRecord] = {}
        self._last_cleanup = time.time()

        self._ip_config = RateLimitConfig(
            max_failures=XINFERENCE_RATE_LIMIT_IP_MAX_FAILURES,
            window_seconds=XINFERENCE_RATE_LIMIT_IP_WINDOW_SECONDS,
            ban_seconds=XINFERENCE_RATE_LIMIT_IP_BAN_SECONDS,
        )
        self._key_config = RateLimitConfig(
            max_failures=XINFERENCE_RATE_LIMIT_KEY_MAX_FAILURES,
            window_seconds=XINFERENCE_RATE_LIMIT_KEY_WINDOW_SECONDS,
            ban_seconds=XINFERENCE_RATE_LIMIT_KEY_BAN_SECONDS,
        )

    # --- IP Layer ---

    def _update_ban_gauges(self) -> None:
        """Update Prometheus gauges. Must be called while holding self._lock."""
        try:
            from ....core.metrics import banned_ips_total, banned_keys_total

            now = time.time()
            ip_count = sum(
                1
                for r in self._ip_records.values()
                if r.banned_until and r.banned_until > now
            )
            key_count = sum(
                1
                for r in self._key_records.values()
                if r.banned_until and r.banned_until > now
            )
            banned_ips_total.set({}, ip_count)
            banned_keys_total.set({}, key_count)
        except Exception:
            pass

    def _cleanup_stale_records(self) -> None:
        """Remove stale records that are not banned and have no recent failures.
        Must be called while holding self._lock.
        """
        now = time.time()
        if now - self._last_cleanup < 300:
            return
        self._last_cleanup = now

        ip_stale = [
            k
            for k, r in self._ip_records.items()
            if not (r.banned_until and r.banned_until > now)
            and (
                not r.timestamps
                or now - r.timestamps[-1] > self._ip_config.window_seconds
            )
        ]
        for ip_k in ip_stale:
            del self._ip_records[ip_k]

        key_stale = [
            k
            for k, r in self._key_records.items()
            if not (r.banned_until and r.banned_until > now)
            and (
                not r.timestamps
                or now - r.timestamps[-1] > self._key_config.window_seconds
            )
        ]
        for key_k in key_stale:
            del self._key_records[key_k]

    def is_ip_banned(self, ip: str) -> bool:
        with self._lock:
            rec = self._ip_records.get(ip)
            if not rec:
                return False
            if rec.banned_until and time.time() < rec.banned_until:
                return True
            if rec.banned_until and time.time() >= rec.banned_until:
                del self._ip_records[ip]
            return False

    def record_invalid_key(self, ip: str) -> None:
        with self._lock:
            self._cleanup_stale_records()
            rec = self._ip_records.setdefault(ip, FailureRecord())
            now = time.time()
            cutoff = now - self._ip_config.window_seconds
            rec.timestamps = [t for t in rec.timestamps if t > cutoff]
            rec.timestamps.append(now)
            if len(rec.timestamps) >= self._ip_config.max_failures:
                rec.banned_until = now + self._ip_config.ban_seconds
                self._update_ban_gauges()

    def reset_ip(self, ip: str) -> None:
        with self._lock:
            self._ip_records.pop(ip, None)

    # --- Key Layer ---

    def is_key_banned(self, ip: str, key_id: int) -> bool:
        with self._lock:
            rec = self._key_records.get((ip, key_id))
            if not rec:
                return False
            if rec.banned_until and time.time() < rec.banned_until:
                return True
            if rec.banned_until and time.time() >= rec.banned_until:
                del self._key_records[(ip, key_id)]
            return False

    def record_key_failure(
        self, ip: str, key_id: int, config: Optional[RateLimitConfig] = None
    ) -> None:
        cfg = config or self._key_config
        with self._lock:
            key = (ip, key_id)
            rec = self._key_records.setdefault(key, FailureRecord())
            now = time.time()
            cutoff = now - cfg.window_seconds
            rec.timestamps = [t for t in rec.timestamps if t > cutoff]
            rec.timestamps.append(now)
            if len(rec.timestamps) >= cfg.max_failures:
                rec.banned_until = now + cfg.ban_seconds
                self._update_ban_gauges()

    def reset_key(self, ip: str, key_id: int) -> None:
        with self._lock:
            self._key_records.pop((ip, key_id), None)

    # --- Admin operations ---

    def get_ip_config(self) -> Dict:
        return {
            "max_failures": self._ip_config.max_failures,
            "window_seconds": self._ip_config.window_seconds,
            "ban_seconds": self._ip_config.ban_seconds,
        }

    def get_key_config(self) -> Dict:
        return {
            "max_failures": self._key_config.max_failures,
            "window_seconds": self._key_config.window_seconds,
            "ban_seconds": self._key_config.ban_seconds,
        }

    def update_ip_config(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._ip_config, k) and v is not None:
                    setattr(self._ip_config, k, int(v))

    def update_key_config(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._key_config, k) and v is not None:
                    setattr(self._key_config, k, int(v))

    def get_banned_ips(self) -> List[Dict]:
        now = time.time()
        result = []
        with self._lock:
            for ip, rec in list(self._ip_records.items()):
                if rec.banned_until and rec.banned_until > now:
                    result.append(
                        {
                            "ip": ip,
                            "banned_until": rec.banned_until,
                            "remaining_seconds": int(rec.banned_until - now),
                        }
                    )
        return result

    def get_banned_keys(self) -> List[Dict]:
        now = time.time()
        result = []
        with self._lock:
            for (ip, key_id), rec in list(self._key_records.items()):
                if rec.banned_until and rec.banned_until > now:
                    result.append(
                        {
                            "ip": ip,
                            "key_id": key_id,
                            "banned_until": rec.banned_until,
                            "remaining_seconds": int(rec.banned_until - now),
                        }
                    )
        return result

    def get_key_bans(self, key_id: int) -> List[Dict]:
        now = time.time()
        result = []
        with self._lock:
            for (ip, kid), rec in list(self._key_records.items()):
                if kid == key_id and rec.banned_until and rec.banned_until > now:
                    result.append(
                        {
                            "ip": ip,
                            "banned_until": rec.banned_until,
                            "remaining_seconds": int(rec.banned_until - now),
                        }
                    )
        return result

    def unban_ip(self, ip: str) -> bool:
        with self._lock:
            result = self._ip_records.pop(ip, None) is not None
            if result:
                self._update_ban_gauges()
        return result

    def unban_all_ips(self) -> int:
        with self._lock:
            count = len(self._ip_records)
            self._ip_records.clear()
            self._update_ban_gauges()
        return count

    def unban_key(self, ip: str, key_id: int) -> bool:
        with self._lock:
            result = self._key_records.pop((ip, key_id), None) is not None
            if result:
                self._update_ban_gauges()
        return result

    def unban_all_keys(self) -> int:
        with self._lock:
            count = len(self._key_records)
            self._key_records.clear()
            self._update_ban_gauges()
        return count

    def unban_key_ip(self, key_id: int, ip: str) -> bool:
        return self.unban_key(ip, key_id)

    def unban_key_all(self, key_id: int) -> int:
        with self._lock:
            to_remove = [k for k in self._key_records if k[1] == key_id]
            for k in to_remove:
                del self._key_records[k]
            self._update_ban_gauges()
        return len(to_remove)
