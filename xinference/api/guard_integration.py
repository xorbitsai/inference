# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional fastapi-guard security middleware wiring for the RESTful API.

fastapi-guard (https://github.com/Guard-Core/fastapi-guard) provides IP
block/allow lists, rate limiting with auto-ban, user-agent blocking,
penetration-attempt detection, passive mode, optional Redis-backed shared
state, and optional IPInfo geo/cloud-provider lookups.

This complements the existing protections without replacing them: the
``XINFERENCE_ALLOWED_IPS`` allowlist stays first, and the auth rate limiter
(bans on failed logins) remains auth-scoped. The guard adds general
per-request rate limiting, blocklists, user-agent blocking, and payload
screening.

Everything here is opt-in: with ``XINFERENCE_GUARD_ENABLED`` unset (the
default), ``attach_guard`` is a no-op and the server behaves exactly as
before. Install the extra with ``pip install "xinference[guard]"``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI
    from guard import SecurityConfig

DEFAULT_EXCLUDED_PATHS = "/health,/docs,/redoc,/openapi.json"
DEFAULT_TRUSTED_PROXIES = "10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"


def _csv(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _build_guard_config() -> SecurityConfig:
    """Build the SecurityConfig from the XINFERENCE_GUARD_* environment variables.

    Environment variables are read at call time (the same reason
    ``is_auth_advanced`` reads at call time: forked child processes).
    """
    from guard import SecurityConfig

    kwargs: dict[str, Any] = {
        "enable_rate_limiting": True,
        "rate_limit": _env_int("XINFERENCE_GUARD_RATE_LIMIT", 100),
        "rate_limit_window": _env_int("XINFERENCE_GUARD_RATE_LIMIT_WINDOW", 60),
        "enable_ip_banning": True,
        "auto_ban_threshold": _env_int("XINFERENCE_GUARD_AUTO_BAN_THRESHOLD", 10),
        "auto_ban_duration": _env_int("XINFERENCE_GUARD_AUTO_BAN_DURATION", 300),
        "enable_penetration_detection": True,
        # In-memory state unless a Redis URL is configured: never implicitly
        # depend on a Redis server being reachable. The RESTful API runs as a
        # single process per supervisor, matching the existing auth rate
        # limiter's per-process design; set XINFERENCE_GUARD_REDIS_URL for
        # multi-instance deployments behind xinference-router.
        "enable_redis": False,
        "passive_mode": _env_bool("XINFERENCE_GUARD_PASSIVE_MODE"),
        "blacklist": _csv(os.environ.get("XINFERENCE_GUARD_BLOCKED_IPS")),
        "blocked_user_agents": list(_csv(os.environ.get("XINFERENCE_GUARD_BLOCKED_USER_AGENTS"))),
        "trusted_proxies": _csv(os.environ.get("XINFERENCE_GUARD_TRUSTED_PROXIES"))
        or _csv(DEFAULT_TRUSTED_PROXIES),
        "trusted_proxy_depth": _env_int("XINFERENCE_GUARD_TRUSTED_PROXY_DEPTH", 1),
        "exclude_paths": list(
            _csv(os.environ.get("XINFERENCE_GUARD_EXCLUDED_PATHS"))
            or DEFAULT_EXCLUDED_PATHS.split(",")
        ),
    }

    if allowed_ips := _csv(os.environ.get("XINFERENCE_GUARD_ALLOWED_IPS")):
        kwargs["whitelist"] = allowed_ips
    if blocked_countries := _csv(os.environ.get("XINFERENCE_GUARD_BLOCKED_COUNTRIES")):
        kwargs["blocked_countries"] = frozenset(blocked_countries)
    if allowed_countries := _csv(os.environ.get("XINFERENCE_GUARD_ALLOWED_COUNTRIES")):
        kwargs["whitelist_countries"] = frozenset(allowed_countries)
    if cloud_providers := _csv(os.environ.get("XINFERENCE_GUARD_BLOCK_CLOUD_PROVIDERS")):
        kwargs["block_cloud_providers"] = frozenset(cloud_providers)

    if redis_url := os.environ.get("XINFERENCE_GUARD_REDIS_URL"):
        kwargs["enable_redis"] = True
        kwargs["redis_url"] = redis_url
        kwargs["redis_prefix"] = "xinference_guard:"

    if ipinfo_token := os.environ.get("XINFERENCE_GUARD_IPINFO_TOKEN"):
        kwargs["ipinfo_token"] = ipinfo_token

    return SecurityConfig(**kwargs)


def attach_guard(app: FastAPI) -> None:
    """Attach the fastapi-guard middleware when XINFERENCE_GUARD_ENABLED is set.

    No-op unless the env flag is on. Fails loudly when the flag is on but
    the package is missing: a misconfiguration must never silently disable
    security.
    """
    if os.environ.get("XINFERENCE_GUARD_ENABLED", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    try:
        from guard import SecurityMiddleware
    except ImportError as exc:
        raise ImportError(
            "XINFERENCE_GUARD_ENABLED requires fastapi-guard. "
            'Install it with: pip install "xinference[guard]"'
        ) from exc

    app.add_middleware(SecurityMiddleware, config=_build_guard_config())
