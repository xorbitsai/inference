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
per-request rate limiting, blocklists, user-agent blocking, and URL/query/
header screening.

Client identity is zero-trust: ``X-Forwarded-For`` is honored only from
proxies explicitly listed in ``XINFERENCE_GUARD_TRUSTED_PROXIES``, by both
Uvicorn (which would otherwise rewrite the client address for any peer
before middleware runs) and guard. Prompt bodies are not signature-scanned
by default (model input is data, not commands this API executes); opt into
that with ``XINFERENCE_GUARD_SCAN_BODY``. The default security headers keep
same-origin microphone/camera capture working for the Web UI.

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


def guard_enabled() -> bool:
    """Whether XINFERENCE_GUARD_ENABLED is set (read at call time so forked
    child processes and late env wiring are seen)."""
    return _env_bool("XINFERENCE_GUARD_ENABLED")


def uvicorn_forwarded_allow_ips() -> str:
    """The Uvicorn ``forwarded_allow_ips`` value matching guard's proxy trust.

    ``RESTfulAPI.serve()`` launches Uvicorn with proxy headers enabled. With
    the previous default ``forwarded_allow_ips="*"``, Uvicorn rewrote
    ``scope["client"]`` from ANY peer's ``X-Forwarded-For`` before any
    middleware ran, so a direct client could forge its address and defeat
    guard's blocklists and per-IP limiter (and the allowlist/audit layers
    reading the same field). When the guard is enabled, only proxies listed
    in ``XINFERENCE_GUARD_TRUSTED_PROXIES`` may rewrite the client address;
    with none listed, Uvicorn trusts loopback only. Guard's own
    ``trusted_proxies`` reads the same variable, so Uvicorn, guard, and the
    existing allowlist/audit consumers resolve one client identity.
    """
    if not guard_enabled():
        return "*"
    return (
        ",".join(_csv(os.environ.get("XINFERENCE_GUARD_TRUSTED_PROXIES")))
        or "127.0.0.1"
    )


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
        # Model input is data, not commands this API executes: signature-
        # scanning prompt bodies false-positives on ordinary code-generation
        # and security-analysis conversations (SQL/HTML/shell text inside
        # chat messages). Detection keeps screening the URL path, query
        # params, and headers; reading and scanning bodies is opt-in via
        # XINFERENCE_GUARD_SCAN_BODY.
        "detection_scan_body": _env_bool("XINFERENCE_GUARD_SCAN_BODY"),
        # The dependency's default Permissions-Policy locks the microphone
        # to none, which silently breaks the Web UI's in-page audio recorder
        # (frontend audio-recorder-upload.tsx calls getUserMedia({audio:
        # true}) on the same origin). Keep the hardening but let same-origin
        # media capture work; operators can override the whole policy.
        "security_headers": {
            "enabled": True,
            "hsts": {"max_age": 31536000, "include_subdomains": True, "preload": False},
            "csp": None,
            "frame_options": "SAMEORIGIN",
            "content_type_options": "nosniff",
            "xss_protection": "1; mode=block",
            "referrer_policy": "strict-origin-when-cross-origin",
            "permissions_policy": os.environ.get("XINFERENCE_GUARD_PERMISSIONS_POLICY")
            or "geolocation=(), microphone=(self), camera=(self)",
            "custom": None,
        },
        # In-memory state unless a Redis URL is configured: never implicitly
        # depend on a Redis server being reachable. The RESTful API runs as a
        # single process per supervisor, matching the existing auth rate
        # limiter's per-process design; set XINFERENCE_GUARD_REDIS_URL for
        # multi-instance deployments behind xinference-router.
        "enable_redis": False,
        "passive_mode": _env_bool("XINFERENCE_GUARD_PASSIVE_MODE"),
        "blacklist": _csv(os.environ.get("XINFERENCE_GUARD_BLOCKED_IPS")),
        "blocked_user_agents": list(
            _csv(os.environ.get("XINFERENCE_GUARD_BLOCKED_USER_AGENTS"))
        ),
        # Zero-trust default: no proxies are trusted, so forwarded headers
        # are ignored and the connecting peer is the client (matching the
        # Uvicorn trust configured by uvicorn_forwarded_allow_ips). List
        # real ingress addresses to enable X-Forwarded-For resolution.
        "trusted_proxies": _csv(os.environ.get("XINFERENCE_GUARD_TRUSTED_PROXIES")),
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
    if cloud_providers := _csv(
        os.environ.get("XINFERENCE_GUARD_BLOCK_CLOUD_PROVIDERS")
    ):
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
    if not guard_enabled():
        return
    try:
        from guard import SecurityMiddleware
    except ImportError as exc:
        raise ImportError(
            "XINFERENCE_GUARD_ENABLED requires fastapi-guard. "
            'Install it with: pip install "xinference[guard]"'
        ) from exc

    app.add_middleware(SecurityMiddleware, config=_build_guard_config())
