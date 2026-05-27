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
"""Admin routes for brute-force protection / rate-limit management."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, Security

from ...responses import JSONResponse
from ..advanced.auth_service import AdvancedAuthService

if TYPE_CHECKING:
    from ...restful_api import RESTfulAPI

logger = logging.getLogger(__name__)


def _get_rate_limiter(request: Request):
    auth_service = getattr(request.app.state, "advanced_auth", None)
    if auth_service is not None and hasattr(auth_service, "_rate_limiter"):
        return auth_service._rate_limiter
    rl = getattr(request.app.state, "rate_limiter", None)
    if rl is None:
        raise HTTPException(status_code=503, detail="Rate limiter not available")
    return rl


# --- Global config ---


async def get_rate_limit_config(request: Request) -> JSONResponse:
    rl = _get_rate_limiter(request)
    return JSONResponse(content={"ip": rl.get_ip_config(), "key": rl.get_key_config()})


async def update_rate_limit_config(request: Request) -> JSONResponse:
    rl = _get_rate_limiter(request)
    body = await request.json()
    if "ip" in body:
        rl.update_ip_config(**body["ip"])
    if "key" in body:
        rl.update_key_config(**body["key"])
    return JSONResponse(content={"ok": True})


# --- Banned IPs ---


async def get_banned_ips(request: Request) -> JSONResponse:
    rl = _get_rate_limiter(request)
    return JSONResponse(content=rl.get_banned_ips())


async def unban_ip(request: Request) -> JSONResponse:
    rl = _get_rate_limiter(request)
    body = await request.json()
    ip = body.get("ip", "")
    if not ip:
        raise HTTPException(status_code=400, detail="ip required")
    rl.unban_ip(ip)
    return JSONResponse(content={"ok": True})


async def unban_all_ips(request: Request) -> JSONResponse:
    rl = _get_rate_limiter(request)
    count = rl.unban_all_ips()
    return JSONResponse(content={"unbanned": count})


# --- Banned Keys ---


async def get_banned_keys(request: Request) -> JSONResponse:
    rl = _get_rate_limiter(request)
    return JSONResponse(content=rl.get_banned_keys())


async def unban_key(request: Request) -> JSONResponse:
    rl = _get_rate_limiter(request)
    body = await request.json()
    ip = body.get("ip", "")
    key_id = body.get("key_id")
    if not ip or key_id is None:
        raise HTTPException(status_code=400, detail="ip and key_id required")
    rl.unban_key(ip, int(key_id))
    return JSONResponse(content={"ok": True})


async def unban_all_keys(request: Request) -> JSONResponse:
    rl = _get_rate_limiter(request)
    count = rl.unban_all_keys()
    return JSONResponse(content={"unbanned": count})


# --- Per-Key bans ---


async def get_key_banned(key_id: int, request: Request) -> JSONResponse:
    rl = _get_rate_limiter(request)
    return JSONResponse(content=rl.get_key_bans(key_id))


async def unban_key_ip(key_id: int, request: Request) -> JSONResponse:
    rl = _get_rate_limiter(request)
    body = await request.json()
    ip = body.get("ip", "")
    if not ip:
        raise HTTPException(status_code=400, detail="ip required")
    rl.unban_key_ip(key_id, ip)
    return JSONResponse(content={"ok": True})


async def unban_key_all(key_id: int, request: Request) -> JSONResponse:
    rl = _get_rate_limiter(request)
    count = rl.unban_key_all(key_id)
    return JSONResponse(content={"unbanned": count})


# --- Route registration ---


def register_security_routes(api: "RESTfulAPI") -> None:
    router = api._router
    auth_service: AdvancedAuthService = api._app.state.advanced_auth

    router.add_api_route(
        "/v1/admin/security/rate-limit",
        get_rate_limit_config,
        methods=["GET"],
        dependencies=[Security(auth_service, scopes=["admin"])],
    )
    router.add_api_route(
        "/v1/admin/security/rate-limit",
        update_rate_limit_config,
        methods=["PUT"],
        dependencies=[Security(auth_service, scopes=["admin"])],
    )
    router.add_api_route(
        "/v1/admin/security/banned-ips",
        get_banned_ips,
        methods=["GET"],
        dependencies=[Security(auth_service, scopes=["admin"])],
    )
    router.add_api_route(
        "/v1/admin/security/unban-ip",
        unban_ip,
        methods=["POST"],
        dependencies=[Security(auth_service, scopes=["admin"])],
    )
    router.add_api_route(
        "/v1/admin/security/unban-all-ips",
        unban_all_ips,
        methods=["POST"],
        dependencies=[Security(auth_service, scopes=["admin"])],
    )
    router.add_api_route(
        "/v1/admin/security/banned-keys",
        get_banned_keys,
        methods=["GET"],
        dependencies=[Security(auth_service, scopes=["admin"])],
    )
    router.add_api_route(
        "/v1/admin/security/unban-key",
        unban_key,
        methods=["POST"],
        dependencies=[Security(auth_service, scopes=["admin"])],
    )
    router.add_api_route(
        "/v1/admin/security/unban-all-keys",
        unban_all_keys,
        methods=["POST"],
        dependencies=[Security(auth_service, scopes=["admin"])],
    )
    router.add_api_route(
        "/v1/admin/keys/{key_id}/banned",
        get_key_banned,
        methods=["GET"],
        dependencies=[Security(auth_service, scopes=["admin"])],
    )
    router.add_api_route(
        "/v1/admin/keys/{key_id}/unban",
        unban_key_ip,
        methods=["POST"],
        dependencies=[Security(auth_service, scopes=["admin"])],
    )
    router.add_api_route(
        "/v1/admin/keys/{key_id}/unban-all",
        unban_key_all,
        methods=["POST"],
        dependencies=[Security(auth_service, scopes=["admin"])],
    )
