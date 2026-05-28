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
"""Admin routes for the advanced authentication system."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from fastapi import HTTPException, Query, Request, Security

from ...responses import JSONResponse
from ..advanced.auth_service import AdvancedAuthService, _get_client_ip
from ..advanced.crypto import get_password_hash

if TYPE_CHECKING:
    from ...restful_api import RESTfulAPI

logger = logging.getLogger(__name__)


def _refresh_key_gauges(auth: AdvancedAuthService) -> None:
    """Update Prometheus gauges for active/expired key counts.

    NOTE: Metrics counters are defined in the security-hardening PR.
    This is a placeholder that will be filled when that PR is merged.
    """
    pass


def get_advanced_auth(request: Request) -> AdvancedAuthService:
    return request.app.state.advanced_auth


def _get_current_user_from_token(request: Request, auth: AdvancedAuthService):
    """Extract current user info from the Authorization header."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        return None, None, []
    payload = auth.verify_access_token(token)
    if not payload:
        return None, None, []
    return payload.get("user_id"), payload.get("sub"), payload.get("scopes", [])


# --- Auth endpoints ---


async def advanced_login(request: Request) -> JSONResponse:
    try:
        from .audit import record_audit_event
    except ImportError:
        record_audit_event = None  # type: ignore[assignment]

    auth: AdvancedAuthService = get_advanced_auth(request)
    body = await request.json()
    username = body.get("username", "")
    password = body.get("password", "")
    client_ip = _get_client_ip(request)
    try:
        result = auth.login(username, password)
    except Exception:
        if record_audit_event is not None:
            record_audit_event(
                user=username,
                api_key_name="",
                api_key_prefix="",
                model_id="",
                model_type="",
                endpoint="/token",
                status="login_failed",
                client_ip=client_ip,
                category="auth",
                auth_type="none",
            )
        raise
    if record_audit_event is not None:
        record_audit_event(
            user=username,
            api_key_name="",
            api_key_prefix="",
            model_id="",
            model_type="",
            endpoint="/token",
            status="success",
            client_ip=client_ip,
            category="auth",
            auth_type="none",
        )
    return JSONResponse(content=result)


async def advanced_refresh(request: Request) -> JSONResponse:
    auth: AdvancedAuthService = get_advanced_auth(request)
    body = await request.json()
    refresh_token = body.get("refresh_token", "")
    if not refresh_token:
        raise HTTPException(status_code=400, detail="refresh_token required")
    result = auth.refresh_access_token(refresh_token)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    return JSONResponse(content=result)


async def advanced_logout(request: Request) -> JSONResponse:
    auth: AdvancedAuthService = get_advanced_auth(request)
    body = await request.json()
    refresh_token = body.get("refresh_token", "")
    auth.logout(refresh_token)
    return JSONResponse(content={"ok": True})


# --- User management ---


async def create_user(request: Request) -> JSONResponse:
    auth: AdvancedAuthService = get_advanced_auth(request)
    body = await request.json()
    username = body.get("username")
    password = body.get("password")
    permissions = body.get("permissions", [])

    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password required")

    existing = auth.db.get_user_by_username(username, "local")
    if existing:
        raise HTTPException(status_code=409, detail="User already exists")

    password_hash = get_password_hash(password)
    user_id = auth.db.create_user(
        username=username,
        password_hash=password_hash,
        source="local",
        permissions=permissions,
    )
    return JSONResponse(content={"id": user_id, "username": username}, status_code=201)


async def list_users(
    request: Request, source: Optional[str] = Query(None)
) -> JSONResponse:
    auth: AdvancedAuthService = get_advanced_auth(request)
    users = auth.db.list_users(source=source)
    result = []
    for u in users:
        result.append(
            {
                "id": u["id"],
                "username": u["username"],
                "source": u["source"],
                "enabled": bool(u["enabled"]),
                "must_change_password": bool(u["must_change_password"]),
                "permissions": u["permissions"],
                "created_at": u.get("created_at"),
            }
        )
    return JSONResponse(content=result)


async def get_user(user_id: int, request: Request) -> JSONResponse:
    auth: AdvancedAuthService = get_advanced_auth(request)
    user = auth.db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return JSONResponse(
        content={
            "id": user["id"],
            "username": user["username"],
            "source": user["source"],
            "enabled": bool(user["enabled"]),
            "must_change_password": bool(user["must_change_password"]),
            "permissions": user["permissions"],
            "created_at": user.get("created_at"),
        }
    )


async def update_user(user_id: int, request: Request) -> JSONResponse:
    auth: AdvancedAuthService = get_advanced_auth(request)
    user = auth.db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    body = await request.json()

    if "enabled" in body:
        enabled = int(body["enabled"])
        if enabled:
            auth.enable_user(user_id)
        else:
            auth.disable_user(user_id)

    if "permissions" in body:
        auth.db.set_user_permissions(user_id, body["permissions"])

    return JSONResponse(content={"ok": True})


async def delete_user(user_id: int, request: Request) -> JSONResponse:
    auth: AdvancedAuthService = get_advanced_auth(request)
    user = auth.db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    auth.db.delete_user(user_id)
    auth.cache.reload()
    return JSONResponse(content={"ok": True})


async def change_password(user_id: int, request: Request) -> JSONResponse:
    auth: AdvancedAuthService = get_advanced_auth(request)
    user = auth.db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    body = await request.json()
    new_password = body.get("new_password")
    if not new_password:
        raise HTTPException(status_code=400, detail="new_password required")
    password_hash = get_password_hash(new_password)
    auth.db.update_user(user_id, password_hash=password_hash, must_change_password=0)
    return JSONResponse(content={"ok": True})


# --- API Key management ---


async def create_api_key(request: Request) -> JSONResponse:
    auth: AdvancedAuthService = get_advanced_auth(request)
    body = await request.json()

    current_user_id, _, scopes = _get_current_user_from_token(request, auth)
    is_admin = "admin" in scopes or "keys:manage" in scopes

    owner_id = body.get("owner")
    if owner_id and not is_admin:
        owner_id = current_user_id
    elif not owner_id:
        owner_id = current_user_id
    if not owner_id:
        raise HTTPException(status_code=400, detail="owner required")

    result = auth.create_api_key_for_user(
        user_id=owner_id,
        name=body.get("name"),
        description=body.get("description"),
        expires_at=body.get("expires_at"),
        model_permissions=body.get("model_permissions"),
        rate_limit_max_failures=body.get("rate_limit_max_failures"),
        rate_limit_window_seconds=body.get("rate_limit_window_seconds"),
        rate_limit_ban_seconds=body.get("rate_limit_ban_seconds"),
    )
    _refresh_key_gauges(auth)
    return JSONResponse(content=result, status_code=201)


async def list_api_keys(
    request: Request, owner: Optional[int] = Query(None)
) -> JSONResponse:
    auth: AdvancedAuthService = get_advanced_auth(request)
    current_user_id, _, scopes = _get_current_user_from_token(request, auth)
    is_admin = "admin" in scopes or "keys:manage" in scopes

    if not is_admin:
        owner = current_user_id

    keys = auth.db.list_api_keys(user_id=owner)
    result = []
    for k in keys:
        result.append(
            {
                "id": k["id"],
                "user_id": k["user_id"],
                "key_prefix": k["key_prefix"],
                "name": k.get("name"),
                "enabled": bool(k.get("enabled", 1)),
                "expires_at": k.get("expires_at"),
                "model_permissions": k.get("model_permissions", []),
                "created_at": k.get("created_at"),
            }
        )
    return JSONResponse(content=result)


async def get_api_key(key_id: int, request: Request) -> JSONResponse:
    auth: AdvancedAuthService = get_advanced_auth(request)
    current_user_id, _, scopes = _get_current_user_from_token(request, auth)
    is_admin = "admin" in scopes or "keys:manage" in scopes

    key = auth.db.get_api_key_by_id(key_id)
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    if not is_admin and key["user_id"] != current_user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return JSONResponse(
        content={
            "id": key["id"],
            "user_id": key["user_id"],
            "key_prefix": key["key_prefix"],
            "name": key.get("name"),
            "enabled": bool(key.get("enabled", 1)),
            "expires_at": key.get("expires_at"),
            "model_permissions": key.get("model_permissions", []),
            "created_at": key.get("created_at"),
        }
    )


async def update_api_key(key_id: int, request: Request) -> JSONResponse:
    auth: AdvancedAuthService = get_advanced_auth(request)
    key = auth.db.get_api_key_by_id(key_id)
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")

    body = await request.json()
    update_fields = {}
    for field in (
        "name",
        "enabled",
        "expires_at",
        "rate_limit_max_failures",
        "rate_limit_window_seconds",
        "rate_limit_ban_seconds",
    ):
        if field in body:
            update_fields[field] = body[field]
    if update_fields:
        auth.db.update_api_key(key_id, **update_fields)

    if "model_permissions" in body:
        auth.db.set_api_key_model_permissions(key_id, body["model_permissions"])

    auth.cache.reload()
    _refresh_key_gauges(auth)
    return JSONResponse(content={"ok": True})


async def delete_api_key(key_id: int, request: Request) -> JSONResponse:
    auth: AdvancedAuthService = get_advanced_auth(request)
    key = auth.db.get_api_key_by_id(key_id)
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    auth.db.delete_api_key(key_id)
    auth.cache.remove(key["key_hash"])
    _refresh_key_gauges(auth)
    return JSONResponse(content={"ok": True})


async def reveal_api_key(key_id: int, request: Request) -> JSONResponse:
    auth: AdvancedAuthService = get_advanced_auth(request)
    plaintext = auth.reveal_api_key(key_id)
    if not plaintext:
        raise HTTPException(status_code=404, detail="API key not found")
    return JSONResponse(content={"key": plaintext})


# --- Permissions ---


async def get_key_permissions(key_id: int, request: Request) -> JSONResponse:
    auth: AdvancedAuthService = get_advanced_auth(request)
    key = auth.db.get_api_key_by_id(key_id)
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    return JSONResponse(content={"model_permissions": key.get("model_permissions", [])})


async def update_key_permissions(key_id: int, request: Request) -> JSONResponse:
    auth: AdvancedAuthService = get_advanced_auth(request)
    key = auth.db.get_api_key_by_id(key_id)
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    body = await request.json()
    permissions = body.get("model_permissions", [])
    auth.db.set_api_key_model_permissions(key_id, permissions)
    auth.cache.reload()
    return JSONResponse(content={"ok": True})


async def get_user_permissions(user_id: int, request: Request) -> JSONResponse:
    auth: AdvancedAuthService = get_advanced_auth(request)
    user = auth.db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return JSONResponse(content={"permissions": user["permissions"]})


async def update_user_permissions(user_id: int, request: Request) -> JSONResponse:
    auth: AdvancedAuthService = get_advanced_auth(request)
    user = auth.db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    body = await request.json()
    permissions = body.get("permissions", [])
    auth.db.set_user_permissions(user_id, permissions)
    return JSONResponse(content={"ok": True})


# --- Route registration ---


def register_advanced_auth_routes(api: "RESTfulAPI") -> None:
    router = api._router
    auth_service: AdvancedAuthService = api._app.state.advanced_auth

    # Store rate_limiter on app state for security routes (if available)
    if auth_service._rate_limiter:
        api._app.state.rate_limiter = auth_service._rate_limiter

    router.add_api_route("/token", advanced_login, methods=["POST"])
    router.add_api_route("/v1/auth/refresh", advanced_refresh, methods=["POST"])
    router.add_api_route("/v1/auth/logout", advanced_logout, methods=["POST"])

    # User management
    router.add_api_route(
        "/v1/admin/users",
        create_user,
        methods=["POST"],
        dependencies=[Security(auth_service, scopes=["users:manage"])],
    )
    router.add_api_route(
        "/v1/admin/users",
        list_users,
        methods=["GET"],
        dependencies=[Security(auth_service, scopes=["users:manage"])],
    )
    router.add_api_route(
        "/v1/admin/users/{user_id}",
        get_user,
        methods=["GET"],
        dependencies=[Security(auth_service, scopes=["users:manage"])],
    )
    router.add_api_route(
        "/v1/admin/users/{user_id}",
        update_user,
        methods=["PUT"],
        dependencies=[Security(auth_service, scopes=["users:manage"])],
    )
    router.add_api_route(
        "/v1/admin/users/{user_id}",
        delete_user,
        methods=["DELETE"],
        dependencies=[Security(auth_service, scopes=["users:manage"])],
    )
    router.add_api_route(
        "/v1/admin/users/{user_id}/password",
        change_password,
        methods=["PUT"],
        dependencies=[Security(auth_service, scopes=["users:manage"])],
    )

    # API Key management
    router.add_api_route(
        "/v1/admin/keys",
        create_api_key,
        methods=["POST"],
        dependencies=[Security(auth_service, scopes=["keys:create"])],
    )
    router.add_api_route(
        "/v1/admin/keys",
        list_api_keys,
        methods=["GET"],
        dependencies=[Security(auth_service, scopes=["keys:create"])],
    )
    router.add_api_route(
        "/v1/admin/keys/{key_id}",
        get_api_key,
        methods=["GET"],
        dependencies=[Security(auth_service, scopes=["keys:create"])],
    )
    router.add_api_route(
        "/v1/admin/keys/{key_id}",
        update_api_key,
        methods=["PUT"],
        dependencies=[Security(auth_service, scopes=["keys:manage"])],
    )
    router.add_api_route(
        "/v1/admin/keys/{key_id}",
        delete_api_key,
        methods=["DELETE"],
        dependencies=[Security(auth_service, scopes=["keys:manage"])],
    )
    router.add_api_route(
        "/v1/admin/keys/{key_id}/reveal",
        reveal_api_key,
        methods=["GET"],
        dependencies=[Security(auth_service, scopes=["admin"])],
    )

    # Permissions
    router.add_api_route(
        "/v1/admin/keys/{key_id}/permissions",
        get_key_permissions,
        methods=["GET"],
        dependencies=[Security(auth_service, scopes=["keys:create"])],
    )
    router.add_api_route(
        "/v1/admin/keys/{key_id}/permissions",
        update_key_permissions,
        methods=["PUT"],
        dependencies=[Security(auth_service, scopes=["keys:manage"])],
    )
    router.add_api_route(
        "/v1/admin/users/{user_id}/permissions",
        get_user_permissions,
        methods=["GET"],
        dependencies=[Security(auth_service, scopes=["users:manage"])],
    )
    router.add_api_route(
        "/v1/admin/users/{user_id}/permissions",
        update_user_permissions,
        methods=["PUT"],
        dependencies=[Security(auth_service, scopes=["users:manage"])],
    )

    # Register security/rate-limit admin routes
    try:
        from .security_routes import register_security_routes

        register_security_routes(api)
    except ImportError:
        pass
