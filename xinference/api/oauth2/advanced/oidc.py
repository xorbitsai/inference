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
"""Keycloak OIDC integration routes."""

from __future__ import annotations

import logging
import secrets
import time
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.responses import Response

from ..advanced.auth_service import AdvancedAuthService

if TYPE_CHECKING:
    from ...restful_api import RESTfulAPI

logger = logging.getLogger(__name__)

_oidc_client = None
_state_store: dict = {}

_JWKS_CACHE_TTL = 86400  # 24 hours


class OIDCConfig:
    def __init__(self, issuer: str):
        self._issuer = issuer
        self._well_known = None
        self._jwks_data = None
        self._jwks_fetched_at = 0.0


def _get_oidc_client():
    global _oidc_client
    if _oidc_client is not None:
        return _oidc_client

    from ....constants import XINFERENCE_OIDC_ISSUER

    _oidc_client = OIDCConfig(XINFERENCE_OIDC_ISSUER)
    return _oidc_client


async def _get_well_known(client):
    if client._well_known is not None:
        return client._well_known
    import httpx

    url = f"{client._issuer}/.well-known/openid-configuration"
    async with httpx.AsyncClient() as http:
        resp = await http.get(url)
        resp.raise_for_status()
        client._well_known = resp.json()
    return client._well_known


async def _get_jwks(client, well_known) -> dict:
    import time

    import httpx

    now = time.time()
    if (
        client._jwks_data is not None
        and (now - client._jwks_fetched_at) < _JWKS_CACHE_TTL
    ):
        return client._jwks_data

    jwks_uri = well_known.get("jwks_uri")
    async with httpx.AsyncClient() as http:
        resp = await http.get(jwks_uri)
        resp.raise_for_status()
        client._jwks_data = resp.json()
        client._jwks_fetched_at = now
    return client._jwks_data


def get_advanced_auth(request: Request) -> AdvancedAuthService:
    return request.app.state.advanced_auth


async def oidc_authorize(request: Request) -> Response:
    # Clean up expired states (older than 10 minutes)
    now = time.time()
    expired_states = [s for s, t in _state_store.items() if now - t > 600]
    for s in expired_states:
        _state_store.pop(s, None)

    client = _get_oidc_client()
    well_known = await _get_well_known(client)
    state = secrets.token_urlsafe(32)
    _state_store[state] = now

    auth_url = well_known["authorization_endpoint"]
    from ....constants import XINFERENCE_OIDC_CLIENT_ID, XINFERENCE_OIDC_REDIRECT_URI

    params = {
        "response_type": "code",
        "client_id": XINFERENCE_OIDC_CLIENT_ID,
        "redirect_uri": XINFERENCE_OIDC_REDIRECT_URI,
        "scope": "openid profile email",
        "state": state,
    }
    url = f"{auth_url}?{urlencode(params)}"
    return RedirectResponse(url=url, status_code=302)


async def oidc_callback(request: Request) -> Response:
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    error = request.query_params.get("error")

    if error:
        raise HTTPException(status_code=400, detail=f"OIDC error: {error}")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")
    if not state or state not in _state_store:
        raise HTTPException(status_code=400, detail="Invalid state parameter")

    _state_store.pop(state, None)

    client = _get_oidc_client()
    well_known = await _get_well_known(client)
    token_url = well_known["token_endpoint"]

    import httpx

    from ....constants import (
        XINFERENCE_OIDC_CLIENT_ID,
        XINFERENCE_OIDC_CLIENT_SECRET,
        XINFERENCE_OIDC_REDIRECT_URI,
    )

    async with httpx.AsyncClient() as http:
        token_resp = await http.post(
            token_url,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": XINFERENCE_OIDC_REDIRECT_URI,
                "client_id": XINFERENCE_OIDC_CLIENT_ID,
                "client_secret": XINFERENCE_OIDC_CLIENT_SECRET,
            },
        )
        if token_resp.status_code != 200:
            logger.error("Token exchange failed: %s", token_resp.text[:500])
            raise HTTPException(status_code=502, detail="Token exchange failed")
        token_data = token_resp.json()

    id_token = token_data.get("id_token")
    if not id_token:
        raise HTTPException(status_code=502, detail="No ID token in response")

    # Decode ID token (verify with JWKS via python-jose)
    try:
        from jose import jwt as jose_jwt

        jwks_data = await _get_jwks(client, well_known)
        access_token_raw = token_data.get("access_token")

        claims = jose_jwt.decode(
            id_token,
            jwks_data,
            algorithms=["RS256"],
            audience=XINFERENCE_OIDC_CLIENT_ID,
            issuer=client._issuer,
            access_token=access_token_raw,
        )
    except Exception as e:
        logger.error("ID token verification failed: %s", e)
        raise HTTPException(status_code=401, detail="ID token verification failed")

    sub = claims.get("sub")
    preferred_username = claims.get("preferred_username") or claims.get("email") or sub
    if not sub:
        raise HTTPException(status_code=400, detail="No 'sub' claim in ID token")

    auth: AdvancedAuthService = get_advanced_auth(request)

    # Find or create user by oidc_sub
    user = auth.db.get_user_by_oidc_sub(sub)
    if not user:
        try:
            user_id = auth.db.create_user(
                username=preferred_username,
                password_hash=None,
                source="oidc",
                oidc_sub=sub,
                permissions=["models:list"],
            )
        except Exception as e:
            logger.exception("Failed to create OIDC user, retrying lookup")
            # Concurrent creation — retry lookup
            user = auth.db.get_user_by_oidc_sub(sub)
            if not user:
                raise HTTPException(
                    status_code=500, detail="Failed to create OIDC user"
                ) from e
            user_id = user["id"]
        user = auth.db.get_user_by_id(user_id)

    if not user or not user["enabled"]:
        raise HTTPException(status_code=403, detail="User account is disabled")

    access_token = auth.create_access_token(
        user["id"], user["username"], user["permissions"]
    )
    refresh_token = auth.create_refresh_token(user["id"])

    # Return HTML that stores tokens and redirects to main page
    html = f"""
    <html><body><script>
    sessionStorage.setItem('token', '{access_token}');
    sessionStorage.setItem('refresh_token', '{refresh_token}');
    document.cookie = 'token={access_token}; path=/';
    window.location.href = '/';
    </script></body></html>
    """
    return HTMLResponse(content=html)


def validate_oidc_config() -> None:
    from ....constants import (
        XINFERENCE_OIDC_CLIENT_ID,
        XINFERENCE_OIDC_CLIENT_SECRET,
        XINFERENCE_OIDC_ENABLED,
        XINFERENCE_OIDC_ISSUER,
        XINFERENCE_OIDC_REDIRECT_URI,
    )

    if not XINFERENCE_OIDC_ENABLED:
        return
    missing = []
    if not XINFERENCE_OIDC_ISSUER:
        missing.append("XINFERENCE_OIDC_ISSUER")
    if not XINFERENCE_OIDC_CLIENT_ID:
        missing.append("XINFERENCE_OIDC_CLIENT_ID")
    if not XINFERENCE_OIDC_CLIENT_SECRET:
        missing.append("XINFERENCE_OIDC_CLIENT_SECRET")
    if not XINFERENCE_OIDC_REDIRECT_URI:
        missing.append("XINFERENCE_OIDC_REDIRECT_URI")
    if missing:
        raise SystemExit(
            f"ERROR: OIDC is enabled but missing required config: {', '.join(missing)}"
        )


def register_oidc_routes(api: "RESTfulAPI") -> None:
    router = api._router
    router.add_api_route("/api/oidc/authorize", oidc_authorize, methods=["GET"])
    router.add_api_route("/api/oidc/callback", oidc_callback, methods=["GET"])
