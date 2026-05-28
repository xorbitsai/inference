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
import base64
import logging
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from jose import JWTError, jwt
from typing_extensions import Annotated

from .cache import ApiKeyCache, ApiKeyCacheEntry
from .crypto import (
    aes_decrypt,
    aes_encrypt,
    derive_encryption_key,
    generate_api_key,
    generate_password,
    get_password_hash,
    sha256_hex,
    verify_password,
)
from .database import Database

logger = logging.getLogger(__name__)


def _get_client_ip(request: Request) -> str:
    if "x-forwarded-for" in request.headers:
        return request.headers["x-forwarded-for"].split(",")[0].strip()
    if "x-real-ip" in request.headers:
        return request.headers["x-real-ip"].strip()
    return request.client.host if request.client else ""


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
JWT_ALGORITHM = "HS256"


class AdvancedAuthService:
    def __init__(self, db_path: str, jwt_secret_key: str, encryption_key: str):
        self._db = Database(db_path)
        self._jwt_secret_key = jwt_secret_key
        self._encryption_key = derive_encryption_key(encryption_key)
        self._cache = ApiKeyCache(self._db)
        self._init_admin()

        try:
            from .rate_limiter import RateLimiter

            self._rate_limiter = RateLimiter()
        except ImportError:
            self._rate_limiter = None

    @property
    def db(self) -> Database:
        return self._db

    @property
    def cache(self) -> ApiKeyCache:
        return self._cache

    def _init_admin(self):
        if self._db.user_count() == 0:
            password = generate_password()
            password_hash = get_password_hash(password)
            admin_perms = [
                "admin",
                "models:list",
                "models:read",
                "models:write",
                "keys:create",
                "keys:manage",
                "users:manage",
                "cache:list",
                "cache:delete",
                "virtualenv:list",
                "virtualenv:delete",
            ]
            self._db.create_user(
                username="admin",
                password_hash=password_hash,
                source="local",
                enabled=1,
                must_change_password=1,
                permissions=admin_perms,
            )
            logger.warning(
                "\n" + "=" * 60 + "\n"
                "  INITIAL ADMIN CREDENTIALS (shown only once)\n"
                "  Username: admin\n"
                "  Password: %s\n"
                "  Please change the password on first login.\n" + "=" * 60,
                password,
            )

    # --- JWT ---

    def create_access_token(
        self, user_id: int, username: str, scopes: List[str]
    ) -> str:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        payload = {
            "sub": username,
            "user_id": user_id,
            "scopes": scopes,
            "exp": expire,
            "type": "access",
        }
        return jwt.encode(payload, self._jwt_secret_key, algorithm=JWT_ALGORITHM)

    def create_refresh_token(self, user_id: int) -> str:
        token = secrets.token_urlsafe(64)
        token_hash = sha256_hex(token)
        expires_at = (
            datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        ).isoformat()
        self._db.create_refresh_token(user_id, token_hash, expires_at)
        return token

    def verify_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        try:
            payload = jwt.decode(
                token,
                self._jwt_secret_key,
                algorithms=[JWT_ALGORITHM],
                options={"verify_exp": True},
            )
            if payload.get("type") != "access":
                return None
            return payload
        except JWTError:
            return None

    def refresh_access_token(self, refresh_token: str) -> Optional[Dict[str, str]]:
        token_hash = sha256_hex(refresh_token)
        rt = self._db.get_refresh_token(token_hash)
        if not rt:
            return None
        expires_at = datetime.fromisoformat(rt["expires_at"])
        if datetime.utcnow() > expires_at:
            self._db.delete_refresh_token(token_hash)
            return None
        user = self._db.get_user_by_id(rt["user_id"])
        if not user or not user["enabled"]:
            self._db.delete_refresh_token(token_hash)
            return None

        # Token rotation: invalidate old token, issue new one
        self._db.delete_refresh_token(token_hash)
        new_refresh_token = self.create_refresh_token(user["id"])

        access_token = self.create_access_token(
            user["id"], user["username"], user["permissions"]
        )
        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
        }

    def logout(self, refresh_token: str) -> bool:
        token_hash = sha256_hex(refresh_token)
        return self._db.delete_refresh_token(token_hash)

    # --- Login ---

    def authenticate_user(
        self, username: str, password: str
    ) -> Optional[Dict[str, Any]]:
        user = self._db.get_user_by_username(username, "local")
        if not user:
            return None
        if not user["enabled"]:
            return None
        if not verify_password(password, user["password_hash"]):
            return None
        return user

    def login(self, username: str, password: str) -> Dict[str, Any]:
        user = self.authenticate_user(username, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token = self.create_access_token(
            user["id"], user["username"], user["permissions"]
        )
        refresh_token = self.create_refresh_token(user["id"])
        result: Dict[str, Any] = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
        }
        if user["must_change_password"]:
            result["must_change_password"] = True
        return result

    # --- API Key validation ---

    def validate_api_key(self, key: str) -> Optional[ApiKeyCacheEntry]:
        key_hash = sha256_hex(key)
        entry = self._cache.get(key_hash)
        if not entry:
            return None
        if not entry.is_valid():
            return None
        return entry

    # --- API Key CRUD ---

    def create_api_key_for_user(
        self,
        user_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        expires_at: Optional[str] = None,
        model_permissions: Optional[List[Dict[str, Optional[str]]]] = None,
        rate_limit_max_failures: Optional[int] = None,
        rate_limit_window_seconds: Optional[int] = None,
        rate_limit_ban_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        plaintext_key = generate_api_key()
        key_hash = sha256_hex(plaintext_key)
        key_encrypted = base64.b64encode(
            aes_encrypt(plaintext_key, self._encryption_key)
        ).decode("utf-8")
        key_prefix = plaintext_key[:7]

        if not model_permissions:
            model_permissions = [{"permission_type": "all", "permission_value": None}]

        key_id = self._db.create_api_key(
            user_id=user_id,
            key_hash=key_hash,
            key_encrypted=key_encrypted,
            key_prefix=key_prefix,
            name=name,
            description=description,
            expires_at=expires_at,
            model_permissions=model_permissions,
            rate_limit_max_failures=rate_limit_max_failures,
            rate_limit_window_seconds=rate_limit_window_seconds,
            rate_limit_ban_seconds=rate_limit_ban_seconds,
        )

        user = self._db.get_user_by_id(user_id)
        cache_data = self._db.get_api_key_by_id(key_id)
        if cache_data is None:
            raise RuntimeError(f"Failed to retrieve newly created API key: {key_id}")
        cache_data["user_enabled"] = user["enabled"] if user else 0
        cache_data["username"] = user["username"] if user else ""
        self._cache.add(cache_data)

        return {"id": key_id, "key": plaintext_key, "prefix": key_prefix, "name": name}

    def reveal_api_key(self, key_id: int) -> Optional[str]:
        key_data = self._db.get_api_key_by_id(key_id)
        if not key_data:
            return None
        encrypted = base64.b64decode(key_data["key_encrypted"])
        return aes_decrypt(encrypted, self._encryption_key)

    # --- FastAPI dependency (callable) ---

    async def __call__(
        self,
        request: Request,
        security_scopes: SecurityScopes,
        token: Annotated[Optional[str], Depends(oauth2_scheme)] = None,
    ):
        try:
            from .audit import (
                classify_endpoint,
                record_audit_event,
                resolve_model_info,
                should_skip_audit,
            )
        except ImportError:
            classify_endpoint = None  # type: ignore[assignment]
            record_audit_event = None  # type: ignore[assignment]
            resolve_model_info = None  # type: ignore[assignment]
            should_skip_audit = None  # type: ignore[assignment]

        if security_scopes.scopes:
            authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
        else:
            authenticate_value = "Bearer"
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": authenticate_value},
        )

        # Get client IP for rate limiting, respecting reverse proxy headers
        client_ip = _get_client_ip(request)

        endpoint = request.url.path
        _skip_audit = (
            should_skip_audit(endpoint) if should_skip_audit is not None else True
        )

        # Extract model from request body for audit (POST endpoints like /v1/chat/completions)
        _request_model = ""
        if not _skip_audit and request.method == "POST":
            content_type = request.headers.get("content-type", "")
            content_length = request.headers.get("content-length")
            if "application/json" in content_type:
                try:
                    if content_length is None or int(content_length) <= 1024 * 1024:
                        body_bytes = await request.body()
                        import json as _json

                        _request_model = _json.loads(body_bytes).get("model", "")
                except Exception:
                    pass

        # Resolve model_name and model_type from cache
        if resolve_model_info is not None:
            _model_name, _model_type = resolve_model_info(_request_model)
        else:
            _model_name, _model_type = "", ""

        def _audit(
            status_val: str,
            user: str = "",
            key_name: str = "",
            key_prefix: str = "",
            auth_type: str = "",
        ):
            if _skip_audit or record_audit_event is None:
                return
            record_audit_event(
                user=user,
                api_key_name=key_name,
                api_key_prefix=key_prefix,
                model_id=_request_model,
                model_name=_model_name,
                model_type=_model_type,
                endpoint=endpoint,
                status=status_val,
                client_ip=client_ip,
                auth_type=auth_type,
            )

        # Check IP ban
        if (
            client_ip
            and self._rate_limiter
            and self._rate_limiter.is_ip_banned(client_ip)
        ):
            _audit("ip_banned")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed attempts. IP temporarily banned.",
            )

        if not token:
            _audit("no_credentials")
            raise credentials_exception

        # Look up key in cache (including disabled/expired) for proper (IP, Key) ban handling
        _key_hash = sha256_hex(token)
        api_key_entry = self._cache.get(_key_hash)
        if api_key_entry:
            user_obj = self._db.get_user_by_id(api_key_entry.user_id)
            _username = user_obj["username"] if user_obj else ""

            if not api_key_entry.is_valid():
                _status = (
                    "key_expired"
                    if (
                        api_key_entry.expires_at
                        and datetime.utcnow() > api_key_entry.expires_at
                    )
                    else "key_disabled"
                )
                _audit(
                    _status,
                    user=_username,
                    key_name=api_key_entry.name or "",
                    key_prefix=api_key_entry.key_prefix,
                    auth_type="api_key",
                )
                if client_ip and self._rate_limiter:
                    try:
                        from .rate_limiter import RateLimitConfig

                        key_data = self._db.get_api_key_by_id(api_key_entry.key_id)
                        per_key_config = None
                        if key_data and key_data.get("rate_limit_max_failures"):
                            per_key_config = RateLimitConfig(
                                max_failures=key_data["rate_limit_max_failures"],
                                window_seconds=key_data.get("rate_limit_window_seconds")
                                or 300,
                                ban_seconds=key_data.get("rate_limit_ban_seconds")
                                or 300,
                            )
                        self._rate_limiter.record_key_failure(
                            client_ip, api_key_entry.key_id, per_key_config
                        )
                    except ImportError:
                        pass
                raise credentials_exception

            # Check (IP, Key) ban
            if (
                client_ip
                and self._rate_limiter
                and self._rate_limiter.is_key_banned(client_ip, api_key_entry.key_id)
            ):
                _audit(
                    "key_banned",
                    user=_username,
                    key_name=api_key_entry.name or "",
                    key_prefix=api_key_entry.key_prefix,
                    auth_type="api_key",
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many failed attempts for this key.",
                )

            _API_KEY_ALLOWED_SCOPES = {"models:read", "models:list"}
            for scope in security_scopes.scopes:
                if scope not in _API_KEY_ALLOWED_SCOPES:
                    _audit(
                        "insufficient_scope",
                        user=_username,
                        key_name=api_key_entry.name or "",
                        key_prefix=api_key_entry.key_prefix,
                        auth_type="api_key",
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="API keys can only access model query and inference endpoints",
                        headers={"WWW-Authenticate": authenticate_value},
                    )
            if not user_obj:
                _audit(
                    "invalid_key",
                    key_prefix=api_key_entry.key_prefix,
                    auth_type="api_key",
                )
                raise credentials_exception
            # Success — reset counters
            if client_ip and self._rate_limiter:
                self._rate_limiter.reset_key(client_ip, api_key_entry.key_id)
            # Record success for non-inference endpoints (inference success is recorded by audit_middleware)
            _category = classify_endpoint(endpoint) if classify_endpoint else ""
            if _category != "inference":
                _audit(
                    "success",
                    user=_username,
                    key_name=api_key_entry.name or "",
                    key_prefix=api_key_entry.key_prefix,
                    auth_type="api_key",
                )
            return user_obj

        # Token not found in API key cache — check prefix
        if token.startswith(("sk-", "xf-")):
            _audit("invalid_key", key_prefix=token[:7], auth_type="api_key")
            if client_ip and self._rate_limiter:
                self._rate_limiter.record_invalid_key(client_ip)
            raise credentials_exception

        payload = self.verify_access_token(token)
        if not payload:
            _audit("invalid_token", auth_type="jwt")
            if client_ip and self._rate_limiter:
                self._rate_limiter.record_invalid_key(client_ip)
            raise credentials_exception

        username: Optional[str] = payload.get("sub")
        token_scopes = payload.get("scopes", [])
        user_id = payload.get("user_id")

        if user_id:
            user = self._db.get_user_by_id(user_id)
        elif username:
            user = self._db.get_user_by_username(username)
        else:
            _audit("invalid_token", auth_type="jwt")
            raise credentials_exception
        if not user:
            _audit("invalid_token", user=username or "", auth_type="jwt")
            raise credentials_exception
        if not user["enabled"]:
            _audit("user_disabled", user=username or "", auth_type="jwt")
            raise credentials_exception

        if "admin" in token_scopes:
            _category = classify_endpoint(endpoint) if classify_endpoint else ""
            if _category != "inference":
                _audit("success", user=username or "", auth_type="jwt")
            return user

        for scope in security_scopes.scopes:
            if scope not in token_scopes:
                _audit("insufficient_scope", user=username or "", auth_type="jwt")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not enough permissions",
                    headers={"WWW-Authenticate": authenticate_value},
                )
        _category = classify_endpoint(endpoint) if classify_endpoint else ""
        if _category != "inference":
            _audit("success", user=username or "", auth_type="jwt")
        return user

    def validate_model_access(
        self, token: str, model_uid: str, model_type: Optional[str] = None
    ) -> bool:
        api_key_entry = self.validate_api_key(token)
        if api_key_entry:
            return api_key_entry.has_model_access(model_uid, model_type)
        payload = self.verify_access_token(token)
        if payload:
            scopes = payload.get("scopes", [])
            if "admin" in scopes or "models:read" in scopes:
                return True
        return False

    # --- User disable cascade ---

    def disable_user(self, user_id: int):
        self._db.update_user(user_id, enabled=0)
        self._cache.invalidate_user_keys(user_id, enabled=False)
        self._db.delete_user_refresh_tokens(user_id)

    def enable_user(self, user_id: int):
        self._db.update_user(user_id, enabled=1)
        self._cache.invalidate_user_keys(user_id, enabled=True)
