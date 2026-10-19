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
"""Storage contract and selection for advanced authentication.

Providers keep the public Database method semantics and row dictionaries. They
must serialize first-user setup, refresh rotation and password/token revocation
as database transactions, including across API processes. API key validation
and name conflicts use the shared helpers/exceptions in database.py.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, cast


class AuthStore(Protocol):
    """Synchronous authentication operations consumed by API and recovery tools."""

    permission_resolver: Optional[Callable[[int, List[str]], List[str]]]

    @property
    def path(self) -> Optional[Path]:
        """Local SQLite path, or None when storage is not a SQLite file."""
        ...

    def create_user(
        self,
        username: str,
        password_hash: Optional[str],
        source: str = "local",
        oidc_sub: Optional[str] = None,
        enabled: int = 1,
        must_change_password: int = 0,
        permissions: Optional[List[str]] = None,
    ) -> int: ...

    def get_or_create_setup_token(self) -> Optional[str]: ...

    def create_first_user(
        self,
        username: str,
        password_hash: Optional[str],
        permissions: Optional[List[str]] = None,
        setup_token: Optional[str] = None,
    ) -> Optional[int]: ...

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]: ...

    def get_user_by_username(
        self, username: str, source: str = "local"
    ) -> Optional[Dict[str, Any]]: ...

    def get_user_by_oidc_sub(self, oidc_sub: str) -> Optional[Dict[str, Any]]: ...

    def list_users(self, source: Optional[str] = None) -> List[Dict[str, Any]]: ...

    def update_user(self, user_id: int, **kwargs) -> bool: ...

    def delete_user(self, user_id: int) -> bool: ...

    def set_user_permissions(self, user_id: int, permissions: List[str]) -> None: ...

    def user_count(self) -> int: ...

    def create_api_key(
        self,
        user_id: int,
        key_hash: str,
        key_encrypted: str,
        key_prefix: str,
        name: str,
        description: Optional[str] = None,
        expires_at: Optional[str] = None,
        rate_limit_max_failures: Optional[int] = None,
        rate_limit_window_seconds: Optional[int] = None,
        rate_limit_ban_seconds: Optional[int] = None,
        model_permissions: Optional[List[Dict[str, Optional[str]]]] = None,
    ) -> int: ...

    def get_api_key_by_id(self, key_id: int) -> Optional[Dict[str, Any]]: ...

    def get_api_key_by_hash(self, key_hash: str) -> Optional[Dict[str, Any]]: ...

    def list_api_keys(self, user_id: Optional[int] = None) -> List[Dict[str, Any]]: ...

    def update_api_key(self, key_id: int, **kwargs) -> bool: ...

    def delete_api_key(self, key_id: int) -> bool: ...

    def set_api_key_model_permissions(
        self, key_id: int, permissions: List[Dict[str, Optional[str]]]
    ) -> None: ...

    def get_all_api_keys_with_users(self) -> List[Dict[str, Any]]: ...

    def create_refresh_token(
        self, user_id: int, token_hash: str, expires_at: str
    ) -> int: ...

    def get_refresh_token(self, token_hash: str) -> Optional[Dict[str, Any]]: ...

    def get_user_refresh_tokens(self, user_id: int) -> List[Dict[str, Any]]: ...

    def delete_refresh_token(self, token_hash: str) -> bool: ...

    def delete_user_refresh_tokens(self, user_id: int) -> None: ...

    def cleanup_expired_refresh_tokens(self) -> None: ...

    def rotate_refresh_token(
        self, old_token_hash: str, new_token_hash: str, expires_at: str
    ) -> Optional[Dict[str, Any]]: ...

    def update_password_and_revoke_tokens(
        self, user_id: int, password_hash: str
    ) -> None: ...

    def get_user_token_version(self, user_id: int) -> Optional[int]: ...

    def get_config(self, key: str) -> Optional[str]: ...

    def set_config(self, key: str, value: str) -> None: ...


def _is_sync_callable(value: Any) -> bool:
    return callable(value) and not (
        inspect.iscoroutinefunction(value)
        or inspect.iscoroutinefunction(getattr(value, "__call__", None))
    )


def validate_auth_store(store: Any) -> AuthStore:
    """Reject incomplete or asynchronous providers before serving requests."""
    for name, member in AuthStore.__dict__.items():
        if name.startswith("_") or not inspect.isfunction(member):
            continue
        method = getattr(store, name, None)
        is_stub = getattr(method, "__func__", method) is member
        if is_stub or not _is_sync_callable(method):
            raise TypeError(f"Auth store requires synchronous {name}()")
    if not hasattr(store, "permission_resolver"):
        raise TypeError("Auth store requires permission_resolver")
    resolver = store.permission_resolver
    if resolver is not None and not _is_sync_callable(resolver):
        raise TypeError("Auth store permission_resolver must be synchronous")
    path = getattr(store, "path", False)
    if path is not None and not isinstance(path, Path):
        raise TypeError("Auth store path must be a pathlib.Path or None")
    return cast(AuthStore, store)


def create_auth_store(
    db_path: str,
    purpose: Literal["service", "reset_password", "migrate"] = "service",
) -> AuthStore:
    """Select one configured provider, retaining SQLite as the default."""
    from ....extensions import AuthStoreContext, resolve_auth_store
    from .database import Database

    store = resolve_auth_store(AuthStoreContext(db_path=db_path, purpose=purpose))
    if store is None:
        return Database(db_path)
    return validate_auth_store(store)
