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

"""Explicit, process-local extension points for distribution wrappers.

XINFERENCE_EXTENSIONS is a comma-separated list of ``module:factory`` entries.
Factories are loaded in every API/actor process at startup, including spawn and
recovery. A configured extension that cannot load is a startup error. Unset
configuration leaves upstream behavior unchanged and requires no extra package.
An optional auth-store hook selects storage before authentication starts and
may perform database initialization during startup.

Extensions may compose the API, contribute worker identity metadata, restrict
worker resources, and expose named control-plane operations. Resource filters
may only narrow visible devices. They run at allocation time, including CPU
launches, and may reject an operation by raising an exception.
"""

from __future__ import annotations

import importlib
import inspect
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Tuple

EXTENSION_API_VERSION = 1


@dataclass(frozen=True)
class APIExtensionContext:
    app: Any
    router: Any
    auth_service: Any
    auth_enabled: bool
    get_supervisor_ref: Callable[[], Awaitable[Any]]


@dataclass(frozen=True)
class AuthStoreContext:
    """Requested auth store and caller, available before API composition."""

    db_path: str
    purpose: Literal["service", "reset_password", "migrate"]


@dataclass(frozen=True)
class ModelFavoritesStoreContext:
    """Configured auth database path and selected store (None without authentication).

    Available during API initialization, before route composition. A provider
    must reuse the selected database and its transaction/backend domain.
    """

    db_path: str
    auth_store: Any


@dataclass(frozen=True)
class LaunchHistoryStoreContext:
    """Startup caller, default history path and selected API auth store.

    The supervisor has no auth service, so its auth_store is None. Providers
    must select the same persistent history backend in both processes.
    """

    db_path: str
    service: Literal["api", "supervisor"]
    auth_store: Any = None


@dataclass(frozen=True)
class WorkerControlAuthorityContext:
    """Worker identity only; an authority must not initialize an auth store.

    The optional provider returns an object with validate(session, phases=...).
    Validation may be async; synchronous validation runs in the Worker's thread
    pool. Every call must check current authority and raise on rejection or
    unavailability. Returning None from the factory preserves legacy admission.
    This hook does not elect or automatically install a controller session.
    """

    address: str
    incarnation: str


@dataclass(frozen=True)
class SupervisorControlAuthorityContext:
    """Process-local runtime capabilities for an optional control extension.

    Providers implement an immutable ControllerSession, validate and close.
    Optional async execute(operation, payload) and worker_registered(ref) hooks
    own extension behavior. runtime exposes local pause/restore/resume primitives;
    it is not serializable. Async callbacks run on the actor loop and may
    overlap; providers synchronize their own services. worker_registered receives
    one reference, not a complete membership set. Synchronous hooks use a thread
    pool. An unconfigured hook preserves ordinary behavior.
    """

    address: str
    runtime: Any = None


@dataclass(frozen=True)
class ModelCatalogCredentials:
    """An ephemeral catalog token whose repr stays safe in actor call logs."""

    token: str = field(repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.token, str)
            or not self.token
            or self.token != self.token.strip()
            or any(ord(c) < 32 or ord(c) == 127 for c in self.token)
        ):
            raise ValueError("Invalid model catalog credential")
        try:
            self.token.encode("latin-1")
        except UnicodeError:
            raise ValueError("Invalid model catalog credential") from None


@dataclass(frozen=True)
class CacheSalt:
    """An authoritative cache scope supplied by an authenticated API extension."""

    value: str = field(repr=False)

    def __post_init__(self) -> None:
        from .cache_isolation import validate_cache_salt

        if validate_cache_salt(self.value) is None:
            raise ValueError("Cache salt provider must supply a cache_salt")


_loaded: Optional[Tuple[str, Tuple[Any, ...]]] = None


def get_extensions() -> Tuple[Any, ...]:
    global _loaded
    configured = os.environ.get("XINFERENCE_EXTENSIONS", "").strip()
    if _loaded is not None and _loaded[0] == configured:
        return _loaded[1]
    instances: List[Any] = []
    names = set()
    for entry in configured.split(","):
        if not entry.strip():
            continue
        entry = entry.strip()
        parts = [part.strip() for part in entry.split(":")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"Extension entry {entry!r} must have the form module:factory"
            )
        module, factory = parts
        try:
            extension = getattr(importlib.import_module(module), factory)()
            _require_sync_result(extension, f"factory {entry!r}")
        except Exception as error:
            raise RuntimeError(
                f"Failed to load extension {entry!r}: {error}"
            ) from error
        name = getattr(extension, "name", None)
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Extension names must be nonempty and unique")
        if getattr(extension, "api_version", None) != EXTENSION_API_VERSION:
            raise ValueError(f"Incompatible extension API version for {name}")
        names.add(name)
        instances.append(extension)
    result = tuple(instances)
    _loaded = (configured, result)
    return result


def _require_sync_result(result: Any, hook: str) -> Any:
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise TypeError(f"Extension {hook} must be synchronous and non-blocking")
    return result


def _call_hook(extension: Any, method: str, *args: Any) -> Any:
    return _require_sync_result(
        getattr(extension, method)(*args), f"{extension.name}.{method}"
    )


def configure_api(context: APIExtensionContext) -> None:
    for extension in get_extensions():
        _call_hook(extension, "configure_api", context)


def resolve_auth_store(context: AuthStoreContext) -> Any:
    """Allow one distribution to supply auth storage during initialization.

    This synchronous startup hook may initialize a database. Returning None
    selects the default SQLite store; provider errors never select a fallback.
    Reject multiple providers before invoking any of them.
    """
    providers = [e for e in get_extensions() if hasattr(e, "create_auth_store")]
    if len(providers) > 1:
        raise ValueError("Multiple auth store providers")
    if not providers:
        return None
    return _call_hook(providers[0], "create_auth_store", context)


def resolve_model_favorites_store(context: ModelFavoritesStoreContext) -> Any:
    """Select at most one optional startup provider; errors never fall back."""
    providers = [
        e for e in get_extensions() if hasattr(e, "create_model_favorites_store")
    ]
    if len(providers) > 1:
        raise ValueError("Multiple model favorites store providers")
    if not providers:
        return None
    return _call_hook(providers[0], "create_model_favorites_store", context)


def resolve_launch_history_store(context: LaunchHistoryStoreContext) -> Any:
    """Select one optional startup provider; errors never select a fallback."""
    providers = [
        e for e in get_extensions() if hasattr(e, "create_launch_history_store")
    ]
    if len(providers) > 1:
        raise ValueError("Multiple launch history store providers")
    if not providers:
        return None
    return _call_hook(providers[0], "create_launch_history_store", context)


def resolve_worker_control_authority(context: WorkerControlAuthorityContext) -> Any:
    providers = [
        e for e in get_extensions() if hasattr(e, "create_worker_control_authority")
    ]
    if len(providers) > 1:
        raise ValueError("Multiple worker control authority providers")
    if not providers:
        return None
    authority = _call_hook(providers[0], "create_worker_control_authority", context)
    if authority is not None and not callable(getattr(authority, "validate", None)):
        raise TypeError("Worker control authority must implement validate")
    return authority


def resolve_supervisor_control_authority(
    context: SupervisorControlAuthorityContext,
) -> Any:
    providers = [
        e for e in get_extensions() if hasattr(e, "create_supervisor_control_authority")
    ]
    if len(providers) > 1:
        raise ValueError("Multiple Supervisor control authority providers")
    if not providers:
        return None
    authority = _call_hook(providers[0], "create_supervisor_control_authority", context)
    if authority is not None:
        from .core.control_session import ControllerSession

        if not isinstance(
            getattr(authority, "session", None), ControllerSession
        ) or not all(
            callable(getattr(authority, name, None)) for name in ("validate", "close")
        ):
            raise TypeError("Invalid Supervisor control authority")
    return authority


def resolve_model_catalog_credentials(
    request: Any, model_type: str
) -> Optional[ModelCatalogCredentials]:
    """Resolve optional server-owned credentials after route authentication.

    Returning None preserves anonymous catalog refresh. Providers may reject a
    request; their failures never cause an anonymous retry. Existing extensions
    do not need to implement this optional hook.
    """
    credentials = None
    for extension in get_extensions():
        if not hasattr(extension, "model_catalog_credentials"):
            continue
        candidate = _call_hook(
            extension, "model_catalog_credentials", request, model_type
        )
        if candidate is None:
            continue
        if not isinstance(candidate, ModelCatalogCredentials):
            raise TypeError(
                "Catalog credential hooks must return ModelCatalogCredentials or None"
            )
        if credentials is not None:
            raise ValueError("Multiple model catalog credential providers")
        credentials = candidate
    return credentials


def worker_extension_metadata() -> Dict[str, Any]:
    return {e.name: _call_hook(e, "worker_metadata") for e in get_extensions()}


def resolve_cache_salt(request: Any) -> Optional[CacheSalt]:
    """Resolve after authentication; a configured provider must fail closed.

    The optional synchronous ``cache_salt(request)`` hook returns CacheSalt.
    None means there is no provider, never a fallback after a provider failure.
    """
    providers = [e for e in get_extensions() if hasattr(e, "cache_salt")]
    if len(providers) > 1:
        raise ValueError("Multiple cache salt providers")
    if not providers:
        return None
    result = _call_hook(providers[0], "cache_salt", request)
    if not isinstance(result, CacheSalt):
        raise TypeError("Cache salt provider must return CacheSalt")
    return result


def filter_worker_resources(
    worker: Dict[str, Any], workers: List[Dict[str, Any]]
) -> List[int]:
    original = list(worker["gpu_devices"])
    allowed = set(original)
    for extension in get_extensions():
        if extension.name not in worker["extensions"]:
            raise RuntimeError(f"Worker is missing required extension {extension.name}")
        filtered = set(
            _call_hook(extension, "filter_worker_resources", worker, workers)
        )
        if not filtered <= set(original):
            raise ValueError("Extensions must not expand worker GPU visibility")
        allowed.intersection_update(filtered)
    return [device for device in original if device in allowed]


def call_extension(name: str, operation: str, payload: Dict[str, Any]) -> Any:
    for extension in get_extensions():
        if extension.name == name:
            return _call_hook(extension, "control_operation", operation, payload)
    raise ValueError(f"Extension {name} is not configured")
