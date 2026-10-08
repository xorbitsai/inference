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

Extensions may compose the API, contribute worker identity metadata, restrict
worker resources, and expose named control-plane operations. Resource filters
may only narrow visible devices. They run at allocation time, including CPU
launches, and may reject an operation by raising an exception.
"""

from __future__ import annotations

import importlib
import inspect
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

EXTENSION_API_VERSION = 1


@dataclass(frozen=True)
class APIExtensionContext:
    app: Any
    router: Any
    auth_service: Any
    auth_enabled: bool
    get_supervisor_ref: Callable[[], Awaitable[Any]]


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


def worker_extension_metadata() -> Dict[str, Any]:
    return {e.name: _call_hook(e, "worker_metadata") for e in get_extensions()}


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
