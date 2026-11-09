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

import os
import subprocess
import sys
from contextlib import contextmanager
from typing import Iterator, Optional


def dependency_import_message(
    module_name: str, exc: ImportError, *, install_hint: Optional[str] = None
) -> str:
    """Describe an import failure and suggest installation for a missing module."""
    details = [f"{type(exc).__name__}: {exc}"]
    seen = {id(exc)}
    cause = exc.__cause__
    while cause is not None and id(cause) not in seen:
        details.append(f"{type(cause).__name__}: {cause}")
        seen.add(id(cause))
        cause = cause.__cause__
    message = f"Failed to import module '{module_name}': " + "\nCaused by: ".join(
        details
    )
    if (
        isinstance(exc, ModuleNotFoundError)
        and exc.name == module_name
        and exc.__cause__ is None
    ):
        hint = install_hint or (
            f"Install '{module_name}' in the active environment "
            f"with `pip install {module_name}`."
        )
        message = f"{message}\n{hint}"
    return message


@contextmanager
def dependency_import_guard(
    module_name: str, *, install_hint: Optional[str] = None
) -> Iterator[None]:
    """Report failures from lazy imports while retaining their original cause."""
    try:
        yield
    except ImportError as exc:
        raise ImportError(
            dependency_import_message(module_name, exc, install_hint=install_hint)
        ) from exc


def cuda_count():
    import torch

    # even if install torch cpu, this interface would return 0.
    return torch.cuda.device_count()


def get_real_path(path: str) -> Optional[str]:
    # parsing soft links
    if os.path.isdir(path):
        files = os.listdir(path)
        # dir has files
        if files:
            resolved_file = os.path.realpath(os.path.join(path, files[0]))
            if resolved_file:
                return os.path.dirname(resolved_file)
        return None
    else:
        return os.path.realpath(path)


def get_pip_config_args() -> dict[str, str | list[str]]:
    """
    Parse pip config and return a dict with keys matching install_packages kwargs:
    index_url, extra_index_url, find_links, trusted_host.
    """
    key_map = {
        "global.index-url": "index_url",
        "global.extra-index-url": "extra_index_url",
        "global.trusted-host": "trusted_host",
        "global.find-links": "find_links",
    }

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "config", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        args: dict[str, str | list[str]] = {}
        for line in result.stdout.splitlines():
            if "=" not in line:
                continue
            raw_key, raw_value = line.split("=", 1)
            key = raw_key.strip()
            value = raw_value.strip().strip("'\"")
            mapped_key = key_map.get(key)
            if mapped_key and value:
                if mapped_key in ("extra_index_url", "find_links", "trusted_host"):
                    existing = args.get(mapped_key)
                    if existing is None:
                        args[mapped_key] = value
                    elif isinstance(existing, list):
                        existing.append(value)
                    else:
                        args[mapped_key] = [existing, value]
                else:
                    args[mapped_key] = value

        return args
    except subprocess.CalledProcessError:
        return {}


def make_hashable(obj):
    """
    Recursively convert an object into a hashable form.

    This function is useful for creating deterministic cache keys or
    comparing complex structures (dicts, lists, sets, etc.) that are
    otherwise unhashable. It ensures consistent ordering for dictionaries
    and sets, so that equivalent objects produce identical hashable results.

    Examples:
        make_hashable({"a": [1, 2], "b": {"x": True}})
        -> (("a", (1, 2)), ("b", (("x", True),)))

    Args:
        obj: Any Python object.

    Returns:
        A hashable version of `obj`, typically composed of tuples.
    """
    if isinstance(obj, (tuple, list)):
        return tuple(make_hashable(o) for o in obj)
    elif isinstance(obj, dict):
        # Sort by key to ensure consistent ordering
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, set):
        # Sort elements for deterministic output
        return tuple(sorted(make_hashable(o) for o in obj))
    else:
        # Assume it's already hashable (int, str, float, bool, None, etc.)
        return obj
