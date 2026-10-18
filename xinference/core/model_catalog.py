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

"""Coordinated publication of full and single-model downloaded catalogs."""

import json
import os
import stat
import sys
import time
import uuid
from importlib import import_module
from pathlib import Path
from typing import Any, Optional

from filelock import FileLock


def validate_catalog_model_name(value: Any) -> Optional[str]:
    if value is not None and (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or value != value.strip()
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError("Invalid catalog model name")
    return value


def store_model_catalog(
    path: Path,
    data: Any,
    *,
    model_name: Optional[str] = None,
    model_type: Optional[str] = None,
) -> None:
    """Replace a category or merge one named model without losing other updates.

    Single-model responses may contain one record or a list of engine/hub
    variants sharing the requested name. A malformed existing file is never
    silently replaced by a partial catalog. Runtime loaders merge the resulting
    downloaded catalog with packaged built-ins using their normal rules.
    """
    validate_catalog_model_name(model_name)
    if model_name is not None:
        records = [data] if isinstance(data, dict) else data
        if (
            not isinstance(records, list)
            or not records
            or any(
                not isinstance(row, dict) or _record_name(row) != model_name
                for row in records
            )
        ):
            raise ValueError("Invalid single-model catalog response")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock inode stable across processes and platforms. Full refreshes
    # use the same lock, while readers see only complete JSON via os.replace.
    with FileLock(str(path) + ".lock", timeout=30, preserve_lock_file=True):
        if model_name is not None:
            existing = (
                json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            )
            if not isinstance(existing, list) or any(
                not isinstance(row, dict) or not isinstance(_record_name(row), str)
                for row in existing
            ):
                raise ValueError("Invalid stored model catalog")
            data = []
            replaced = False
            for row in existing:
                if _record_name(row) == model_name:
                    if not replaced:
                        data.extend(records)
                        replaced = True
                else:
                    data.append(row)
            if not replaced:
                data.extend(records)
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
        temporary = path.parent / f".catalog-{uuid.uuid4().hex}.json"
        # Unlike mkstemp's fixed 0600, exclusive creation applies the process
        # umask to 0666 just like the previous open(path, "w") implementation.
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            if model_name is not None and model_type is not None:
                _validate_runtime_catalog(temporary, model_type)
            if mode is not None:
                os.chmod(temporary, mode)
            _replace_catalog(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _record_name(row):
    return row.get("model_name") or row.get("model_family")


def _validate_runtime_catalog(path: Path, model_type: str) -> None:
    """Use the real loaders and family schemas without publishing registries."""
    module = import_module(f"xinference.model.{model_type.lower()}")
    if model_type.lower() == "llm":
        module.load_model_family_from_json(str(path), [], register_metadata=False)
    else:
        module.load_model_family_from_json(str(path), {})


def _replace_catalog(source: Path, target: Path) -> None:
    # Windows readers may briefly deny replacement while their handle is open.
    for attempt in range(5):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if sys.platform != "win32" or attempt == 4:
                raise
            time.sleep(0.05)
