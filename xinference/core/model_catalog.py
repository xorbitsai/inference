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
import re
import stat
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from filelock import FileLock

from .._model_catalog import MODEL_KINDS, model_filename, validate_model_file

_GITHUB_API = "https://api.github.com/repos/xorbitsai/inference"
_GITHUB_RAW = "https://raw.githubusercontent.com/xorbitsai/inference"
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


def github_catalog_path(model_type: str) -> Path:
    from ..constants import XINFERENCE_MODEL_DIR

    return (
        Path(XINFERENCE_MODEL_DIR)
        / "v2"
        / "github"
        / model_type
        / f"{model_type}_models.json"
    )


def read_github_catalog(path: Path) -> Tuple[bool, List[Dict[str, Any]]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, []
    if not isinstance(data, dict) or not isinstance(data.get("complete"), bool):
        raise ValueError("Invalid stored GitHub catalog")
    records = data.get("models")
    if not isinstance(records, list) or not _valid_catalog(records):
        raise ValueError("Invalid stored GitHub model records")
    return data["complete"], records


def download_github_catalog(
    model_type: str, model_name: Optional[str] = None
) -> Tuple[str, Optional[str], List[Dict[str, Any]]]:
    """Fetch one snapshot from the public repository and return its commit."""
    if model_type not in MODEL_KINDS:
        raise ValueError(f"Unsupported catalog type: {model_type}")
    requested_file = model_filename(model_name) if model_name is not None else None

    def read_json(url: str, **kwargs: Any) -> Any:
        response = requests.get(url, timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()

    commit = read_json(f"{_GITHUB_API}/commits/main")["sha"]
    if not isinstance(commit, str) or not _COMMIT_SHA.fullmatch(commit):
        raise ValueError("Invalid GitHub catalog commit")
    directory = f"xinference/model/{model_type}/models"

    def read_model(filename: str) -> List[Dict[str, Any]]:
        data = read_json(f"{_GITHUB_RAW}/{commit}/{directory}/{filename}")
        records = validate_model_file(filename, data)
        if not records:
            raise ValueError(f"Empty GitHub catalog file: {filename}")
        return records

    if requested_file is not None and model_type != "world":
        return commit, model_name, read_model(requested_file)

    entries = read_json(f"{_GITHUB_API}/contents/{directory}", params={"ref": commit})
    if not isinstance(entries, list) or not 0 < len(entries) < 1000:
        raise ValueError("Invalid GitHub catalog directory")
    filenames = []
    for entry in entries:
        name = entry["name"]
        if not name.endswith(".json"):
            continue
        if entry["type"] != "file" or model_filename(name[:-5]) != name:
            raise ValueError("Invalid GitHub catalog file")
        filenames.append(name)
    if not filenames or len(filenames) != len({name.casefold() for name in filenames}):
        raise ValueError("Invalid GitHub catalog filenames")
    if requested_file is not None and requested_file in filenames:
        return commit, model_name, read_model(requested_file)

    with ThreadPoolExecutor(max_workers=8) as pool:
        catalogs = list(pool.map(read_model, sorted(filenames, key=str.casefold)))
    if model_name is None:
        return commit, None, [row for catalog in catalogs for row in catalog]

    # World model cards can name a variant stored in a model family file.
    for catalog in catalogs:
        for row in catalog:
            if any(spec.get("model_name") == model_name for spec in row["model_specs"]):
                return commit, _record_name(row), catalog
    raise ValueError("Model not found in the GitHub catalog")


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
    reload_catalog: Optional[Callable[[], None]] = None,
    github_commit: Optional[str] = None,
) -> None:
    """Replace a category or merge one named model without losing other updates.

    Single-model responses may contain one record or a list of engine/hub
    variants sharing the requested name. A malformed existing file is never
    silently replaced by a partial catalog. Runtime loaders merge the resulting
    downloaded catalog with packaged built-ins using their normal rules.
    """
    validate_catalog_model_name(model_name)
    github = github_commit is not None
    if github_commit is not None and not _COMMIT_SHA.fullmatch(github_commit):
        raise ValueError("Invalid GitHub catalog commit")
    if model_name is None and not _valid_catalog(data):
        raise ValueError("Invalid catalog response")
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
        complete = model_name is None
        if model_name is not None:
            if github:
                complete, existing = read_github_catalog(path)
            else:
                existing = (
                    json.loads(path.read_text(encoding="utf-8"))
                    if path.exists()
                    else []
                )
            if not _valid_catalog(existing):
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
        backup = temporary.with_suffix(".bak")
        # Unlike mkstemp's fixed 0600, exclusive creation applies the process
        # umask to 0666 just like the previous open(path, "w") implementation.
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                # A single-model refresh can leave other entries from past
                # fetches. Only a full refresh represents one category commit.
                payload = (
                    {
                        "last_update_commit": github_commit,
                        "complete": complete,
                        "models": data,
                    }
                    if github
                    else data
                )
                json.dump(payload, stream, indent=2, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            if model_type is not None:
                _validate_runtime_catalog(data if github else temporary, model_type)
            if mode is not None:
                os.chmod(temporary, mode)
            if reload_catalog is not None and mode is not None:
                os.link(path, backup)
            _replace_catalog(temporary, path)
            if reload_catalog is not None:
                try:
                    reload_catalog()
                except Exception:
                    if backup.exists():
                        _replace_catalog(backup, path)
                    else:
                        path.unlink()
                    reload_catalog()
                    raise
        finally:
            temporary.unlink(missing_ok=True)
            backup.unlink(missing_ok=True)


def _record_name(row):
    return row.get("model_name") or row.get("model_family")


def _valid_catalog(data: Any) -> bool:
    return isinstance(data, list) and all(
        isinstance(row, dict) and isinstance(_record_name(row), str) for row in data
    )


def _validate_runtime_catalog(source: Any, model_type: str) -> None:
    """Use the real loaders and family schemas without publishing registries."""
    module = import_module(f"xinference.model.{model_type.lower()}")
    if isinstance(source, Path):
        source = str(source)
    else:
        # Family loaders normalize records in place.
        from copy import deepcopy

        source = deepcopy(source)
    if model_type.lower() == "llm":
        module.load_model_family_from_json(source, [], register_metadata=False)
    else:
        module.load_model_family_from_json(source, {})


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
