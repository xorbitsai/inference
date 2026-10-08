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

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

DEFAULT_TOKENIZER_ASSET_FILES = (
    "tokenizer.json",
    "encoding/encoding_dsv4.py",
)


def aggregate_tokenizer_asset_fingerprint(file_digests: Mapping[str, str]) -> str:
    """Build the fingerprint used by both the registry and Router workers."""
    aggregate = hashlib.sha256()
    for relative_name in sorted(file_digests):
        aggregate.update(relative_name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(file_digests[relative_name].encode("ascii"))
        aggregate.update(b"\0")
    return aggregate.hexdigest()


def compute_tokenizer_asset_fingerprint(
    path: Path, required_files: Iterable[str] = DEFAULT_TOKENIZER_ASSET_FILES
) -> str:
    """Hash the files used by a Router tokenizer worker."""
    path = path.expanduser().resolve()
    file_digests: dict[str, str] = {}
    for relative_name in sorted(set(required_files)):
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe required file path: {relative_name}")
        file_path = (path / relative_path).resolve()
        if file_path != path and not file_path.is_relative_to(path):
            raise ValueError(f"Required file escapes asset directory: {relative_name}")
        if not file_path.is_file():
            raise ValueError(f"Missing required Tokenizer file: {relative_name}")
        digest = hashlib.sha256()
        with file_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        file_digests[relative_name] = digest.hexdigest()
    return f"sha256:{aggregate_tokenizer_asset_fingerprint(file_digests)}"


def read_tokenizer_asset_revision(path: Path) -> str:
    """Read the revision declared by the manifest at the loaded asset path.

    The manifest is parsed statically (JSON only, no code execution), so this
    is safe to run inside the Supervisor. Returns an empty string when the
    path is not a registered asset (no ``asset.json``) or the manifest cannot
    be parsed.
    """
    path = path.expanduser().resolve()
    manifest_path = path / "asset.json"
    if not manifest_path.is_file():
        return ""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(manifest, dict):
        return ""
    revision = manifest.get("revision")
    return str(revision).strip() if isinstance(revision, str) else ""
