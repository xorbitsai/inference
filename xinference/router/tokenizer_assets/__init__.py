# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Built-in Tokenizer assets shipped with Xinference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

_BUILTIN_ASSET_PACKAGES = {
    "deepseek-v4-flash-0731": "deepseek_v4_flash_0731",
}


class BuiltinTokenizerAssetError(ValueError):
    """Raised when a built-in Tokenizer asset is unavailable or inconsistent."""


def builtin_tokenizer_asset_entries() -> Dict[str, Dict[str, Any]]:
    """Return Registry entries for assets included in this installation."""
    root = Path(__file__).resolve().parent
    return {
        asset_id: {
            "asset_id": asset_id,
            "path": str((root / package_name).resolve()),
            "enabled": True,
            "origin": "builtin",
        }
        for asset_id, package_name in _BUILTIN_ASSET_PACKAGES.items()
    }


def resolve_builtin_tokenizer_asset(asset_id: str) -> Dict[str, str]:
    """Resolve and verify a built-in asset from the local Xinference package."""
    entry = builtin_tokenizer_asset_entries().get(asset_id)
    if entry is None:
        raise KeyError(asset_id)
    path = Path(str(entry["path"]))
    manifest_path = path / "asset.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuiltinTokenizerAssetError(
            f"Cannot read built-in Tokenizer asset manifest: {asset_id}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise BuiltinTokenizerAssetError(
            f"Built-in Tokenizer asset manifest must be an object: {asset_id}"
        )
    if str(manifest.get("asset_id", "")).strip() != asset_id:
        raise BuiltinTokenizerAssetError(
            f"Built-in Tokenizer asset manifest ID mismatch: {asset_id}"
        )
    if str(manifest.get("origin", "")).strip() != "builtin":
        raise BuiltinTokenizerAssetError(
            f"Built-in Tokenizer asset origin mismatch: {asset_id}"
        )

    required_files = manifest.get("required_files")
    checksums = manifest.get("checksums")
    if not isinstance(required_files, list) or not required_files:
        raise BuiltinTokenizerAssetError(
            f"Built-in Tokenizer asset required_files is invalid: {asset_id}"
        )
    if not isinstance(checksums, dict):
        raise BuiltinTokenizerAssetError(
            f"Built-in Tokenizer asset checksums is invalid: {asset_id}"
        )

    aggregate = hashlib.sha256()
    for relative_name in sorted(str(item).strip() for item in required_files):
        relative_path = Path(relative_name)
        if (
            not relative_name
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise BuiltinTokenizerAssetError(
                f"Unsafe built-in Tokenizer asset file path: {relative_name}"
            )
        file_path = (path / relative_path).resolve()
        if file_path != path and not file_path.is_relative_to(path):
            raise BuiltinTokenizerAssetError(
                f"Built-in Tokenizer asset file escapes its directory: {relative_name}"
            )
        if not file_path.is_file():
            raise BuiltinTokenizerAssetError(
                f"Missing built-in Tokenizer asset file: {relative_name}"
            )
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        expected = str(checksums.get(relative_name, "")).removeprefix("sha256:")
        if not expected or digest.lower() != expected.lower():
            raise BuiltinTokenizerAssetError(
                f"SHA-256 mismatch for built-in Tokenizer asset file: {relative_name}"
            )
        aggregate.update(relative_name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\0")

    return {
        "tokenizer_asset_id": asset_id,
        "tokenizer_path": str(path),
        "tokenizer_asset_origin": "builtin",
        "tokenizer_asset_revision": str(manifest.get("revision", "")),
        "tokenizer_asset_fingerprint": f"sha256:{aggregate.hexdigest()}",
    }
