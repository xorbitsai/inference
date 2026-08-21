# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Registered Tokenizer assets for Token-aware Routers."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # type: ignore[import-untyped]

from ..constants import XINFERENCE_TOKENIZER_ASSET_CONFIG
from ..router.tokenizer_asset import (
    DEFAULT_TOKENIZER_ASSET_FILES,
    aggregate_tokenizer_asset_fingerprint,
)

logger = logging.getLogger(__name__)

_DEFAULT_REQUIRED_FILES = DEFAULT_TOKENIZER_ASSET_FILES
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_TOKENIZER_BYTES = 4 * 1024 * 1024 * 1024
_MAX_ENCODING_BYTES = 16 * 1024 * 1024


class TokenizerAssetError(ValueError):
    """Raised when a registered Tokenizer asset cannot be used safely."""


class TokenizerAssetRegistry:
    """Load and validate the Supervisor-side Tokenizer asset whitelist."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        configured_path = config_path
        if configured_path is None:
            configured_path = XINFERENCE_TOKENIZER_ASSET_CONFIG
        self._config_path = (
            Path(configured_path).expanduser() if configured_path else None
        )
        self._asset_roots: List[Path] = []
        self._entries: Dict[str, Dict[str, Any]] = {}
        # Preserve the pre-registry API when no production registry is configured.
        self._allow_custom_path = True
        self._config_error = ""
        self._last_validated_at: Dict[str, str] = {}
        self.reload()

    @property
    def allow_custom_path(self) -> bool:
        return self._allow_custom_path

    @property
    def config_error(self) -> str:
        return self._config_error

    def reload(self) -> None:
        self._asset_roots = []
        self._entries = {}
        self._allow_custom_path = True
        self._config_error = ""
        if self._config_path is None:
            return
        if not self._config_path.is_file():
            self._allow_custom_path = False
            self._config_error = "Tokenizer asset config does not exist"
            logger.error("%s: %s", self._config_error, self._config_path)
            return

        try:
            raw = yaml.safe_load(self._config_path.read_text(encoding="utf-8")) or {}
            if not isinstance(raw, dict):
                raise TokenizerAssetError("Tokenizer asset config must be an object")
            if int(raw.get("schema_version", 1)) != 1:
                raise TokenizerAssetError("Unsupported Tokenizer asset schema_version")

            roots = raw.get("asset_roots", [])
            if not isinstance(roots, list):
                raise TokenizerAssetError("asset_roots must be a list")
            self._asset_roots = [
                Path(str(root)).expanduser().resolve() for root in roots
            ]
            self._allow_custom_path = bool(raw.get("allow_custom_path", False))

            assets = raw.get("assets", [])
            if not isinstance(assets, list):
                raise TokenizerAssetError("assets must be a list")
            for entry in assets:
                if not isinstance(entry, dict):
                    raise TokenizerAssetError("Each Tokenizer asset must be an object")
                asset_id = str(entry.get("asset_id", "")).strip()
                if not asset_id:
                    raise TokenizerAssetError("Tokenizer asset_id is required")
                if asset_id in self._entries:
                    raise TokenizerAssetError(
                        f"Duplicate Tokenizer asset_id: {asset_id}"
                    )
                self._entries[asset_id] = dict(entry)
        except Exception as exc:
            self._config_error = str(exc)
            self._asset_roots = []
            self._entries = {}
            self._allow_custom_path = False
            logger.error(
                "Failed to load Tokenizer asset config %s: %s",
                self._config_path,
                exc,
            )

    def _entry_path(self, entry: Dict[str, Any]) -> Path:
        raw_path = str(entry.get("path", "")).strip()
        if not raw_path:
            raise TokenizerAssetError("Tokenizer asset path is required")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            if not self._asset_roots:
                raise TokenizerAssetError(
                    "Relative Tokenizer asset path requires asset_roots"
                )
            path = self._asset_roots[0] / path
        return path.resolve()

    def _ensure_allowed_root(self, path: Path) -> None:
        if not self._asset_roots:
            return
        if not any(
            path == root or path.is_relative_to(root) for root in self._asset_roots
        ):
            raise TokenizerAssetError(
                "Tokenizer asset path escapes configured asset_roots"
            )

    @staticmethod
    def _read_manifest(path: Path) -> Dict[str, Any]:
        manifest_path = path / "asset.json"
        if not manifest_path.is_file():
            raise TokenizerAssetError("Missing Tokenizer asset manifest")
        resolved_manifest_path = manifest_path.resolve()
        if (
            resolved_manifest_path != path
            and not resolved_manifest_path.is_relative_to(path)
        ):
            raise TokenizerAssetError(
                "Tokenizer asset manifest escapes asset directory"
            )
        if resolved_manifest_path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise TokenizerAssetError("Tokenizer asset manifest is too large")
        try:
            manifest = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise TokenizerAssetError(
                f"Cannot read Tokenizer asset manifest: {exc.strerror or exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise TokenizerAssetError(
                f"Invalid Tokenizer asset manifest JSON: {exc.msg}"
            ) from exc
        if not isinstance(manifest, dict):
            raise TokenizerAssetError("Tokenizer asset manifest must be an object")
        if int(manifest.get("schema_version", 1)) != 1:
            raise TokenizerAssetError("Unsupported Tokenizer asset manifest version")
        return manifest

    @staticmethod
    def _required_files(manifest: Optional[Dict[str, Any]]) -> List[str]:
        if manifest is None:
            return list(_DEFAULT_REQUIRED_FILES)
        required = manifest.get("required_files", list(_DEFAULT_REQUIRED_FILES))
        if not isinstance(required, list) or not required:
            raise TokenizerAssetError("required_files must be a non-empty list")
        result = [str(item).strip() for item in required]
        if any(not item for item in result):
            raise TokenizerAssetError("required_files cannot contain an empty path")
        for required_file in _DEFAULT_REQUIRED_FILES:
            if required_file not in result:
                result.append(required_file)
        return result

    @staticmethod
    def _validate_manifest_metadata(manifest: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        for key in ("model_family", "model_name", "revision", "encoding_type"):
            if not isinstance(manifest.get(key), str) or not manifest[key].strip():
                errors.append(f"Tokenizer asset manifest {key} is required")

        compatible_models = manifest.get("compatible_models")
        if (
            not isinstance(compatible_models, list)
            or not compatible_models
            or any(
                not isinstance(item, str) or not item.strip()
                for item in compatible_models
            )
        ):
            errors.append(
                "Tokenizer asset manifest compatible_models must be a non-empty string list"
            )

        capabilities = manifest.get("capabilities")
        if not isinstance(capabilities, dict):
            errors.append("Tokenizer asset manifest capabilities must be an object")
        else:
            for name in ("chat", "tools", "thinking"):
                if not isinstance(capabilities.get(name), bool):
                    errors.append(
                        f"Tokenizer asset manifest capability {name} must be boolean"
                    )
            if capabilities.get("chat") is not True:
                errors.append("Tokenizer asset manifest capability chat must be true")
        return errors

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _validate_files(
        self, path: Path, manifest: Optional[Dict[str, Any]]
    ) -> tuple[List[str], str]:
        errors: List[str] = []
        if not path.is_dir():
            return ["Tokenizer asset directory does not exist"], ""

        try:
            required_files = self._required_files(manifest)
        except TokenizerAssetError as exc:
            return [str(exc)], ""

        checksums = manifest.get("checksums", {}) if manifest else {}
        require_checksums = manifest is not None
        if not isinstance(checksums, dict):
            errors.append("checksums must be an object")
            checksums = {}
        elif require_checksums:
            for relative_name in required_files:
                expected = str(checksums.get(relative_name, "")).removeprefix("sha256:")
                if not expected:
                    errors.append(
                        f"Missing SHA-256 checksum for Tokenizer file: {relative_name}"
                    )
                elif len(expected) != 64 or any(
                    char not in "0123456789abcdefABCDEF" for char in expected
                ):
                    errors.append(
                        f"Invalid SHA-256 checksum for Tokenizer file: {relative_name}"
                    )

        file_digests: Dict[str, str] = {}
        for relative_name in sorted(required_files):
            relative_path = Path(relative_name)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                errors.append(f"Unsafe required file path: {relative_name}")
                continue
            file_path = (path / relative_path).resolve()
            if file_path != path and not file_path.is_relative_to(path):
                errors.append(f"Required file escapes asset directory: {relative_name}")
                continue
            if not file_path.is_file():
                errors.append(f"Missing required Tokenizer file: {relative_name}")
                continue
            try:
                file_size = file_path.stat().st_size
            except OSError as exc:
                errors.append(f"Cannot stat Tokenizer file {relative_name}: {exc}")
                continue
            max_size = (
                _MAX_ENCODING_BYTES
                if relative_name == "encoding/encoding_dsv4.py"
                else _MAX_TOKENIZER_BYTES
            )
            if file_size <= 0:
                errors.append(f"Tokenizer file is empty: {relative_name}")
                continue
            if file_size > max_size:
                errors.append(f"Tokenizer file is too large: {relative_name}")
                continue
            try:
                digest = self._file_sha256(file_path)
            except OSError as exc:
                errors.append(f"Cannot read Tokenizer file {relative_name}: {exc}")
                continue
            expected = str(checksums.get(relative_name, "")).removeprefix("sha256:")
            if expected and digest.lower() != expected.lower():
                errors.append(f"SHA-256 mismatch for Tokenizer file: {relative_name}")
            file_digests[relative_name] = digest
        return (
            errors,
            aggregate_tokenizer_asset_fingerprint(file_digests) if not errors else "",
        )

    def _inspect_entry(self, asset_id: str, *, smoke_test: bool) -> Dict[str, Any]:
        entry = self._entries.get(asset_id)
        if entry is None:
            raise KeyError(asset_id)

        result: Dict[str, Any] = {
            "asset_id": asset_id,
            "display_name": asset_id,
            "model_family": "",
            "model_name": "",
            "revision": "",
            "encoding_type": "",
            "compatible_models": [],
            "capabilities": {},
            "enabled": bool(entry.get("enabled", True)),
            "status": "invalid",
            "fingerprint": "",
            "errors": [],
            "checks": {},
            "validated_at": self._last_validated_at.get(asset_id, ""),
        }
        try:
            path = self._entry_path(entry)
            self._ensure_allowed_root(path)
            result["path"] = str(path)
            manifest = self._read_manifest(path)
            if str(manifest.get("asset_id", "")).strip() != asset_id:
                raise TokenizerAssetError(
                    "Tokenizer asset manifest asset_id does not match registry"
                )
            for key in (
                "display_name",
                "model_family",
                "model_name",
                "revision",
                "encoding_type",
                "compatible_models",
                "capabilities",
            ):
                if key in manifest:
                    result[key] = manifest[key]
            metadata_errors = self._validate_manifest_metadata(manifest)
            if not metadata_errors:
                result["checks"]["manifest"] = "ok"
            result["errors"].extend(metadata_errors)
            result["required_files"] = self._required_files(manifest)
            errors, fingerprint = self._validate_files(path, manifest)
            result["errors"].extend(errors)
            if not errors:
                result["fingerprint"] = f"sha256:{fingerprint}"
                result["checks"]["required_files"] = "ok"
                result["checks"]["checksums"] = "ok"
        except Exception as exc:
            result["errors"].append(str(exc))

        if not result["enabled"]:
            result["status"] = "disabled"
        elif result["errors"]:
            path_value = result.get("path")
            result["status"] = (
                "missing" if path_value and not Path(path_value).is_dir() else "invalid"
            )
        else:
            result["status"] = "available"
        result["valid"] = result["status"] == "available"
        return result

    @staticmethod
    def _public_asset(asset: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(asset)
        result.pop("path", None)
        return result

    def list_assets(self) -> Dict[str, Any]:
        return {
            "items": [
                self._public_asset(self._inspect_entry(asset_id, smoke_test=False))
                for asset_id in sorted(self._entries)
            ],
            "allow_custom_path": self._allow_custom_path,
            "config_error": self._config_error,
        }

    def get_asset(self, asset_id: str) -> Dict[str, Any]:
        return self._public_asset(self._inspect_entry(asset_id, smoke_test=False))

    def validate_asset(self, asset_id: str) -> Dict[str, Any]:
        validated_at = datetime.now(timezone.utc).isoformat()
        self._last_validated_at[asset_id] = validated_at
        result = self._inspect_entry(asset_id, smoke_test=True)
        result["validated_at"] = validated_at
        return self._public_asset(result)

    def match_path(self, tokenizer_path: str) -> Optional[Dict[str, Any]]:
        try:
            requested = Path(tokenizer_path).expanduser().resolve()
        except (OSError, RuntimeError):
            return None
        for asset_id in self._entries:
            asset = self._inspect_entry(asset_id, smoke_test=False)
            if asset.get("path") == str(requested):
                return asset
        return None

    def resolve(
        self, asset_id: str, tokenizer_path: Optional[str] = None
    ) -> Dict[str, Any]:
        asset = self._inspect_entry(asset_id, smoke_test=True)
        if asset["status"] != "available":
            detail = "; ".join(asset["errors"]) or asset["status"]
            raise TokenizerAssetError(
                f"Tokenizer asset is not available: {asset_id}: {detail}"
            )
        resolved_path = str(asset["path"])
        if tokenizer_path:
            supplied = str(Path(tokenizer_path).expanduser().resolve())
            if supplied != resolved_path:
                raise TokenizerAssetError(
                    "tokenizer_asset_id and tokenizer_path resolve to different directories"
                )
        capabilities = asset.get("capabilities") or {}
        return {
            "tokenizer_asset_id": asset_id,
            "tokenizer_path": resolved_path,
            "tokenizer_asset_revision": str(asset.get("revision", "")),
            "tokenizer_asset_fingerprint": str(asset.get("fingerprint", "")),
            "tokenizer_asset_files": list(
                asset.get("required_files") or DEFAULT_TOKENIZER_ASSET_FILES
            ),
            "tokenizer_asset_capabilities": {
                str(name): bool(enabled) for name, enabled in capabilities.items()
            },
        }

    def validate_path(self, tokenizer_path: str, *, smoke_test: bool) -> Dict[str, Any]:
        # ``smoke_test`` is retained for API compatibility. Executing asset
        # Python in the Supervisor is intentionally not supported.
        path = Path(tokenizer_path).expanduser().resolve()
        errors, fingerprint = self._validate_files(path, None)
        checks: Dict[str, str] = {}
        if not errors:
            checks["required_files"] = "ok"
        return {
            "valid": not errors,
            "status": "available" if not errors else "invalid",
            "fingerprint": f"sha256:{fingerprint}" if fingerprint else "",
            "required_files": list(DEFAULT_TOKENIZER_ASSET_FILES),
            "capabilities": {"chat": True, "tools": True, "thinking": True},
            "checks": checks,
            "errors": errors,
        }
