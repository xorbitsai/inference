# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Tokenizer Asset desired-state reconciliation for Router Agents."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ..tokenizer_assets import resolve_builtin_tokenizer_asset
from .control_plane import RouterAgentControlPlaneClient

logger = logging.getLogger(__name__)


class RouterAgentAssetManager:
    """Prepare and validate Assets without giving the Agent database access."""

    def __init__(
        self,
        node_id: str,
        control_plane: RouterAgentControlPlaneClient,
        *,
        inventory_path: Optional[str] = None,
    ) -> None:
        self.node_id = node_id
        self.control_plane = control_plane
        self.inventory_path = Path(inventory_path) if inventory_path else None
        self._inventory: Dict[str, Dict[str, Any]] = self._load_inventory()

    def _load_inventory(self) -> Dict[str, Dict[str, Any]]:
        if self.inventory_path is None or not self.inventory_path.is_file():
            return {}
        try:
            raw = json.loads(self.inventory_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "Ignoring invalid Router Agent Asset inventory", exc_info=True
            )
            return {}

    def _save_inventory(self) -> None:
        if self.inventory_path is None:
            return
        self.inventory_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.inventory_path.with_suffix(self.inventory_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._inventory, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.inventory_path)

    async def _report(
        self,
        binding: Dict[str, Any],
        observed_state: str,
        *,
        observed_revision: str = "",
        observed_fingerprint: str = "",
        local_path: str = "",
        last_error_code: str = "",
        last_error: str = "",
    ) -> Dict[str, Any]:
        return await self.control_plane.report_asset_binding_status(
            self.node_id,
            asset_id=str(binding["asset_id"]),
            generation=int(binding["generation"]),
            observed_state=observed_state,
            observed_revision=observed_revision,
            observed_fingerprint=observed_fingerprint,
            local_path=local_path,
            last_error_code=last_error_code,
            last_error=last_error,
        )

    @staticmethod
    def _shared_path(asset: Dict[str, Any]) -> Path:
        source = asset.get("source", {})
        path = str(source.get("path") or "").strip()
        if not path:
            raise ValueError("Tokenizer Asset source path is required")
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"Tokenizer Asset directory does not exist: {resolved}")
        return resolved

    @staticmethod
    def _validate_shared_asset(
        asset_id: str, path: Path, desired_revision: str, desired_fingerprint: str
    ) -> Dict[str, str]:
        from ...core.tokenizer_asset_registry import TokenizerAssetRegistry

        validation = TokenizerAssetRegistry().validate_path(str(path), smoke_test=False)
        if not validation.get("valid"):
            raise ValueError("; ".join(validation.get("errors", [])) or "invalid Asset")
        observed_revision = desired_revision
        manifest_path = path / "asset.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_id = str(manifest.get("asset_id") or asset_id)
            if manifest_id != asset_id:
                raise ValueError("Tokenizer Asset manifest asset_id mismatch")
            observed_revision = str(manifest.get("revision") or desired_revision)
        observed_fingerprint = str(validation.get("fingerprint") or "").lower()
        if observed_revision != desired_revision:
            raise ValueError("Tokenizer Asset revision mismatch")
        if observed_fingerprint != desired_fingerprint.lower():
            raise ValueError("Tokenizer Asset fingerprint mismatch")
        return {
            "path": str(path),
            "revision": observed_revision,
            "fingerprint": observed_fingerprint,
        }

    def _resolve(self, binding: Dict[str, Any]) -> Dict[str, str]:
        asset = binding.get("asset", {})
        source_type = str(
            asset.get("source", {}).get("type") or asset.get("origin") or ""
        )
        asset_id = str(binding["asset_id"])
        desired_revision = str(binding["desired_revision"])
        desired_fingerprint = str(binding["desired_fingerprint"]).lower()
        if source_type == "builtin":
            resolved = resolve_builtin_tokenizer_asset(asset_id)
            result = {
                "path": resolved["tokenizer_path"],
                "revision": resolved["tokenizer_asset_revision"],
                "fingerprint": resolved["tokenizer_asset_fingerprint"].lower(),
            }
            if result["revision"] != desired_revision:
                raise ValueError("Built-in Tokenizer Asset revision mismatch")
            if result["fingerprint"] != desired_fingerprint:
                raise ValueError("Built-in Tokenizer Asset fingerprint mismatch")
            return result
        if source_type in {"shared_fs", "local", "external"}:
            return self._validate_shared_asset(
                asset_id,
                self._shared_path(asset),
                desired_revision,
                desired_fingerprint,
            )
        raise ValueError(
            f"Unsupported Tokenizer Asset source type: {source_type or 'unknown'}"
        )

    async def _reconcile_one(self, binding: Dict[str, Any]) -> None:
        asset_id = str(binding["asset_id"])
        generation = int(binding["generation"])
        if binding.get("desired_state") == "absent":
            await self._report(binding, "removing")
            self._inventory.pop(asset_id, None)
            self._save_inventory()
            await self._report(binding, "absent")
            return
        await self._report(binding, "preparing")
        try:
            await self._report(binding, "validating")
            resolved = self._resolve(binding)
            self._inventory[asset_id] = {
                **resolved,
                "generation": generation,
            }
            self._save_inventory()
            await self._report(
                binding,
                "ready",
                observed_revision=resolved["revision"],
                observed_fingerprint=resolved["fingerprint"],
                local_path=resolved["path"],
            )
        except Exception as exc:
            self._inventory.pop(asset_id, None)
            self._save_inventory()
            await self._report(
                binding,
                "error",
                last_error_code="asset_validation_failed",
                last_error=str(exc),
            )

    async def reconcile(self, bindings: Iterable[Dict[str, Any]]) -> None:
        desired = {
            (str(item["asset_id"]), int(item["generation"])): item for item in bindings
        }
        for binding in desired.values():
            cached = self._inventory.get(str(binding["asset_id"]))
            if (
                binding.get("desired_state") == "present"
                and binding.get("observed_state") == "ready"
                and cached is not None
                and int(cached.get("generation", 0)) == int(binding["generation"])
                and cached.get("revision") == binding.get("desired_revision")
                and str(cached.get("fingerprint", "")).lower()
                == str(binding.get("desired_fingerprint", "")).lower()
                and Path(str(cached.get("path", ""))).is_dir()
            ):
                continue
            await self._reconcile_one(binding)


def asset_binding_snapshot(payload: Optional[Dict[str, Any]]) -> list[Dict[str, Any]]:
    if not payload:
        return []
    bindings = payload.get("bindings", [])
    if not isinstance(bindings, list):
        raise ValueError("Supervisor Asset Binding snapshot must be a list")
    return [dict(item) for item in bindings]
