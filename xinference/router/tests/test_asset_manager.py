# Copyright 2022-2026 Xinference Holdings Pte. Ltd

from __future__ import annotations

import pytest

from xinference.router.agent.asset_manager import RouterAgentAssetManager

FINGERPRINT = "sha256:" + "a" * 64


class _ControlPlane:
    def __init__(self):
        self.reports = []

    async def report_asset_binding_status(self, node_id, **payload):
        self.reports.append((node_id, payload))
        return payload


def _binding(tmp_path, *, desired_state="present", fingerprint=FINGERPRINT):
    asset_path = tmp_path / "asset-a"
    asset_path.mkdir(exist_ok=True)
    return {
        "asset_id": "asset-a",
        "node_id": "node-a",
        "desired_state": desired_state,
        "observed_state": "pending",
        "desired_revision": "v1",
        "desired_fingerprint": fingerprint,
        "generation": 1,
        "asset": {
            "asset_id": "asset-a",
            "origin": "shared_fs",
            "source": {"type": "shared_fs", "path": str(asset_path)},
        },
    }


@pytest.mark.asyncio
async def test_shared_asset_reconciles_to_ready(monkeypatch, tmp_path):
    control = _ControlPlane()
    manager = RouterAgentAssetManager(
        "node-a", control, inventory_path=str(tmp_path / "inventory.json")
    )
    binding = _binding(tmp_path)
    monkeypatch.setattr(
        manager,
        "_validate_shared_asset",
        lambda *args: {
            "path": str(tmp_path / "asset-a"),
            "revision": "v1",
            "fingerprint": FINGERPRINT,
        },
    )

    await manager.reconcile([binding])

    assert [item[1]["observed_state"] for item in control.reports] == [
        "preparing",
        "validating",
        "ready",
    ]
    assert control.reports[-1][1]["local_path"] == str(tmp_path / "asset-a")
    assert (tmp_path / "inventory.json").is_file()


@pytest.mark.asyncio
async def test_builtin_asset_mismatch_reports_error(monkeypatch, tmp_path):
    control = _ControlPlane()
    manager = RouterAgentAssetManager("node-a", control)
    binding = _binding(tmp_path)
    binding["asset"] = {
        "asset_id": "asset-a",
        "origin": "builtin",
        "source": {"type": "builtin"},
    }
    monkeypatch.setattr(
        "xinference.router.agent.asset_manager.resolve_builtin_tokenizer_asset",
        lambda asset_id: {
            "tokenizer_path": str(tmp_path / "asset-a"),
            "tokenizer_asset_revision": "v1",
            "tokenizer_asset_fingerprint": "sha256:" + "b" * 64,
        },
    )

    await manager.reconcile([binding])

    assert control.reports[-1][1]["observed_state"] == "error"
    assert control.reports[-1][1]["last_error_code"] == "asset_validation_failed"
    assert "fingerprint mismatch" in control.reports[-1][1]["last_error"]


@pytest.mark.asyncio
async def test_absent_binding_removes_inventory(tmp_path):
    control = _ControlPlane()
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        '{"asset-a":{"path":"/tmp/a","revision":"v1",'
        f'"fingerprint":"{FINGERPRINT}","generation":1}}'
    )
    manager = RouterAgentAssetManager("node-a", control, inventory_path=str(inventory))

    await manager.reconcile([_binding(tmp_path, desired_state="absent")])

    assert [item[1]["observed_state"] for item in control.reports] == [
        "removing",
        "absent",
    ]
    assert manager._inventory == {}
