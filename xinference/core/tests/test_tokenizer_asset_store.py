# Copyright 2022-2026 Xinference Holdings Pte. Ltd

from __future__ import annotations

import pytest

from xinference.core.router_node_store import RouterNodeStore
from xinference.core.tokenizer_asset_store import TokenizerAssetStore

FINGERPRINT_A = "sha256:" + "a" * 64
FINGERPRINT_B = "sha256:" + "b" * 64


def _stores(tmp_path):
    db_path = str(tmp_path / "routers.db")
    nodes = RouterNodeStore(db_path)
    for node_id, host in (("node-a", "127.0.0.1"), ("node-b", "127.0.0.2")):
        nodes.register(
            {
                "node_id": node_id,
                "advertise_host": host,
                "port_range_start": 12080,
                "port_range_end": 12089,
                "max_instances": 5,
            }
        )
    return db_path, TokenizerAssetStore(db_path)


def _asset(asset_id="asset-a", revision="v1", fingerprint=FINGERPRINT_A):
    return {
        "asset_id": asset_id,
        "origin": "shared_fs",
        "revision": revision,
        "fingerprint": fingerprint,
        "source": {"type": "shared_fs", "path": f"/assets/{asset_id}"},
        "capabilities": {"model_families": ["deepseek"]},
        "display_name": asset_id,
        "metadata": {},
        "enabled": True,
    }


def test_assets_and_many_to_many_bindings_are_persistent(tmp_path):
    db_path, store = _stores(tmp_path)
    store.create_asset(_asset("asset-a"))
    store.create_asset(_asset("asset-b"))
    store.upsert_binding("asset-a", "node-a")
    store.upsert_binding("asset-a", "node-b")
    store.upsert_binding("asset-b", "node-a")

    reopened = TokenizerAssetStore(db_path)

    assert [item["asset_id"] for item in reopened.list_assets()] == [
        "asset-a",
        "asset-b",
    ]
    assert {
        (item["asset_id"], item["node_id"]) for item in reopened.list_bindings()
    } == {
        ("asset-a", "node-a"),
        ("asset-a", "node-b"),
        ("asset-b", "node-a"),
    }


def test_legacy_binding_mode_is_rejected(tmp_path):
    _, store = _stores(tmp_path)
    store.create_asset(_asset())

    with pytest.raises(ValueError, match="Invalid Tokenizer Asset Binding mode"):
        store.upsert_binding("asset-a", "node-a", binding_mode="legacy")


def test_binding_generation_fences_status_and_asset_updates(tmp_path):
    _, store = _stores(tmp_path)
    store.create_asset(_asset())
    binding = store.upsert_binding("asset-a", "node-a")

    with pytest.raises(ValueError, match="Stale"):
        store.report_binding_status(
            "asset-a", "node-a", binding["generation"] + 1, "preparing"
        )
    with pytest.raises(ValueError, match="revision or fingerprint"):
        store.report_binding_status(
            "asset-a",
            "node-a",
            binding["generation"],
            "ready",
            observed_revision="wrong",
            observed_fingerprint=FINGERPRINT_A,
        )

    ready = store.report_binding_status(
        "asset-a",
        "node-a",
        binding["generation"],
        "ready",
        observed_revision="v1",
        observed_fingerprint=FINGERPRINT_A,
        local_path="/assets/asset-a",
    )
    store.update_asset("asset-a", {"revision": "v2", "fingerprint": FINGERPRINT_B})
    updated = store.get_binding("asset-a", "node-a")

    assert updated is not None
    assert updated["generation"] == ready["generation"] + 1
    assert updated["observed_state"] == "pending"
    assert updated["desired_revision"] == "v2"
    assert updated["desired_fingerprint"] == FINGERPRINT_B


def test_import_is_idempotent_and_preserves_enabled_state(tmp_path):
    _, store = _stores(tmp_path)
    imported = store.import_asset(_asset())
    disabled = store.update_asset("asset-a", {"enabled": False}, username="admin")

    result = store.import_asset(_asset())

    assert result["enabled"] is False
    assert result["updated_at"] == disabled["updated_at"]
    assert result["created_at"] == imported["created_at"]


def test_revalidate_and_delete_guards(tmp_path):
    _, store = _stores(tmp_path)
    store.create_asset(_asset())
    binding = store.upsert_binding("asset-a", "node-a")

    with pytest.raises(ValueError, match="bindings"):
        store.delete_asset("asset-a")
    with pytest.raises(ValueError, match="absent"):
        store.delete_binding("asset-a", "node-a")

    revalidating = store.revalidate_binding("asset-a", "node-a")
    assert revalidating["generation"] == binding["generation"] + 1
    assert revalidating["observed_state"] == "pending"

    absent = store.upsert_binding("asset-a", "node-a", desired_state="absent")
    store.report_binding_status("asset-a", "node-a", absent["generation"], "absent")
    assert store.delete_binding("asset-a", "node-a") is True
    assert store.delete_asset("asset-a") is True
