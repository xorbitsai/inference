# Copyright 2022-2026 Xinference Holdings Pte. Ltd

from xinference.core.router_node_store import RouterNodeStore


def _node(node_id: str) -> dict:
    return {
        "node_id": node_id,
        "advertise_host": "127.0.0.1",
        "port_range_start": 12080,
        "port_range_end": 12089,
        "max_instances": 5,
        "labels": {"legacy": "value"},
        "capabilities": {},
    }


def test_register_uses_legacy_labels_only_when_reported_labels_is_omitted(tmp_path):
    store = RouterNodeStore(str(tmp_path / "nodes.db"))

    legacy = store.register(_node("legacy"))
    explicit_empty = store.register({**_node("explicit-empty"), "reported_labels": {}})

    assert legacy["reported_labels"] == {"legacy": "value"}
    assert explicit_empty["reported_labels"] == {}
    assert explicit_empty["labels"] == {}


def test_register_preserves_managed_labels_with_explicit_empty_reported_labels(
    tmp_path,
):
    store = RouterNodeStore(str(tmp_path / "nodes.db"))
    store.register(_node("node-a"))
    store.set_managed_labels("node-a", {"environment": "test"})

    updated = store.register({**_node("node-a"), "reported_labels": {}})

    assert updated["reported_labels"] == {}
    assert updated["managed_labels"] == {"environment": "test"}
    assert updated["labels"] == {"environment": "test"}
