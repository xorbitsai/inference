# Copyright 2022-2026 Xinference Holdings Pte. Ltd

from xinference.core.router_assignment_store import RouterAssignmentStore


def _assignment():
    return {
        "assignment_id": "router-a-0",
        "router_uid": "router-a",
        "replica_index": 0,
        "node_id": "node-a",
        "listen_host": "127.0.0.1",
        "listen_port": 12080,
        "public_endpoint": "http://127.0.0.1:12080",
        "desired_state": "running",
        "assignment_generation": 1,
        "config_revision": 1,
    }


def test_report_status_without_instance_id_preserves_existing_value(tmp_path):
    store = RouterAssignmentStore(str(tmp_path / "assignments.db"))
    store.create(_assignment())

    store.report_status(
        "router-a-0",
        1,
        "starting",
        instance_id="router-agent-1-13fdc0e8-34b3-4e7d-ad06-a1db60427557",
    )
    updated = store.report_status("router-a-0", 1, "ready", pid=1234)
    preserved = store.report_status("router-a-0", 1, "ready")

    assert updated["instance_id"] == (
        "router-agent-1-13fdc0e8-34b3-4e7d-ad06-a1db60427557"
    )
    assert preserved["instance_id"] == updated["instance_id"]
    assert preserved["pid"] == 1234
