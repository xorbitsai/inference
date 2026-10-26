# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# CI: linux

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event

import pytest

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


@contextmanager
def _pause_before_write(store, monkeypatch, operation, *args, **kwargs):
    read_finished = Event()
    write_allowed = Event()
    now = store._now

    def pause_after_read():
        read_finished.set()
        assert write_allowed.wait(5)
        return now()

    monkeypatch.setattr(store, "_now", pause_after_read)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(operation, "router-a-0", *args, **kwargs)
        try:
            assert read_finished.wait(5)
            yield result
        finally:
            write_allowed.set()


@pytest.mark.parametrize("change", ["reassign", "report", "delete"])
def test_report_status_preserves_concurrent_changes(tmp_path, monkeypatch, change):
    path = str(tmp_path / "assignments.db")
    store = RouterAssignmentStore(path)
    other = RouterAssignmentStore(path)
    store.create(_assignment())
    with _pause_before_write(
        store, monkeypatch, store.report_status, 1, "ready"
    ) as report:
        if change == "reassign":
            other.update_desired("router-a-0", node_id="node-b")
        elif change == "report":
            other.report_status(
                "router-a-0",
                1,
                "starting",
                pid=1234,
                instance_id="new-instance",
                observed={"runtime_status": "starting"},
            )
        else:
            other.delete("router-a-0")
    if change == "report":
        result = report.result(timeout=5)
        assert result["observed_state"] == "ready"
        assert result["pid"] == 1234
        assert result["instance_id"] == "new-instance"
        assert result["observed"] == {"runtime_status": "starting"}
    else:
        with pytest.raises(ValueError, match="Stale Assignment generation"):
            report.result(timeout=5)

    reopened = RouterAssignmentStore(path).get("router-a-0")
    if change == "reassign":
        assert reopened["assignment_generation"] == 2
        assert reopened["node_id"] == "node-b"
        assert reopened["observed_state"] == "pending"
        assert reopened["pid"] is None
    elif change == "report":
        assert reopened == result
    else:
        assert reopened is None


@pytest.mark.parametrize("change", ["stop_late", "reassign_late", "ready", "two_bumps"])
def test_update_desired_preserves_concurrent_changes(tmp_path, monkeypatch, change):
    path = str(tmp_path / "assignments.db")
    store, other = RouterAssignmentStore(path), RouterAssignmentStore(path)
    store.create(_assignment())
    changes = {
        "stop_late": {"desired_state": "stopped"},
        "reassign_late": {
            "node_id": "node-b",
            "desired_state": "running",
            "bump_generation": True,
        },
        "ready": {"config_revision": 2},
        "two_bumps": {"node_id": "node-b", "bump_generation": True},
    }[change]
    with _pause_before_write(
        store, monkeypatch, store.update_desired, **changes
    ) as update:
        if change == "stop_late":
            other.update_desired("router-a-0", node_id="node-b", bump_generation=True)
        elif change == "reassign_late":
            other.update_desired("router-a-0", desired_state="stopped")
        elif change == "ready":
            other.report_status("router-a-0", 1, "ready")
        else:
            other.update_desired("router-a-0", node_id="node-c", bump_generation=True)
    if change == "ready":
        result = update.result(timeout=5)
        assert result["observed_state"] == "ready"
        assert result["config_revision"] == 2
        assert result["assignment_generation"] == 1
    else:
        with pytest.raises(ValueError, match="Stale Assignment desired state"):
            update.result(timeout=5)
    reopened = RouterAssignmentStore(path).get("router-a-0")
    if change == "stop_late":
        result = other.update_desired("router-a-0", desired_state="stopped")
        assert reopened["node_id"] == result["node_id"] == "node-b"
        assert reopened["assignment_generation"] == result["assignment_generation"] == 2
        assert result["desired_state"] == "stopped"
    elif change == "reassign_late":
        assert reopened["node_id"] == "node-a"
        assert reopened["desired_state"] == "stopped"
        assert reopened["assignment_generation"] == 1
    elif change == "two_bumps":
        assert reopened["node_id"] == "node-c"
        assert reopened["assignment_generation"] == 2
        result = other.update_desired("router-a-0", **changes)
        assert result["assignment_generation"] == 3
        with pytest.raises(ValueError, match="Stale Assignment generation"):
            other.report_status("router-a-0", 2, "ready")
    else:
        assert reopened == result
