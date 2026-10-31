# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
"""Journal selection preserves the existing protocols and failure semantics."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from xinference import extensions

from ..control_session import ControllerSession
from ..supervisor import SupervisorActor
from ..supervisor_journal import supervisor_journal
from ..worker_registry import SupervisorHandoffMixin, WorkerOwnership
from ..worker_relaunch import WorkerRelaunchManager


class MemoryStore:
    def __init__(self):
        self.payload = None

    def load(self, default):
        return copy.deepcopy(self.payload if self.payload is not None else default)

    def save(self, payload):
        self.payload = copy.deepcopy(payload)


@pytest.mark.asyncio
async def test_startup_loads_ownership_before_support_actor_awaits(
    monkeypatch, tmp_path
):
    store = MemoryStore()
    WorkerOwnership(store=store).checkpoint(retired={"worker:1234": {"old-boot"}})
    monkeypatch.setattr(
        extensions,
        "get_extensions",
        lambda: (
            SimpleNamespace(
                name="test", create_supervisor_journal_store=lambda ctx: store
            ),
        ),
    )
    monkeypatch.setattr("xinference.core.supervisor.XINFERENCE_ENABLE_OTEL", False)
    owner = SupervisorActor()
    owner.address = "supervisor:1234"

    async def first_support_actor(*args, **kwargs):
        registry = owner._ownership_registry()
        with pytest.raises(RuntimeError, match="ownership was transferred"):
            registry.check("worker:1234", "old-boot", registration=True)
        assert owner._ownership_registry() is registry
        raise RuntimeError("stop after startup check")

    monkeypatch.setattr(
        "xinference.core.supervisor.xo.create_actor", first_support_actor
    )
    with pytest.raises(RuntimeError, match="stop after startup check"):
        await owner.__post_create__()


def test_provider_read_error_uses_worker_journal_failure_contract(
    tmp_path, monkeypatch
):
    from ..worker_state import WorkerJournalError

    provider = MemoryStore()
    provider.load = Mock(side_effect=RuntimeError("provider connection detail"))
    monkeypatch.setattr(
        extensions,
        "get_extensions",
        lambda: (
            SimpleNamespace(
                name="test", create_supervisor_journal_store=lambda ctx: provider
            ),
        ),
    )
    store = supervisor_journal(
        SimpleNamespace(address="supervisor:1234"),
        "worker_relaunch",
        tmp_path / "unused.json",
    )
    with pytest.raises(
        WorkerJournalError, match="Supervisor journal load failed"
    ) as error:
        WorkerRelaunchManager(None, tmp_path / "unused.json", store=store)
    assert "provider connection detail" not in str(error.value)


def test_factory_gets_session_and_default_remains_local(tmp_path, monkeypatch):
    session = ControllerSession("cluster", "candidate", "process", 1)
    owner = SimpleNamespace(
        control_address="supervisor:1234",
        _controller_control=SimpleNamespace(session=session),
    )
    store = MemoryStore()
    create = Mock(return_value=store)
    monkeypatch.setattr(
        extensions,
        "get_extensions",
        lambda: (SimpleNamespace(name="test", create_supervisor_journal_store=create),),
    )
    path = tmp_path / "state.json"
    selected = supervisor_journal(owner, "worker_ownership", path)
    selected.save({"version": 1})
    assert store.payload == {"version": 1}
    context = create.call_args.args[0]
    assert (
        context.control_session is session and context.address == owner.control_address
    )
    assert context.path == str(path) and not path.exists()
    create.return_value = None
    local = supervisor_journal(owner, "worker_ownership", path)
    local.save({"version": 1})
    assert local.load({}) == {"version": 1}


@pytest.mark.parametrize("failure", ["multiple", "factory", "invalid", "async"])
def test_provider_errors_never_select_local(tmp_path, monkeypatch, failure):
    async def async_factory(context):
        return MemoryStore()

    create = (
        async_factory
        if failure == "async"
        else Mock(return_value=object() if failure == "invalid" else MemoryStore())
    )
    if failure == "factory":
        create.side_effect = RuntimeError("store unavailable")
    provider = SimpleNamespace(name="test", create_supervisor_journal_store=create)
    monkeypatch.setattr(
        extensions,
        "get_extensions",
        lambda: (provider, provider) if failure == "multiple" else (provider,),
    )
    path = tmp_path / "state.json"
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        supervisor_journal(
            SimpleNamespace(control_address="supervisor:1234"), "worker_ownership", path
        )
    assert not path.exists()
    if failure == "multiple":
        create.assert_not_called()


def test_ownership_roundtrip_and_failed_save_preserves_memory(tmp_path):
    store = MemoryStore()
    path = tmp_path / "unused.json"
    first = WorkerOwnership(path, store=store)
    first.checkpoint(
        retired={"worker:1234": {"old-boot"}},
        completed={"worker:1234": ("ticket", "new:1234")},
    )
    loaded = WorkerOwnership(path, store=store)
    assert loaded.retired == first.retired and loaded.completed == first.completed
    store.save = Mock(side_effect=RuntimeError("write rejected"))
    with pytest.raises(RuntimeError):
        loaded.checkpoint(retired={})
    assert loaded.retired == first.retired
    assert not path.exists()


def test_relaunch_roundtrip_and_failed_save_preserves_memory(tmp_path):
    store = MemoryStore()
    path = tmp_path / "unused.json"
    first = WorkerRelaunchManager(None, path, store=store)
    first._commit(
        {
            "op": dict(
                operation_id="op",
                status="starting",
                source_worker="a",
                source_incarnation="boot",
                target_worker=None,
                created_at=1,
                python_executable=None,
            )
        }
    )
    loaded = WorkerRelaunchManager(None, path, store=store)
    assert loaded.records == first.records
    store.save = Mock(side_effect=RuntimeError("write rejected"))
    with pytest.raises(RuntimeError):
        loaded.update("op", status="failed")
    assert loaded.records == first.records
    assert not path.exists()


def test_production_ownership_initializer_uses_provider(monkeypatch):
    store = MemoryStore()
    create = Mock(return_value=store)
    monkeypatch.setattr(
        extensions,
        "get_extensions",
        lambda: (SimpleNamespace(name="test", create_supervisor_journal_store=create),),
    )
    owner = SimpleNamespace(control_address="supervisor:1234")
    SupervisorHandoffMixin._initialize_worker_ownership(owner)
    owner._worker_ownership.checkpoint(retired={"worker:1234": {"retired"}})
    assert store.payload["retired"] == {"worker:1234": ["retired"]}


def test_async_save_cannot_publish_unpersisted_state(tmp_path, monkeypatch):
    store = MemoryStore()

    async def save(payload):
        store.payload = payload

    store.save = save
    monkeypatch.setattr(
        extensions,
        "get_extensions",
        lambda: (
            SimpleNamespace(
                name="test", create_supervisor_journal_store=lambda context: store
            ),
        ),
    )
    selected = supervisor_journal(
        SimpleNamespace(control_address="supervisor:1234"),
        "worker_ownership",
        tmp_path / "unused.json",
    )
    ownership = WorkerOwnership(store=selected)
    with pytest.raises(TypeError, match="synchronous"):
        ownership.checkpoint(retired={"worker:1234": {"old"}})
    assert ownership.retired == {} and store.payload is None
