# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
import asyncio
import contextlib
import copy
import json
import os
import pickle
import sys
import threading
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import xoscar as xo
from xoscar.backends.router import Router

from xinference import extensions

from .. import model as model_module
from .. import supervisor as supervisor_module
from .. import worker as worker_module
from ..cache_tracker import CacheTrackerActor
from ..control_session import (
    CONTROL_CONTEXT_KEY,
    READ_ONLY_WORKER_RPCS,
    ControlContext,
    ControllerSession,
    ControlSessionRejected,
    WorkerControlFence,
    check_control_session,
    run_worker_control_callback,
)
from ..lifecycle import (
    WorkerLifecycle,
    WorkerPhase,
    management_cleanup,
    management_operation,
    spawn_management_task,
)
from ..model import ModelActor
from ..supervisor import SupervisorActor, WorkerStatus
from ..worker import ModelStatus, WorkerActor
from .test_worker_handoff import CounterModel, HandoffSupervisor, HandoffWorker


@pytest.mark.asyncio
async def test_catalog_replacement_does_not_yield_after_ready_publication(monkeypatch):
    from ..worker_handoff import CustomModelState

    fence, authority = setup_fence()
    worker = WorkerActor("supervisor:1234", None, SimpleNamespace(), [])
    worker._control_fence, worker._lifecycle = fence, fence.lifecycle
    restored, exposed = [], []
    validate = fence._validate

    async def observed(owner, phases):
        await validate(owner, phases)
        if fence.ready:
            # A validation await after readiness would let queued commands see
            # the catalog before its replacement.
            await asyncio.sleep(0)
            exposed.append(bool(restored))

    def restore(owner, models, **kwargs):
        assert fence.ready and asyncio.current_task() in fence.lifecycle._tasks
        restored.append(models)

    monkeypatch.setattr(fence, "_validate", observed)
    monkeypatch.setattr(CustomModelState, "restore", restore)
    assert (
        await worker.install_controller_session(
            authority.owner.to_dict(), custom_models={}
        )
    )["ready"]
    assert restored == [{}]
    assert all(exposed)


@pytest.mark.asyncio
@pytest.mark.parametrize("same_session", [False, True])
async def test_failed_catalog_install_closes_admission(same_session):
    fence, authority = setup_fence()
    worker = WorkerActor("supervisor:1234", None, SimpleNamespace(), [])
    worker._control_fence, worker._lifecycle = fence, fence.lifecycle
    if same_session:
        await worker.install_controller_session(authority.owner.to_dict())
    with pytest.raises(ValueError, match="custom model catalog"):
        await worker.install_controller_session(
            authority.owner.to_dict(), custom_models={"unknown": {}}
        )
    assert not worker.controller_session_status()["ready"]
    with pytest.raises(ControlSessionRejected, match="fence_pending"):
        async with fence.command(command(authority.owner)):
            pytest.fail("A failed catalog install must close command admission")


def session(term=1, **changes):
    return replace(
        ControllerSession("cluster", "candidate", f"boot-{term}", term), **changes
    )


def command(owner, operation="operation"):
    return ControlContext(owner, operation)


class Authority:
    def __init__(self, owner, phase="active"):
        self.owner, self.phase = owner, phase
        self.unavailable = False
        self.calls = 0

    async def validate(self, owner, *, phases):
        self.calls += 1
        if self.unavailable:
            raise RuntimeError("private-database-url-do-not-expose")
        if owner != self.owner or self.phase not in phases:
            raise ControlSessionRejected("authority_rejected")


def setup_fence(owner=None, phase="active"):
    authority = Authority(owner or session(), phase)
    return WorkerControlFence(authority, WorkerLifecycle(active=True)), authority


def test_strict_immutable_protocol_and_remote_diagnostic():
    owner = session()
    context = command(owner)
    assert ControllerSession.from_dict(owner.to_dict()) == owner
    assert ControlContext.from_dict(context.to_dict()) == context
    with pytest.raises(FrozenInstanceError):
        owner.term = 3
    for change in (
        {"term": True},
        {"term": 0},
        {"term": 1 << 63},
        {"owner_session": "bad/value"},
    ):
        with pytest.raises(ValueError):
            replace(owner, **change)
    error = pickle.loads(pickle.dumps(ControlSessionRejected("stale_session")))
    assert error.reason == "stale_session"
    assert str(error) == "Worker control rejected: stale_session"


@pytest.mark.parametrize("value", [None, {}, {"version": True}, {"version": 2}])
def test_protocol_rejects_missing_and_unknown_versions(value):
    for parser in (ControllerSession.from_dict, ControlContext.from_dict):
        with pytest.raises(ValueError):
            parser(value)


@pytest.mark.asyncio
async def test_installation_does_not_activate_a_controller_or_cache_authority():
    fence, authority = setup_fence(phase="fencing")
    with pytest.raises(ControlSessionRejected):
        async with fence.command(command(authority.owner)):
            pytest.fail("uninstalled session admitted")
    assert (await fence.install(authority.owner, 1))["ready"]
    with pytest.raises(ControlSessionRejected, match="authority_rejected"):
        async with fence.command(command(authority.owner)):
            pytest.fail("merely elected controller admitted")
    authority.phase = "active"
    for _ in range(2):
        async with fence.command(command(authority.owner)):
            pass
    count = authority.calls
    authority.unavailable = True
    with pytest.raises(ControlSessionRejected, match="authority_unavailable") as error:
        async with fence.command(command(authority.owner)):
            pass
    assert "private-database" not in str(error.value)
    assert authority.calls == count + 1


@pytest.mark.asyncio
async def test_new_session_drains_old_work_and_rejects_delayed_commit_and_callback():
    fence, authority = setup_fence()
    old = authority.owner
    await fence.install(old, 1)
    entered, release = asyncio.Event(), asyncio.Event()
    committed = []

    async def old_work():
        async with fence.command(command(old)):
            entered.set()
            await release.wait()
            await check_control_session(SimpleNamespace(_control_fence=fence))
            committed.append("late completion")

    task = asyncio.create_task(old_work())
    await entered.wait()
    authority.owner = new = session(2)
    installing = asyncio.create_task(fence.install(new, 1))
    while fence.session != new:
        await asyncio.sleep(0)
    assert not installing.done() and not fence.ready
    with pytest.raises(ControlSessionRejected, match="stale_session"):
        async with fence.command(command(old)):
            pass
    with pytest.raises(ControlSessionRejected, match="fence_pending"):
        async with fence.command(command(new)):
            pass
    release.set()
    with pytest.raises(ControlSessionRejected, match="stale_session"):
        await task
    assert (await installing)["ready"]
    assert not committed
    async with fence.command(command(new)):
        pass


@pytest.mark.asyncio
async def test_failed_drain_never_restores_old_authority():
    fence, authority = setup_fence()
    old = authority.owner
    await fence.install(old, 1)
    entered, release = asyncio.Event(), asyncio.Event()

    async def old_work():
        async with fence.command(command(old)):
            entered.set()
            await release.wait()

    task = asyncio.create_task(old_work())
    await entered.wait()
    authority.owner = new = session(2)
    with pytest.raises(asyncio.TimeoutError):
        await fence.install(new, 0.01)
    assert fence.session == new and not fence.ready
    assert fence.lifecycle.phase == WorkerPhase.ACTIVE
    for owner in (old, new):
        with pytest.raises(ControlSessionRejected):
            async with fence.command(command(owner)):
                pass
    release.set()
    await task
    assert (await fence.install(new, 1))["ready"]


@pytest.mark.asyncio
async def test_install_order_rejects_old_term_reused_session_and_changed_cluster():
    fence, authority = setup_fence(session(2))
    await fence.install(authority.owner, 1)
    for stale in (
        session(),
        session(2, owner_session="another-boot"),
        session(3, cluster_id="other"),
    ):
        with pytest.raises(ControlSessionRejected):
            await fence.install(stale, 1)
    assert (await fence.install(authority.owner, 1))["term"] == 2


@pytest.mark.asyncio
async def test_authority_failure_after_drain_keeps_admission_closed():
    fence, authority = setup_fence()
    original = fence.lifecycle.drain

    async def fail_after_drain(timeout):
        await original(timeout)
        authority.unavailable = True

    fence.lifecycle.drain = fail_after_drain
    with pytest.raises(ControlSessionRejected, match="authority_unavailable"):
        await fence.install(authority.owner, 1)
    assert not fence.ready
    with pytest.raises(ControlSessionRejected, match="fence_pending"):
        async with fence.command(command(authority.owner)):
            pass
    fence.lifecycle.drain = original
    authority.unavailable = False
    assert (await fence.install(authority.owner, 1))["ready"]


@pytest.mark.asyncio
async def test_cancelled_installation_finishes_without_rolling_back_the_term():
    fence, authority = setup_fence()
    old = authority.owner
    await fence.install(old, 1)
    entered, release = asyncio.Event(), asyncio.Event()

    async def work():
        async with fence.command(command(old)):
            entered.set()
            await release.wait()

    task = asyncio.create_task(work())
    await entered.wait()
    authority.owner = new = session(2)
    owner = WorkerActor("supervisor", None, SimpleNamespace(), [])
    owner._control_fence = fence
    owner._lifecycle = fence.lifecycle
    installing = asyncio.create_task(
        WorkerActor.install_controller_session(owner, new.to_dict(), 1)
    )
    while fence.session != new:
        if installing.done():
            await installing
        await asyncio.sleep(0)
    installing.cancel()
    await asyncio.sleep(0)
    installing.cancel()
    assert not installing.done()
    release.set()
    await task
    with pytest.raises(asyncio.CancelledError):
        await installing
    assert fence.session == new and fence.ready


@pytest.mark.asyncio
async def test_scheduled_management_tasks_keep_the_original_context():
    fence, authority = setup_fence()
    old = authority.owner
    await fence.install(old, 1)
    owner = SimpleNamespace(_control_fence=fence, _lifecycle=fence.lifecycle)
    ran = []

    async def callback():
        ran.append(True)

    async with fence.command(command(old)):
        task = spawn_management_task(owner, callback())
        # No await: the child has not run but already participates in drain.
        fence.session = session(2)
        fence.ready = False
    with pytest.raises(ControlSessionRejected, match="stale_session"):
        await task
    await asyncio.sleep(0)
    assert not ran and not fence.lifecycle._tasks


@pytest.mark.asyncio
async def test_local_management_tasks_validate_authority_and_nested_release():
    fence, authority = setup_fence()
    await fence.install(authority.owner, 1)

    class Owner:
        _control_fence = fence
        _lifecycle = fence.lifecycle
        released = False

        @management_operation
        def release(self):
            self.released = True

        @management_operation
        async def work(self):
            self.release()

    owner = Owner()
    with pytest.raises(ControlSessionRejected, match="missing_context"):
        owner.release()
    with pytest.raises(ControlSessionRejected, match="missing_context"):
        await owner.work()
    with fence.local_scope(command(authority.owner)):
        await owner.work()
    assert owner.released
    authority.unavailable = True
    with fence.local_scope(command(authority.owner)):
        with pytest.raises(ControlSessionRejected, match="authority_unavailable"):
            await owner.work()


@pytest.mark.asyncio
async def test_failed_old_work_cleans_up_before_new_owner_is_acknowledged():
    fence, authority = setup_fence()
    old = authority.owner
    await fence.install(old, 1)
    entered, release = asyncio.Event(), asyncio.Event()

    class Owner:
        _control_fence = fence
        _lifecycle = fence.lifecycle
        allocations = 1

        @management_cleanup
        def cleanup(self):
            self.allocations -= 1

    owner = Owner()

    async def work():
        async with fence.command(command(old)):
            entered.set()
            try:
                await release.wait()
                await check_control_session(owner)
            finally:
                owner.cleanup()

    task = asyncio.create_task(work())
    await entered.wait()
    authority.owner = new = session(2)
    installing = asyncio.create_task(fence.install(new, 1))
    while fence.session != new:
        await asyncio.sleep(0)
    assert not installing.done() and owner.allocations == 1
    release.set()
    with pytest.raises(ControlSessionRejected, match="stale_session"):
        await task
    assert (await installing)["ready"]
    assert owner.allocations == 0
    # This models a delayed cleanup after a replacement already owns resources.
    owner.allocations = 1
    with fence.local_scope(command(old)):
        with pytest.raises(ControlSessionRejected, match="cleanup_not_admitted"):
            owner.cleanup()
    assert owner.allocations == 1


def test_worker_control_authority_hook_is_optional_exclusive_and_fail_closed(
    monkeypatch,
):
    monkeypatch.delenv("XINFERENCE_EXTENSIONS", raising=False)
    monkeypatch.setattr(extensions, "_loaded", None)
    context = extensions.WorkerControlAuthorityContext("127.0.0.1:1234", "worker-boot")
    assert extensions.resolve_worker_control_authority(context) is None
    module = ModuleType("test_control_authority_extension")
    seen = []

    def factory():
        def provider(context):
            seen.append(context)
            return Authority(session())

        return SimpleNamespace(
            name="control", api_version=1, create_worker_control_authority=provider
        )

    module.factory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("XINFERENCE_EXTENSIONS", module.__name__ + ":factory")
    assert isinstance(extensions.resolve_worker_control_authority(context), Authority)
    assert seen == [context]
    monkeypatch.setattr(extensions, "_loaded", None)
    duplicate = factory()
    duplicate.name = "other"
    module.other = lambda: duplicate
    monkeypatch.setenv(
        "XINFERENCE_EXTENSIONS",
        module.__name__ + ":factory," + module.__name__ + ":other",
    )
    with pytest.raises(ValueError, match="Multiple"):
        extensions.resolve_worker_control_authority(context)
    assert seen == [context]


@pytest.mark.asyncio
async def test_sync_authority_runs_off_loop_and_never_falls_back():
    actor_thread = threading.get_ident()
    calls = []

    class SyncAuthority:
        available = True

        def validate(self, owner, *, phases):
            calls.append(threading.get_ident())
            if not self.available:
                raise RuntimeError("private-connection-string")

    authority = SyncAuthority()
    fence = WorkerControlFence(authority, WorkerLifecycle(active=True))
    await fence.install(session(), 1)
    async with fence.command(command(session())):
        pass
    assert calls and all(identity != actor_thread for identity in calls)
    authority.available = False
    with pytest.raises(ControlSessionRejected, match="authority_unavailable") as error:
        async with fence.command(command(session())):
            pass
    assert "private-connection" not in str(error.value)


@pytest.mark.asyncio
async def test_hung_authority_check_has_a_bounded_failure():
    class HangingAuthority:
        async def validate(self, owner, *, phases):
            await asyncio.Event().wait()

    fence = WorkerControlFence(
        HangingAuthority(), WorkerLifecycle(active=True), validation_timeout=0.01
    )
    with pytest.raises(ControlSessionRejected, match="authority_unavailable"):
        await asyncio.wait_for(fence.install(session(), 1), 1)
    assert fence.session is None and not fence.ready


@pytest.mark.asyncio
async def test_stopped_worker_never_acknowledges_a_retried_install():
    fence, authority = setup_fence()
    await fence.install(authority.owner, 1)
    fence.lifecycle.phase = WorkerPhase.STOPPED
    with pytest.raises(ControlSessionRejected, match="worker_not_active"):
        await fence.install(authority.owner, 1)


@pytest.mark.parametrize("invalid", ["object", "error", "async"])
def test_invalid_or_failed_authority_factory_never_selects_legacy_admission(
    monkeypatch, invalid
):
    monkeypatch.setattr(extensions, "_loaded", None)
    module = ModuleType("test_failed_control_authority")

    def provider(context):
        if invalid == "error":
            raise RuntimeError("configured authority failed")
        if invalid == "async":

            async def coroutine():
                return None

            return coroutine()
        return object()

    module.factory = lambda: SimpleNamespace(
        name="failed", api_version=1, create_worker_control_authority=provider
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("XINFERENCE_EXTENSIONS", module.__name__ + ":factory")
    with pytest.raises((RuntimeError, TypeError)):
        extensions.resolve_worker_control_authority(
            extensions.WorkerControlAuthorityContext("worker:1234", "boot")
        )


class AuthorityActor(xo.StatelessActor):
    def __init__(self):
        self.owner = session()
        self.phase = "active"

    def select(self, value, phase="active"):
        self.owner = ControllerSession.from_dict(value)
        self.phase = phase

    def validate(self, value, phases):
        if self.phase == "unavailable":
            raise RuntimeError("private-database-url")
        if ControllerSession.from_dict(value) != self.owner or self.phase not in phases:
            raise ControlSessionRejected("authority_rejected")


@pytest.mark.parametrize("value", [None, {}, {"version": True}, {"version": 2}])
@pytest.mark.asyncio
async def test_mutating_rpc_rejects_invalid_context_before_business_code(value):
    fence, authority = setup_fence()
    await fence.install(authority.owner, 1)
    worker = WorkerActor("supervisor", None, SimpleNamespace(), [])
    worker._control_fence = fence
    with pytest.raises(ControlSessionRejected) as rejected:
        await worker.__on_receive__(
            ("unknown_mutation", 0, (), {CONTROL_CONTEXT_KEY: value})
        )
    assert rejected.value.reason in {"missing_context", "invalid_context"}
    with pytest.raises(ControlSessionRejected, match="unsupported_batch"):
        await worker.__on_receive__(("unknown_mutation", 1, (), {}))


@pytest.mark.asyncio
async def test_context_never_silently_selects_an_unprotected_worker():
    worker = WorkerActor("supervisor", None, SimpleNamespace(), [])
    with pytest.raises(ControlSessionRejected, match="authority_not_configured"):
        await worker.__on_receive__(
            (
                "unknown_mutation",
                0,
                (),
                {CONTROL_CONTEXT_KEY: command(session()).to_dict()},
            )
        )

    class LocalWorker(WorkerActor):
        async def mutation(self, **kwargs):
            return "legacy"

    local_worker = LocalWorker("supervisor", None, SimpleNamespace(), [])
    assert await local_worker.mutation() == "legacy"
    with pytest.raises(ControlSessionRejected, match="authority_not_configured"):
        await local_worker.mutation(
            **{CONTROL_CONTEXT_KEY: command(session()).to_dict()}
        )


@pytest.mark.asyncio
async def test_instance_callable_keeps_its_interface_but_cannot_bypass_a_fence():
    worker = WorkerActor("supervisor", None, SimpleNamespace(), [])
    mutation = AsyncMock(return_value="mutated")
    worker.future_mutation = mutation
    assert worker.future_mutation.__wrapped__ is mutation
    assert worker.future_mutation.__self__ is worker
    assert await worker.future_mutation("legacy") == "mutated"
    worker.future_mutation.assert_awaited_once_with("legacy")
    worker.future_mutation.reset_mock()

    fence, authority = setup_fence()
    worker._control_fence, worker._lifecycle = fence, fence.lifecycle
    old = authority.owner
    await worker.install_controller_session(old.to_dict())
    with pytest.raises(ControlSessionRejected, match="missing_context"):
        await worker.future_mutation("missing")
    mutation.assert_not_awaited()
    assert (
        await worker.future_mutation(
            "current", **{CONTROL_CONTEXT_KEY: command(old).to_dict()}
        )
        == "mutated"
    )
    mutation.assert_awaited_once_with("current")

    async with fence.command(command(old)):
        authority.owner = session(2)
        installing = asyncio.create_task(
            worker.install_controller_session(authority.owner.to_dict())
        )
        while fence.session != authority.owner:
            if installing.done():
                await installing
            await asyncio.sleep(0)
        with pytest.raises(ControlSessionRejected, match="stale_session"):
            # A dynamic attribute on an ordinary callable is not a cleanup tag.
            await worker.future_mutation("stale-bound")
    await installing
    with pytest.raises(ControlSessionRejected, match="stale_session"):
        await worker.future_mutation(
            "stale", **{CONTROL_CONTEXT_KEY: command(old).to_dict()}
        )
    mutation.assert_awaited_once_with("current")


@pytest.mark.asyncio
async def test_false_authority_result_is_a_rejection():
    class Denied:
        def validate(self, owner, *, phases):
            return False

    fence = WorkerControlFence(Denied(), WorkerLifecycle(active=True))
    with pytest.raises(ControlSessionRejected, match="authority_rejected"):
        await fence.install(session(), 1)
    assert not fence.ready and fence.session is None


class ActorAuthority:
    def __init__(self, ref):
        self.ref = ref

    async def validate(self, owner, *, phases):
        await self.ref.validate(owner.to_dict(), phases)


class FencedWorker(HandoffWorker):
    def __init__(self, authority, **kwargs):
        super().__init__(**kwargs)
        self._control_fence = WorkerControlFence(
            ActorAuthority(authority), self._lifecycle
        )

    async def future_mutation(self):
        # Deliberately undecorated: unclassified RPCs must still be guarded.
        return "mutated"

    def future_sync_mutation(self, value):
        self._test_value = value
        return value

    @staticmethod
    def future_static_mutation(value):
        return value

    async def callback_mutation(self):
        return await spawn_management_task(self, self.future_mutation())


@pytest.mark.asyncio
async def test_real_worker_rejects_old_controller_and_retains_model_process(tmp_path):
    pools = [await xo.create_actor_pool("127.0.0.1", n_process=0) for _ in range(2)]
    try:
        supervisor = await xo.create_actor(
            HandoffSupervisor,
            uid="supervisor",
            address=pools[0].external_address,
            ownership_path=tmp_path / "handoffs.json",
        )
        authority = await xo.create_actor(
            AuthorityActor, uid="authority", address=pools[0].external_address
        )
        worker = await xo.create_actor(
            FencedWorker,
            authority,
            uid="worker",
            address=pools[1].external_address,
            supervisor_address=supervisor.address,
            supervisor_endpoint=None,
            main_pool=pools[1],
            gpu_devices=[],
        )
        old = session()
        await worker.install_controller_session(old.to_dict())
        local = await xo.actor_ref(address=worker.address, uid=worker.uid)
        assert type(local).__name__ == "LocalActorRef"
        old_kwargs = {CONTROL_CONTEXT_KEY: command(old, "launch").to_dict()}
        model = await worker.install_model(**old_kwargs)
        value, pid, _ = await model.add(5)
        new = session(2)
        await authority.select(new.to_dict(), "fencing")
        await worker.install_controller_session(new.to_dict())
        assert await worker.ping()
        assert (await model.add(1))[:2] == (value + 1, pid)
        for ref in (worker, local):
            for kwargs, reason in (
                ({}, "missing_context"),
                (old_kwargs, "stale_session"),
            ):
                for name, args in (
                    ("terminate_model", ("counter-rep0",)),
                    ("future_mutation", ()),
                    ("future_sync_mutation", (7,)),
                    ("future_static_mutation", (7,)),
                    ("callback_mutation", ()),
                    ("release_devices", ("counter-rep0",)),
                ):
                    with pytest.raises(ControlSessionRejected) as rejected:
                        await getattr(ref, name)(*args, **kwargs)
                    assert rejected.value.reason == reason
        assert (await model.add(1))[:2] == (value + 2, pid)
        current = {CONTROL_CONTEXT_KEY: command(new, "terminate").to_dict()}
        with pytest.raises(ControlSessionRejected, match="authority_rejected"):
            await worker.future_mutation(**current)
        await authority.select(new.to_dict())
        assert await worker.future_mutation(**current) == "mutated"
        assert await local.future_mutation(**current) == "mutated"
        assert await local.future_sync_mutation(7, **current) == 7
        assert await local.future_static_mutation(7, **current) == 7
        assert await local.callback_mutation(**current) == "mutated"
        for method, args in (
            ("replace_worker", (worker.address,)),
            ("reload_model", ("counter-rep0", {}, 30, "reload")),
        ):
            with pytest.raises(
                ControlSessionRejected, match="handoff_requires_session_transfer"
            ):
                await getattr(worker, method)(*args, **current)
        assert (await model.add(0))[:2] == (value + 2, pid)
        await authority.select(new.to_dict(), "unavailable")
        with pytest.raises(
            ControlSessionRejected, match="authority_unavailable"
        ) as rejected:
            await worker.terminate_model("counter-rep0", **current)
        assert "private-database" not in str(rejected.value)
        assert (await model.add(1))[:2] == (value + 3, pid)
        await authority.select(new.to_dict())
        await worker.terminate_model("counter-rep0", **current)
        assert "counter-rep0" not in await worker.list_models()
    finally:
        for pool in reversed(pools):
            with contextlib.suppress(Exception):
                await pool.stop()
            pool._actors.clear()
        Router.set_instance(None)


def test_read_only_allowlist_cannot_accidentally_name_missing_worker_methods():
    assert all(
        callable(getattr(WorkerActor, method, None)) for method in READ_ONLY_WORKER_RPCS
    )


@pytest.mark.parametrize("phase", ["uninstalled", "draining", "timeout", "finished"])
@pytest.mark.asyncio
async def test_heartbeats_preserve_worker_liveness_while_control_is_fenced(
    monkeypatch, phase
):
    worker = WorkerActor("supervisor", None, SimpleNamespace(), [])
    worker.address = "worker:1234"
    worker._actor_loop = asyncio.get_running_loop()
    fence, authority = setup_fence()
    worker._control_fence, worker._lifecycle = fence, fence.lifecycle
    supervisor = HandoffSupervisor()
    supervisor.receive_heartbeat = MethodType(
        SupervisorActor.receive_heartbeat, supervisor
    )
    supervisor._worker_address_to_worker[worker.address] = worker
    supervisor._worker_ownership.incarnations[worker.address] = worker._incarnation
    worker._supervisor_ownership_supported = True
    status = WorkerStatus(update_time=0, failure_remaining_count=5, status={})
    supervisor._worker_status[worker.address] = status
    supervisor._replica_model_uid_to_worker["counter-rep0"] = worker
    supervisor._remove_dead_worker = AsyncMock(return_value=False)
    worker._supervisor_ref = supervisor
    worker._registered = True
    recover = AsyncMock()
    publish = AsyncMock()
    monkeypatch.setattr(worker, "_retry_control_faults", recover)
    monkeypatch.setattr(worker, "_record_model_versions_best_effort", publish)
    monkeypatch.setattr(supervisor, "add_worker", AsyncMock())
    release, entered = asyncio.Event(), asyncio.Event()
    held = installing = None
    old = authority.owner

    async def hold():
        async with fence.command(command(old)):
            entered.set()
            await release.wait()

    try:
        if phase != "uninstalled":
            await fence.install(old, 1)
            held = asyncio.create_task(hold())
            await entered.wait()
            authority.owner = session(2)
            installing = asyncio.create_task(
                fence.install(authority.owner, 10 if phase == "draining" else 0.02)
            )
            while fence.ready:
                await asyncio.sleep(0)
            if phase in {"timeout", "finished"}:
                with pytest.raises(asyncio.TimeoutError):
                    await installing
            if phase == "finished":
                release.set()
                await held
        assert not fence.ready
        validations = authority.calls
        clock = [0.0]
        original_sleep = asyncio.sleep

        async def one_tick(delay):
            if delay == worker_module.XINFERENCE_HEALTH_CHECK_INTERVAL:
                raise asyncio.CancelledError
            await original_sleep(delay)

        monkeypatch.setattr(supervisor_module.time, "time", lambda: clock[0])
        monkeypatch.setattr(worker_module.asyncio, "sleep", one_tick)
        for _ in range(8):
            clock[0] += worker_module.XINFERENCE_HEALTH_CHECK_INTERVAL
            # Production schedules this loop on a separate status-isolation loop.
            await asyncio.to_thread(
                lambda: asyncio.run(worker._periodical_report_status())
            )
            assert status.update_time == clock[0]
            # A stale inherited binding must not turn liveness into management.
            with fence.local_scope(command(old)):
                await worker.heartbeat()
            with pytest.raises(asyncio.CancelledError):
                await supervisor._check_dead_nodes()
        supervisor._remove_dead_worker.assert_not_awaited()
        assert supervisor._replica_model_uid_to_worker["counter-rep0"] is worker
        assert authority.calls == validations
        supervisor.add_worker.assert_not_awaited()
        recover.assert_not_awaited()
        publish.assert_not_awaited()
        with pytest.raises(ControlSessionRejected, match="missing_context"):
            await worker.report_status()
    finally:
        release.set()
        if held is not None:
            await held
        if installing is not None:
            await asyncio.gather(installing, return_exceptions=True)


@pytest.mark.parametrize("dispatch", ["local", "receive"])
@pytest.mark.asyncio
async def test_heartbeat_dispatch_needs_no_controller_or_registration(
    monkeypatch, dispatch
):
    worker = WorkerActor("supervisor", None, SimpleNamespace(), [])
    worker.address = "worker:1234"
    worker._control_fence, _ = setup_fence()
    call = AsyncMock()
    monkeypatch.setattr(worker, "_call_supervisor", call)
    if dispatch == "local":
        await worker.heartbeat()
    else:
        await worker.__on_receive__(("heartbeat", 0, (), {}))
    call.assert_awaited_once_with("receive_heartbeat", worker.address, add_worker=False)


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.asyncio
async def test_failed_launch_status_cleanup_preserves_legacy_behavior(
    monkeypatch, configured
):
    worker = WorkerActor("supervisor", None, SimpleNamespace(), [])
    ref = object()
    uid = "counter-rep0"
    worker._model_uid_to_model[uid] = ref
    worker._model_uid_to_model_status[uid] = ModelStatus(model_state="loading")
    monkeypatch.setattr(
        worker, "terminate_model", AsyncMock(side_effect=RuntimeError("failed cleanup"))
    )
    monkeypatch.setattr(
        worker,
        "_terminate_model",
        AsyncMock(side_effect=RuntimeError("failed cleanup")),
    )
    scope = contextlib.nullcontext()
    if configured:
        fence, authority = setup_fence()
        worker._control_fence, worker._lifecycle = fence, fence.lifecycle
        await fence.install(authority.owner, 1)
        scope = fence.command(command(authority.owner))
    async with scope:
        await worker._clean_up_failed_launch(uid, ref)
    assert (uid in worker._model_uid_to_model_status) == configured
    assert worker._model_uid_to_model[uid] is ref


@pytest.mark.parametrize("dispatch", ["local", "receive"])
@pytest.mark.parametrize(
    "method", ["get_gpu_allocation_status", "is_model_vllm_backend"]
)
@pytest.mark.asyncio
async def test_gpu_reads_need_no_control_context_or_worker_registration(
    monkeypatch, dispatch, method
):
    worker = WorkerActor("supervisor", None, SimpleNamespace(), [0, 1])
    fence, authority = setup_fence()
    worker._control_fence, worker._lifecycle = fence, fence.lifecycle
    await worker.install_controller_session(authority.owner.to_dict())
    supervisor = SimpleNamespace(
        get_worker_resource_policy=AsyncMock(return_value=[1]),
        get_model=AsyncMock(
            return_value=SimpleNamespace(is_vllm_backend=AsyncMock(return_value=True))
        ),
        add_worker=AsyncMock(),
    )
    lookup = AsyncMock(return_value=(supervisor, 1))
    monkeypatch.setattr(worker, "_get_supervisor_ref_with_generation", lookup)
    monkeypatch.setattr(extensions, "get_extensions", lambda: (object(),))
    args = ("counter-rep0",) if method == "is_model_vllm_backend" else ()
    if dispatch == "local":
        result = await getattr(worker, method)(*args)
    else:
        result = await worker.__on_receive__((method, 0, args, {}))
    assert result is True if args else result["total"] == [1]
    lookup.assert_awaited_once_with(add_worker=False)
    supervisor.add_worker.assert_not_awaited()
    with pytest.raises(ControlSessionRejected, match="missing_context"):
        await worker.get_supervisor_ref()


@pytest.mark.asyncio
async def test_native_monitor_captures_each_fault_and_drains_inflight_recovery(
    monkeypatch,
):
    pool = SimpleNamespace(remove_sub_pool=AsyncMock())
    worker = WorkerActor("supervisor", None, pool, [])
    fence, authority = setup_fence()
    worker._control_fence, worker._lifecycle = fence, fence.lifecycle
    old = authority.owner
    await worker.install_controller_session(old.to_dict())
    faults = asyncio.Queue()
    seen = []

    async def monitor():
        while True:
            address, done = await faults.get()
            try:
                await pool.recover_sub_pool(address)
            except Exception as error:
                done.set_exception(error)
            else:
                done.set_result(True)

    monitor_task = None

    async def start_monitor():
        nonlocal monitor_task
        monitor_task = asyncio.create_task(monitor())
        pool._monitor_task = monitor_task

    pool.start_monitor = start_monitor
    worker._model_uid_to_model = {"counter-rep0": SimpleNamespace()}
    worker._model_uid_to_addr = {"counter-rep0": "dead"}
    worker._model_uid_to_launch_args = {"counter-rep0": {"model_uid": "counter-rep0"}}
    worker._model_uid_to_recover_count = {"counter-rep0": 0}

    async def record(*args, **kwargs):
        seen.append(fence.bound_context().session)

    terminate = AsyncMock(side_effect=record)
    evict = AsyncMock(side_effect=record)
    monkeypatch.setattr(worker, "terminate_model", terminate)
    monkeypatch.setattr(worker, "_evict_replica_from_supervisor", evict)
    try:
        async with fence.command(command(old)):
            await worker._ensure_subpool_monitor()
        authority.owner = second = session(2)
        await worker.install_controller_session(second.to_dict())
        completed = asyncio.get_running_loop().create_future()
        await faults.put(("dead", completed))
        assert await asyncio.wait_for(completed, 2)
        terminate.assert_awaited_once_with("counter-rep0", is_model_die=True)
        evict.assert_awaited_once_with("counter-rep0")
        assert seen == [second, second]

        entered, release = asyncio.Event(), asyncio.Event()

        async def recover(address):
            seen.append(fence.bound_context().session)
            entered.set()
            await release.wait()

        monkeypatch.setattr(worker, "recover_sub_pool", recover)
        completed = asyncio.get_running_loop().create_future()
        await faults.put(("another-dead-process", completed))
        await asyncio.wait_for(entered.wait(), 2)
        authority.owner = third = session(3)
        installing = asyncio.create_task(
            worker.install_controller_session(third.to_dict())
        )
        while fence.session != third:
            if installing.done():
                await installing
            await asyncio.sleep(0)
        assert not installing.done()
        release.set()
        assert await asyncio.wait_for(completed, 2)
        assert (await asyncio.wait_for(installing, 2))["ready"]
        assert seen[-1] == second
        with fence.local_scope(command(old)):
            with pytest.raises(ControlSessionRejected, match="stale_session"):
                await worker.recover_sub_pool("stale-callback")
        with pytest.raises(ControlSessionRejected, match="untrusted_recovery_event"):
            await pool.recover_sub_pool("forged-fault")
        completed = asyncio.get_running_loop().create_future()
        await faults.put(("third-fault", completed))
        assert await asyncio.wait_for(completed, 2)
        assert seen[-1] == third
    finally:
        if monitor_task is not None:
            monitor_task.cancel()
            await asyncio.gather(monitor_task, return_exceptions=True)


@pytest.mark.parametrize("phase", ["fencing", "reconciling", "queued"])
@pytest.mark.asyncio
async def test_startup_cache_inventory_survives_until_active_callback(
    monkeypatch, phase
):
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    try:
        tracker = await xo.create_actor(
            CacheTrackerActor, address=pool.external_address
        )
        worker = WorkerActor("supervisor", None, SimpleNamespace(), [])
        worker.address = "worker:1234"
        worker._actor_loop = asyncio.get_running_loop()
        fence, authority = setup_fence(phase="active" if phase == "queued" else phase)
        worker._control_fence, worker._lifecycle = fence, fence.lifecycle
        supervisor = SimpleNamespace(
            add_worker=AsyncMock(), report_worker_status=AsyncMock()
        )
        worker._supervisor_ref = supervisor
        worker._supervisor_ref_address = "supervisor"

        async def actor_ref(*, address, uid):
            return (
                tracker if uid == CacheTrackerActor.default_uid() else SimpleNamespace()
            )

        monkeypatch.setattr(xo, "actor_ref", actor_ref)
        inventory = {
            "cached-model": [
                {
                    "model_version": "cached-v1",
                    "model_type": "embedding",
                    "cache_status": True,
                    "model_file_location": "/cache/existing",
                }
            ]
        }
        versions = MagicMock(side_effect=lambda: copy.deepcopy(inventory))
        monkeypatch.setattr(worker_module, "get_model_version_infos", versions)
        monkeypatch.setattr(worker_module, "gather_node_info", lambda: {})
        if phase == "queued":
            await fence.install(authority.owner, 1)
            async with fence.command(command(authority.owner)):
                await worker._get_supervisor_ref_with_generation(add_worker=True)
                versions.assert_not_called()
                authority.owner, authority.phase = session(2), "fencing"
        else:
            await worker._get_supervisor_ref_with_generation(add_worker=True)
        supervisor.add_worker.assert_awaited_once()
        assert not await tracker.get_model_versions("cached-model")
        await worker.install_controller_session(authority.owner.to_dict())
        assert not await tracker.get_model_versions("cached-model")
        authority.phase = "active"
        await run_worker_control_callback(worker, worker.report_status)
        entries = await tracker.get_model_versions("cached-model")
        assert entries[0]["model_file_location"] == {worker.address: "/cache/existing"}
        cached = await tracker.list_cached_models(worker.address)
        assert cached[0]["model_file_location"] == "/cache/existing"
        assert (
            await tracker.list_deletable_models("cached-v1", worker.address)
            == "/cache/existing"
        )
        versions.assert_called_once()
        await worker.install_controller_session(authority.owner.to_dict())
        await run_worker_control_callback(worker, worker.report_status)
        versions.assert_called_once()
    finally:
        await pool.stop()
        pool._actors.clear()
        Router.set_instance(None)


@pytest.mark.parametrize("local_ref", [False, True])
@pytest.mark.asyncio
async def test_model_oom_report_checks_current_incarnation_without_controller_context(
    monkeypatch, local_ref
):
    async def post_create(self):
        pass

    monkeypatch.setattr(WorkerActor, "__post_create__", post_create)
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    try:
        ref = await xo.create_actor(
            WorkerActor, "supervisor", None, pool, [], address=pool.external_address
        )
        worker = pool._actors[ref.uid]
        fence, authority = setup_fence()
        worker._control_fence, worker._lifecycle = fence, fence.lifecycle
        await ref.install_controller_session(authority.owner.to_dict())
        uid, token = "counter-rep0", "1" * 32
        worker._model_uid_to_model[uid] = SimpleNamespace()
        worker._model_uid_to_model_status[uid] = ModelStatus(model_state="ready")
        worker._model_uid_to_status_token = {uid: token}
        authority.owner = session(2)
        await ref.install_controller_session(authority.owner.to_dict())
        if local_ref:
            ref = await xo.actor_ref(address=ref.address, uid=ref.uid)
        model = ModelActor.__new__(ModelActor)
        model._model = object()
        model._replica_model_uid = uid
        model._worker_status_token = token
        model.model_uid = lambda: uid
        model._get_worker_ref = AsyncMock(return_value=ref)
        model.stop = AsyncMock()
        exits = []
        monkeypatch.setattr(model_module.os, "_exit", exits.append)
        await model._handle_oom_error(MemoryError("diagnostic-test"))
        status = await ref.get_model_status(uid)
        assert "out of memory" in status.last_error
        assert "diagnostic-test" in status.last_error
        assert exits == [1]
        model.stop.assert_awaited_once()
        with pytest.raises(Exception, match="out of memory"):
            await ref.get_model(uid)
        worker._model_uid_to_status_token[uid] = "2" * 32
        worker._model_uid_to_model_status[uid] = ModelStatus(model_state="ready")
        assert not await ref.report_model_oom(uid, token, "late old-model OOM")
        assert not await ref.report_model_oom(uid, "\u00e9" * 32, "invalid token")
        assert (await ref.get_model_status(uid)).last_error == ""
        with pytest.raises(ControlSessionRejected, match="missing_context"):
            await ref.update_model_status(uid, last_error="unscoped setter")
        assert await ref.report_model_oom(uid, "2" * 32, "current model OOM")
        assert (await ref.get_model_status(uid)).model_state == "ready"
    finally:
        await pool.stop()
        pool._actors.clear()
        Router.set_instance(None)


class LaunchCounterModel(CounterModel):
    """Real subprocess actor without a model engine/download dependency."""

    def __init__(self, worker_address, model, **kwargs):
        super().__init__(worker_address)
        self.cleanup_path = model.cleanup_path
        self._test_status_token = kwargs.get("worker_status_token")

    def get_status_token_for_test(self):
        return self._test_status_token

    async def load(self):
        pass

    async def wait_for_load(self):
        pass

    def need_create_pools(self):
        return False

    def get_pool_addresses(self):
        return []

    async def stop(self):
        if self.cleanup_path:
            Path(self.cleanup_path + ".entered").touch()
            while not Path(self.cleanup_path + ".release").exists():
                await asyncio.sleep(0.01)


class LaunchProgress(contextlib.AbstractContextManager):
    def __exit__(self, *args):
        pass

    def split_stages(self, *args, **kwargs):
        pass

    def activate_stage(self):
        pass

    def set_progress(self, *args, **kwargs):
        pass


def make_launch_worker(pool, tmp_path, monkeypatch, *, cleanup=False):
    """Retain production launch/terminate/persistence; replace engine I/O only."""
    worker = WorkerActor("supervisor", None, pool, [])
    worker.address = pool.external_address
    worker._actor_loop = asyncio.get_running_loop()
    fence, authority = setup_fence()
    worker._lifecycle = fence.lifecycle
    worker._control_fence = fence
    supervisor = SimpleNamespace(
        describe_model=AsyncMock(return_value={"model_name": "counter"}),
        can_recover_model=AsyncMock(return_value=True),
    )
    worker._supervisor_ref = supervisor
    worker._status_guard_ref = SimpleNamespace(
        update_instance_info=AsyncMock(), update_replica_info=AsyncMock()
    )
    monkeypatch.setattr(worker_module, "ModelActor", LaunchCounterModel)
    monkeypatch.setattr(worker, "_check_model_is_valid", lambda *args: None)
    monkeypatch.setattr(worker, "_create_virtual_env_manager", lambda *args: None)
    monkeypatch.setattr(
        worker, "_get_progressor", AsyncMock(return_value=LaunchProgress())
    )
    monkeypatch.setattr(
        worker, "_get_recovery_file_path", lambda: str(tmp_path / "models.json")
    )
    monkeypatch.setattr(
        worker,
        "_get_supervisor_ref_with_generation",
        AsyncMock(return_value=(supervisor, 1)),
    )
    monkeypatch.setattr(worker, "_get_model_ability", AsyncMock(return_value=["chat"]))
    family = SimpleNamespace(
        to_description=lambda: {"model_name": "counter", "model_type": "LLM"}
    )
    model = SimpleNamespace(
        model_family=family,
        cleanup_path=str(tmp_path / "cleanup") if cleanup else None,
    )
    monkeypatch.setattr(
        worker,
        "_download_model_files",
        AsyncMock(return_value=(model, SimpleNamespace(cancelled=False))),
    )

    async def allocate(model_uid, *args, **kwargs):
        # Exercise reservation release without requiring a physical GPU.
        worker._gpu_to_model_uids[0].add(model_uid)
        return {}, []

    monkeypatch.setattr(worker, "_allocate_subpool_devices", allocate)
    return worker, authority


def counter_launch_args(uid="counter-rep0"):
    return dict(
        model_uid=uid,
        model_name="counter",
        model_size_in_billions=None,
        model_format=None,
        quantization=None,
        model_engine=None,
        n_gpu=None,
        enable_virtual_env=False,
    )


@pytest.mark.parametrize("pause", ["status", "stop"])
@pytest.mark.asyncio
async def test_admitted_termination_finishes_cleanup_during_handoff(
    tmp_path, monkeypatch, pause
):
    import psutil

    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    release = asyncio.Event()
    try:
        worker, authority = make_launch_worker(
            pool, tmp_path, monkeypatch, cleanup=pause == "stop"
        )
        old, new = authority.owner, session(2)
        await worker.install_controller_session(old.to_dict())
        await worker.launch_builtin_model(
            **counter_launch_args(), **{CONTROL_CONTEXT_KEY: command(old).to_dict()}
        )
        uid = "counter-rep0"
        pid = worker._model_uid_to_pid[uid]
        entered = asyncio.Event()

        async def status(*args):
            entered.set()
            await release.wait()

        if pause == "status":
            worker._status_guard_ref.update_instance_info = status
        terminating = asyncio.create_task(
            worker.terminate_model(uid, **{CONTROL_CONTEXT_KEY: command(old).to_dict()})
        )
        if pause == "status":
            await asyncio.wait_for(entered.wait(), 10)
        else:

            async def stopping():
                while not (tmp_path / "cleanup.entered").exists():
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(stopping(), 10)
        authority.owner = new
        installing = asyncio.create_task(
            worker.install_controller_session(new.to_dict())
        )
        while worker._control_fence.session != new:
            await asyncio.sleep(0)
        assert not installing.done()
        with pytest.raises(ControlSessionRejected, match="stale_session"):
            await worker.terminate_model(
                uid, **{CONTROL_CONTEXT_KEY: command(old).to_dict()}
            )
        release.set()
        (tmp_path / "cleanup.release").touch()
        await asyncio.wait_for(terminating, 10)
        assert (await asyncio.wait_for(installing, 10))["ready"]
        assert not psutil.pid_exists(pid) and not pool.sub_processes
        for name in (
            "model",
            "addr",
            "launch_args",
            "model_spec",
            "model_status",
            "status_token",
            "pid",
            "subpool_addresses",
            "subpool_pids",
        ):
            assert uid not in getattr(worker, f"_model_uid_to_{name}")
        assert uid not in worker._gpu_to_model_uids[0]
        assert uid not in json.loads((tmp_path / "models.json").read_text())
        current = {CONTROL_CONTEXT_KEY: command(new).to_dict()}
        await worker.launch_builtin_model(**counter_launch_args(), **current)
        await worker.terminate_model(uid, **current)
    finally:
        release.set()
        (tmp_path / "cleanup.release").touch()
        await pool.stop()
        pool._actors.clear()
        Router.set_instance(None)


@pytest.mark.parametrize(
    "pause,recover,continuation",
    [
        ("remove", False, None),
        ("remove", True, None),
        ("cleanup", False, None),
        ("cleanup", True, None),
        ("launch", True, None),
        ("cleanup", True, "delete"),
        ("cleanup", True, "replace"),
        ("cleanup", True, "status"),
    ],
)
@pytest.mark.asyncio
async def test_native_fault_intent_survives_inflight_handoff(
    tmp_path, monkeypatch, pause, recover, continuation
):
    import psutil

    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    release = asyncio.Event()
    try:
        worker, authority = make_launch_worker(pool, tmp_path, monkeypatch)
        old, new = authority.owner, session(2)
        await worker.install_controller_session(old.to_dict())
        await worker.launch_builtin_model(
            **counter_launch_args(), **{CONTROL_CONTEXT_KEY: command(old).to_dict()}
        )
        uid = "counter-rep0"
        pid = worker._model_uid_to_pid[uid]
        address = worker._model_uid_to_addr[uid]
        worker._model_uid_to_recover_count[uid] = 1 if recover else 0
        worker._supervisor_ref.mark_replica_dead = AsyncMock()
        entered = asyncio.Event()
        target, name = {
            "remove": (pool, "remove_sub_pool"),
            "cleanup": (worker, "_terminate_model"),
            "launch": (worker, "_get_model_ability"),
        }[pause]
        original = getattr(target, name)

        async def paused(*args, **kwargs):
            result = await original(*args, **kwargs)
            if not entered.is_set():
                entered.set()
                await release.wait()
            return result

        monkeypatch.setattr(target, name, paused)
        psutil.Process(pid).kill()
        await asyncio.wait_for(entered.wait(), 20)
        assert address not in pool.sub_processes
        authority.owner = new
        if continuation is not None:
            authority.phase = "fencing"
        installing = asyncio.create_task(
            worker.install_controller_session(new.to_dict())
        )
        while worker._control_fence.session != new:
            await asyncio.sleep(0)
        assert not installing.done()
        release.set()
        assert (await asyncio.wait_for(installing, 20))["ready"]
        if continuation is not None:
            assert worker._pending_control_faults
            authority.phase = "active"
            current = {CONTROL_CONTEXT_KEY: command(new).to_dict()}
            if continuation == "delete":
                await worker.terminate_model(uid, **current)
            elif continuation == "replace":
                await worker.launch_builtin_model(**counter_launch_args(), **current)
                await worker.wait_for_load(uid, **current)
                replacement = worker._model_uid_to_model[uid]
            if continuation == "status":
                monkeypatch.setattr(worker, "_call_supervisor", AsyncMock())
                monkeypatch.setattr(worker_module, "gather_node_info", lambda: {})
                await worker.report_status(**current)
            else:
                await worker.install_controller_session(new.to_dict())
            if continuation == "delete":
                assert uid not in worker._model_uid_to_model
                assert not worker._pending_control_faults and not pool.sub_processes
                worker._supervisor_ref.mark_replica_dead.assert_not_awaited()
                return
            if continuation == "replace":
                assert worker._model_uid_to_model[uid] is replacement
        if recover:
            restored = worker._model_uid_to_model[uid]
            value, new_pid, _ = await restored.add(1)
            assert value == 1 and new_pid != pid
            assert worker._model_uid_to_model_status[uid].model_state == "ready"
            worker._supervisor_ref.mark_replica_dead.assert_not_awaited()
        else:
            assert uid not in worker._model_uid_to_model
            assert uid not in worker._gpu_to_model_uids[0]
            assert not pool.sub_processes
            worker._supervisor_ref.mark_replica_dead.assert_awaited_once_with(uid)
        assert not worker._pending_control_faults
    finally:
        release.set()
        await pool.stop()
        pool._actors.clear()
        Router.set_instance(None)


@pytest.mark.parametrize("recover", [False, True], ids=["evict", "recover"])
@pytest.mark.asyncio
async def test_real_native_monitor_handles_fault_after_session_change(
    tmp_path, monkeypatch, recover
):
    import psutil

    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    try:
        worker, authority = make_launch_worker(pool, tmp_path, monkeypatch)
        old = authority.owner
        await worker.install_controller_session(old.to_dict())
        await worker.launch_builtin_model(
            **counter_launch_args(), **{CONTROL_CONTEXT_KEY: command(old).to_dict()}
        )
        uid = "counter-rep0"
        worker._model_uid_to_recover_count[uid] = 1 if recover else 0
        old_token = worker._model_uid_to_status_token[uid]
        pid = worker._model_uid_to_pid[uid]
        assert pid != os.getpid()
        event = asyncio.Event()
        sessions = []

        async def mark_dead(model_uid):
            assert model_uid == uid
            sessions.append(worker._control_fence.bound_context().session)
            event.set()

        worker._supervisor_ref.mark_replica_dead = AsyncMock(side_effect=mark_dead)
        authority.owner = current = session(2)
        await worker.install_controller_session(current.to_dict())
        psutil.Process(pid).kill()
        if recover:

            async def ready():
                while (
                    worker._model_uid_to_pid.get(uid) in (None, pid)
                    or getattr(
                        worker._model_uid_to_model_status.get(uid), "model_state", None
                    )
                    != "ready"
                ):
                    await asyncio.sleep(0.02)

            await asyncio.wait_for(ready(), 20)
            restored = worker._model_uid_to_model[uid]
            value, new_pid, _ = await restored.add(1)
            assert value == 1 and new_pid != pid
            assert worker._model_uid_to_status_token[uid] != old_token
            assert (
                await restored.get_status_token_for_test()
                == worker._model_uid_to_status_token[uid]
            )
            assert not worker.report_model_oom(uid, old_token, "late old-model OOM")
            worker._supervisor_ref.mark_replica_dead.assert_not_awaited()
        else:
            await asyncio.wait_for(event.wait(), 20)
            assert sessions == [current]
            assert uid not in worker._model_uid_to_model
            assert uid not in worker._model_uid_to_status_token
            assert not pool.sub_processes
    finally:
        await pool.stop()
        pool._actors.clear()
        Router.set_instance(None)


@pytest.mark.asyncio
async def test_production_launch_rolls_back_before_new_session_ack(
    tmp_path, monkeypatch
):
    import psutil

    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    try:
        worker, authority = make_launch_worker(
            pool, tmp_path, monkeypatch, cleanup=True
        )
        old, new = authority.owner, session(2)
        await worker.install_controller_session(old.to_dict())
        entered, release = asyncio.Event(), asyncio.Event()

        async def ability(*args):
            entered.set()
            await release.wait()
            return ["chat"]

        monkeypatch.setattr(worker, "_get_model_ability", ability)
        uid = "counter-rep0"
        launching = asyncio.create_task(
            worker.launch_builtin_model(
                **counter_launch_args(),
                **{CONTROL_CONTEXT_KEY: command(old).to_dict()},
            )
        )
        await asyncio.wait_for(entered.wait(), 20)
        ref = worker._model_uid_to_model[uid]
        old_status_token = worker._model_uid_to_status_token[uid]
        assert await ref.get_status_token_for_test() == old_status_token
        pid = await ref.get_pid()
        assert pid != os.getpid() and psutil.pid_exists(pid)
        assert uid in json.loads((tmp_path / "models.json").read_text())
        authority.owner = new
        installing = asyncio.create_task(
            worker.install_controller_session(new.to_dict())
        )
        while worker._control_fence.session != new:
            await asyncio.sleep(0)
        release.set()

        async def wait_cleanup():
            while not (tmp_path / "cleanup.entered").exists():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_cleanup(), 10)
        assert not installing.done()
        assert worker._model_uid_to_model[uid] is ref
        assert uid in worker._gpu_to_model_uids[0]
        # Normal stale termination cannot acquire the rollback privilege.
        with pytest.raises(ControlSessionRejected, match="stale_session"):
            await worker.terminate_model(
                uid, **{CONTROL_CONTEXT_KEY: command(old).to_dict()}
            )
        (tmp_path / "cleanup.release").touch()
        with pytest.raises(ControlSessionRejected, match="stale_session"):
            await asyncio.wait_for(launching, 10)
        assert (await asyncio.wait_for(installing, 10))["ready"]
        assert not psutil.pid_exists(pid)
        assert not pool.sub_processes
        for table in (
            worker._model_uid_to_model,
            worker._model_uid_to_addr,
            worker._model_uid_to_launch_args,
            worker._model_uid_to_model_spec,
            worker._model_uid_to_model_status,
            worker._model_uid_to_status_token,
            worker._model_uid_to_pid,
            worker._model_uid_to_subpool_addresses,
            worker._model_uid_to_subpool_pids,
        ):
            assert uid not in table
        assert uid not in worker._gpu_to_model_uids[0]
        assert uid not in json.loads((tmp_path / "models.json").read_text())
        current = {CONTROL_CONTEXT_KEY: command(new).to_dict()}
        await worker.launch_builtin_model(**counter_launch_args(), **current)
        assert worker._model_uid_to_status_token[uid] != old_status_token
        assert not worker.report_model_oom(uid, old_status_token, "late OOM")
        # A delayed cleanup must not touch this newer registration.
        async with worker._control_fence.command(command(new)):
            with pytest.raises(
                ControlSessionRejected, match="cleanup_resource_changed"
            ):
                await worker._terminate_model(uid, True, failed_launch_ref=ref)
        await worker.terminate_model(uid, True, **current)
    finally:
        await pool.stop()
        pool._actors.clear()
        Router.set_instance(None)


@pytest.mark.parametrize("interruption", ["none", "fence", "cancel"])
@pytest.mark.asyncio
async def test_startup_recovery_waits_for_explicit_active_session(
    tmp_path, monkeypatch, interruption
):
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    try:
        worker, authority = make_launch_worker(pool, tmp_path, monkeypatch)
        saved = {
            "counter-rep0": counter_launch_args(),
            "other-rep0": counter_launch_args("other-rep0"),
        }
        path = tmp_path / "models.json"
        path.write_text(json.dumps(saved))
        monkeypatch.setattr(worker, "_start_metrics_service", lambda: None)
        monkeypatch.setattr(worker_module, "purge_dir", lambda *args: None)
        monkeypatch.setattr(worker_module, "XINFERENCE_DISABLE_HEALTH_CHECK", True)
        monkeypatch.setattr(
            asyncio.get_running_loop(), "add_signal_handler", lambda *args: None
        )
        await worker._activate_worker(recover=True)
        assert worker._pending_recovery_launch_args == saved
        assert json.loads(path.read_text()) == saved
        assert not pool.sub_processes
        # A different persistence trigger must retain the pending snapshot.
        worker._persist_launch_args()
        assert json.loads(path.read_text()) == saved
        authority.phase = "fencing"
        assert (await worker.install_controller_session(authority.owner.to_dict()))[
            "ready"
        ]
        assert not pool.sub_processes and json.loads(path.read_text()) == saved
        authority.phase = "active"
        if interruption != "none":
            entered, release = asyncio.Event(), asyncio.Event()

            async def ability(*args):
                entered.set()
                await release.wait()
                return ["chat"]

            monkeypatch.setattr(worker, "_get_model_ability", ability)
            resuming = asyncio.create_task(
                worker.install_controller_session(authority.owner.to_dict())
            )
            await asyncio.wait_for(entered.wait(), 20)
            if interruption == "cancel":
                resuming.cancel()
                await asyncio.sleep(0)
                assert not resuming.done()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(resuming, 20)
                assert not worker._pending_recovery_launch_args
                assert set(worker._model_uid_to_model) == set(saved)
            else:
                authority.owner, authority.phase = session(2), "fencing"
                fencing = asyncio.create_task(
                    worker.install_controller_session(authority.owner.to_dict())
                )
                while worker._control_fence.session != authority.owner:
                    await asyncio.sleep(0)
                release.set()
                await asyncio.wait_for(resuming, 10)
                await asyncio.wait_for(fencing, 10)
                assert not pool.sub_processes
                assert worker._pending_recovery_launch_args == saved
                assert json.loads(path.read_text()) == saved
                authority.phase = "active"
        await worker.install_controller_session(authority.owner.to_dict())
        assert not worker._pending_recovery_launch_args
        assert "counter-rep0" in await worker.list_models()
        assert worker._model_uid_to_model_status["counter-rep0"].model_state == "ready"
        assert "counter-rep0" in json.loads(path.read_text())
        await worker.terminate_model(
            "counter-rep0",
            True,
            **{CONTROL_CONTEXT_KEY: command(authority.owner).to_dict()},
        )
        await worker.terminate_model(
            "other-rep0",
            True,
            **{CONTROL_CONTEXT_KEY: command(authority.owner).to_dict()},
        )
    finally:
        await pool.stop()
        pool._actors.clear()
        Router.set_instance(None)


@pytest.mark.asyncio
async def test_ready_registration_recovers_persisted_models_after_releasing_lock(
    tmp_path, monkeypatch
):
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    worker = None
    try:
        worker, authority = make_launch_worker(pool, tmp_path, monkeypatch)
        saved = {"counter-rep0": counter_launch_args()}
        path = tmp_path / "models.json"
        path.write_text(json.dumps(saved))
        supervisor = worker._supervisor_ref
        # Registration can remove the last live route. Recovery still consults
        # the real Supervisor's retained unexpected-loss record, not inference.
        live = SupervisorActor()
        live._unexpected_down_replicas[("counter", 0)] = "counter"
        from ..supervisor_control import SupervisorControl

        provider = SimpleNamespace(session=authority.owner, validate=authority.validate)
        live._controller_control = SupervisorControl(live, provider)
        live._controller_control.fence.ready = True
        live._controller_control.has_served = True
        supervisor.describe_model = live.describe_model
        supervisor.can_recover_model = live.can_recover_model
        worker._supervisor_init_lock = asyncio.Lock()
        monkeypatch.setattr(worker, "_start_metrics_service", lambda: None)
        monkeypatch.setattr(worker, "_cleanup_gpu_orphans_on_startup", AsyncMock())
        monkeypatch.setattr(worker_module, "purge_dir", lambda *args: None)
        monkeypatch.setattr(worker_module, "XINFERENCE_DISABLE_HEALTH_CHECK", True)
        monkeypatch.setattr(
            asyncio.get_running_loop(), "add_signal_handler", lambda *args: None
        )

        async def connect(add_worker):
            if not worker._registered:
                async with worker._supervisor_init_lock:
                    await worker.install_controller_session(authority.owner.to_dict())
                    assert not worker._pending_recovery_launch_args
                    worker._registered = True
            return supervisor, 1

        monkeypatch.setattr(worker, "_get_supervisor_ref_with_generation", connect)
        await asyncio.wait_for(worker._activate_worker(recover=True), 15)
        assert not worker._pending_recovery_launch_args
        assert worker._model_uid_to_model_status["counter-rep0"].model_state == "ready"
        assert "counter-rep0" in json.loads(path.read_text())
        await worker.terminate_model(
            "counter-rep0",
            True,
            **{CONTROL_CONTEXT_KEY: command(authority.owner).to_dict()},
        )
    finally:
        if worker is not None:
            worker._lifecycle.stop_background()
        await pool.stop()
        pool._actors.clear()
        Router.set_instance(None)


@pytest.mark.asyncio
async def test_fenced_description_keeps_recovery_intent_for_retry(
    tmp_path, monkeypatch
):
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    try:
        worker, authority = make_launch_worker(pool, tmp_path, monkeypatch)
        saved = {"counter-rep0": counter_launch_args()}
        path = tmp_path / "models.json"
        path.write_text(json.dumps(saved))
        worker._pending_recovery_launch_args = saved.copy()
        await worker._control_fence.install(authority.owner, 1)
        worker._supervisor_ref.describe_model.side_effect = ControlSessionRejected(
            "fence_pending"
        )
        await worker._resume_pending_recovery()
        assert worker._pending_recovery_launch_args == saved
        assert json.loads(path.read_text()) == saved
        assert not pool.sub_processes
    finally:
        await pool.stop()
        pool._actors.clear()
        Router.set_instance(None)
