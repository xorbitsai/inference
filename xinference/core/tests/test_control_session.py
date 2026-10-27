# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
import asyncio
import contextlib
import pickle
import sys
import threading
from dataclasses import FrozenInstanceError, replace
from types import ModuleType, SimpleNamespace

import pytest
import xoscar as xo
from xoscar.backends.router import Router

from xinference import extensions

from ..control_session import (
    CONTROL_CONTEXT_KEY,
    READ_ONLY_WORKER_RPCS,
    ControlContext,
    ControllerSession,
    ControlSessionRejected,
    WorkerControlFence,
    check_control_session,
)
from ..lifecycle import (
    WorkerLifecycle,
    WorkerPhase,
    management_cleanup,
    management_operation,
    spawn_management_task,
)
from ..worker import WorkerActor
from .test_worker_handoff import HandoffSupervisor, HandoffWorker


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
    owner = SimpleNamespace(_control_fence=fence)
    installing = asyncio.create_task(
        WorkerActor.install_controller_session(owner, new.to_dict(), 1)
    )
    while fence.session != new:
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
    await owner.work()
    assert owner.released
    authority.unavailable = True
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
        old_kwargs = {CONTROL_CONTEXT_KEY: command(old, "launch").to_dict()}
        model = await worker.install_model(**old_kwargs)
        value, pid, _ = await model.add(5)
        new = session(2)
        await authority.select(new.to_dict(), "fencing")
        await worker.install_controller_session(new.to_dict())
        assert await worker.ping()
        assert (await model.add(1))[:2] == (value + 1, pid)
        for kwargs, reason in (({}, "missing_context"), (old_kwargs, "stale_session")):
            with pytest.raises(ControlSessionRejected) as rejected:
                await worker.terminate_model("counter-rep0", **kwargs)
            assert rejected.value.reason == reason
        assert (await model.add(1))[:2] == (value + 2, pid)
        current = {CONTROL_CONTEXT_KEY: command(new, "terminate").to_dict()}
        with pytest.raises(ControlSessionRejected, match="authority_rejected"):
            await worker.future_mutation(**current)
        await authority.select(new.to_dict())
        assert await worker.future_mutation(**current) == "mutated"
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
