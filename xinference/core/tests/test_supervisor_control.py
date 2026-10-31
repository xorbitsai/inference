# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
import asyncio
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import xoscar as xo

from xinference import extensions

from ..control_session import CONTROL_CONTEXT_KEY, ControlSessionRejected
from ..supervisor import SupervisorActor
from ..supervisor_control import SupervisorControl, SupervisorControlPort
from ..worker import WorkerActor, _bounded_control_report
from .test_control_session import command, session, setup_fence


class Provider:
    def __init__(self, state, owner):
        self.session, self.state = owner, state

    async def validate(self, owner, *, phases):
        if self.state.failed:
            raise RuntimeError("private connection must not leak")
        if owner != self.state.owner or self.state.phase not in phases:
            raise ControlSessionRejected("stale_session")

    async def close(self):
        pass


class InterceptedSupervisor(SupervisorActor):
    def __init__(self):
        xo.StatelessActor.__init__(self)
        state = SimpleNamespace(owner=session(), phase="active", failed=False)
        authority = Provider(state, state.owner)
        authority.around_supervisor_call = self._intercept
        self._controller_control = SupervisorControl(self, authority)
        self._controller_control.fence.ready = True
        self._controller_control.has_served = True
        self.events = []

    async def __post_create__(self):
        pass

    async def __pre_destroy__(self):
        pass

    async def _intercept(self, invocation, call):
        self.events.append(("entry", invocation.method, invocation.command.to_dict()))
        try:
            result = await call()
        except Exception:
            self.events.append(("error", invocation.method))
            raise
        self.events.append(("complete", invocation.method))
        return result

    def nested_cleanup(self, uid):
        return uid

    async def mark_replica_dead(self, uid):
        assert self.nested_cleanup(uid) == uid
        if uid == "fail":
            raise ValueError("Callback failed")
        return uid

    def get_status(self):
        return self.events


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["local", "remote", "port"])
@pytest.mark.parametrize("uid", ["model-rep0", "fail"])
async def test_supervisor_intercepts_worker_callbacks_once(transport, uid):
    from xoscar.core import LocalActorRef

    from ..control_session import SessionBoundSupervisorRef

    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    try:
        ref = await xo.create_actor(
            InterceptedSupervisor, address=pool.external_address, uid="supervisor"
        )
        fence, authority = setup_fence()
        await fence.install(authority.owner, 1)
        original = command(authority.owner, "worker-callback")
        actor = pool._actors[ref.uid]
        if transport == "port":
            call = SupervisorControlPort(actor).call("mark_replica_dead", uid)
        else:
            target = (
                xo.ActorRef(ref.address, ref.uid)
                if transport == "remote"
                else LocalActorRef(actor)
            )
            with fence.local_scope(original):
                call = SessionBoundSupervisorRef(target, fence).mark_replica_dead(uid)
        if uid == "fail":
            with pytest.raises(ValueError, match="Callback failed"):
                await call
        else:
            assert await call == uid
        events = await ref.get_status()
        if transport == "port":
            assert events == []
        else:
            assert events == [
                ("entry", "mark_replica_dead", original.to_dict()),
                ("error" if uid == "fail" else "complete", "mark_replica_dead"),
            ]
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_persistent_monitor_does_not_inherit_launch_hook_suppression():
    from ..control_invocation import ControlInvocation, invoke_control_hook

    seen = []
    wake = asyncio.Event()
    monitor = None
    authority = SimpleNamespace()
    first = ControlInvocation(
        "worker:1234", "launch_builtin_model", command(session()), (), {}
    )
    second = ControlInvocation(
        "worker:1234", "launch_builtin_model", command(session(), "recovery"), (), {}
    )

    async def intercept(invocation, call):
        seen.append(invocation.command.operation_id)
        return await call()

    authority.around_control_call = intercept

    async def recover():
        await wake.wait()
        return await invoke_control_hook(
            authority,
            "around_control_call",
            second,
            AsyncMock(return_value="recovered"),
        )

    async def launch():
        nonlocal monitor
        monitor = asyncio.create_task(recover())
        return await invoke_control_hook(
            authority, "around_control_call", first, AsyncMock(return_value="launched")
        )

    try:
        assert (
            await invoke_control_hook(authority, "around_control_call", first, launch)
            == "launched"
        )
        wake.set()
        assert await monitor == "recovered"
        assert seen == ["operation", "recovery"]
    finally:
        if monitor is not None:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)


@pytest.mark.asyncio
async def test_worker_proxy_captures_original_session_and_never_borrows_successor():
    old = session()
    state = SimpleNamespace(owner=old, phase="active", failed=False)
    control = SupervisorControl(SimpleNamespace(), Provider(state, old))
    control.fence.ready = True
    ref = SimpleNamespace(
        address="worker:1234",
        uid=b"worker",
        terminate_model=AsyncMock(),
        list_models=AsyncMock(return_value={}),
    )
    wrapped = control.worker(ref)
    await wrapped.terminate_model("model")
    assert (
        ref.terminate_model.await_args.kwargs[CONTROL_CONTEXT_KEY]["session"]
        == old.to_dict()
    )
    delayed = wrapped.terminate_model("model")
    state.owner = session(2)
    with pytest.raises(ControlSessionRejected, match="stale_session"):
        await delayed
    assert await wrapped.list_models() == {}
    with pytest.raises(ControlSessionRejected, match="caller_control_override"):
        wrapped.terminate_model(
            "model", **{CONTROL_CONTEXT_KEY: command(state.owner).to_dict()}
        )
    state.owner, state.failed = old, True
    with pytest.raises(ControlSessionRejected, match="authority_unavailable"):
        await wrapped.terminate_model("model")
    ref.terminate_model.assert_awaited_once()


def test_supervisor_hook_is_optional_exclusive_and_validated(monkeypatch):
    context = extensions.SupervisorControlAuthorityContext("supervisor:1234")
    monkeypatch.setattr(extensions, "get_extensions", lambda: ())
    assert extensions.resolve_supervisor_control_authority(context) is None
    bad = SimpleNamespace(
        name="bad", create_supervisor_control_authority=lambda c: object()
    )
    monkeypatch.setattr(extensions, "get_extensions", lambda: (bad,))
    with pytest.raises(TypeError):
        extensions.resolve_supervisor_control_authority(context)
    monkeypatch.setattr(extensions, "get_extensions", lambda: (bad, bad))
    with pytest.raises(ValueError, match="Multiple"):
        extensions.resolve_supervisor_control_authority(context)


def controlled_owner():
    state = SimpleNamespace(owner=session(), phase="fencing", failed=False)
    owner = SimpleNamespace(
        _upgrade_state=SimpleNamespace(
            _require_idle=Mock(), pause=AsyncMock(), resume=Mock()
        ),
        _activate_supervisor=AsyncMock(),
        _schedule_autostart=Mock(),
        _worker_address_to_worker={},
        _ownership_registry=Mock(return_value=SimpleNamespace(incarnations={})),
        _restore_worker_inventory=AsyncMock(),
        _system_settings={"example": True},
    )
    owner._controller_control = SupervisorControl(owner, Provider(state, state.owner))
    return owner, state, SupervisorControlPort(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [float("nan"), 0, True, [], None])
async def test_invalid_pause_preserves_admission(timeout):
    owner, _, port = controlled_owner()
    owner._controller_control.fence.ready = True
    with pytest.raises(ValueError):
        await port.pause(timeout)
    assert port.status()["ready"]


@pytest.mark.asyncio
async def test_extension_port_only_applies_requested_local_primitives():
    owner, state, port = controlled_owner()
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(port)
    ref = SimpleNamespace(address="worker:1234", uid=b"worker")
    with pytest.raises(ControlSessionRejected, match="control_not_paused"):
        await port.restore_worker(ref, [], "worker-boot")
    await port.pause(1)
    assert not port.status()["ready"]
    await port.restore_worker(ref, [], "worker-boot")
    assert owner._ownership_registry().incarnations == {ref.address: "worker-boot"}
    owner._restore_worker_inventory.assert_awaited_once()
    # Runtime primitives never decide whether authority can become active.
    with pytest.raises(ControlSessionRejected, match="stale_session"):
        await port.resume()
    owner._activate_supervisor.assert_not_awaited()
    state.phase = "active"
    assert (await port.resume())["ready"]
    owner._activate_supervisor.assert_awaited_once()
    owner._upgrade_state.resume.assert_called_once()
    port.close_admission()
    assert not port.status()["ready"]


@pytest.mark.asyncio
async def test_busy_local_pause_preserves_admission():
    owner, _, port = controlled_owner()
    owner._controller_control.fence.ready = True
    owner._upgrade_state._require_idle.side_effect = RuntimeError("busy")
    with pytest.raises(RuntimeError, match="busy"):
        await port.pause(1)
    assert port.status()["ready"]
    assert not owner._controller_control.prepared


@pytest.mark.asyncio
async def test_generic_operation_requires_provider_and_does_not_implement_workflows():
    assert not hasattr(SupervisorActor, "reconcile_controller_workers")
    with pytest.raises(ControlSessionRejected, match="authority_not_configured"):
        await SupervisorActor.supervisor_extension_operation(
            SimpleNamespace(), "example", {}
        )
    owner, state, _ = controlled_owner()
    control = owner._controller_control
    with pytest.raises(ControlSessionRejected, match="extension_operation_unsupported"):
        await control.operation("example", {})
    control.authority.execute = AsyncMock(return_value={"ok": True})
    assert await control.operation("example", {"value": 1}) == {"ok": True}
    control.authority.execute.assert_awaited_once_with("example", {"value": 1})
    state.failed = True
    with pytest.raises(ControlSessionRejected, match="authority_unavailable"):
        await control.operation("example", {})
    control.authority.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_registration_delegates_without_installing_or_probing_workers():
    owner, _, _ = controlled_owner()
    control = owner._controller_control
    ref = SimpleNamespace(
        address="worker:1234",
        uid=b"worker",
        install_controller_session=AsyncMock(),
        controller_inventory=AsyncMock(),
    )
    control.authority.worker_registered = AsyncMock()
    assert (await control.register(ref)).address == ref.address
    control.authority.worker_registered.assert_awaited_once_with(ref)
    ref.install_controller_session.assert_not_awaited()
    ref.controller_inventory.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_endpoint_does_not_wait_for_or_publish_obsolete_handshake(
    monkeypatch,
):
    fence, authority = setup_fence()
    worker = WorkerActor("old:1234", None, SimpleNamespace(), [])
    worker._control_fence, worker._lifecycle = fence, fence.lifecycle
    entered, release = asyncio.Event(), asyncio.Event()

    async def old_protocol():
        entered.set()
        await release.wait()
        return 1

    old_ref = SimpleNamespace(worker_handoff_protocol=old_protocol)
    new_ref = SimpleNamespace(worker_handoff_protocol=AsyncMock(return_value=1))
    monkeypatch.setattr(
        xo,
        "actor_ref",
        AsyncMock(
            side_effect=lambda *, address, uid: (
                old_ref if address == "old:1234" else new_ref
            )
        ),
    )
    obsolete = asyncio.create_task(worker._get_supervisor_ref_on_actor_loop(False))
    try:
        await entered.wait()
        await WorkerActor.install_controller_session(
            worker, authority.owner.to_dict(), supervisor_address="new:1234"
        )
        current, generation = await asyncio.wait_for(
            worker._get_supervisor_ref_on_actor_loop(False), 1
        )
        assert current is new_ref and not obsolete.done()
        release.set()
        with pytest.raises(RuntimeError, match="address changed"):
            await obsolete
        assert worker._supervisor_ref is new_ref
        assert worker._supervisor_ref_generation == generation
    finally:
        release.set()
        await asyncio.gather(obsolete, return_exceptions=True)


@pytest.mark.asyncio
async def test_suspended_status_reply_does_not_block_drain_or_accumulate_retries():
    fence, authority = setup_fence()
    await fence.install(authority.owner, 1)
    owner = SimpleNamespace(_control_fence=fence, _supervisor_address="old-endpoint")
    cancelled, release = asyncio.Event(), asyncio.Event()

    async def report():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()  # Simulate waiting for a remote cancellation ACK.

    try:
        async with fence.command(command(authority.owner)):
            with pytest.raises(asyncio.TimeoutError):
                await _bounded_control_report(owner, "old-endpoint", report, 0.01)
        await cancelled.wait()
        with pytest.raises(asyncio.TimeoutError):
            await _bounded_control_report(owner, "old-endpoint", report, 0.01)
        assert len(owner._control_report_calls) == 1
        authority.owner = session(2)
        assert (await fence.install(authority.owner, 0.1))["ready"]
        assert await _bounded_control_report(
            owner, "new-endpoint", AsyncMock(return_value=True), 0.1
        )
    finally:
        release.set()
        await asyncio.gather(*list(owner._control_report_calls.values()))


@pytest.mark.asyncio
async def test_management_pause_preserves_routes_after_first_activation():
    owner, state, port = controlled_owner()
    receiver = SupervisorActor.__new__(SupervisorActor)
    receiver._controller_control = owner._controller_control
    receiver.get_model = AsyncMock(return_value="existing-route")
    with pytest.raises(ControlSessionRejected):
        await receiver.get_model("model")
    await port.pause(1)
    state.phase = "active"
    await port.resume()
    owner._upgrade_state.pause.side_effect = RuntimeError("busy")
    with pytest.raises(RuntimeError, match="busy"):
        await port.pause(1)
    assert not port.status()["ready"]
    assert await receiver.get_model("model") == "existing-route"
    settings = port.settings()
    settings["example"] = False
    assert owner._system_settings["example"]


@pytest.mark.asyncio
async def test_busy_pause_leaves_existing_operation_and_cleanup_admitted():
    owner, state, port = controlled_owner()
    control = owner._controller_control
    control.fence.ready, state.phase = True, "active"
    entered, release, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def launch():
        async with control.fence.command(control.capture()):
            entered.set()
            await release.wait()
            async with control.fence.command(control.capture()):
                cleaned.set()

    work = asyncio.create_task(launch())
    await entered.wait()
    with pytest.raises(RuntimeError, match="active management"):
        await port.pause(1)
    assert control.ready
    release.set()
    await work
    await port.pause(1)
    assert cleaned.is_set() and not control.ready


@pytest.mark.asyncio
async def test_registration_is_not_subject_to_the_report_timeout(monkeypatch):
    from .. import worker as module

    fence, authority = setup_fence()
    await fence.install(authority.owner, 1)
    worker = WorkerActor("supervisor:1234", None, SimpleNamespace(), [])
    worker._control_fence = fence
    supervisor = SimpleNamespace(report_worker_status=AsyncMock(return_value=True))
    entered, release = asyncio.Event(), asyncio.Event()

    async def register(_):
        entered.set()
        await release.wait()
        return supervisor, 1

    worker._get_supervisor_ref_with_generation = register
    monkeypatch.setattr(module, "XINFERENCE_TCP_REQUEST_TIMEOUT", 0.01)
    task = asyncio.create_task(
        worker._call_supervisor("report_worker_status", {}, add_worker=True)
    )
    await entered.wait()
    await asyncio.sleep(0.04)
    assert not task.done()
    supervisor.report_worker_status.assert_not_awaited()
    release.set()
    assert await task


@pytest.mark.asyncio
async def test_concurrent_reports_join_a_live_read_instead_of_failing():
    fence, _ = setup_fence()
    owner = SimpleNamespace(_control_fence=fence, _supervisor_address="supervisor:1234")
    entered, release = asyncio.Event(), asyncio.Event()

    async def report():
        entered.set()
        await release.wait()
        return True

    call = AsyncMock(side_effect=report)
    first = asyncio.create_task(_bounded_control_report(owner, "report", call, 1))
    await entered.wait()
    second = asyncio.create_task(_bounded_control_report(owner, "report", call, 1))
    await asyncio.sleep(0)
    release.set()
    assert await first and await second
    call.assert_awaited_once()


@pytest.mark.asyncio
async def test_health_authority_callbacks_return_to_the_actor_loop():
    from xinference.isolation import Isolation

    loop = asyncio.get_running_loop()
    calls = []

    async def check():
        assert asyncio.get_running_loop() is loop
        future = loop.create_future()
        loop.call_soon(future.set_result, True)
        await future
        calls.append(True)

    owner = SimpleNamespace(
        _actor_loop=loop, _controller_control=SimpleNamespace(check=check)
    )
    isolation = Isolation(asyncio.new_event_loop(), threaded=True)
    isolation.start()
    try:
        future = asyncio.run_coroutine_threadsafe(
            SupervisorActor._control_available(owner), isolation.loop
        )
        assert await asyncio.wrap_future(future)
        assert calls == [True]
    finally:
        await asyncio.to_thread(isolation.stop)
        isolation.loop.close()


@pytest.mark.asyncio
async def test_rejected_provider_registration_removes_stale_capacity(monkeypatch):
    monkeypatch.setattr(extensions, "get_extensions", lambda: ())
    worker = SimpleNamespace(
        get_extension_resources=AsyncMock(return_value={"extensions": {}})
    )
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=worker))
    owner = SupervisorActor()
    owner._worker_address_to_worker["worker"] = object()
    owner._extension_worker_resources["worker"] = {"old": True}
    owner._worker_metadata["worker"] = {"old": True}
    owner._controller_control = SimpleNamespace(
        register=AsyncMock(side_effect=ControlSessionRejected("rejected"))
    )
    with pytest.raises(ControlSessionRejected, match="rejected"):
        await owner.add_worker("worker")
    assert "worker" not in owner._worker_address_to_worker
    assert "worker" not in owner._extension_worker_resources
    assert "worker" not in owner._worker_metadata


@pytest.mark.asyncio
async def test_provider_close_is_attempted_when_runtime_cleanup_fails():
    owner = SupervisorActor()
    close = AsyncMock()
    owner._controller_control = SimpleNamespace(
        fence=SimpleNamespace(ready=True), invoke=close
    )
    owner._worker_relaunch_manager = SimpleNamespace(
        close=AsyncMock(side_effect=RuntimeError("cleanup failed"))
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await owner.__pre_destroy__()
    close.assert_awaited_once_with("close")


@pytest.mark.asyncio
async def test_recovery_description_only_keeps_known_unexpected_losses():
    owner = SupervisorActor()
    owner._unexpected_down_replicas[("counter", 0)] = "counter"
    with pytest.raises(ValueError, match="not found"):
        await owner.describe_model("counter")
    assert await owner.describe_model("counter", for_recovery=True) == {
        "model_name": "counter"
    }
    owner._clear_unexpected_down_replicas("counter")
    with pytest.raises(ValueError, match="not found"):
        await owner.describe_model("counter", for_recovery=True)


@pytest.mark.asyncio
async def test_worker_callback_keeps_original_session_after_reinstallation():
    from ..control_session import SessionBoundSupervisorRef

    fence, authority = setup_fence()
    first = session()
    # setup_fence is intentionally synchronous; installation validates authority.
    authority.owner = first
    await fence.install(first, 1)
    target = SimpleNamespace(
        terminate_model=AsyncMock(), list_models=AsyncMock(return_value={})
    )
    ref = SessionBoundSupervisorRef(target, fence)
    callback = ref.terminate_model
    successor = session(2)
    authority.owner = successor
    await fence.install(successor, 1)
    with pytest.raises(ControlSessionRejected):
        await callback("model")
    target.terminate_model.assert_not_awaited()
    await ref.terminate_model("model")
    assert (
        target.terminate_model.await_args.kwargs[CONTROL_CONTEXT_KEY]["session"]
        == successor.to_dict()
    )
    assert await ref.list_models() == {}


@pytest.mark.asyncio
async def test_service_placement_is_process_local_and_scoped_per_launch():
    from ..supervisor_control import model_service_address

    owner, _, port = controlled_owner()
    owner.control_address = "default:1234"
    started, proceed = asyncio.Event(), asyncio.Event()

    async def selected():
        assert model_service_address(owner) == "worker:1234"
        started.set()
        await proceed.wait()
        assert model_service_address(owner) == "worker:1234"

    task = asyncio.create_task(port.with_service_address("worker:1234", selected))
    await started.wait()
    assert model_service_address(owner) == owner.control_address
    proceed.set()
    await task
    assert model_service_address(owner) == owner.control_address


def test_model_service_port_restores_reference_graph_without_orchestrating():
    from ..supervisor_control import MODEL_SERVICE_FIELDS

    owner, _, port = controlled_owner()
    for name in MODEL_SERVICE_FIELDS:
        setattr(owner, name, {})
    ref = SimpleNamespace(address="worker:1234", uid=b"pd-model")
    port.restore_model_services(
        "model", {"_pd_model_mapping": ref, "_pd_roles": {0: "prefill", 1: "decode"}}
    )
    assert port.model_services("model")["_pd_model_mapping"] is ref
    assert port.model_services("model")["_pd_roles"][0] == "prefill"
    with pytest.raises(ValueError):
        port.restore_model_services("model", {"arbitrary_private_state": {}})


def test_service_inventory_includes_pd_shards_and_rank_zero_only_on_request():
    from ..worker import ModelStatus

    worker = WorkerActor("supervisor:1234", None, SimpleNamespace(), [])
    worker._model_uid_to_model_spec = {
        "pd-rep0": {"model_name": "pd"},
        "shards-rep0": {"model_name": "shards"},
    }
    worker._model_uid_to_launch_args = {
        "pd-rep0": {"_nixl_config": {"role": "prefill"}},
        "shards-rep0": {"n_worker": 2, "shard": 1},
    }
    worker._model_uid_to_model = {
        uid: object() for uid in ("pd-rep0", "shards-rep0", "pd-rank0")
    }
    worker._model_uid_to_model_status = {
        uid: ModelStatus(model_state="ready") for uid in worker._model_uid_to_model_spec
    }
    legacy = worker.controller_inventory()
    assert set(legacy["unsupported"]) == {"pd-rep0", "shards-rep0"}
    assert "coordinators" not in legacy
    actual = worker.controller_inventory(include_services=True)
    assert not actual["unsupported"]
    assert actual["coordinators"] == ["pd-rank0"]
    assert {row["replica_model_uid"] for row in actual["replicas"]} == {
        "pd-rep0",
        "shards-rep0",
    }


@pytest.mark.asyncio
async def test_runtime_actor_mutations_keep_worker_session_admission(monkeypatch):
    fence, authority = setup_fence()
    await fence.install(authority.owner, 1)
    worker = WorkerActor("supervisor:1234", None, SimpleNamespace(), [])
    worker._control_fence, worker._lifecycle = fence, fence.lifecycle
    create = AsyncMock(return_value="created")
    lookup = AsyncMock()
    monkeypatch.setattr(xo, "create_actor", create)
    monkeypatch.setattr(xo, "actor_ref", lookup)
    original = command(authority.owner).to_dict()
    assert (
        await worker.create_runtime_actor(
            object, (), {}, "service", **{CONTROL_CONTEXT_KEY: original}
        )
        == "created"
    )
    create.assert_awaited_once()
    authority.owner = session(2)
    with pytest.raises(ControlSessionRejected):
        await worker.destroy_runtime_actor("service", **{CONTROL_CONTEXT_KEY: original})
    lookup.assert_not_awaited()
