# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ..replica_config import DeviceConfig, ReplicaConfig
from ..status_guard import InstanceInfo, StatusGuardActor
from ..supervisor import InitialModelLaunch, SupervisorActor


class _Worker:
    def __init__(self, address, guard, stage=None):
        self.address = address
        self.guard = guard
        self.stage = stage
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.models = set()
        self.launching = set()
        self.terminated = []
        self.fail_termination = False

    async def launch_builtin_model(self, model_uid, **kwargs):
        self.launching.add(model_uid)
        if self.stage == "dispatch":
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                # Simulate an RPC acknowledging cancellation only after its
                # native work finishes, possibly publishing a late model.
                await self.release.wait()
        self.models.add(model_uid)
        self.launching.discard(model_uid)
        return "subpool"

    async def wait_for_load(self, model_uid, **kwargs):
        if self.stage == "load":
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                await self.release.wait()
        self.guard.update_instance_info("demo", {"status": "READY"})

    async def cancel_launch_model(self, model_uid):
        if model_uid not in self.launching:
            raise RuntimeError("Model is not launching")
        self.release.set()

    async def terminate_model(self, model_uid, **kwargs):
        if self.fail_termination:
            raise RuntimeError("worker unavailable")
        self.models.discard(model_uid)
        self.launching.discard(model_uid)
        self.terminated.append(model_uid)
        self.release.set()

    async def get_launch_args(self, model_uid):
        return {"model_name": "demo", "n_gpu": None}


@pytest.fixture
def runtime():
    guard = StatusGuardActor()
    supervisor = SupervisorActor()
    supervisor.address = "supervisor:9999"
    supervisor._collective_manager_mapping = {}
    supervisor._block_tracker_mapping = {}
    supervisor._status_guard_ref = SimpleNamespace(
        **{
            name: AsyncMock(side_effect=getattr(guard, name))
            for name in (
                "set_instance_info",
                "get_instance_info",
                "update_instance_info",
                "update_replica_status",
                "get_replica_statuses",
            )
        }
    )
    workers = [_Worker(f"worker-{i}:1234", guard) for i in range(2)]
    supervisor._worker_address_to_worker = {w.address: w for w in workers}
    supervisor._resolve_replica_config = AsyncMock(
        return_value=([(w, None, None) for w in workers], {})
    )
    supervisor._resolve_download_hub_from_workers = AsyncMock(
        return_value="huggingface"
    )
    kwargs = dict(
        model_uid="demo",
        model_name="demo",
        model_size_in_billions=1,
        model_format="pytorch",
        quantization=None,
        model_engine="vLLM",
        model_type="LLM",
        replica=2,
        replica_config=[
            ReplicaConfig(devices=[DeviceConfig(worker_ip=w.address, n_gpu=0)])
            for w in workers
        ],
    )
    return supervisor, guard, workers, kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel", "resume_failure"])
async def test_initial_resume_keeps_public_calls_closed_until_activation_commits(
    runtime, monkeypatch, outcome
):
    from ..control_session import ControlSessionRejected
    from ..supervisor_control import SupervisorControl, SupervisorControlPort
    from .test_control_session import session
    from .test_supervisor_control import Provider

    supervisor, _, workers, kwargs = runtime
    state = SimpleNamespace(owner=session(), phase="active", failed=False)
    control = SupervisorControl(supervisor, Provider(state, state.owner))
    supervisor._controller_control = control
    supervisor._control_fence = control.fence
    supervisor._standby = True
    supervisor._schedule_autostart = Mock()
    supervisor._relaunch_manager = Mock()
    supervisor.list_models = AsyncMock(return_value={"existing-model": {}})
    monkeypatch.setattr(
        "xinference.core.supervisor.XINFERENCE_DISABLE_HEALTH_CHECK", True
    )
    for worker in workers:
        worker.uid = b"worker"
        worker.stage = "dispatch"
        worker.launch_builtin_model = AsyncMock(side_effect=worker.launch_builtin_model)
    refs = [control.worker(worker) for worker in workers]
    supervisor._worker_address_to_worker = {ref.address: ref for ref in refs}
    supervisor._resolve_replica_config.return_value = (
        [(ref, None, None) for ref in refs],
        {},
    )
    entered, release = asyncio.Event(), asyncio.Event()

    async def prepare(*args):
        entered.set()
        await release.wait()
        if outcome == "failure":
            raise RuntimeError("cache tracker unavailable")

    supervisor._cache_tracker_ref = SimpleNamespace(
        record_model_version=AsyncMock(side_effect=prepare)
    )
    if outcome == "resume_failure":
        supervisor._upgrade_state.resume = Mock(
            side_effect=RuntimeError("resume failed")
        )
    port = SupervisorControlPort(supervisor)
    await port.pause(1)
    resuming = asyncio.create_task(port.resume())
    launch = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        read = (await asyncio.gather(supervisor.list_models(), return_exceptions=True))[
            0
        ]
        request = (
            await asyncio.gather(
                supervisor.launch_builtin_model(**kwargs, wait_ready=False),
                return_exceptions=True,
            )
        )[0]
        if not isinstance(request, ControlSessionRejected):
            launch = supervisor._initial_model_launches["demo"].task
            await asyncio.wait_for(
                asyncio.gather(*(w.entered.wait() for w in workers)), 2
            )
        if outcome == "cancel":
            resuming.cancel()
        else:
            release.set()
        result = (await asyncio.gather(resuming, return_exceptions=True))[0]
        for worker in workers:
            worker.release.set()
        if launch is not None:
            await asyncio.gather(launch, return_exceptions=True)
        assert isinstance(read, ControlSessionRejected)
        assert isinstance(request, ControlSessionRejected)
        for worker in workers:
            worker.launch_builtin_model.assert_not_awaited()
            assert not worker.models and not worker.launching
        assert not supervisor._initial_model_launches
        assert not supervisor._model_uid_to_replica_info
        assert not supervisor._replica_model_uid_to_worker
        if outcome == "success":
            assert result["ready"] and control.has_served
            assert await supervisor.launch_builtin_model(**kwargs) == "demo"
            assert all(worker.models for worker in workers)
            await port.pause(1)
            assert await supervisor.list_models() == {"existing-model": {}}
            with pytest.raises(ControlSessionRejected):
                await supervisor.launch_builtin_model(**kwargs, wait_ready=False)
        else:
            assert isinstance(
                result, asyncio.CancelledError if outcome == "cancel" else RuntimeError
            )
            assert not control.ready and not control.has_served
    finally:
        release.set()
        for worker in workers:
            worker.release.set()
        resuming.cancel()
        await asyncio.gather(
            resuming, *([launch] if launch else []), return_exceptions=True
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_ready", [False, True])
@pytest.mark.parametrize("load_fails", [False, True])
async def test_control_pause_rejects_busy_spawned_launch_before_closing_admission(
    runtime, wait_ready, load_fails
):
    from ..supervisor_control import SupervisorControl, SupervisorControlPort
    from .test_control_session import session
    from .test_supervisor_control import Provider

    supervisor, guard, workers, kwargs = runtime
    state = SimpleNamespace(owner=session(), phase="active", failed=False)
    control = SupervisorControl(supervisor, Provider(state, state.owner))
    supervisor._controller_control = control
    supervisor._control_fence = control.fence
    control.fence.ready = control.has_served = True
    for worker in workers:
        worker.uid = b"worker"
    refs = [control.worker(worker) for worker in workers]
    supervisor._worker_address_to_worker = {ref.address: ref for ref in refs}
    supervisor._resolve_replica_config.return_value = (
        [(ref, None, None) for ref in refs],
        {},
    )
    workers[1].stage = "dispatch"
    if load_fails:
        workers[1].wait_for_load = AsyncMock(side_effect=RuntimeError("load failed"))
    request = asyncio.create_task(
        supervisor.launch_builtin_model(**kwargs, wait_ready=wait_ready)
    )
    launch = None
    try:
        await asyncio.wait_for(workers[1].entered.wait(), 2)
        launch = supervisor._initial_model_launches["demo"].task
        # This is the real create_task/gather launch path, with the Worker RPC
        # still suspended. A rejected pause must not fence readiness or rollback.
        with pytest.raises(RuntimeError, match="active"):
            await SupervisorControlPort(supervisor).pause(0.2)
        assert control.ready
        workers[1].release.set()
        results = await asyncio.gather(request, launch, return_exceptions=True)
        assert not supervisor._initial_model_launches
        assert not supervisor._workers_launching
        if load_fails:
            assert isinstance(results[1], RuntimeError)
            assert str(results[1]) == "load failed"
            assert all(not worker.models and not worker.launching for worker in workers)
            assert not supervisor._model_uid_to_replica_info
            assert not supervisor._replica_model_uid_to_worker
            assert guard.get_instance_info(model_uid="demo")[0].status == "ERROR"
        else:
            assert results == ["demo", None]
            assert all(worker.models for worker in workers)
            assert all(r.status == "READY" for r in guard.get_replica_statuses("demo"))
        await SupervisorControlPort(supervisor).pause(1)
        assert not control.ready
    finally:
        workers[1].release.set()
        await asyncio.gather(
            request, *([launch] if launch else []), return_exceptions=True
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_ready", [False, True])
@pytest.mark.parametrize("stage", ["dispatch", "load"])
async def test_cancel_mixed_initial_readiness_drains_dispatch_and_all_replicas(
    runtime, stage, wait_ready
):
    supervisor, guard, workers, kwargs = runtime
    for worker in workers:
        worker.models.add("established-rep0")
    workers[1].stage = stage
    request = asyncio.create_task(
        supervisor._launch_builtin_model(**kwargs, wait_ready=wait_ready)
    )
    await asyncio.wait_for(workers[1].entered.wait(), 2)
    info = (await supervisor.get_instance_info(None, "demo"))[0]
    assert info["status"] == "READY"
    assert info["can_cancel_launch"]
    assert guard.get_replica_statuses("demo")[0].status == "READY"
    with pytest.raises(ValueError, match="Initial launch"):
        await supervisor.add_model_replica("demo")
    with pytest.raises(ValueError, match="Initial launch"):
        await supervisor.terminate_model("demo")

    await asyncio.wait_for(supervisor.cancel_launch_builtin_model("demo"), 2)
    if wait_ready:
        with pytest.raises(RuntimeError, match="launch cancelled"):
            await request
    else:
        assert await request == "demo"
    assert not supervisor._initial_model_launches
    assert not supervisor._model_uid_to_replica_info
    assert not supervisor._replica_model_uid_to_worker
    assert not supervisor._workers_launching
    assert not supervisor._async_launch_tasks
    for i, worker in enumerate(workers):
        assert worker.models == {"established-rep0"}
        assert not worker.launching
        assert f"demo-rep{i}" in worker.terminated
    assert (await supervisor.get_instance_info(None, "demo"))[0]["status"] == "ERROR"
    assert not (await supervisor.get_instance_info(None, "demo"))[0][
        "can_cancel_launch"
    ]

    # Cancellation does not poison reuse of the UID or a later complete model.
    workers[1].stage = None
    assert await supervisor._launch_builtin_model(**kwargs) == "demo"
    terminated = [list(w.terminated) for w in workers]
    with pytest.raises(ValueError, match="No initial launch"):
        await supervisor.cancel_launch_builtin_model("demo")
    assert terminated == [w.terminated for w in workers]
    assert all(w.models for w in workers)


@pytest.mark.asyncio
async def test_failed_cancel_retains_uid_ownership_until_retry(runtime):
    supervisor, _, workers, kwargs = runtime
    workers[1].stage = "dispatch"
    await supervisor._launch_builtin_model(**kwargs, wait_ready=False)
    await asyncio.wait_for(workers[1].entered.wait(), 2)
    workers[0].fail_termination = True
    with pytest.raises(RuntimeError, match="worker unavailable"):
        await asyncio.wait_for(supervisor.cancel_launch_builtin_model("demo"), 2)
    assert (await supervisor.get_instance_info(None, "demo"))[0]["can_cancel_launch"]
    with pytest.raises(ValueError, match="already in the model list"):
        await supervisor._launch_builtin_model(**kwargs)
    workers[0].fail_termination = False
    await supervisor.cancel_launch_builtin_model("demo")
    assert not supervisor._initial_model_launches
    assert all(not w.models for w in workers)


@pytest.mark.asyncio
async def test_queued_cancel_cannot_target_a_replacement_launch(runtime):
    supervisor, _, workers, _ = runtime
    tasks = [asyncio.create_task(asyncio.Event().wait()) for _ in range(2)]
    supervisor._initial_model_launches["demo"] = InitialModelLaunch(tasks[0])
    lock = supervisor._get_model_replica_lock("demo")
    await lock.acquire()
    cancel = asyncio.create_task(supervisor.cancel_launch_builtin_model("demo"))
    await asyncio.sleep(0)
    replacement = InitialModelLaunch(tasks[1])
    supervisor._initial_model_launches["demo"] = replacement
    lock.release()
    try:
        with pytest.raises(ValueError, match="No initial launch"):
            await cancel
        assert supervisor._initial_model_launches["demo"] is replacement
        assert all(not w.terminated for w in workers)
        assert not tasks[1].done()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_rejects_scale_up_without_touching_established_replicas(runtime):
    supervisor, guard, workers, _ = runtime
    guard.set_instance_info(
        "demo",
        InstanceInfo(
            model_uid="demo",
            model_name="demo",
            model_version=None,
            model_ability=["chat"],
            replica=1,
            status="READY",
            instance_created_ts=1,
        ),
    )
    guard.update_replica_status("demo", 0, {"status": "READY"})
    supervisor._model_uid_to_replica_info["demo"] = supervisor._build_replica_info(1)
    supervisor._replica_model_uid_to_worker["demo-rep0"] = workers[0]
    workers[0].models.add("demo-rep0")
    workers[1].stage = "dispatch"
    supervisor._select_worker_for_scale_up = AsyncMock(return_value=(workers[1], None))
    scale = asyncio.create_task(supervisor.add_model_replica("demo"))
    await asyncio.wait_for(workers[1].entered.wait(), 2)
    try:
        info = (await supervisor.get_instance_info(None, "demo"))[0]
        assert not info["can_cancel_launch"]
        with pytest.raises(ValueError, match="No initial launch"):
            await asyncio.wait_for(supervisor.cancel_launch_builtin_model("demo"), 1)
        assert workers[0].models == {"demo-rep0"}
        assert not workers[0].terminated
        assert not workers[1].terminated
    finally:
        workers[1].release.set()
        await scale
    assert supervisor._model_uid_to_replica_info["demo"].active_replica_ids == [0, 1]
