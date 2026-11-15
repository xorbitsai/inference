# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
import asyncio
import itertools
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import xoscar as xo

from xinference import extensions

from ..control_session import CONTROL_CONTEXT_KEY, ControlSessionRejected
from ..model import ModelActor
from ..replica_config import DeviceConfig, ReplicaConfig
from ..status_guard import StatusGuardActor
from ..supervisor import ReplicaInfo, SupervisorActor, WorkerStatus
from ..supervisor_control import SupervisorControl, SupervisorControlPort
from ..supervisor_upgrade import SupervisorGateway
from ..utils import CancelMixin, build_replica_model_uid
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


class _LaunchWorker:
    def __init__(self, runtime, index):
        self.runtime, self.index = runtime, index
        self.address, self.uid = f"worker-{index}:1234", b"worker"
        self.models = set()
        self.gpu_models = set()
        self.terminated = []
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.fail_uid, self.fail_stage = None, "load"
        self.cleanup_error = None

    async def get_extension_resources(self, **kwargs):
        return {"extensions": {}}

    async def get_node_metadata(self, **kwargs):
        return {"software_version": "test"}

    async def update_system_settings(self, settings, **kwargs):
        pass

    async def get_model_registration(self, *args, **kwargs):
        return None

    async def get_model_count(self, **kwargs):
        return len(self.models)

    async def list_models(self, **kwargs):
        return {uid: {} for uid in self.models if not uid.endswith("-rank0")}

    async def get_model_launch_status(self, model_uid, **kwargs):
        return "READY" if model_uid in self.models else None

    async def get_gpu_allocation_status(self, **kwargs):
        return {
            "total": [0],
            "models": {0: list(self.gpu_models)},
            "user_specified": {},
        }

    async def resolve_download_hub(self, download_hub, model_path, **kwargs):
        return download_hub or "huggingface"

    async def _fail(self, uid, stage):
        if uid == self.fail_uid and stage == self.fail_stage:
            self.entered.set()
            await self.release.wait()
            self.runtime.state.failed = self.runtime.loss == "rollback"
            raise self.runtime.error

    async def launch_builtin_model(self, model_uid, **kwargs):
        if self.runtime.replacing:
            assert not any(w.models for w in self.runtime.workers)
            self.runtime.replacing = False
        assert model_uid not in self.models
        self.models.add(model_uid)
        if kwargs.get("gpu_idx"):
            self.gpu_models.add(model_uid)
        await self._fail(model_uid, "dispatch")
        return (
            ("subpool", {"driver": "info"}) if kwargs.get("shard") == 0 else "subpool"
        )

    async def launch_rank0_model(self, model_uid, config, **kwargs):
        self.models.add(model_uid)
        await self._fail(model_uid, "dispatch")
        return "worker-0:4321", 9876

    async def wait_for_load(self, model_uid, **kwargs):
        await self._fail(model_uid, "load")
        self.runtime.guard.update_instance_info("failed", {"status": "READY"})

    async def terminate_model(self, model_uid, **kwargs):
        self.runtime.guard.update_instance_info("failed", {"status": "TERMINATING"})
        if self.runtime.loss == "worker":
            self.runtime.state.failed = True
            raise ControlSessionRejected("authority_unavailable")
        if self.cleanup_error is not None:
            raise self.cleanup_error
        self.models.discard(model_uid)
        self.gpu_models.discard(model_uid)
        self.terminated.append(model_uid)
        if self.runtime.loss == "service" and not any(
            uid
            for worker in self.runtime.workers
            for uid in worker.models
            if not uid.endswith("-rank0")
        ):
            self.runtime.state.failed = True

    async def start_transfer_for_vllm(self, *args, **kwargs):
        pass

    async def get_model(self, **kwargs):
        return AsyncMock()


@pytest.fixture
async def failed_launch_runtime(monkeypatch, tmp_path):
    from .. import supervisor as supervisor_module
    from ..worker_registry import WorkerOwnership

    monkeypatch.setattr(extensions, "get_extensions", lambda: ())
    for name in ("DOWNLOAD_TASK", "LAUNCH_HISTORY", "TOKEN_ROUTER"):
        monkeypatch.setattr(
            supervisor_module, f"XINFERENCE_{name}_DB_PATH", str(tmp_path / name)
        )
    owner, guard = SupervisorActor(), StatusGuardActor()
    owner.address = "supervisor:1234"
    owner._collective_manager_mapping, owner._block_tracker_mapping = {}, {}
    owner._worker_ownership = WorkerOwnership(str(tmp_path / "ownership.json"))
    owner._schedule_autostart = Mock()
    owner._status_guard_ref = SimpleNamespace(
        **{
            name: AsyncMock(side_effect=getattr(guard, name))
            for name in (
                "set_instance_info",
                "get_instance_info",
                "update_instance_info",
                "update_replica_status",
                "get_replica_statuses",
                "remove_failed_instance_info",
            )
        }
    )
    state = SimpleNamespace(owner=session(), phase="active", failed=False)
    authority = Provider(state, state.owner)
    control = SupervisorControl(owner, authority)
    control.fence.ready = control.has_served = True
    owner._controller_control = control
    runtime = SimpleNamespace(
        owner=owner,
        guard=guard,
        path=tmp_path,
        state=state,
        authority=authority,
        loss="rollback",
        replacing=False,
        actors={},
        destroyed=[],
        error=RuntimeError("original launch failure"),
    )
    workers = runtime.workers = [_LaunchWorker(runtime, i) for i in range(2)]
    monkeypatch.setattr(
        xo,
        "actor_ref",
        AsyncMock(
            side_effect=lambda **kw: next(
                w for w in workers if w.address == kw["address"]
            )
        ),
    )
    for worker in workers:
        await owner.add_worker(worker.address)

    async def create(cls, args, kwargs, address, uid):
        assert uid not in runtime.actors
        ref = AsyncMock(address=address, uid=uid.encode())
        runtime.actors[uid] = ref
        return ref

    async def destroy(ref):
        if runtime.reject_destroy:
            state.failed = True
            if runtime.reject_destroy == "uncertain":
                del runtime.actors[ref.uid.decode()]
                raise asyncio.TimeoutError()
            if runtime.reject_destroy == "dependency":
                raise xo.ActorNotExist("dependency")
            raise ControlSessionRejected("authority_unavailable")
        if ref.uid.decode() not in runtime.actors:
            raise xo.ActorNotExist(ref.uid)
        runtime.destroyed.append(ref.uid.decode())
        del runtime.actors[ref.uid.decode()]

    authority.create_model_service, authority.destroy_model_service = create, destroy
    monkeypatch.setattr(
        xo,
        "has_actor",
        AsyncMock(side_effect=lambda ref: ref.uid.decode() in runtime.actors),
    )
    runtime.reject_destroy = False
    runtime.kwargs = dict(
        model_uid="failed",
        model_name="qwen3",
        model_size_in_billions="0_6",
        model_format="pytorch",
        quantization="none",
        model_engine="Transformers",
        model_type="LLM",
        n_gpu=None,
        replica=2,
    )
    return runtime


async def _failed_launch(runtime, wait_ready=True, sharded=False):
    worker = runtime.workers[1]
    worker.fail_uid = "failed-rep1"
    launch = asyncio.create_task(
        runtime.owner.launch_builtin_model(**runtime.kwargs, wait_ready=wait_ready)
    )
    await asyncio.wait_for(worker.entered.wait(), 5)
    operation = runtime.owner._initial_model_launches["failed"]
    if sharded:
        assert "failed-rep0" in runtime.owner._replica_model_uid_to_worker
        assert "failed-rep1" not in runtime.owner._replica_model_uid_to_worker
    worker.release.set()
    if not wait_ready:
        assert await launch == "failed"
        launch = operation.task
    result = (await asyncio.gather(launch, return_exceptions=True))[0]
    assert result is runtime.error
    return operation


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_ready", [True, False])
@pytest.mark.parametrize("recovery", ["launch", "terminate"])
@pytest.mark.parametrize(
    "loss,stage,engine,model_format,quantization",
    [
        ("rollback", "load", "Transformers", "pytorch", "none"),
        ("worker", "load", "vLLM", "pytorch", "none"),
        ("healthy", "dispatch", "llama.cpp", "ggufv2", "Q4_K_M"),
        ("legacy", "dispatch", "Transformers", "pytorch", "none"),
        ("rollback", "dispatch", "SGLang", "pytorch", "none"),
        ("rollback", "load", "MLX", "mlx", "bf16"),
        ("worker", "dispatch", "TRT-LLM", "pytorch", "none"),
    ],
)
async def test_failed_initial_launch_recovery(
    failed_launch_runtime,
    wait_ready,
    recovery,
    loss,
    stage,
    engine,
    model_format,
    quantization,
):
    rt = failed_launch_runtime
    rt.loss, rt.workers[1].fail_stage = loss, stage
    rt.kwargs.update(
        model_engine=engine, model_format=model_format, quantization=quantization
    )
    targets = [
        ReplicaConfig(devices=[DeviceConfig(worker_ip=w.address, gpu_idx=[0])])
        for w in rt.workers
    ]
    if engine == "vLLM":
        rt.kwargs.pop("n_gpu")
        rt.kwargs["replica_config"] = targets
    if loss == "legacy":
        rt.owner._controller_control = None
        rt.owner._worker_address_to_worker = {w.address: w for w in rt.workers}
    operation = await _failed_launch(rt, wait_ready)
    assert rt.guard.get_instance_info(model_uid="failed")[0].status == "ERROR"
    if loss in ("rollback", "worker"):
        assert rt.owner._initial_model_launches["failed"] is operation
        assert operation.cleanup_pending and not operation.cancelling
        assert any(w.models for w in rt.workers)
        before = [list(w.terminated) for w in rt.workers]
        with pytest.raises(ControlSessionRejected):
            await rt.owner.terminate_model("failed", suppress_exception=True)
        assert before == [w.terminated for w in rt.workers]
    else:
        assert all(not w.models for w in rt.workers)
        assert "failed" not in rt.owner._initial_model_launches
    rt.state.failed, rt.loss = False, "healthy"
    rt.workers[1].fail_uid = None
    if operation.cleanup_pending:
        for worker in rt.workers:
            rt.owner._rebuild_worker_replica_state(
                rt.owner._worker_address_to_worker[worker.address], []
            )
        assert any(operation.workers.values())
    if recovery == "launch":
        # Validate real GPU occupancy only after recovery releases old models.
        rt.kwargs.pop("n_gpu", None)
        rt.kwargs["replica_config"] = targets
        rt.replacing = True
        assert await rt.owner.launch_builtin_model(**rt.kwargs) == "failed"
        assert all(w.models for w in rt.workers)
        with pytest.raises(ValueError, match="already"):
            await rt.owner.launch_builtin_model(**{**rt.kwargs, "replica_config": None})
        await rt.owner.terminate_model("failed")
    else:
        await rt.owner.terminate_model("failed")
        assert not rt.guard.get_instance_info(model_uid="failed")
    assert all(not w.models for w in rt.workers)
    assert not rt.owner._initial_model_launches
    assert not rt.owner._model_uid_to_replica_info
    assert not rt.owner._replica_model_uid_to_worker


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery", ["launch", "terminate"])
async def test_failed_sharded_initial_launch_recovery(failed_launch_runtime, recovery):
    rt = failed_launch_runtime
    rt.kwargs.update(model_engine="vLLM", n_worker=2)
    rt.workers[1].fail_stage = "dispatch"
    operation = await _failed_launch(rt, sharded=True)
    assert rt.owner._initial_model_launches["failed"] is operation
    assert all(operation.workers.values())
    assert rt.guard.get_instance_info(model_uid="failed")[0].status == "ERROR"
    rt.state.failed, rt.loss, rt.workers[1].fail_uid = False, "healthy", None
    if recovery == "launch":
        rt.replacing = True
        await rt.owner.launch_builtin_model(**rt.kwargs)
        assert all(w.models == {"failed-rep0", "failed-rep1"} for w in rt.workers)
    await rt.owner.terminate_model("failed")
    assert all(not w.models for w in rt.workers)
    assert not rt.owner._initial_model_launches


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "engine,shape",
    [
        ("vLLM", "collective"),
        ("vLLM", "rank0"),
        ("vLLM", "pd"),
        ("SGLang", "pd"),
        ("MLX", "pd"),
        ("TRT-LLM", "pd"),
        ("vLLM", "cache"),
        ("SGLang", "cache"),
        ("MLX", "cache"),
        ("TRT-LLM", "cache"),
        ("vLLM", "history-host"),
        ("vLLM", "uncertain"),
        ("vLLM", "dependency"),
        ("vLLM", "nixl"),
        ("SGLang", "nixl"),
        ("TRT-LLM", "nixl"),
    ],
)
async def test_failed_initial_launch_retains_services(
    failed_launch_runtime, engine, shape
):
    rt = failed_launch_runtime
    destruction = shape
    if shape in ("uncertain", "dependency"):
        shape = "history-host"
    rt.kwargs.update(model_engine=engine, enable_xavier=shape != "nixl")
    if engine == "MLX":
        rt.kwargs.update(model_format="mlx", quantization="bf16")
    if shape in ("pd", "nixl", "history-host"):
        rt.kwargs.pop("n_gpu")
        rt.kwargs["replica_config"] = [
            ReplicaConfig(
                role=role,
                devices=[DeviceConfig(worker_ip=w.address, gpu_idx=[0])],
                model_engine="MLX" if shape == "history-host" and i == 0 else engine,
                engine_config=(
                    {"model_format": "mlx"}
                    if shape == "history-host" and i == 0
                    else {}
                ),
            )
            for i, (w, role) in enumerate(zip(rt.workers, ("prefill", "decode")))
        ]
    if shape == "nixl":
        rt.kwargs["vllm_transfer_backend_type"] = "nixl"
    if shape == "cache":
        rt.kwargs.update(xavier_cache_bytes=1024)
        if engine == "vLLM":
            rt.kwargs["xavier_gpu_cache_bytes"] = 0
    if shape == "history-host":
        rt.kwargs.update(
            xavier_cache_backend="lmcache",
            xavier_cache_backend_config={
                "cpu_bytes": 1024,
                "disk_bytes": 4096,
                "disk_path": str(rt.path / "cache"),
            },
        )

    async def fail_notification(uid):
        rt.state.failed = True
        raise rt.error

    rt.authority.model_services_changed = fail_notification
    rt.error = ControlSessionRejected("support launch failed")
    if shape == "rank0":
        rt.workers[0].fail_uid, rt.workers[0].fail_stage = "failed-rank0", "dispatch"
        rt.workers[0].release.set()
    with pytest.raises(ControlSessionRejected, match="support launch failed"):
        await rt.owner.launch_builtin_model(**rt.kwargs)
    operation = rt.owner._initial_model_launches["failed"]
    assert operation.cleanup_pending and rt.actors
    if engine == "vLLM" and shape in ("collective", "pd"):
        assert "failed-rank0" in rt.workers[0].models
    if shape == "rank0":
        assert "failed-rank0" not in rt.owner._replica_model_uid_to_worker
    if shape == "history-host":
        assert rt.owner._xavier_source_mapping["failed"]
        assert len(rt.owner._xavier_history_mapping["failed"]) == 2
    assert {id(ref) for refs in operation.services.values() for ref in refs} == {
        id(ref) for ref in rt.actors.values()
    }
    before = dict(rt.actors)
    for loss in ("service", "healthy"):
        rt.state.failed, rt.loss, rt.reject_destroy = False, loss, loss == "healthy"
        if destruction in ("uncertain", "dependency") and loss == "healthy":
            rt.reject_destroy = destruction
        with pytest.raises(
            (ControlSessionRejected, asyncio.TimeoutError, xo.ActorNotExist)
        ):
            await rt.owner.terminate_model("failed", suppress_exception=True)
        assert not rt.destroyed
        assert len(rt.actors) == len(before) - int(
            destruction == "uncertain" and loss == "healthy"
        )
        assert rt.owner._initial_model_launches["failed"] is operation
        assert rt.guard.get_instance_info(model_uid="failed")[0].status == "ERROR"
    # Recovery still owns the actual actors after routing/service maps change.
    rt.owner._pd_model_mapping.clear()
    rt.owner._xavier_cache_mapping.clear()
    rt.state.failed, rt.reject_destroy = False, False
    await rt.owner.terminate_model("failed")
    assert not rt.actors and all(not w.models for w in rt.workers)
    assert not rt.owner._initial_model_launches
    assert not rt.guard.get_instance_info(model_uid="failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", ["launch", "terminate"])
@pytest.mark.parametrize(
    "cleanup_error", [RuntimeError, asyncio.CancelledError, asyncio.TimeoutError]
)
async def test_failed_initial_launch_recovery_ownership(
    failed_launch_runtime, queued, cleanup_error
):
    rt = failed_launch_runtime
    operation = await _failed_launch(rt)
    assert rt.owner._initial_model_launches["failed"] is operation
    rt.state.failed, rt.loss, rt.workers[1].fail_uid = False, "healthy", None
    rt.workers[1].cleanup_error = cleanup_error()
    with pytest.raises(cleanup_error):
        if queued == "launch":
            await rt.owner.launch_builtin_model(**rt.kwargs)
        else:
            await rt.owner.terminate_model("failed", suppress_exception=True)
    assert rt.workers[1].models and operation.workers["failed-rep1"]
    assert not rt.workers[0].models and not operation.workers["failed-rep0"]
    assert rt.guard.get_instance_info(model_uid="failed")[0].status == "ERROR"
    rt.workers[1].cleanup_error = None
    # Queue both requests while they can observe only the failed operation.
    entered = asyncio.Queue()

    async def intercept(invocation, call):
        entered.put_nowait(invocation.method)
        return await call()

    rt.authority.around_supervisor_call = intercept
    lock = rt.owner._get_model_replica_lock("failed")
    await lock.acquire()
    first = asyncio.create_task(rt.owner.launch_builtin_model(**rt.kwargs))
    await asyncio.wait_for(entered.get(), 5)
    second = asyncio.create_task(
        rt.owner.launch_builtin_model(**rt.kwargs)
        if queued == "launch"
        else rt.owner.terminate_model("failed")
    )
    await asyncio.wait_for(entered.get(), 5)
    lock.release()
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert results[0] == "failed"
    assert isinstance(results[1], ValueError)
    assert "ownership changed" in str(results[1])
    assert all(w.models for w in rt.workers)
    assert rt.guard.get_instance_info(model_uid="failed")[0].status == "READY"
    await rt.owner.terminate_model("failed")
    assert all(not w.models for w in rt.workers)


@pytest.mark.asyncio
async def test_failed_initial_launch_retired_worker(failed_launch_runtime):
    rt = failed_launch_runtime
    operation = await _failed_launch(rt)
    rt.state.failed, rt.loss = False, "healthy"
    retired = rt.workers[1]
    await rt.owner.remove_worker(retired.address)
    assert retired.address not in rt.owner._worker_address_to_worker
    # Permanent shutdown has removed the models as well as the registration.
    retired.models.clear()
    retired.cleanup_error = ConnectionRefusedError("retired worker")
    assert rt.owner._initial_model_launches["failed"] is operation

    await rt.owner.terminate_model("failed")
    assert all(not w.models for w in rt.workers)
    assert not rt.owner._initial_model_launches
    assert not rt.guard.get_instance_info(model_uid="failed")
    rt.kwargs.update(replica=1, worker_ip=rt.workers[0].address)
    rt.replacing = True
    assert await rt.owner.launch_builtin_model(**rt.kwargs) == "failed"
    assert rt.workers[0].models == {"failed-rep0"}
    assert not retired.models
    await rt.owner.terminate_model("failed")
    assert all(not w.models for w in rt.workers)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", ["heartbeat", "reverse_ping"])
@pytest.mark.parametrize("resolution", ["termination", "inventory"])
async def test_failed_initial_launch_live_worker_health_timeout(
    failed_launch_runtime, monkeypatch, timeout, resolution
):
    from .. import supervisor as supervisor_module

    rt = failed_launch_runtime
    operation = await _failed_launch(rt)
    rt.state.failed, rt.loss = False, "healthy"
    healthy, worker = rt.workers
    rt.owner._actor_loop = asyncio.get_running_loop()
    worker.cleanup_error = asyncio.TimeoutError("reverse link unavailable")
    for current in rt.workers:
        monkeypatch.setattr(current, "ping", AsyncMock(), raising=False)
        rt.owner._worker_status[current.address] = WorkerStatus(
            update_time=supervisor_module.time.time(),
            failure_remaining_count=2,
            status={},
        )
    worker.ping.side_effect = asyncio.TimeoutError("reverse link unavailable")
    if timeout == "heartbeat":
        rt.owner._worker_status[worker.address].update_time = 0
    with monkeypatch.context() as health:
        health.setattr(
            supervisor_module, "XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD", 2
        )
        health.setattr(
            supervisor_module.asyncio,
            "sleep",
            AsyncMock(side_effect=[None, asyncio.CancelledError()]),
        )
        with pytest.raises(asyncio.CancelledError):
            await rt.owner._check_dead_nodes()
    assert worker.ping.await_count == (2 if timeout == "reverse_ping" else 1)
    assert worker.address not in rt.owner._worker_address_to_worker
    assert healthy.address in rt.owner._worker_address_to_worker
    assert worker.models == {"failed-rep1"}
    assert operation.workers["failed-rep1"]

    with pytest.raises(asyncio.TimeoutError) as exc:
        await rt.owner.terminate_model("failed")
    assert exc.value is worker.cleanup_error
    assert rt.owner._initial_model_launches["failed"] is operation
    assert operation.workers["failed-rep1"] and worker.models
    assert not healthy.models and not operation.workers["failed-rep0"]
    rt.kwargs.update(replica=1, worker_ip=healthy.address)
    rt.replacing = True
    with pytest.raises(asyncio.TimeoutError) as exc:
        await rt.owner.launch_builtin_model(**rt.kwargs)
    assert exc.value is worker.cleanup_error
    assert rt.owner._initial_model_launches["failed"] is operation
    assert operation.workers["failed-rep1"] and worker.models
    assert not healthy.models

    if resolution == "inventory":
        await rt.owner.add_worker(worker.address)
        # Re-registration alone does not establish absence of the old replica.
        with pytest.raises(asyncio.TimeoutError):
            await rt.owner.terminate_model("failed")
        assert operation.workers["failed-rep1"] and worker.models
        inventory = AsyncMock(wraps=worker.list_models)
        launch_status = AsyncMock(wraps=worker.get_model_launch_status)
        monkeypatch.setattr(worker, "list_models", inventory)
        monkeypatch.setattr(worker, "get_model_launch_status", launch_status)
        worker.models.clear()
    else:
        worker.cleanup_error = None
    await rt.owner.terminate_model("failed")
    if resolution == "inventory":
        inventory.assert_awaited_once()
        launch_status.assert_awaited_once_with("failed-rep1")
    assert not operation.workers["failed-rep1"]
    assert not rt.owner._initial_model_launches
    assert not rt.guard.get_instance_info(model_uid="failed")
    assert all(not w.models for w in rt.workers)
    assert await rt.owner.launch_builtin_model(**rt.kwargs) == "failed"
    assert healthy.models == {"failed-rep0"} and not worker.models
    await rt.owner.terminate_model("failed")
    assert all(not w.models for w in rt.workers)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "inventory_error", [ConnectionRefusedError, asyncio.TimeoutError]
)
async def test_failed_initial_launch_registered_worker_inventory(
    failed_launch_runtime, monkeypatch, inventory_error
):
    rt = failed_launch_runtime
    operation = await _failed_launch(rt)
    rt.state.failed, rt.loss = False, "healthy"
    worker = rt.workers[1]
    worker.cleanup_error = ConnectionRefusedError("registered worker")
    inventory = AsyncMock(side_effect=inventory_error("inventory unavailable"))
    monkeypatch.setattr(worker, "list_models", inventory)
    for unavailable in (True, False):
        if not unavailable:
            inventory.side_effect = None
            inventory.return_value = {"failed-rep1": {}}
        with pytest.raises(ConnectionRefusedError) as exc:
            await rt.owner.terminate_model("failed", suppress_exception=True)
        assert exc.value is worker.cleanup_error
        assert worker.address in rt.owner._worker_address_to_worker
        assert rt.owner._initial_model_launches["failed"] is operation
        assert operation.workers["failed-rep1"] and worker.models
        assert not operation.workers["failed-rep0"] and not rt.workers[0].models

    # Failed re-registration removes the registry entry without proving death.
    resources = worker.get_extension_resources
    monkeypatch.setattr(
        worker,
        "get_extension_resources",
        AsyncMock(side_effect=RuntimeError("registration failed")),
    )
    with pytest.raises(RuntimeError, match="registration failed"):
        await rt.owner.add_worker(worker.address)
    assert worker.address not in rt.owner._worker_address_to_worker
    with pytest.raises(ConnectionRefusedError) as exc:
        await rt.owner.terminate_model("failed")
    assert exc.value is worker.cleanup_error
    assert rt.owner._initial_model_launches["failed"] is operation
    assert operation.workers["failed-rep1"] and worker.models
    monkeypatch.setattr(worker, "get_extension_resources", resources)
    await rt.owner.add_worker(worker.address)

    # Only a fresh successful inventory for this registration proves absence.
    worker.models.clear()
    inventory.return_value = {}
    await rt.owner.terminate_model("failed")
    assert not rt.owner._initial_model_launches
    assert all(not w.models for w in rt.workers)
    assert not rt.guard.get_instance_info(model_uid="failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_write", ["first", "final", "all"])
async def test_failed_initial_launch_status_write_failure(
    failed_launch_runtime, failed_write
):
    rt = failed_launch_runtime
    rt.loss = "healthy"
    error_writes = 0

    def update(uid, info):
        nonlocal error_writes
        if info.get("status") == "ERROR":
            error_writes += 1
            if (
                failed_write == "all"
                or (failed_write == "first" and error_writes == 1)
                or (failed_write == "final" and error_writes == 2)
            ):
                raise RuntimeError("status write failed")
        return rt.guard.update_instance_info(uid, info)

    rt.owner._status_guard_ref.update_instance_info.side_effect = update
    # The helper asserts identity of the propagated original launch exception.
    await _failed_launch(rt)
    assert error_writes >= (2 if failed_write == "final" else 1)
    assert all(not w.models for w in rt.workers)
    assert not rt.owner._initial_model_launches
    assert not rt.owner._model_uid_to_replica_info
    assert not rt.owner._replica_model_uid_to_worker


@pytest.mark.asyncio
async def test_failed_initial_launch_status_failure_preserves_cleanup_error(
    failed_launch_runtime,
):
    rt = failed_launch_runtime
    operation = await _failed_launch(rt)
    rt.state.failed, rt.loss = False, "healthy"
    worker = rt.workers[1]
    worker.cleanup_error = RuntimeError("worker cleanup failed")
    update = rt.owner._status_guard_ref.update_instance_info
    update.side_effect = RuntimeError("status write failed")
    with pytest.raises(RuntimeError) as exc:
        await rt.owner.terminate_model("failed")
    assert exc.value is worker.cleanup_error
    assert rt.owner._initial_model_launches["failed"] is operation
    assert operation.workers["failed-rep1"] and worker.models
    update.side_effect = rt.guard.update_instance_info
    worker.cleanup_error = None
    await rt.owner.terminate_model("failed")
    assert all(not w.models for w in rt.workers)
    assert not rt.owner._initial_model_launches


@pytest.mark.asyncio
@pytest.mark.parametrize("hidden", ["launching", "rank0"])
async def test_failed_initial_launch_catalog_absence_is_not_cleanup(
    failed_launch_runtime, monkeypatch, hidden
):
    rt = failed_launch_runtime
    if hidden == "rank0":
        rt.kwargs.update(model_engine="vLLM", enable_xavier=True)
    operation = await _failed_launch(rt)
    rt.state.failed, rt.loss = False, "healthy"
    worker = rt.workers[0 if hidden == "rank0" else 1]
    uid = "failed-rank0" if hidden == "rank0" else "failed-rep1"
    assert uid in worker.models
    error = RuntimeError("cleanup unresolved")
    terminate = worker.terminate_model

    async def stop(model_uid, **kwargs):
        if model_uid == uid:
            raise error
        await terminate(model_uid, **kwargs)

    # Use the production catalog and launch-status semantics for hidden targets.
    inventory = SimpleNamespace(
        _model_uid_to_model_spec={},
        _model_uid_to_model={uid: object()} if hidden == "rank0" else {},
        _model_uid_launching_guard={uid: object()} if hidden == "launching" else {},
    )

    async def list_models(**kwargs):
        return await WorkerActor.list_models(inventory)

    async def launch_status(model_uid, **kwargs):
        return WorkerActor.get_model_launch_status(inventory, model_uid)

    monkeypatch.setattr(worker, "terminate_model", stop)
    monkeypatch.setattr(worker, "list_models", list_models)
    monkeypatch.setattr(worker, "get_model_launch_status", launch_status)
    with pytest.raises(RuntimeError) as exc:
        await rt.owner.terminate_model("failed")
    assert exc.value is error
    assert rt.owner._initial_model_launches["failed"] is operation
    assert operation.workers[uid] and uid in worker.models
    if hidden == "rank0":
        assert rt.actors

    worker.models.discard(uid)
    inventory._model_uid_to_model.clear()
    inventory._model_uid_launching_guard.clear()
    await rt.owner.terminate_model("failed")
    assert not rt.owner._initial_model_launches
    assert not rt.actors and all(not w.models for w in rt.workers)


@pytest.mark.asyncio
async def test_failed_initial_launch_stale_inventory(
    failed_launch_runtime, monkeypatch
):
    rt = failed_launch_runtime
    operation = await _failed_launch(rt)
    rt.state.failed, rt.loss = False, "healthy"
    worker = rt.workers[1]
    worker.cleanup_error = ConnectionRefusedError("cleanup unresolved")
    worker.models.clear()
    entered, release = asyncio.Event(), asyncio.Event()

    async def inventory(**kwargs):
        entered.set()
        await release.wait()
        return {}

    monkeypatch.setattr(worker, "list_models", inventory)
    cleanup = asyncio.create_task(rt.owner.terminate_model("failed"))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await rt.owner.add_worker(worker.address)
    finally:
        release.set()
    with pytest.raises(ConnectionRefusedError) as exc:
        await cleanup
    assert exc.value is worker.cleanup_error
    assert rt.owner._initial_model_launches["failed"] is operation
    assert operation.workers["failed-rep1"]
    worker.cleanup_error = None
    await rt.owner.terminate_model("failed")
    assert not rt.owner._initial_model_launches
    assert all(not w.models for w in rt.workers)


@pytest.mark.asyncio
async def test_failed_initial_launch_stale_retirement(
    failed_launch_runtime, monkeypatch
):
    rt = failed_launch_runtime
    operation = await _failed_launch(rt)
    rt.state.failed, rt.loss = False, "healthy"
    worker = rt.workers[1]
    registered = rt.owner._worker_address_to_worker[worker.address]
    generation = rt.owner._worker_metadata_generation[worker.address]
    entered, release = asyncio.Event(), asyncio.Event()
    resources = worker.get_extension_resources

    async def register(**kwargs):
        entered.set()
        await release.wait()
        return await resources(**kwargs)

    monkeypatch.setattr(worker, "get_extension_resources", register)
    registration = asyncio.create_task(rt.owner.add_worker(worker.address))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        removal = asyncio.create_task(
            rt.owner._remove_dead_worker(worker.address, registered)
        )
        await asyncio.sleep(0)
        assert not removal.done()
    finally:
        release.set()
    await registration
    assert rt.owner._worker_address_to_worker[worker.address] is registered
    assert rt.owner._worker_metadata_generation[worker.address] != generation
    assert await removal is False
    assert rt.owner._worker_address_to_worker[worker.address] is registered
    assert rt.owner._initial_model_launches["failed"] is operation
    assert operation.workers["failed-rep1"] and worker.models
    await rt.owner.terminate_model("failed")
    assert not rt.owner._initial_model_launches
    assert all(not w.models for w in rt.workers)


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
        _filter_worker_inventory=AsyncMock(side_effect=lambda ref, rows: rows),
        _replica_model_uid_to_worker={},
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


class ServingModel(ModelActor):
    """Use production request tracking and abort without a model engine."""

    def __init__(self):
        xo.StatelessActor.__init__(self)
        CancelMixin.__init__(self)
        self._model = SimpleNamespace()
        self._replica_model_uid = build_replica_model_uid("model", 0)
        self.entered = asyncio.Event()

    async def __post_create__(self):
        pass

    async def __pre_destroy__(self):
        pass

    async def run_request(self, request_id: str) -> None:
        self._add_running_task(request_id)
        self.entered.set()
        await asyncio.Event().wait()


class ServingWorker(WorkerActor):
    def __init__(self, model):
        xo.StatelessActor.__init__(self)
        self._model_uid_to_model = {build_replica_model_uid("model", 0): model}
        self._model_uid_to_model_status = {}
        self._control_fence, _ = setup_fence()

    async def __post_create__(self):
        pass

    async def __pre_destroy__(self):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["get_chat_model", "abort_request"])
@pytest.mark.parametrize("gate", ["pause", "gateway"])
async def test_serving_calls_remain_available_during_management(method, gate):
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    running = None
    try:
        model = await xo.create_actor(ServingModel, address=pool.external_address)
        worker = await xo.create_actor(
            ServingWorker, model, address=pool.external_address
        )
        supervisor = await xo.create_actor(
            InterceptedSupervisor, address=pool.external_address
        )
        receiver = pool._actors[supervisor.uid]
        owner, state, port = controlled_owner()
        receiver._controller_control = owner._controller_control
        receiver._pd_model_mapping = {}
        receiver._model_uid_to_replica_info = {
            "model": ReplicaInfo(1, itertools.cycle([0]), active_replica_ids=[0])
        }
        receiver._replica_model_uid_to_worker = {
            build_replica_model_uid("model", 0): worker
        }
        args, kwargs = (
            (("model", "key", "reservation"), {})
            if method == "get_chat_model"
            else (("model", "active-request"), {"block_duration": 0})
        )
        with pytest.raises(ControlSessionRejected, match="fence_pending"):
            await getattr(supervisor, method)(*args, **kwargs)
        await port.pause(1)
        state.phase = "active"
        await port.resume()
        if method == "abort_request":
            running = asyncio.create_task(model.run_request("active-request"))
            await asyncio.wait_for(pool._actors[model.uid].entered.wait(), 1)
        target = supervisor
        if gate == "pause":
            await port.pause(1)
            assert not port.status()["ready"]
            with pytest.raises(ControlSessionRejected, match="fence_pending"):
                await target.nested_cleanup("model-rep0")
        else:
            target = await xo.create_actor(
                SupervisorGateway, address=pool.external_address
            )
            await target.select_supervisor(supervisor)
            await target.prepare_supervisor_switch()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(target.nested_cleanup("model-rep0"), 0.1)
        assert await asyncio.wait_for(target.get_model("model"), 1) == model
        if method == "abort_request":
            assert not running.done()
        result = await asyncio.wait_for(getattr(target, method)(*args, **kwargs), 1)
        if method == "get_chat_model":
            assert result == (model, False)
        else:
            assert result == {"msg": "DONE"}
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(running, 1)
            assert running.cancelled()
    finally:
        if running is not None:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        await pool.stop()


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
    fence, _ = setup_fence()
    fence.authority.include_model_services = True
    worker._control_fence = fence
    assert worker.controller_inventory() == actual


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


def short_provider_deadline(control):
    invoke = control.invoke

    async def shortened(name, *args, **kwargs):
        kwargs.setdefault("timeout", 0.001)
        return await invoke(name, *args, **kwargs)

    control.invoke = shortened


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "destroy"])
async def test_service_actor_callbacks_preserve_errors_and_native_lifetime(operation):
    owner, _, _ = controlled_owner()
    control = owner._controller_control
    short_provider_deadline(control)
    entered, release = asyncio.Event(), asyncio.Event()

    async def callback(*args):
        entered.set()
        await release.wait()
        return "actor-result"

    setattr(control.authority, operation + "_model_service", callback)

    async def run():
        if operation == "create":
            return await SupervisorActor._create_model_service_actor(
                owner, object, address="worker:1234", uid="service"
            )
        return await SupervisorActor._destroy_model_service_actor(owner, "ref")

    task = asyncio.create_task(run())
    try:
        await entered.wait()
        await asyncio.sleep(0.01)
        assert not task.done()
        release.set()
        assert await task == "actor-result"
        setattr(
            control.authority,
            operation + "_model_service",
            AsyncMock(side_effect=ValueError("actor failure")),
        )
        with pytest.raises(ValueError, match="actor failure"):
            await run()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "timeout", "invalid"])
async def test_inventory_filter_failure_does_not_publish_worker(monkeypatch, failure):
    monkeypatch.setattr(extensions, "get_extensions", lambda: ())
    ref = SimpleNamespace(
        address="worker:1234",
        uid=b"worker",
        get_extension_resources=AsyncMock(return_value={"extensions": {}}),
    )
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=ref))
    owner = SupervisorActor()
    state = SimpleNamespace(owner=session(), phase="active", failed=False)
    authority = Provider(state, state.owner)
    if failure == "timeout":

        async def filter_inventory(*args):
            await asyncio.Event().wait()

        authority.filter_worker_inventory = filter_inventory
    else:
        authority.filter_worker_inventory = AsyncMock(
            side_effect=ValueError("provider detail") if failure == "error" else None,
            return_value=[{"replica_model_uid": "invented"}],
        )
    owner._controller_control = SupervisorControl(owner, authority)
    short_provider_deadline(owner._controller_control)
    owner._worker_address_to_worker[ref.address] = object()
    owner._extension_worker_resources[ref.address] = {"old": True}
    owner._worker_metadata[ref.address] = {"old": True}
    error = ValueError if failure == "invalid" else ControlSessionRejected
    with pytest.raises(error):
        await owner.add_worker(ref.address, replica_states=[])
    assert ref.address not in owner._worker_address_to_worker
    assert ref.address not in owner._extension_worker_resources
    assert ref.address not in owner._worker_metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("coordinators", ["model-rank0", None, ["bad"], [1]])
async def test_coordinator_validation_precedes_registry_mutation(coordinators):
    owner, _, port = controlled_owner()
    await port.pause(1)
    with pytest.raises(ValueError, match="coordinator"):
        await port.restore_worker(
            SimpleNamespace(address="worker:1234", uid=b"worker"),
            [],
            "boot",
            coordinators,
        )
    assert not owner._worker_address_to_worker
    assert not owner._ownership_registry().incarnations
    owner._filter_worker_inventory.assert_not_awaited()
    owner._restore_worker_inventory.assert_not_awaited()


@pytest.mark.asyncio
async def test_coordinator_restore_applies_validated_references():
    owner, _, port = controlled_owner()
    await port.pause(1)
    ref = SimpleNamespace(address="worker:1234", uid=b"worker")
    await port.restore_worker(ref, [], "boot", ["model-rank0"])
    assert owner._replica_model_uid_to_worker["model-rank0"].address == ref.address
    owner._restore_worker_inventory.assert_awaited_once()


@pytest.mark.asyncio
async def test_restore_worker_normalizes_missing_replicas_before_filtering():
    owner, _, port = controlled_owner()
    await port.pause(1)
    ref = SimpleNamespace(address="worker:1234", uid=b"worker")
    await port.restore_worker(ref, None, "boot")
    assert owner._filter_worker_inventory.await_args.args[1] == []
    assert owner._restore_worker_inventory.await_args.kwargs["replica_states"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["local", "remote"])
async def test_invalid_supervisor_context_has_stable_diagnostic(transport):
    from xoscar.core import LocalActorRef

    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    try:
        ref = await xo.create_actor(
            InterceptedSupervisor, address=pool.external_address, uid="supervisor"
        )
        target = LocalActorRef(pool._actors[ref.uid]) if transport == "local" else ref
        with pytest.raises(ControlSessionRejected, match="invalid_context"):
            await target.mark_replica_dead(
                "model-rep0", **{CONTROL_CONTEXT_KEY: {"version": 99}}
            )
        assert await ref.get_status() == []
    finally:
        await pool.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["local", "remote"])
@pytest.mark.parametrize("phase", ["fencing", "reconciling"])
async def test_worker_registration_metadata_requires_active_phase(
    transport, phase, tmp_path
):
    from xoscar.core import LocalActorRef

    from ..worker_registry import WorkerOwnership

    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    try:
        ref = await xo.create_actor(
            InterceptedSupervisor, address=pool.external_address, uid="supervisor"
        )
        actor = pool._actors[ref.uid]
        actor._worker_ownership = WorkerOwnership(str(tmp_path / "ownership.json"))
        actor._worker_address_to_worker = {}
        actor._controller_control.authority.state.phase = phase
        target = LocalActorRef(actor) if transport == "local" else ref
        with pytest.raises(ControlSessionRejected):
            await target.add_worker(
                "worker:1234", **{CONTROL_CONTEXT_KEY: command(session()).to_dict()}
            )
        assert actor._worker_address_to_worker == {}
    finally:
        await pool.stop()


class BatchSupervisor(SupervisorActor):
    def __init__(self):
        xo.StatelessActor.__init__(self)
        self._controller_control = None

    @xo.extensible
    def echo(self, value):
        return value


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [False, True])
async def test_active_local_registration_retains_one_admitted_call(
    monkeypatch, tmp_path, invalid
):
    from ..worker_registry import WorkerOwnership

    monkeypatch.setattr(extensions, "get_extensions", lambda: ())
    worker = SimpleNamespace(
        address="worker:1234",
        uid=b"worker",
        get_extension_resources=AsyncMock(return_value={"extensions": {}}),
        update_system_settings=AsyncMock(),
    )
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=worker))
    owner = SupervisorActor()
    owner.address = "supervisor:1234"
    owner._worker_ownership = WorkerOwnership(str(tmp_path / "ownership.json"))
    owner._refresh_worker_metadata = AsyncMock(return_value=True)
    owner._schedule_autostart = Mock()
    owner._status_guard_ref = None
    state = SimpleNamespace(owner=session(), phase="active", failed=False)
    authority = Provider(state, state.owner)
    authority.worker_registered = AsyncMock()
    authority.filter_worker_inventory = AsyncMock(return_value=[])
    owner._controller_control = SupervisorControl(owner, authority)
    owner._controller_control.fence.ready = True
    payload = {"version": 99} if invalid else command(state.owner).to_dict()
    call = owner.add_worker(worker.address, **{CONTROL_CONTEXT_KEY: payload})
    if invalid:
        with pytest.raises(ControlSessionRejected, match="invalid_context"):
            await call
        worker.get_extension_resources.assert_not_awaited()
    else:
        await asyncio.wait_for(call, 1)
        assert worker.address in owner._worker_address_to_worker
        authority.worker_registered.assert_awaited_once()
        authority.filter_worker_inventory.assert_awaited_once()


@pytest.mark.asyncio
async def test_unconfigured_supervisor_accepts_batch_without_kwargs():
    actor = BatchSupervisor()
    assert await actor.__on_receive__(("echo", 1, ([(1,), (2,)], None), None)) == [1, 2]


@pytest.mark.asyncio
async def test_invocation_kwargs_snapshot_does_not_mutate_default_call():
    from ..control_invocation import ControlInvocation, invoke_control_hook

    kwargs = {"original": 1}
    invocation = ControlInvocation(
        "worker:1234", "method", command(session()), (), kwargs
    )

    async def callback(invocation, call):
        invocation.kwargs["extra"] = 2
        return await call()

    assert await invoke_control_hook(
        SimpleNamespace(hook=callback),
        "hook",
        invocation,
        AsyncMock(side_effect=lambda: dict(kwargs)),
    ) == {"original": 1}
    assert kwargs == {"original": 1}
