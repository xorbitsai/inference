# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ...model.llm.tests.test_weight_cache import FakeCachedModel
from ...model.llm.weight_cache import ModelReloadError
from ..exceptions import ModelNotReadyError
from ..launch_history_store import LaunchHistoryStore
from ..model import ModelActor, request_limit
from ..supervisor import SupervisorActor
from ..worker import ModelStatus, WorkerActor


def make_actor():
    actor = ModelActor.__new__(ModelActor)
    actor._model = FakeCachedModel()
    actor._model_state = "ready"
    actor._serve_count = 0
    return actor


def make_pipeline(actor):
    ref = SimpleNamespace(
        reload=actor.reload,
        validate_reload=actor.validate_reload,
        get_reload_config=AsyncMock(side_effect=actor.get_reload_config),
        get_reload_status=AsyncMock(side_effect=actor.get_reload_status),
    )
    worker = WorkerActor.__new__(WorkerActor)
    worker._model_uid_to_model = {"test-replica": ref}
    worker._model_uid_to_launch_args = {
        "test-replica": {
            "model_uid": "test-replica",
            "gpu_idx": [0],
            "max_num_seqs": 16,
        }
    }
    worker._model_uid_to_model_status = {}
    worker._status_guard_ref = None
    worker._persist_launch_args = MagicMock()
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._model_uid_to_replica_info = {"test": object()}
    supervisor._iter_active_replica_model_uids = lambda uid: iter(["test-replica"])
    supervisor._replica_model_uid_to_worker = {"test-replica": worker}
    supervisor._model_reload_status = {}
    supervisor._model_reload_tasks = {}
    supervisor._autostart_store_lock = asyncio.Lock()
    supervisor._launch_history_store = SimpleNamespace(
        update_autostart_launch_config=MagicMock()
    )
    lock = asyncio.Lock()
    supervisor._get_model_replica_lock = lambda uid: lock
    supervisor._invalidate_list_models_debounce_cache = MagicMock()
    return supervisor, worker


@pytest.mark.asyncio
async def test_reload_job_drains_requests_then_reuses_identity():
    actor = make_actor()
    actor._serve_count = 1  # A stream still owns its serving slot.
    supervisor, worker = make_pipeline(actor)
    old_engine = actor._model.engine
    weights = actor._model._weight_cache.weights
    job = await supervisor.reload_model("test", {"max_num_seqs": 32}, drain_timeout=2)
    assert job["status"] == "reloading"
    task = supervisor._model_reload_tasks["test"]
    await asyncio.sleep(0.05)
    assert actor._model.engine is old_engine
    assert (await supervisor.get_model_reload_status("test"))["stage"] == "draining"
    with pytest.raises(RuntimeError, match="already"):
        # Avoid waiting for the placement lock: pending jobs must be rejected.
        await asyncio.wait_for(
            supervisor.reload_model("test", {"max_num_seqs": 8}), timeout=0.1
        )
    actor._serve_count = 0
    await task
    status = await supervisor.get_model_reload_status("test")
    assert status["status"] == "ready" and status["weights_reused"]
    assert status["operation_id"] == job["operation_id"]
    assert actor._model._weight_cache.weights is weights
    assert worker._model_uid_to_model["test-replica"] is not None
    assert worker._model_uid_to_launch_args["test-replica"] == {
        "model_uid": "test-replica",
        "gpu_idx": [0],
        "max_num_seqs": 32,
    }
    worker._persist_launch_args.assert_called_once()


@pytest.mark.asyncio
async def test_preflight_leaves_original_engine_serving():
    actor = make_actor()
    supervisor, _ = make_pipeline(actor)
    with pytest.raises(ValueError):
        await supervisor.reload_model("test", {"tensor_parallel_size": 2})
    assert actor._model_state == "ready"
    assert actor._model.stops == 0
    assert not supervisor._model_reload_tasks


@pytest.mark.asyncio
async def test_drain_timeout_preserves_engine():
    actor = make_actor()
    actor._serve_count = 1
    with pytest.raises(ModelReloadError) as exc:
        await actor.reload({"max_num_seqs": 32}, drain_timeout=0.01)
    assert exc.value.restored
    assert actor._model_state == "ready"
    assert actor._model.stops == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_restore", [False, True])
async def test_failed_job_reports_restoration_and_does_not_persist(fail_restore):
    actor = make_actor()
    actor._model.fail_restore = fail_restore
    supervisor, worker = make_pipeline(actor)
    await supervisor.reload_model("test", {"max_num_seqs": 64})
    await supervisor._model_reload_tasks["test"]
    status = await supervisor.get_model_reload_status("test")
    assert status["status"] == "error"
    assert status["restored"] is not fail_restore
    assert actor._model_state == ("error" if fail_restore else "ready")
    assert worker._model_uid_to_launch_args["test-replica"]["max_num_seqs"] == 16
    worker._persist_launch_args.assert_not_called()


@pytest.mark.asyncio
async def test_requests_rejected_during_reload():
    actor = make_actor()
    actor._model_state = "reloading"
    called = AsyncMock()
    with pytest.raises(ModelNotReadyError):
        await request_limit(called)(actor)
    called.assert_not_called()
    assert actor._serve_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"replica": 2},
        {"n_worker": 2},
        {"model_engine": "Transformers"},
        {"model_type": "embedding"},
        {"enable_weight_cache": "yes"},
    ],
)
async def test_invalid_cache_launch_rejected_before_gpu_allocation(overrides):
    supervisor = SupervisorActor.__new__(SupervisorActor)
    args = dict(
        model_uid="test",
        model_name="qwen2.5-instruct",
        model_size_in_billions="0_5",
        model_format="pytorch",
        quantization="none",
        model_engine="vLLM",
        model_type="LLM",
        enable_weight_cache=True,
    )
    args.update(overrides)
    with pytest.raises(ValueError):
        await supervisor.launch_builtin_model(**args)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["draining", "loading"])
async def test_termination_interrupts_reload_without_waiting_for_engine(stage):
    actor = make_actor()
    supervisor, worker = make_pipeline(actor)
    entered = asyncio.Event()
    if stage == "draining":
        actor._serve_count = 1
    else:

        async def hung_load(*args):
            actor._model_state = "reloading"
            entered.set()
            await asyncio.Event().wait()

        worker._model_uid_to_model["test-replica"].reload = hung_load

    async def terminate(uid, suppress_exception):
        assert (
            worker._model_uid_to_model_status["test-replica"].model_state == "reloading"
        )
        await worker._update_model_state("test-replica", "stopping")
        worker._model_uid_to_model.clear()
        worker._model_uid_to_model_status.clear()

    supervisor._terminate_model = terminate
    await supervisor.reload_model("test", {"max_num_seqs": 32}, drain_timeout=3600)
    task = supervisor._model_reload_tasks["test"]
    if stage == "loading":
        await entered.wait()
    else:
        await asyncio.sleep(0.05)
    await asyncio.wait_for(supervisor.terminate_model("test"), timeout=0.2)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not worker._model_uid_to_model_status
    assert not supervisor._model_reload_tasks
    worker._persist_launch_args.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["error", "recovered", "terminated", "stopping"])
async def test_unknown_reload_failure_never_revives_stale_actor(outcome):
    actor = make_actor()
    _, worker = make_pipeline(actor)

    async def fail(*args):
        if outcome == "recovered":
            worker._model_uid_to_model["test-replica"] = object()
            worker._model_uid_to_model_status["test-replica"] = SimpleNamespace(
                model_state="loading"
            )
        elif outcome == "terminated":
            worker._model_uid_to_model.clear()
            worker._model_uid_to_model_status.clear()
        elif outcome == "stopping":
            await worker._update_model_state("test-replica", "stopping")
        raise KeyError("actor disappeared")

    worker._model_uid_to_model["test-replica"].reload = fail
    with pytest.raises(KeyError):
        await worker.reload_model("test-replica", {"max_num_seqs": 32}, 1)
    status = worker._model_uid_to_model_status.get("test-replica")
    assert (status.model_state if status else None) == {
        "error": "error",
        "recovered": "loading",
        "terminated": None,
        "stopping": "stopping",
    }[outcome]
    worker._persist_launch_args.assert_not_called()


@pytest.mark.asyncio
async def test_abort_request_can_finish_draining_reload(monkeypatch):
    actor = make_actor()
    supervisor, worker = make_pipeline(actor)
    supervisor._pd_model_mapping = {}
    actor._serve_count = 1

    async def abort(request_id, block_duration):
        actor._serve_count = 0
        return "DONE"

    worker._model_uid_to_model["test-replica"].abort_request = abort

    async def call(ref, method, *args, **kwargs):
        kwargs.pop("_rpc_operation_request_id", None)
        value = getattr(ref, method)(*args, **kwargs)
        return await value if asyncio.iscoroutine(value) else value

    monkeypatch.setattr("xinference.core.supervisor.actor_call", call)
    await supervisor.reload_model("test", {"max_num_seqs": 32}, drain_timeout=2)
    task = supervisor._model_reload_tasks["test"]
    await asyncio.sleep(0.05)
    with pytest.raises(ModelNotReadyError):
        worker.get_model("test-replica")
    assert await supervisor.abort_request("test", "request") == {"msg": "DONE"}
    await task
    assert actor._model_state == "ready"


@pytest.mark.asyncio
async def test_reload_updates_autostart_only_after_commit(tmp_path):
    supervisor, worker = make_pipeline(make_actor())
    store = LaunchHistoryStore(str(tmp_path / "history.db"))
    supervisor._launch_history_store = store
    store.upsert_autostart(
        {
            "launch": {"model_uid": "test", "model_name": "qwen", "max_num_seqs": 16},
            "priority": 42,
        },
        "owner",
    )
    await supervisor.reload_model("test", {"max_num_seqs": 64})
    await supervisor._model_reload_tasks["test"]
    assert store.list_autostart()[0]["launch"]["max_num_seqs"] == 16
    await supervisor.reload_model("test", {"max_num_seqs": 32})
    await supervisor._model_reload_tasks["test"]
    entry = store.list_autostart()[0]
    assert entry["launch"]["max_num_seqs"] == 32
    assert entry["created_by"] == "owner" and entry["priority"] == 42
    worker._persist_launch_args.assert_called_once()


def test_description_exposes_sharded_worker_count():
    worker = WorkerActor.__new__(WorkerActor)
    worker._model_uid_to_model_spec = {"test": {"model_engine": "vLLM"}}
    worker._model_uid_to_launch_args = {"test": {"n_worker": 2}}
    assert worker.describe_model("test")["n_worker"] == 2
    assert "n_worker" not in worker._model_uid_to_model_spec["test"]


@pytest.mark.asyncio
async def test_worker_forces_pool_removal_during_engine_rebuild():
    pool = SimpleNamespace(remove_sub_pool=AsyncMock())
    worker = WorkerActor("supervisor", None, pool, [])
    worker.get_supervisor_ref = AsyncMock()
    worker._status_guard_ref = SimpleNamespace(
        update_instance_info=AsyncMock(), update_replica_status=AsyncMock()
    )
    worker._model_uid_to_model_status["test-0"] = ModelStatus(model_state="reloading")
    ref = SimpleNamespace(stop=AsyncMock(), get_pool_addresses=AsyncMock())
    worker._model_uid_to_model["test-0"] = ref
    worker._model_uid_to_addr["test-0"] = "model-pool"
    worker._remove_persisted_launch_args = MagicMock()
    await asyncio.wait_for(worker.terminate_model("test-0"), timeout=0.2)
    ref.stop.assert_not_called()
    ref.get_pool_addresses.assert_not_called()
    pool.remove_sub_pool.assert_awaited_once_with("model-pool", force=True)
    assert not worker._model_uid_to_model_status and not worker._model_uid_to_model
