# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# CI: linux

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ...model.llm.tests.test_weight_cache import FakeCachedModel
from ...model.llm.weight_cache import ModelReloadError
from ..exceptions import ModelNotReadyError
from ..launch_history_store import LaunchHistoryStore
from ..model import ModelActor, request_limit
from ..rpc_context import actor_call, rpc_context
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
    worker._model_uid_to_model_status = {
        "test-replica": ModelStatus(model_state="ready")
    }

    # Isolate supervisor/persistence orchestration from process creation. The
    # real generation transport and rollback are covered in test_model_generation.
    async def replace(
        uid, config, python_executable=None, operation_id=None, drain_timeout=300
    ):
        return await worker._model_uid_to_model[uid].reload(config, 300)

    worker._replace_model_generation = replace
    worker._status_guard_ref = None
    worker._persist_launch_args = MagicMock()
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._model_uid_to_replica_info = {"test": object()}
    supervisor._iter_active_replica_model_uids = lambda uid: iter(["test-replica"])
    supervisor._replica_model_uid_to_worker = {"test-replica": worker}
    supervisor._model_reload_status = {}
    supervisor._model_reload_tasks = {}
    supervisor._initial_model_launches = {}
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
@pytest.mark.parametrize("cached", [False, True])
async def test_reasoning_update_drains_streams_without_restarting(cached):
    from ...model.llm.reasoning_parser import ReasoningParser
    from ...model.llm.weight_cache import WeightCachedModel

    actor = make_actor()
    model = actor._model
    model._enable_weight_cache = cached
    if not cached:
        model._weight_cache = None
    parser = model.reasoning_parser = ReasoningParser(False, "<think>", "</think>")
    model.validate_reload = lambda patch: WeightCachedModel.validate_reload(
        model, patch
    )
    actor._serve_count = 1
    supervisor, worker = make_pipeline(actor)
    engine = model.engine
    await supervisor.reload_model("test", {"reasoning_content": True}, drain_timeout=2)
    await asyncio.sleep(0.05)
    assert model.reasoning_parser is parser
    assert not parser.reasoning_content
    actor._serve_count = 0
    await supervisor._model_reload_tasks["test"]
    assert model.engine is engine
    assert model.stops == 0 and not model.loads
    assert model.reasoning_parser.reasoning_content
    assert not parser.reasoning_content
    status = await supervisor.get_model_reload_status("test")
    assert status["status"] == "ready" and status["weights_reused"] is False
    assert worker._model_uid_to_launch_args["test-replica"]["reasoning_content"] is True
    supervisor._launch_history_store.update_autostart_launch_config.assert_called_once_with(
        "test", {"reasoning_content": True}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend,field",
    [("vllm", "limit_mm_per_prompt"), ("sglang", "limit_mm_data_per_request")],
)
async def test_normalized_image_limits_are_persisted_for_recovery_and_autostart(
    monkeypatch,
    backend,
    field,
):
    from ...model.llm.tests.test_runtime_settings import make_vision_model

    actor = make_actor()
    actor._model = make_vision_model(monkeypatch, backend)
    supervisor, worker = make_pipeline(actor)
    worker._model_uid_to_launch_args["test-replica"][field] = {
        "image": 2,
        "video": 0,
    }
    await supervisor.reload_model("test", {field: {"image": 8}})
    await supervisor._model_reload_tasks["test"]
    assert worker._model_uid_to_launch_args["test-replica"][field] == {
        "image": 8,
        "video": 0,
    }
    supervisor._launch_history_store.update_autostart_launch_config.assert_called_once_with(
        "test", {field: {"image": 8, "video": 0}}
    )


@pytest.mark.asyncio
async def test_merged_rich_limits_fail_before_scheduling_reload(monkeypatch):
    from ...model.llm.tests.test_runtime_settings import make_vision_model

    actor = make_actor()
    actor._model = make_vision_model(monkeypatch)
    actor._model._reload_config["limit_mm_per_prompt"]["video"] = {"count": 1}
    supervisor, worker = make_pipeline(actor)
    with pytest.raises(ValueError, match="image_limits"):
        await supervisor.reload_model("test", {"limit_mm_per_prompt": {"image": 8}})
    assert actor._model_state == "ready" and actor._model.stops == 0
    assert not supervisor._model_reload_tasks
    worker._persist_launch_args.assert_not_called()


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
        assert "test-replica" in worker._model_reload_active
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
        await worker.reload_model(
            "test-replica", {"max_num_seqs": 32}, 1, operation_id="one"
        )
    status = worker._model_uid_to_model_status.get("test-replica")
    assert (status.model_state if status else None) == {
        "error": "ready",
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

    @rpc_context
    async def abort(request_id, block_duration):
        actor._serve_count = 0
        return "DONE"

    worker._model_uid_to_model["test-replica"].abort_request = abort

    async def call(ref, method, *args, **kwargs):
        # Use the real metadata envelope; only the transport await is local.
        value = actor_call(ref, method, *args, **kwargs)
        return await value if asyncio.iscoroutine(value) else value

    monkeypatch.setattr("xinference.core.supervisor.actor_call", call)
    await supervisor.reload_model("test", {"max_num_seqs": 32}, drain_timeout=2)
    task = supervisor._model_reload_tasks["test"]
    await asyncio.sleep(0.05)
    assert (
        worker.get_model("test-replica") is worker._model_uid_to_model["test-replica"]
    )
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


@pytest.mark.asyncio
async def test_committed_environment_is_persisted_for_recovery_and_autostart(tmp_path):
    supervisor, worker = make_pipeline(make_actor())
    store = LaunchHistoryStore(str(tmp_path / "history.db"))
    store.upsert_autostart(
        {"launch": {"model_uid": "test", "model_name": "qwen"}}, "owner"
    )
    supervisor._launch_history_store = store
    worker._replace_model_generation = AsyncMock(return_value=True)
    await supervisor.reload_model(
        "test", {"max_num_seqs": 32}, python_executable="/opt/new/bin/python"
    )
    await supervisor._model_reload_tasks["test"]
    assert (
        worker._model_uid_to_launch_args["test-replica"]["model_python_executable"]
        == "/opt/new/bin/python"
    )
    launch = LaunchHistoryStore(store._db_path).list_autostart()[0]["launch"]
    assert launch["max_num_seqs"] == 32
    assert launch["model_python_executable"] == "/opt/new/bin/python"
    assert "model_python_executable" not in store.list(username="owner")[0]["data"]


@pytest.mark.asyncio
async def test_reload_status_survives_supervisor_operation_cache_loss():
    actor = make_actor()
    actor._reload_status = {
        "operation_id": "one",
        "status": "ready",
        "weights_reused": True,
    }
    supervisor, _ = make_pipeline(actor)
    assert (await supervisor.get_model_reload_status("test"))["operation_id"] == "one"


@pytest.mark.asyncio
async def test_reload_status_reports_runtime_rollback_after_cutover():
    actor = make_actor()
    actor._reload_status = {"operation_id": "one", "status": "error", "restored": True}
    supervisor, _ = make_pipeline(actor)
    supervisor._model_reload_status["test"] = {"operation_id": "one", "status": "ready"}
    status = await supervisor.get_model_reload_status("test")
    assert status["status"] == "error" and status["restored"]


def test_description_exposes_sharded_worker_count():
    worker = WorkerActor.__new__(WorkerActor)
    worker._model_uid_to_model_spec = {"test": {"model_engine": "vLLM"}}
    worker._model_uid_to_launch_args = {"test": {"n_worker": 2}}
    assert worker.describe_model("test")["n_worker"] == 2
    assert "n_worker" not in worker._model_uid_to_model_spec["test"]


@pytest.mark.asyncio
async def test_worker_forces_pool_removal_during_engine_rebuild():
    pool = SimpleNamespace(remove_sub_pool=AsyncMock(), sub_processes={})
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


@pytest.mark.asyncio
async def test_late_cancelled_reload_cannot_remove_relaunched_model_job():
    supervisor, worker = make_pipeline(make_actor())
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed_rpc(*args):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            # Simulate an actor RPC that cannot promptly acknowledge cancel.
            await release.wait()

    worker._model_uid_to_model["test-replica"].reload = delayed_rpc

    async def terminate(*args):
        worker._model_uid_to_model.clear()
        worker._model_uid_to_model_status.clear()

    supervisor._terminate_model = terminate
    await supervisor.reload_model("test", {"max_num_seqs": 32})
    old_task = supervisor._model_reload_tasks["test"]
    await entered.wait()
    await supervisor.terminate_model("test")
    assert "test" not in supervisor._model_reload_tasks
    new_task = asyncio.create_task(asyncio.Event().wait())
    supervisor._model_reload_tasks["test"] = new_task
    try:
        release.set()
        await old_task
        assert supervisor._model_reload_tasks["test"] is new_task
    finally:
        new_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await new_task


@pytest.mark.asyncio
async def test_force_termination_reaps_native_children_after_parent_dies(tmp_path):
    import subprocess
    import sys
    import time

    import psutil

    pid_file = tmp_path / "native.pid"
    child_code = (
        "import os, time\nfrom pathlib import Path\n"
        f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "while True: time.sleep(0.1)\n"
    )
    parent_code = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        "while True: time.sleep(0.1)\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", parent_code])
    child = None
    try:
        deadline = time.monotonic() + 10
        while not pid_file.exists():
            assert time.monotonic() < deadline
            await asyncio.sleep(0.05)
        child = psutil.Process(int(pid_file.read_text()))

        async def remove_pool(address, force):
            assert address == "model-pool" and force
            parent.kill()
            parent.wait(timeout=5)
            # The pool's parent is now gone; its native child is still alive.
            assert child.is_running()

        pool = SimpleNamespace(
            remove_sub_pool=AsyncMock(side_effect=remove_pool),
            sub_processes={"model-pool": SimpleNamespace(pid=parent.pid)},
        )
        worker = WorkerActor("supervisor", None, pool, [])
        worker.get_supervisor_ref = AsyncMock()
        worker._status_guard_ref = SimpleNamespace(
            update_instance_info=AsyncMock(), update_replica_status=AsyncMock()
        )
        worker._model_uid_to_model_status["test-0"] = ModelStatus(
            model_state="reloading"
        )
        worker._model_uid_to_model["test-0"] = SimpleNamespace(stop=AsyncMock())
        worker._model_uid_to_addr["test-0"] = "model-pool"
        worker._remove_persisted_launch_args = MagicMock()
        await asyncio.wait_for(worker.terminate_model("test-0"), timeout=10)
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)
        if child is not None and child.is_running():
            child.kill()


@pytest.mark.asyncio
async def test_idle_reload_status_survives_worker_handoff():
    supervisor, worker = make_pipeline(make_actor())
    worker.get_model_reload_status = AsyncMock(side_effect=ConnectionError("handoff"))
    assert await supervisor.get_model_reload_status("test") == {"status": "idle"}


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [None, "/old/python"])
async def test_restored_generation_updates_worker_and_autostart_environment(override):
    supervisor, worker = make_pipeline(make_actor())
    supervisor._model_reload_status["test"] = {
        "operation_id": "upgrade",
        "status": "ready",
    }
    status = {"operation_id": "upgrade", "status": "error", "restored": True}
    ref = worker._model_uid_to_model["test-replica"]
    worker._model_uid_to_model["test-rep0"] = worker._model_uid_to_model.pop(
        "test-replica"
    )
    worker._model_uid_to_launch_args["test-rep0"] = (
        worker._model_uid_to_launch_args.pop("test-replica")
    )
    ref.export_model_generation = AsyncMock(
        return_value={
            "config": {"max_num_seqs": 16},
            "python_executable": "/old/python",
            "model_python_executable": override,
        }
    )
    ref.get_reload_status = AsyncMock(return_value=status)
    worker.get_supervisor_ref = AsyncMock(return_value=supervisor)
    worker._model_uid_to_launch_args["test-rep0"].update(
        max_num_seqs=32, model_python_executable="/failed/python"
    )
    await worker._restore_model_generation_config("test-rep0", ref)
    assert worker._model_uid_to_launch_args["test-rep0"]["max_num_seqs"] == 16
    launch = worker._model_uid_to_launch_args["test-rep0"]
    assert launch.get("model_python_executable") == override
    if override is None:
        assert "model_python_executable" not in launch
    supervisor._launch_history_store.update_autostart_launch_config.assert_called_once_with(
        "test", {"max_num_seqs": 16, "model_python_executable": override}
    )
    assert supervisor._model_reload_status["test"]["restored"]
    await supervisor.restore_model_reload_config(
        "test-rep0", {}, {"operation_id": "stale"}
    )
    assert (
        supervisor._launch_history_store.update_autostart_launch_config.call_count == 1
    )


@pytest.mark.asyncio
async def test_parser_reload_after_failed_upgrade_ignores_old_anchor_status():
    from ...model.llm.reasoning_parser import ReasoningParser

    anchor_actor, active_actor = make_actor(), make_actor()
    active_actor._model.reasoning_parser = ReasoningParser(False, "<think>", "</think>")
    supervisor, worker = make_pipeline(anchor_actor)
    ref = worker._model_uid_to_model["test-replica"]
    ref.reload = active_actor.reload
    ref.validate_reload = active_actor.validate_reload

    async def replace(uid, config, python_executable, operation_id, drain_timeout):
        anchor_actor._reload_status = {"operation_id": operation_id, "status": "ready"}
        if config["max_num_seqs"] == 64:
            anchor_actor._reload_status.update(status="error", restored=True)
            raise ModelReloadError("rejected replacement", restored=True)
        return True

    worker._replace_model_generation = replace
    await supervisor.reload_model("test", {"max_num_seqs": 32})
    await supervisor._model_reload_tasks["test"]
    assert (await supervisor.get_model_reload_status("test"))["status"] == "ready"
    await supervisor.reload_model("test", {"max_num_seqs": 64})
    await supervisor._model_reload_tasks["test"]
    assert (await supervisor.get_model_reload_status("test"))["restored"]
    await supervisor.reload_model("test", {"reasoning_content": True})
    await supervisor._model_reload_tasks["test"]
    assert (await supervisor.get_model_reload_status("test"))["status"] == "ready"
    assert worker._model_uid_to_launch_args["test-replica"]["reasoning_content"] is True
    assert worker._persist_launch_args.call_count == 2
    assert (
        supervisor._launch_history_store.update_autostart_launch_config.call_args.args
        == ("test", {"reasoning_content": True})
    )


@pytest.mark.asyncio
async def test_legacy_supervisor_cannot_replace_engine_without_operation_id():
    from ...model.llm.reasoning_parser import ReasoningParser

    actor = make_actor()
    actor._model.reasoning_parser = ReasoningParser(False, "<think>", "</think>")
    _, worker = make_pipeline(actor)
    worker._replace_model_generation = AsyncMock()
    with pytest.raises(ValueError, match="upgrade them first"):
        await worker.reload_model("test-replica", {"max_num_seqs": 32}, 1)
    worker._replace_model_generation.assert_not_called()
    worker._persist_launch_args.assert_not_called()
    assert (
        await worker.reload_model("test-replica", {"reasoning_content": True}, 1)
        is False
    )


@pytest.mark.asyncio
async def test_recovery_survives_concurrent_map_changes_and_model_failures():
    pool = SimpleNamespace(remove_sub_pool=AsyncMock())
    worker = WorkerActor("supervisor", None, pool, [])
    first = {"dead"}
    second = {"dead"}
    worker._model_uid_to_subpool_addresses = {"one": first, "two": second}

    async def fail(address):
        worker._model_uid_to_subpool_addresses["new"] = {"new-pool"}
        worker._model_uid_to_subpool_addresses.pop("one")
        raise ConnectionError("anchor unavailable")

    worker._model_uid_to_model = {
        "one": SimpleNamespace(handle_generation_pool_exit=AsyncMock(side_effect=fail)),
        "two": SimpleNamespace(
            handle_generation_pool_exit=AsyncMock(return_value="restored")
        ),
    }
    worker._restore_model_generation_config = AsyncMock(
        side_effect=RuntimeError("persist failed")
    )
    await worker.recover_sub_pool("dead")
    worker._restore_model_generation_config.assert_awaited_once()
    assert not first and not second
    assert worker._model_uid_to_subpool_addresses["new"] == {"new-pool"}


@pytest.mark.asyncio
async def test_hung_anchor_does_not_block_other_subpool_recovery(monkeypatch):
    from .. import worker as module

    wait_for = module.xo.wait_for

    async def bounded_rpc(awaitable, timeout):
        assert timeout == 5
        return await wait_for(awaitable, 0.01)

    async def hung_rpc(address):
        await asyncio.Event().wait()

    monkeypatch.setattr(module.xo, "wait_for", bounded_rpc)
    worker = WorkerActor(
        "supervisor", None, SimpleNamespace(remove_sub_pool=AsyncMock()), []
    )
    worker._model_uid_to_subpool_addresses = {"hung": {"dead"}, "live": {"dead"}}
    worker._model_uid_to_model = {
        "hung": SimpleNamespace(
            handle_generation_pool_exit=AsyncMock(side_effect=hung_rpc)
        ),
        "live": SimpleNamespace(
            handle_generation_pool_exit=AsyncMock(return_value="restored")
        ),
    }
    worker._restore_model_generation_config = AsyncMock()
    await asyncio.wait_for(worker.recover_sub_pool("dead"), 1)
    worker._restore_model_generation_config.assert_awaited_once_with(
        "live", worker._model_uid_to_model["live"]
    )
    assert not any(worker._model_uid_to_subpool_addresses.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("previous_failure", ["unreachable", "timeout"])
@pytest.mark.parametrize("dead_address", ["active", "candidate"])
async def test_dead_active_generation_recovers_when_previous_probe_fails(
    monkeypatch, previous_failure, dead_address
):
    from .. import worker as module
    from .test_model_generation import ServingGeneration

    wait_for = module.xo.wait_for

    async def bounded_rpc(awaitable, timeout):
        return await wait_for(awaitable, 1 if timeout == 5 else 0.01)

    async def probe_previous():
        if previous_failure == "unreachable":
            raise ConnectionError("previous generation unavailable")
        await asyncio.Event().wait()

    monkeypatch.setattr(module.xo, "wait_for", bounded_rpc)
    monkeypatch.setattr(module, "_parse_gpu_indices", lambda _: [])
    # Keep this CPU-only regression independent of installed NVML/drivers.
    monkeypatch.setitem(sys.modules, "pynvml", None)
    anchor = ServingGeneration("anchor")
    anchor._serving_generation = SimpleNamespace(uid=b"active", address="active")
    anchor._retiring_generations = [
        SimpleNamespace(
            uid=b"previous",
            address="previous",
            generation_serve_count=AsyncMock(side_effect=probe_previous),
        )
    ]
    worker = WorkerActor(
        "supervisor", None, SimpleNamespace(remove_sub_pool=AsyncMock()), []
    )
    worker._model_uid_to_model = {"test-rep0": anchor}
    worker._model_uid_to_addr = {"test-rep0": "anchor"}
    worker._model_uid_to_subpool_addresses = {
        "test-rep0": {"active", "previous", "candidate"}
    }
    launch = {"model_uid": "test-rep0"}
    worker._model_uid_to_launch_args = {"test-rep0": launch}
    worker.terminate_model = AsyncMock()
    worker.recover_model = AsyncMock()
    await worker.recover_sub_pool(dead_address)
    if dead_address == "active":
        worker.terminate_model.assert_awaited_once_with("test-rep0", is_model_die=True)
        worker.recover_model.assert_awaited_once_with(launch)
    else:
        worker.terminate_model.assert_not_awaited()
        worker.recover_model.assert_not_awaited()
        anchor._retiring_generations[0].generation_serve_count.assert_not_awaited()


@pytest.mark.asyncio
async def test_terminate_captures_native_children_in_every_generation(monkeypatch):
    import psutil

    from .. import worker as module

    pool = SimpleNamespace(
        remove_sub_pool=AsyncMock(),
        sub_processes={
            name: SimpleNamespace(pid=pid)
            for name, pid in (("anchor", 100), ("serving", 200), ("candidate", 300))
        },
    )
    parents = {pid: MagicMock(pid=pid) for pid in (100, 200, 300)}
    children = [MagicMock(pid=pid) for pid in (101, 201, 301)]
    for parent, child in zip(parents.values(), children):
        parent.children.return_value = [child]
    monkeypatch.setattr(psutil, "Process", lambda pid: parents[pid])
    stop_children = MagicMock()
    monkeypatch.setattr(module, "_stop_reload_children", stop_children)
    worker = WorkerActor("supervisor", None, pool, [])
    worker.get_supervisor_ref = AsyncMock()
    worker._status_guard_ref = SimpleNamespace(
        update_instance_info=AsyncMock(), update_replica_status=AsyncMock()
    )
    worker._model_uid_to_model_status["test-rep0"] = ModelStatus(
        model_state="reloading"
    )
    worker._model_uid_to_model["test-rep0"] = SimpleNamespace()
    worker._model_uid_to_addr["test-rep0"] = "anchor"
    worker._model_uid_to_subpool_addresses["test-rep0"] = set(pool.sub_processes)
    worker._remove_persisted_launch_args = MagicMock()
    await worker.terminate_model("test-rep0")
    for parent in parents.values():
        parent.suspend.assert_called_once()
        parent.kill.assert_called_once()
    assert {child.pid for child in stop_children.call_args.args[0]} == {101, 201, 301}
    assert {call.args[0] for call in pool.remove_sub_pool.await_args_list} == set(
        pool.sub_processes
    )
