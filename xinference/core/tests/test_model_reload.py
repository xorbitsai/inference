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
from ..model import ModelActor, request_limit
from ..supervisor import SupervisorActor
from ..worker import WorkerActor


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
    worker._update_model_state = AsyncMock()
    worker._persist_launch_args = MagicMock()
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._model_uid_to_replica_info = {"test": object()}
    supervisor._iter_active_replica_model_uids = lambda uid: iter(["test-replica"])
    supervisor._replica_model_uid_to_worker = {"test-replica": worker}
    supervisor._model_reload_status = {}
    supervisor._model_reload_tasks = {}
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
