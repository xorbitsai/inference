# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# CI: linux
"""Replica orchestration uses real model/Worker commit and rollback paths."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from ..launch_history_store import LaunchHistoryStore
from ..utils import build_replica_model_uid
from .test_model_reload import make_actor, make_pipeline


def make_replicas(count=2):
    actors, workers = [], []
    uids = [build_replica_model_uid("test", index) for index in range(count)]
    for uid in uids:
        actor = make_actor()
        owner, worker = make_pipeline(actor)
        for name in (
            "_model_uid_to_model",
            "_model_uid_to_launch_args",
            "_model_uid_to_model_status",
        ):
            mapping = getattr(worker, name)
            mapping[uid] = mapping.pop("test-replica")
        worker._model_uid_to_launch_args[uid]["model_uid"] = uid
        actors.append(actor)
        workers.append(worker)
    owner._iter_active_replica_model_uids = lambda model_uid: iter(uids)
    owner._replica_model_uid_to_worker = dict(zip(uids, workers))
    return owner, workers, actors, uids


@pytest.mark.asyncio
async def test_all_replicas_preflight_before_any_engine_changes():
    owner, workers, actors, _ = make_replicas()
    workers[1].validate_model_reload = AsyncMock(
        side_effect=ValueError("unsupported patch")
    )
    engines = [actor._model.engine for actor in actors]
    for actor in actors:
        actor._reload_status = {"operation_id": "previous", "status": "ready"}
    with pytest.raises(ValueError, match="unsupported patch"):
        await owner.reload_model("test", {"max_num_seqs": 32})
    assert not owner._model_reload_tasks
    assert [actor._model.engine for actor in actors] == engines
    assert all(not actor._model.stops for actor in actors)
    assert all(
        actor._reload_status == {"operation_id": "previous", "status": "ready"}
        for actor in actors
    )


@pytest.mark.asyncio
async def test_reload_is_sequential_and_fences_scale():
    owner, workers, actors, uids = make_replicas()
    entered, release = asyncio.Event(), asyncio.Event()
    replace = workers[0]._replace_model_generation

    async def gated(*args, **kwargs):
        entered.set()
        await release.wait()
        return await replace(*args, **kwargs)

    workers[0]._replace_model_generation = gated
    await owner.reload_model("test", {"max_num_seqs": 32})
    task = owner._model_reload_tasks["test"]
    await entered.wait()
    assert actors[1]._model._reload_config["max_num_seqs"] == 16
    assert actors[1]._model.stops == 0
    assert (await owner.get_model_reload_status("test"))["completed_replicas"] == 0
    with pytest.raises(RuntimeError, match="reload in progress"):
        await owner.add_model_replica("test")
    with pytest.raises(RuntimeError, match="reload in progress"):
        await owner.terminate_model_replica("test", 1)
    release.set()
    await task
    status = await owner.get_model_reload_status("test")
    assert status["status"] == "ready"
    assert status["completed_replicas"] == status["replica_count"] == 2
    assert [row["replica_uid"] for row in status["replicas"]] == uids
    assert all(row["status"] == "ready" for row in status["replicas"])
    assert all(actor._model._reload_config["max_num_seqs"] == 32 for actor in actors)
    owner._launch_history_store.update_autostart_launch_config.assert_called_once()


@pytest.mark.asyncio
async def test_failed_replica_stops_rollout_and_retry_commits_desired_config(tmp_path):
    owner, workers, actors, uids = make_replicas(3)
    store = LaunchHistoryStore(str(tmp_path / "history.db"))
    store.upsert_autostart(
        {
            "launch": {
                "model_uid": "test",
                "model_name": "qwen",
                "replica": 3,
                "max_num_seqs": 16,
            }
        },
        "owner",
    )
    owner._launch_history_store = store
    load = actors[1]._model.load

    def fail_candidate():
        if actors[1]._model._model_config["max_num_seqs"] == 32:
            raise RuntimeError("candidate failed")
        load()

    actors[1]._model.load = fail_candidate
    await owner.reload_model("test", {"max_num_seqs": 32})
    await owner._model_reload_tasks["test"]
    status = await owner.get_model_reload_status("test")
    assert status["status"] == "error" and status["partial"]
    assert status["completed_replicas"] == 1
    assert [row["status"] for row in status["replicas"]] == [
        "ready",
        "error",
        "skipped",
    ]
    assert status["replicas"][1]["restored"]
    assert [
        workers[i]._model_uid_to_launch_args[uid]["max_num_seqs"]
        for i, uid in enumerate(uids)
    ] == [32, 16, 16]
    assert actors[2]._model.stops == 0
    assert store.list_autostart()[0]["launch"]["max_num_seqs"] == 16
    actors[1]._model.load = load
    await owner.reload_model("test", {"max_num_seqs": 32})
    await owner._model_reload_tasks["test"]
    assert (await owner.get_model_reload_status("test"))["status"] == "ready"
    assert store.list_autostart()[0]["launch"]["max_num_seqs"] == 32


@pytest.mark.asyncio
async def test_metadata_exposes_common_and_mixed_values_without_nested_python_paths():
    owner, _, actors, uids = make_replicas()
    actors[1]._model._reload_config["max_num_seqs"] = 32
    metadata = await owner.get_model_reload_config("test")
    assert metadata["enabled"]
    assert "max_num_seqs" not in metadata["model_config"]
    assert metadata["mixed_parameters"] == ["max_num_seqs"]
    assert [row["replica_uid"] for row in metadata["replicas"]] == uids
    assert all("python_executable" not in row for row in metadata["replicas"])


@pytest.mark.asyncio
async def test_mixed_engines_accept_shared_parser_changes_through_rest():
    from unittest.mock import MagicMock

    from ...api.restful_api import RESTfulAPI
    from ...model.llm.reasoning_parser import ReasoningParser

    owner, _, actors, _ = make_replicas()
    actors[1]._model._weight_cache_engine = "sglang"
    for actor in actors:
        actor._model.reasoning_parser = ReasoningParser(False, "<think>", "</think>")
    metadata = await owner.get_model_reload_config("test")
    assert metadata["engine"] == "vllm"
    assert metadata["parameters"] == {"reasoning_content": "bool"}
    api = RESTfulAPI.__new__(RESTfulAPI)
    api._get_supervisor_ref = AsyncMock(return_value=owner)
    response = await api.reload_model(
        "test",
        MagicMock(
            json=AsyncMock(return_value={"model_config": {"reasoning_content": True}})
        ),
    )
    assert response.status_code == 202
    await owner._model_reload_tasks["test"]
    assert (await owner.get_model_reload_status("test"))["completed_replicas"] == 2


@pytest.mark.asyncio
async def test_environment_upgrade_preflights_retention_on_every_replica():
    owner, _, actors, _ = make_replicas()
    actors[1]._model._enable_weight_cache = False
    actors[1]._model._weight_cache = None
    with pytest.raises(ValueError, match="no retained GPU weights"):
        await owner.reload_model("test", {}, python_executable="/new/python")
    assert not owner._model_reload_tasks
    assert all(actor._model.stops == 0 for actor in actors)


@pytest.mark.asyncio
async def test_normalized_replica_limits_and_model_autostart_preserve_other_modalities(
    tmp_path,
):
    owner, workers, _, uids = make_replicas()
    field = "limit_mm_per_prompt"
    store = LaunchHistoryStore(str(tmp_path / "history.db"))
    store.upsert_autostart(
        {
            "launch": {
                "model_uid": "test",
                "model_name": "qwen",
                field: {"image": 2, "video": 3},
            }
        },
        "owner",
    )
    owner._launch_history_store = store
    for index, worker in enumerate(workers):
        worker.validate_model_reload = AsyncMock(
            return_value={field: {"image": 8, "video": index}}
        )
        worker._replace_model_generation = AsyncMock(return_value=True)
    await owner.reload_model("test", {field: {"image": 8}})
    await owner._model_reload_tasks["test"]
    assert (await owner.get_model_reload_status("test"))["status"] == "ready"
    for index, (worker, uid) in enumerate(zip(workers, uids)):
        assert worker._model_uid_to_launch_args[uid][field] == {
            "image": 8,
            "video": index,
        }
    assert store.list_autostart()[0]["launch"][field] == {"image": 8, "video": 3}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend,field",
    [("vllm", "limit_mm_per_prompt"), ("sglang", "limit_mm_data_per_request")],
)
@pytest.mark.parametrize(
    "stored", [None, '{"image": 2, "video": 2, "audio": 1}', ["legacy"]]
)
async def test_replica_reload_persists_normalized_defaults_when_autostart_limits_are_not_a_dict(
    monkeypatch, tmp_path, backend, field, stored
):
    from ...model.llm.tests.test_runtime_settings import make_vision_model

    owner, workers, actors, uids = make_replicas()
    launch = {"model_uid": "test", "model_name": "vision", "replica": 2}
    if stored is not None:
        launch[field] = stored
    store = LaunchHistoryStore(str(tmp_path / "history.db"))
    store.upsert_autostart({"launch": launch}, "owner")
    owner._launch_history_store = store
    for actor in actors:
        actor._model = make_vision_model(monkeypatch, backend)
        actor._model._reload_config[field] = {"image": 2, "video": 2, "audio": 1}
    await owner.reload_model("test", {field: {"image": 8}})
    await owner._model_reload_tasks["test"]
    expected = {"image": 8, "video": 2, "audio": 1}
    assert (await owner.get_model_reload_status("test"))["status"] == "ready"
    assert store.list_autostart()[0]["launch"][field] == expected
    assert all(
        worker._model_uid_to_launch_args[uid][field] == expected
        for worker, uid in zip(workers, uids)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("lose_status", [False, True])
async def test_late_rollback_after_scale_out_does_not_replace_model_autostart(
    lose_status,
):
    owner, _, actors, uids = make_replicas()
    owner._iter_active_replica_model_uids = lambda model_uid: iter(uids[:1])
    job = await owner.reload_model("test", {"max_num_seqs": 32})
    await owner._model_reload_tasks["test"]
    owner._iter_active_replica_model_uids = lambda model_uid: iter(uids)
    actors[1]._model._reload_config["max_num_seqs"] = 32
    if lose_status:
        owner._model_reload_status.clear()
    observed = {
        "operation_id": job["operation_id"],
        "status": "error",
        "restored": True,
        "error": "candidate exited",
    }
    actors[0]._reload_status = observed
    await owner.restore_model_reload_config(uids[0], {"max_num_seqs": 16}, observed)
    owner._launch_history_store.update_autostart_launch_config.assert_called_once_with(
        "test", {"max_num_seqs": 32}
    )
    assert actors[1]._model._reload_config["max_num_seqs"] == 32


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining", [0, 1])
async def test_late_rollback_after_scale_down_updates_remaining_model_autostart(
    remaining,
):
    owner, _, actors, uids = make_replicas()
    job = await owner.reload_model("test", {"max_num_seqs": 32})
    await owner._model_reload_tasks["test"]
    original_uids = uids.copy()
    uids[:] = [uids[remaining]]
    observed = {
        "operation_id": job["operation_id"],
        "status": "error",
        "restored": True,
        "error": "candidate exited",
    }
    actors[remaining]._reload_status = observed
    await owner.restore_model_reload_config(uids[0], {"max_num_seqs": 16}, observed)
    owner._launch_history_store.update_autostart_launch_config.assert_called_with(
        "test", {"max_num_seqs": 16}
    )
    status = await owner.get_model_reload_status("test")
    assert status["status"] == "error" and status["partial"]
    assert status["completed_replicas"] == 1
    assert [row["replica_uid"] for row in status["replicas"]] == original_uids
    assert status["replicas"][remaining]["status"] == "error"
    assert status["replicas"][1 - remaining]["status"] == "ready"


@pytest.mark.asyncio
async def test_generation_commit_preserves_operation_order_for_reconstruction(
    monkeypatch,
):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from ...model.llm.reasoning_parser import ReasoningParser

    owner, _, actors, uids = make_replicas()
    actors[1]._model.reasoning_parser = ReasoningParser(False, "<think>", "</think>")
    monkeypatch.setattr("xinference.core.model.time.time", lambda: 10)
    await actors[1].reload({"reasoning_content": True}, 1, "older-parser")
    monkeypatch.setattr("xinference.core.model_generation.time.time", lambda: 20)
    anchor = actors[0]
    anchor._start_generation_retirement = MagicMock()
    anchor._serving_generation = SimpleNamespace(uid="previous")
    candidate = SimpleNamespace(uid="candidate", generation_ready=AsyncMock())
    await anchor.begin_generation_reload("newer-engine")
    monkeypatch.setattr("xinference.core.model_generation.time.time", lambda: 30)
    await anchor.commit_model_generation(candidate, "newer-engine", True, "cache")
    assert anchor.get_reload_status()["started_at"] == 20
    # The peer has not received the new rollout yet when the Supervisor is lost.
    status = await owner.get_model_reload_status("test")
    assert status["operation_id"] == "newer-engine"
    assert status["status"] == "error" and status["outcome_unknown"]
    assert [row["replica_uid"] for row in status["replicas"]] == uids[:1]


@pytest.mark.asyncio
@pytest.mark.parametrize("after_commit", [False, True])
async def test_placement_change_preserves_committed_rows_and_skips_untouched_replicas(
    after_commit,
):
    owner, workers, actors, uids = make_replicas()
    first_uid = uids[0]
    if after_commit:
        replace = workers[0]._replace_model_generation

        async def move_after_replace(*args, **kwargs):
            result = await replace(*args, **kwargs)
            uids.pop(0)
            return result

        workers[0]._replace_model_generation = move_after_replace
    await owner.reload_model("test", {"max_num_seqs": 32})
    task = owner._model_reload_tasks["test"]
    if not after_commit:
        uids.pop(0)
    await task
    status = await owner.get_model_reload_status("test")
    assert status["status"] == "error" and "placement changed" in status["error"]
    assert [row["status"] for row in status["replicas"]] == [
        "ready" if after_commit else "skipped",
        "skipped",
    ]
    assert status["completed_replicas"] == int(after_commit)
    assert status["partial"] is after_commit
    assert workers[0]._model_uid_to_launch_args[first_uid]["max_num_seqs"] == (
        32 if after_commit else 16
    )
    assert actors[1]._model.stops == 0
    owner._launch_history_store.update_autostart_launch_config.assert_not_called()


@pytest.mark.asyncio
async def test_one_late_rollback_does_not_overwrite_other_replicas_or_desired_config():
    owner, _, actors, uids = make_replicas()
    job = await owner.reload_model("test", {"max_num_seqs": 32})
    await owner._model_reload_tasks["test"]
    observed = {
        "operation_id": job["operation_id"],
        "status": "error",
        "stage": "error",
        "restored": True,
        "error": "candidate exited",
    }
    actors[0]._reload_status = observed
    await owner.restore_model_reload_config(uids[0], {"max_num_seqs": 16}, observed)
    status = await owner.get_model_reload_status("test")
    assert status["status"] == "error" and status["partial"]
    assert status["replicas"][1]["status"] == "ready"
    owner._launch_history_store.update_autostart_launch_config.assert_called_once_with(
        "test", {"max_num_seqs": 32}
    )


@pytest.mark.asyncio
async def test_ready_actors_do_not_prove_modelwide_commit_after_supervisor_cache_loss():
    owner, _, actors, _ = make_replicas()
    job = await owner.reload_model("test", {"max_num_seqs": 32})
    await owner._model_reload_tasks["test"]
    for actor in actors:
        actor._reload_status = {
            "operation_id": job["operation_id"],
            "status": "ready",
            "stage": "ready",
            "weights_reused": True,
        }
    owner._model_reload_status.clear()
    status = await owner.get_model_reload_status("test")
    assert status["status"] == "error" and status["outcome_unknown"]
    assert not status["weights_reused"]
    assert not status["partial"]
    assert status["completed_replicas"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "other_status", [{}, {"operation_id": "older", "started_at": 1, "status": "ready"}]
)
async def test_reconstructed_status_excludes_new_or_recreated_non_participants(
    other_status,
):
    owner, _, actors, uids = make_replicas()
    actors[0]._reload_status = {
        "operation_id": "latest",
        "started_at": 2,
        "status": "ready",
        "weights_reused": True,
    }
    actors[1]._reload_status = other_status
    for _ in range(2):
        status = await owner.get_model_reload_status("test")
        assert status["status"] == "error" and status["outcome_unknown"]
        assert not status["partial"]
        assert status["completed_replicas"] == status["replica_count"] == 1
        assert [row["replica_uid"] for row in status["replicas"]] == uids[:1]


@pytest.mark.asyncio
@pytest.mark.parametrize("registered", [1, 2])
@pytest.mark.parametrize(
    "previous", [{}, {"operation_id": "older", "started_at": 1, "status": "ready"}]
)
async def test_supervisor_loss_before_an_admitted_target_runs_never_reports_success(
    registered, previous
):
    owner, workers, actors, uids = make_replicas()
    actors[1]._reload_status = previous
    entered = asyncio.Event()
    replace = workers[0]._replace_model_generation

    async def commit_with_operation(
        uid, patch, python_executable, operation_id, drain_timeout
    ):
        result = await replace(
            uid, patch, python_executable, operation_id, drain_timeout
        )
        actors[0]._reload_status["operation_id"] = operation_id
        return result

    async def undispatched(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    workers[0]._replace_model_generation = commit_with_operation
    workers[1].reload_model = AsyncMock(side_effect=undispatched)
    operation = await owner.reload_model("test", {"max_num_seqs": 32})
    task = owner._model_reload_tasks["test"]
    try:
        await asyncio.wait_for(entered.wait(), 1)
        recovered, _, _, _ = make_replicas()
        recovered._model_uid_to_replica_info = owner._model_uid_to_replica_info
        recovered._replica_model_uid_to_worker = owner._replica_model_uid_to_worker
        recovered._iter_active_replica_model_uids = lambda model_uid: iter(
            uids[:registered]
        )
        for _ in range(2):
            status = await recovered.get_model_reload_status("test")
            assert status["operation_id"] == operation["operation_id"]
            assert status["status"] == "error" and status["outcome_unknown"]
            assert not status["partial"] and not status["weights_reused"]
            assert [row["replica_uid"] for row in status["replicas"]] == uids[:1]
            assert status["replicas"][0]["status"] == "ready"
        assert [actor._model._reload_config["max_num_seqs"] for actor in actors] == [
            32,
            16,
        ]
        assert all(actor._model_state == "ready" for actor in actors)
        owner._launch_history_store.update_autostart_launch_config.assert_not_called()
        workers[1].reload_model.assert_awaited_once()
        assert actors[1]._model.stops == 0
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_recovered_inflight_actor_finishes_without_claiming_modelwide_success():
    owner, _, actors, _ = make_replicas()
    actors[0]._reload_status = {
        "operation_id": "interrupted",
        "started_at": 1,
        "status": "reloading",
        "stage": "loading",
    }
    assert (await owner.get_model_reload_status("test"))["status"] == "reloading"
    actors[0]._reload_status.update(status="ready", stage="ready", weights_reused=True)
    status = await owner.get_model_reload_status("test")
    assert status["status"] == "error" and status["outcome_unknown"]
    assert not status["weights_reused"]


@pytest.mark.asyncio
async def test_reconstructed_idle_status_does_not_invent_an_operation():
    owner, _, _, _ = make_replicas()
    assert await owner.get_model_reload_status("test") == {"status": "idle"}


@pytest.mark.asyncio
async def test_recovered_member_restoration_does_not_prove_modelwide_restoration():
    owner, _, actors, _ = make_replicas()
    actors[0]._reload_status = {
        "operation_id": "interrupted",
        "status": "error",
        "restored": True,
        "error": "candidate failed",
    }
    status = await owner.get_model_reload_status("test")
    assert status["status"] == "error" and status["outcome_unknown"]
    assert status["replicas"][0]["restored"]
    assert not status["restored"]


@pytest.mark.asyncio
async def test_failed_rollout_after_supervisor_cache_loss_reports_partial_error():
    owner, _, actors, _ = make_replicas()
    actors[0]._reload_status = {
        "operation_id": "interrupted",
        "started_at": 1,
        "status": "ready",
        "stage": "ready",
        "weights_reused": True,
    }
    actors[1]._reload_status = {
        "operation_id": "interrupted",
        "started_at": 1,
        "status": "error",
        "stage": "error",
        "restored": True,
        "error": "Reload interrupted",
    }
    status = await owner.get_model_reload_status("test")
    assert status["status"] == "error" and status["partial"]
    assert status["completed_replicas"] == 1
    assert "interrupted" in status["error"].lower()


@pytest.mark.asyncio
async def test_autostart_failure_does_not_misreport_a_committed_replica_as_failed():
    owner, _, _, _ = make_replicas()
    owner._launch_history_store.update_autostart_launch_config.side_effect = (
        RuntimeError("database unavailable")
    )
    await owner.reload_model("test", {"max_num_seqs": 32})
    await owner._model_reload_tasks["test"]
    status = await owner.get_model_reload_status("test")
    assert status["status"] == "error" and "database unavailable" in status["error"]
    assert status["completed_replicas"] == 2
    assert all(row["status"] == "ready" for row in status["replicas"])


@pytest.mark.asyncio
async def test_force_termination_cancels_rollout_before_later_replicas_start():
    owner, workers, _, _ = make_replicas()
    entered = asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    workers[0]._replace_model_generation = blocked
    workers[1]._replace_model_generation = AsyncMock(return_value=True)
    owner._terminate_model = AsyncMock()
    await owner.reload_model("test", {"max_num_seqs": 32})
    task = owner._model_reload_tasks["test"]
    await entered.wait()
    await asyncio.wait_for(owner.terminate_model("test"), 1)
    with pytest.raises(asyncio.CancelledError):
        await task
    status = await owner.get_model_reload_status("test")
    assert status["status"] == "cancelled"
    assert [member["status"] for member in status["replicas"]] == [
        "cancelled",
        "cancelled",
    ]
    workers[1]._replace_model_generation.assert_not_called()
    owner._terminate_model.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_preflight_rejects_an_existing_reload_without_resetting_status():
    _, workers, actors, uids = make_replicas()
    workers[0]._model_reload_active = {uids[0]}
    actors[0]._reload_status = {"operation_id": "active", "status": "reloading"}
    from ..exceptions import ModelReloadInProgress

    with pytest.raises(ModelReloadInProgress, match="reload in progress"):
        await workers[0].validate_model_reload(uids[0], {"max_num_seqs": 32})
    assert actors[0]._reload_status["operation_id"] == "active"


@pytest.mark.asyncio
async def test_late_failure_stops_remaining_members_and_waits_for_current_member():
    owner, workers, actors, uids = make_replicas(3)
    entered, release = asyncio.Event(), asyncio.Event()
    replace = workers[1]._replace_model_generation

    async def gated(*args, **kwargs):
        entered.set()
        await release.wait()
        return await replace(*args, **kwargs)

    workers[1]._replace_model_generation = gated
    job = await owner.reload_model("test", {"max_num_seqs": 32})
    task = owner._model_reload_tasks["test"]
    await entered.wait()
    observed = {
        "operation_id": job["operation_id"],
        "status": "error",
        "stage": "error",
        "restored": True,
        "error": "candidate exited",
    }
    actors[0]._reload_status = observed
    await owner.restore_model_reload_config(uids[0], {"max_num_seqs": 16}, observed)
    assert (await owner.get_model_reload_status("test"))["status"] == "reloading"
    release.set()
    await task
    status = await owner.get_model_reload_status("test")
    assert status["status"] == "error" and status["partial"]
    assert [member["status"] for member in status["replicas"]] == [
        "error",
        "ready",
        "skipped",
    ]
    assert actors[2]._model.stops == 0
    owner._launch_history_store.update_autostart_launch_config.assert_not_called()


@pytest.mark.asyncio
async def test_parser_reload_keeps_operation_identity_on_each_replica():
    from ...model.llm.reasoning_parser import ReasoningParser

    owner, _, actors, _ = make_replicas()
    for actor in actors:
        actor._model.reasoning_parser = ReasoningParser(False, "<think>", "</think>")
    job = await owner.reload_model("test", {"reasoning_content": True})
    await owner._model_reload_tasks["test"]
    assert all(
        actor._reload_status["operation_id"] == job["operation_id"] for actor in actors
    )
    owner._model_reload_status.clear()
    status = await owner.get_model_reload_status("test")
    assert status["status"] == "error" and status["outcome_unknown"]
    assert status["completed_replicas"] == 2
    assert not status["weights_reused"]
