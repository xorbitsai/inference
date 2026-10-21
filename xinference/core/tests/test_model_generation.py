# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
"""Real actor transport verifies that iterators remain on their admitting engine."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import xoscar as xo

from ..model_generation import ModelGenerationMixin, replace_model_generation


class ServingGeneration(ModelGenerationMixin, xo.StatelessActor):
    def __init__(self, name, healthy=True):
        super().__init__()
        self.name = name
        self.healthy = healthy
        self._serve_count = 0
        self._replica_model_uid = "model-rep0"
        self.stopped = False
        self._reload_status = {}
        self._model = SimpleNamespace(
            _stop_engine=self.stop_engine,
            _weight_cache=SimpleNamespace(directory="cache", client_config=lambda: {}),
        )

    def _require_ready(self):
        if not self.healthy:
            raise ValueError("Candidate failed health check")

    def _get_weight_cached_model(self):
        return self._model

    def stop_engine(self):
        self.stopped = True

    def get_reload_status(self):
        return self._reload_status

    async def _get_worker_ref(self):
        return SimpleNamespace(retire_model_generation_pool=AsyncMock())

    @xo.generator
    async def chat(self, stream=False):
        if self.stopped:
            raise RuntimeError("Engine is stopped")
        if not stream:
            return self.name
        self._serve_count += 1

        async def chunks():
            try:
                for index in range(10):
                    await asyncio.sleep(0.01)
                    yield f"{self.name}:{index}"
            finally:
                self._serve_count -= 1

        return chunks()


def test_another_replacement_waits_for_previous_generation_to_finish():
    actor = ServingGeneration("current")
    actor._retiring_generations = [None]
    with pytest.raises(RuntimeError, match="still draining"):
        actor.begin_generation_reload("another-upgrade")
    assert not actor.stopped


def test_prepared_environment_is_isolated_on_upgrade_and_recovery():
    from ..utils import build_subpool_envs_for_python

    original = {
        "PYTHONPATH": "/old/source",
        "PYTHONHOME": "/old",
        "CUDA_VISIBLE_DEVICES": "1",
        "PATH": "/usr/bin",
    }
    env = build_subpool_envs_for_python(original, "/new/bin/python")
    assert env["PYTHONPATH"] == env["PYTHONHOME"] == env["VIRTUAL_ENV"] == ""
    assert env["PATH"] == "/new/bin:/usr/bin"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert original["PYTHONPATH"] == "/old/source"


@pytest.mark.asyncio
async def test_abort_reaches_new_generation_while_old_stream_is_still_serving():
    from ..model import ModelActor
    from .test_model_reload import make_actor

    actor = make_actor()
    actor._replica_model_uid = "test-replica"
    actor._serve_count = 1
    actor._running_tasks = {}
    actor._cancel_running_task = MagicMock()
    actor._model.abort_request = AsyncMock(return_value="NO_OP")
    actor._serving_generation = SimpleNamespace(
        abort_request=AsyncMock(return_value="DONE")
    )
    assert await ModelActor.abort_request(actor, "new-request") == "DONE"


@pytest.mark.asyncio
async def test_retirement_removes_only_private_generation_pool():
    from .test_model_reload import make_actor, make_pipeline

    _, worker = make_pipeline(make_actor())
    worker._model_uid_to_addr = {"test-replica": "public"}
    worker._model_uid_to_subpool_addresses = {"test-replica": {"public", "retired"}}
    worker._main_pool = SimpleNamespace(remove_sub_pool=AsyncMock())
    await worker.retire_model_generation_pool("test-replica", "retired")
    worker._main_pool.remove_sub_pool.assert_awaited_once_with("retired", force=True)
    assert worker._model_uid_to_subpool_addresses["test-replica"] == {"public"}
    with pytest.raises(ValueError, match="public model"):
        await worker.retire_model_generation_pool("test-replica", "public")


@pytest.mark.asyncio
async def test_cutover_keeps_stream_on_old_generation_and_new_calls_on_ready_generation():
    async with await xo.create_actor_pool("127.0.0.1", n_process=0):
        # Use separate actor addresses, as production generations do.
        async with await xo.create_actor_pool("127.0.0.1", n_process=0) as target_pool:
            async with await xo.create_actor_pool(
                "127.0.0.1", n_process=0
            ) as source_pool:
                source = await xo.create_actor(
                    ServingGeneration,
                    "old",
                    uid="model",
                    address=source_pool.external_address,
                )
                target = await xo.create_actor(
                    ServingGeneration,
                    "new",
                    uid="model",
                    address=target_pool.external_address,
                )
                stream = await source.chat(stream=True)
                first = await stream.__anext__()
                assert first == "old:0"
                await source.commit_model_generation(target, "upgrade", True, "cache")
                # A retry must not retire the active target.
                await source.commit_model_generation(target, "upgrade", True, "cache")
                assert await source.chat() == "new"
                assert await source.generation_serve_count() == 1
                assert [chunk async for chunk in stream] == [
                    f"old:{i}" for i in range(1, 10)
                ]
                await asyncio.sleep(0.1)
                assert await source.chat() == "new"
                assert (await source.get_reload_status())["status"] == "ready"


@pytest.mark.asyncio
async def test_unhealthy_candidate_does_not_change_route():
    async with await xo.create_actor_pool("127.0.0.1", n_process=0) as pool:
        source = await xo.create_actor(
            ServingGeneration, "old", uid="old", address=pool.external_address
        )
        candidate = await xo.create_actor(
            ServingGeneration, "bad", False, uid="bad", address=pool.external_address
        )
        with pytest.raises(ValueError, match="health"):
            await source.commit_model_generation(candidate, "upgrade", True, "cache")
        assert await source.chat() == "old"


@pytest.mark.asyncio
async def test_candidate_exit_preserves_old_stream_and_cancels_its_retirement():
    async with await xo.create_actor_pool("127.0.0.1", n_process=0) as source_pool:
        async with await xo.create_actor_pool("127.0.0.1", n_process=0) as target_pool:
            source = await xo.create_actor(
                ServingGeneration,
                "old",
                uid="old",
                address=source_pool.external_address,
            )
            target = await xo.create_actor(
                ServingGeneration,
                "new",
                uid="new",
                address=target_pool.external_address,
            )
            stream = await source.chat(stream=True)
            assert await stream.__anext__() == "old:0"
            await source.commit_model_generation(target, "upgrade", True, "cache")
            await xo.destroy_actor(target)
            assert (
                await source.handle_generation_pool_exit(target.address) == "restored"
            )
            assert await source.chat() == "old"
            assert len([chunk async for chunk in stream]) == 9
            await asyncio.sleep(0.1)
            # Finishing the old stream must not run the cancelled retirement
            # and stop the engine that now serves all requests again.
            assert await source.chat() == "old"
            assert (await source.get_reload_status())["restored"]
            await source.close_model_generations()


@pytest.mark.asyncio
async def test_new_cache_receives_engine_config_without_wrapper_options(
    monkeypatch, tmp_path
):
    from ...model.llm import weight_cache

    cache = SimpleNamespace(
        directory=tmp_path,
        start=MagicMock(),
        stop=MagicMock(),
        export=lambda: {"directory": str(tmp_path)},
    )
    constructor = MagicMock(return_value=cache)
    monkeypatch.setattr(weight_cache, "WeightCacheDaemon", constructor)
    actor = ServingGeneration("old")
    snapshot = {
        "engine": "sglang",
        "model_path": "/models/test",
        "cache": {"config": {"tp_size": 1, "max_running_requests": 4}},
        "config": {"enable_thinking": False, "max_running_requests": 8},
    }
    await actor.create_generation_cache(snapshot, "/new/python", {})
    assert constructor.call_args.args[2] == {"tp_size": 1, "max_running_requests": 8}


@pytest.mark.asyncio
async def test_in_place_parser_reload_queues_new_requests_instead_of_rejecting():
    from ..model import request_limit
    from .test_model_reload import make_actor

    actor = make_actor()
    actor._serve_count = 1
    actor._request_limits = float("inf")
    actor._metrics_labels = {}
    actor.record_metrics = AsyncMock()
    task = asyncio.create_task(actor.reload({"max_num_seqs": 32}, 2))
    for _ in range(100):
        if actor._model_state == "reloading":
            break
        await asyncio.sleep(0.01)
    called = False

    @request_limit
    async def serve(self):
        nonlocal called
        called = True
        return "ok"

    queued = asyncio.create_task(serve(actor))
    await asyncio.sleep(0.02)
    assert not called and not queued.done()
    actor._serve_count = 0
    await task
    assert await queued == "ok"
    assert actor._serve_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["environment", "capacity", "load", "ownership", None]
)
async def test_private_preparation_never_stops_public_engine(monkeypatch, failure):
    from .. import model_generation as module

    snapshot = {
        "engine": "vllm",
        "engine_version": "0.31.0",
        "torch_version": "2.13.0",
        "python_executable": "/python",
        "env": {"CUDA_VISIBLE_DEVICES": "0"},
        "config": {},
        "cache": {"directory": "/cache"},
    }
    status = {}

    async def commit(*args):
        status.update(operation_id="job", status="ready")

    anchor = SimpleNamespace(
        model_generation_protocol=AsyncMock(return_value=1),
        export_model_generation=AsyncMock(return_value=snapshot),
        begin_generation_reload=AsyncMock(),
        check_generation_capacity=AsyncMock(
            side_effect=(
                ValueError("Insufficient GPU memory") if failure == "capacity" else None
            )
        ),
        commit_model_generation=AsyncMock(side_effect=commit),
        get_reload_status=AsyncMock(side_effect=lambda: status),
        discard_generation_cache=AsyncMock(),
        stop=AsyncMock(),
    )
    worker = SimpleNamespace(
        address="worker",
        _supervisor_address="supervisor",
        _model_uid_to_model={"model": anchor},
        _model_uid_to_launch_args={"model": {}},
        _model_uid_to_subpool_addresses={},
        _append_sub_pool_protected=AsyncMock(return_value="candidate-pool"),
        _main_pool=SimpleNamespace(remove_sub_pool=AsyncMock()),
        _ensure_subpool_monitor=AsyncMock(),
    )

    async def load():
        if failure == "load":
            raise RuntimeError("Candidate OOM")
        if failure == "ownership":
            worker._model_uid_to_model.clear()

    candidate = SimpleNamespace(
        load=AsyncMock(side_effect=load), wait_for_load=AsyncMock()
    )
    monkeypatch.setattr(
        module,
        "probe_environment",
        AsyncMock(
            side_effect=(
                ValueError("Missing environment") if failure == "environment" else None
            ),
            return_value={"vllm": "0.31.0", "torch": "2.13.0"},
        ),
    )
    monkeypatch.setattr(xo, "create_actor", AsyncMock(return_value=candidate))
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            await replace_model_generation(worker, "model", {}, operation_id="job")
        anchor.commit_model_generation.assert_not_called()
        if failure not in {"environment", "capacity"}:
            worker._main_pool.remove_sub_pool.assert_awaited_once_with(
                "candidate-pool", force=True
            )
        else:
            worker._append_sub_pool_protected.assert_not_called()
    else:
        assert await replace_model_generation(worker, "model", {}, operation_id="job")
        worker._main_pool.remove_sub_pool.assert_not_called()
        # After a prior upgrade, the active model's interpreter can differ
        # from the Worker's even when this request omits python_executable.
        assert module.probe_environment.call_args.args[2]["PYTHONPATH"] == ""
    anchor.stop.assert_not_called()


@pytest.mark.parametrize(
    "reused,free,allowed",
    [(True, 9, True), (False, 9, False), (False, 12, True), (True, 7, False)],
)
def test_capacity_preserves_old_runtime_headroom_and_counts_shared_weights_once(
    monkeypatch, reused, free, allowed
):
    import os

    import psutil

    from ...model.llm import weight_cache
    from ..model_generation import _check_generation_capacity

    gib = 1024**3
    # Current runtime uses 4 GiB and cache uses 2 GiB. Its configured 8 GiB
    # budget needs another 2 GiB left free even if the current workload is small.
    model = SimpleNamespace(
        _weight_cache=SimpleNamespace(process=SimpleNamespace(pid=12345), config={}),
        _weight_cache_engine="vllm",
        _reload_config={"gpu_memory_utilization": 0.4},
    )
    monkeypatch.setattr(
        psutil, "Process", lambda pid: SimpleNamespace(children=lambda recursive: [])
    )
    monkeypatch.setattr(
        weight_cache,
        "_gpu_memory_snapshot",
        lambda: {0: (20 * gib, free * gib, {os.getpid(): 4 * gib, 12345: 2 * gib})},
    )
    if allowed:
        _check_generation_capacity(model, [], {"gpu_memory_utilization": 0.4}, reused)
    else:
        with pytest.raises(ValueError, match="Insufficient GPU memory"):
            _check_generation_capacity(
                model, [], {"gpu_memory_utilization": 0.4}, reused
            )


def test_capacity_refuses_unknown_driver_pid_namespace(monkeypatch):
    import psutil

    from ...model.llm import weight_cache
    from ..model_generation import _check_generation_capacity

    model = SimpleNamespace(
        _weight_cache=SimpleNamespace(process=SimpleNamespace(pid=12345), config={}),
        _weight_cache_engine="sglang",
        _reload_config={},
    )
    monkeypatch.setattr(
        psutil, "Process", lambda pid: SimpleNamespace(children=lambda recursive: [])
    )
    monkeypatch.setattr(
        weight_cache,
        "_gpu_memory_snapshot",
        lambda: {0: (24 * 1024**3, 20 * 1024**3, {99999: 1024**3})},
    )
    with pytest.raises(ValueError, match="every serving GPU"):
        _check_generation_capacity(model, [], {}, False)


@pytest.mark.parametrize("unknown", [None, 2**64 - 1, 0])
def test_capacity_refuses_unavailable_driver_memory(monkeypatch, unknown):
    import os

    import psutil

    from ...model.llm import weight_cache
    from ..model_generation import _check_generation_capacity

    model = SimpleNamespace(
        _weight_cache=SimpleNamespace(process=SimpleNamespace(pid=12345), config={}),
        _weight_cache_engine="vllm",
        _reload_config={},
    )
    monkeypatch.setattr(
        psutil, "Process", lambda pid: SimpleNamespace(children=lambda recursive: [])
    )
    monkeypatch.setattr(
        weight_cache,
        "_gpu_memory_snapshot",
        lambda: {
            0: (24 * 1024**3, 20 * 1024**3, {os.getpid(): 1024**3, 12345: unknown})
        },
    )
    with pytest.raises(ValueError, match="Cannot safely account"):
        _check_generation_capacity(model, [], {}, True)
