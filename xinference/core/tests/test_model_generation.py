# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
"""Real actor transport verifies that iterators remain on their admitting engine."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import xoscar as xo

from ...api.restful_api import _release_model_stream
from ..model import ModelActor, request_limit
from ..model_generation import ModelGenerationMixin, replace_model_generation


class ServingGeneration(ModelGenerationMixin, xo.StatelessActor):
    def __init__(self, name, healthy=True):
        super().__init__()
        self.name = name
        self.healthy = healthy
        self._serve_count = 0
        self._request_limits = 2
        self._metrics_labels = {}
        self._replica_model_uid = "model-rep0"
        self.stopped = False
        self._reload_status = {}
        self._model = SimpleNamespace(
            _stop_engine=self.stop_engine,
            _weight_cache=SimpleNamespace(directory="cache", client_config=lambda: {}),
        )

    async def hold_next_chat(self):
        self.admission_entered = asyncio.Event()
        self.admission_gate = asyncio.Event()

    async def wait_for_pending_chat(self):
        await self.admission_entered.wait()

    def release_pending_chat(self):
        self.admission_gate.set()

    async def __on_receive__(self, message):
        if message[0] == "chat" and hasattr(self, "admission_gate"):
            self.admission_entered.set()
            await self.admission_gate.wait()
        return await super().__on_receive__(message)

    def _require_ready(self):
        if not self.healthy:
            raise ValueError("Candidate failed health check")

    def _get_weight_cached_model(self):
        return self._model

    def stop_engine(self):
        self.stopped = True

    def model_uid(self):
        return self.name

    async def record_metrics(self, *args):
        pass

    decrease_serve_count = ModelActor.decrease_serve_count

    def get_reload_status(self):
        return {**self._reload_status, "engine_stopped": self.stopped}

    async def _get_worker_ref(self):
        return SimpleNamespace(retire_model_generation_pool=AsyncMock())

    @request_limit
    @xo.generator
    async def chat(self, stream=False):
        if self.stopped:
            raise RuntimeError("Engine is stopped")
        if not stream:
            return self.name

        async def chunks():
            for index in range(10):
                if self.stopped:
                    raise RuntimeError("Engine is stopped")
                await asyncio.sleep(0.01)
                yield f"{self.name}:{index}"

        return chunks()


@pytest.mark.asyncio
async def test_another_replacement_waits_for_previous_generation_to_finish():
    actor = ServingGeneration("current")
    actor._retiring_generations = [None]

    async def retire():
        await asyncio.sleep(0.02)
        actor._retiring_generations.clear()

    actor._generation_retirement_tasks = {asyncio.create_task(retire())}
    preparing = asyncio.create_task(actor.begin_generation_reload("another-upgrade"))
    await asyncio.sleep(0)
    assert actor._reload_status["stage"] == "draining" and not preparing.done()
    assert not actor.stopped
    await preparing
    assert actor._reload_status["stage"] == "loading"


@pytest.mark.asyncio
async def test_cancelling_next_upgrade_does_not_cancel_previous_retirement():
    actor = ServingGeneration("current")
    actor._retiring_generations = [None]
    release = asyncio.Event()
    retiring = asyncio.create_task(release.wait())
    actor._generation_retirement_tasks = {retiring}
    preparing = asyncio.create_task(actor.begin_generation_reload("cancelled"))
    await asyncio.sleep(0)
    preparing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await preparing
    assert not retiring.cancelled() and not actor.stopped
    assert actor._reload_status["restored"]
    release.set()
    await retiring


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
                new_stream = await source.chat(stream=True)
                assert len([chunk async for chunk in new_stream]) == 10
                await _release_model_stream(source, new_stream)
                await new_stream.destroy()
                new_stream._gc_destroy = False
                assert await target.generation_serve_count() == 0
                assert await source.generation_serve_count() == 1
                assert not (await source.get_reload_status())["engine_stopped"]
                with pytest.raises(asyncio.TimeoutError):
                    await source.begin_generation_reload("waiting", 0.01)
                assert [chunk async for chunk in stream] == [
                    f"old:{i}" for i in range(1, 10)
                ]
                await _release_model_stream(source, stream)
                await stream.destroy()
                stream._gc_destroy = False
                await asyncio.sleep(0.1)
                assert (await source.get_reload_status())["engine_stopped"]
                assert await source.chat() == "new"
                await source.begin_generation_reload("second-upgrade", 1)
                third = await xo.create_actor(
                    ServingGeneration,
                    "third",
                    uid="third",
                    address=target_pool.external_address,
                )
                await source.commit_model_generation(
                    third, "second-upgrade", True, "cache"
                )
                await asyncio.sleep(0.1)
                assert (await target.get_reload_status())["engine_stopped"]
                assert await source.chat() == "third"
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
            await _release_model_stream(source, stream)
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
@pytest.mark.parametrize("reused", [False, True])
async def test_private_preparation_never_stops_public_engine(
    monkeypatch, failure, reused
):
    from .. import model_generation as module

    monkeypatch.setenv("INHERITED_TEST_VARIABLE", "inherited" * 1000)
    snapshot = {
        "generation_uid": b"source",
        "engine": "vllm",
        "engine_version": "0.31.0",
        "torch_version": "2.13.0",
        "python_executable": "/python",
        "runtime_env": {"VIRTUAL_ENV": "/model-env", "PYTHONPATH": "/model-env/site"},
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
        fail_generation_reload=AsyncMock(),
        get_reload_status=AsyncMock(side_effect=lambda: status),
        discard_generation_cache=AsyncMock(),
        create_generation_cache=AsyncMock(return_value={"directory": "/new-cache"}),
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
            return_value={"vllm": "0.31.0" if reused else "0.32.0", "torch": "2.13.0"},
        ),
    )
    monkeypatch.setattr(xo, "create_actor", AsyncMock(return_value=candidate))
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            await replace_model_generation(worker, "model", {}, operation_id="job")
        anchor.commit_model_generation.assert_not_called()
        anchor.fail_generation_reload.assert_awaited_once()
        assert anchor.fail_generation_reload.await_args.args[0] == "job"
        if failure not in {"environment", "capacity"}:
            worker._main_pool.remove_sub_pool.assert_awaited_once_with(
                "candidate-pool", force=True
            )
            if not reused:
                anchor.discard_generation_cache.assert_awaited_once_with("/new-cache")
        else:
            worker._append_sub_pool_protected.assert_not_called()
    else:
        assert (
            await replace_model_generation(worker, "model", {}, operation_id="job")
            is reused
        )
        worker._main_pool.remove_sub_pool.assert_not_called()
        anchor.fail_generation_reload.assert_not_called()
        # After a prior upgrade, the active model's interpreter can differ
        # from the Worker's even when this request omits python_executable.
        assert (
            module.probe_environment.call_args.args[2]["PYTHONPATH"]
            == "/model-env/site"
        )
        assert module.probe_environment.call_args.args[2]["VIRTUAL_ENV"] == "/model-env"
        overrides = worker._append_sub_pool_protected.call_args.kwargs["env"]
        assert "INHERITED_TEST_VARIABLE" not in overrides
        assert overrides["XINFERENCE_MODEL_UID"] == "model"
    anchor.stop.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pool_error", [KeyError("already removed"), RuntimeError("remove failed")]
)
async def test_failed_candidate_cache_cleanup_survives_pool_and_status_errors(
    monkeypatch, pool_error
):
    from .. import model_generation as module

    anchor = SimpleNamespace(
        model_generation_protocol=AsyncMock(return_value=1),
        export_model_generation=AsyncMock(
            return_value={
                "generation_uid": b"source",
                "engine": "vllm",
                "engine_version": "old",
                "torch_version": "old",
                "python_executable": "/python",
                "env": {},
                "config": {},
                "cache": {},
            }
        ),
        begin_generation_reload=AsyncMock(),
        check_generation_capacity=AsyncMock(),
        create_generation_cache=AsyncMock(return_value={"directory": "/new-cache"}),
        get_reload_status=AsyncMock(return_value={}),
        fail_generation_reload=AsyncMock(),
        discard_generation_cache=AsyncMock(),
    )
    worker = SimpleNamespace(
        address="worker",
        _supervisor_address="supervisor",
        _model_uid_to_model={"model": anchor},
        _model_uid_to_launch_args={"model": {}},
        _model_uid_to_subpool_addresses={},
        _append_sub_pool_protected=AsyncMock(return_value="candidate"),
        _main_pool=SimpleNamespace(remove_sub_pool=AsyncMock(side_effect=pool_error)),
    )

    async def load():
        anchor.get_reload_status.side_effect = ConnectionError("anchor unreachable")
        raise RuntimeError("original load error")

    monkeypatch.setattr(
        module,
        "probe_environment",
        AsyncMock(return_value={"vllm": "new", "torch": "new"}),
    )
    monkeypatch.setattr(
        xo, "create_actor", AsyncMock(return_value=SimpleNamespace(load=load))
    )
    with pytest.raises(RuntimeError, match="original load error"):
        await replace_model_generation(worker, "model", {}, operation_id="job")
    anchor.discard_generation_cache.assert_awaited_once_with("/new-cache")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["engine", "cache"])
async def test_failed_retirement_retries_and_releases_only_unused_cache(failure):
    actor = ServingGeneration("current")
    old = SimpleNamespace(
        uid=b"old",
        address="old-pool",
        generation_serve_count=AsyncMock(return_value=0),
        retire_generation_engine=AsyncMock(
            side_effect=(
                [RuntimeError("stop failed"), None] if failure == "engine" else None
            )
        ),
    )
    cache = SimpleNamespace(
        stop=MagicMock(
            side_effect=(
                [RuntimeError("cache stop failed"), None]
                if failure == "cache"
                else None
            )
        )
    )
    actor._retiring_generations = [old]
    actor._generation_caches = {"old-cache": cache}
    actor._generation_cache_leases = {b"old": "old-cache", b"new": "new-cache"}
    await actor.retire_model_generation(old)
    assert old in actor._retiring_generations
    if failure == "engine":
        cache.stop.assert_not_called()
    assert actor._generation_cache_leases[b"old"] == "old-cache"
    await actor.begin_generation_reload("retry", 1)
    assert not actor._retiring_generations
    assert actor._generation_cache_leases == {b"new": "new-cache"}
    assert cache.stop.call_count == (2 if failure == "cache" else 1)


@pytest.mark.asyncio
async def test_generation_exit_recovery_retirement_and_worker_rebinding():
    actor = ServingGeneration("current")
    active = SimpleNamespace(uid=b"active", address="active", rebind_worker=AsyncMock())
    old = SimpleNamespace(uid=b"old", address="old", rebind_worker=AsyncMock())
    actor._serving_generation = active
    actor._retiring_generations = [old]
    actor._start_generation_retirement = MagicMock()
    await actor.rebind_model_generations("source", "target", 2)
    active.rebind_worker.assert_awaited_once_with("source", "target", 2)
    old.rebind_worker.assert_awaited_once_with("source", "target", 2)
    assert await actor.handle_generation_pool_exit("old") == "retiring"
    actor._start_generation_retirement.assert_called_once_with(old)
    assert await actor.handle_generation_pool_exit("active") == "recover"


@pytest.mark.asyncio
async def test_retiring_abort_failure_does_not_hide_active_result():
    from .test_model_reload import make_actor

    actor = make_actor()
    actor._replica_model_uid = "test-rep0"
    actor._serving_generation = SimpleNamespace(
        abort_request=AsyncMock(return_value="DONE")
    )
    actor._retiring_generations = [
        SimpleNamespace(
            abort_request=AsyncMock(side_effect=ValueError("retired engine"))
        )
    ]
    assert await ModelActor.abort_request(actor, "active-request") == "DONE"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "xoscar_version,protocol", [("0.12.1", 1), ("0.12.0", 1), ("0.12.1", 0)]
)
async def test_environment_probe_accepts_startup_output_and_validates_protocol(
    monkeypatch, xoscar_version, protocol
):
    import json

    from ..model_generation import probe_environment

    versions = {
        "xinference": "4.99.0",
        "xoscar": xoscar_version,
        "torch": "2.13.0",
        "vllm": "0.32.0",
        "generation_protocol": protocol,
    }
    process = SimpleNamespace(
        returncode=0,
        communicate=AsyncMock(
            return_value=(
                b"sitecustomize output\n" + json.dumps(versions).encode() + b"\n",
                b"",
            )
        ),
    )
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
    )
    if xoscar_version == "0.12.1" and protocol == 1:
        assert await probe_environment("/python", "vllm", {}) == versions
    else:
        with pytest.raises(ValueError, match="compatible Xinference"):
            await probe_environment("/python", "vllm", {})


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
            0: (
                24 * 1024**3,
                20 * 1024**3,
                {os.getpid(): 1024**3, 12345: unknown},
            )
        },
    )
    with pytest.raises(ValueError, match="Cannot safely account"):
        _check_generation_capacity(model, [], {}, True)


@pytest.mark.asyncio
async def test_forwarded_request_before_target_admission_prevents_retirement():
    async with await xo.create_actor_pool("127.0.0.1", n_process=0) as pool:
        anchor, old, new = [
            await xo.create_actor(
                ServingGeneration, name, uid=name, address=pool.external_address
            )
            for name in ("anchor", "old", "new")
        ]
        await anchor.commit_model_generation(old, "first", True, "cache")
        await anchor.begin_generation_reload("second", 1)
        await old.hold_next_chat()
        pending = asyncio.create_task(anchor.chat())
        try:
            await asyncio.wait_for(old.wait_for_pending_chat(), 1)
            assert await old.generation_serve_count() == 0
            await anchor.commit_model_generation(new, "second", True, "cache")
            await asyncio.sleep(0.1)
            assert not (await old.get_reload_status())["engine_stopped"]
            assert await anchor.chat() == "new"
            await old.release_pending_chat()
            assert await pending == "old"
            await anchor.begin_generation_reload("third", 1)
            assert (await old.get_reload_status())["engine_stopped"]
        finally:
            await old.release_pending_chat()
            await asyncio.gather(pending, return_exceptions=True)
            await anchor.close_model_generations()


@pytest.mark.asyncio
async def test_commit_refuses_snapshot_from_another_serving_generation():
    actor = ServingGeneration("anchor")
    previous = SimpleNamespace(uid=b"restored")
    actor._serving_generation = previous
    candidate = SimpleNamespace(generation_ready=AsyncMock())
    with pytest.raises(RuntimeError, match="Serving generation changed"):
        await actor.commit_model_generation(
            candidate, "upgrade", True, "cache", b"failed"
        )
    assert actor._serving_generation is previous
    assert not actor.__dict__.get("_generation_retirement_tasks")


@pytest.mark.asyncio
async def test_snapshot_is_exported_after_waiting_for_previous_retirement():
    anchor = SimpleNamespace(
        model_generation_protocol=AsyncMock(return_value=1),
        get_reload_status=AsyncMock(return_value={}),
        fail_generation_reload=AsyncMock(),
    )
    ready = asyncio.Event()
    restored = asyncio.Event()

    async def begin(*args):
        ready.set()
        await restored.wait()

    async def export(patch):
        assert restored.is_set()
        raise ValueError("restored snapshot")

    anchor.begin_generation_reload = begin
    anchor.export_model_generation = AsyncMock(side_effect=export)
    worker = SimpleNamespace(_model_uid_to_model={"model": anchor})
    replacing = asyncio.create_task(
        replace_model_generation(worker, "model", {}, operation_id="job")
    )
    await ready.wait()
    anchor.export_model_generation.assert_not_called()
    restored.set()
    with pytest.raises(ValueError, match="restored snapshot"):
        await replacing
    anchor.fail_generation_reload.assert_awaited_once_with("job", "restored snapshot")


@pytest.mark.asyncio
async def test_snapshot_preserves_replica_uid_and_explicit_interpreter(monkeypatch):
    import importlib.metadata

    from .test_model_reload import make_actor

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.0")
    actor = make_actor()
    actor._n_worker = 1
    actor._request_limits = float("inf")
    actor._model.raw_model_uid = "model-rep3"
    actor._model.model_uid = "model"
    actor._model.model_family = SimpleNamespace(dict=lambda: {})
    actor._model.model_path = "/model"
    actor._model._weight_cache.export = lambda: {}
    snapshot = await actor.export_model_generation({})
    assert snapshot["model_uid"] == "model-rep3"
    assert snapshot["model_python_executable"] is None
    actor._generation_python_executable = "/explicit/python"
    assert (await actor.export_model_generation({}))[
        "model_python_executable"
    ] == "/explicit/python"
