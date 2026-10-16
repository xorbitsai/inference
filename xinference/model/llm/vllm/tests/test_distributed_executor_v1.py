# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import importlib.util
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def executor_module(monkeypatch):
    """Exercise the executor on CPU with only the vLLM import boundary stubbed."""

    class Executor:
        def __init__(self, config):
            self.vllm_config = config
            self.parallel_config = config.parallel_config
            self.scheduler_config = config.scheduler_config
            self._init_executor()

    monkeypatch.setitem(
        sys.modules, "vllm.v1.executor.abstract", SimpleNamespace(Executor=Executor)
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(envs=SimpleNamespace(VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=5)),
    )
    name = "xinference.model.llm.vllm._cpu_distributed_executor_v1"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "distributed_executor_v1.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


class Worker:
    def __init__(self):
        self.output = Future()
        self.calls = []
        self.killed = False

    def execute_method(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        return self.output

    def kill(self):
        self.killed = True
        result = Future()
        result.set_result(None)
        return result


@pytest.fixture
def executor(executor_module):
    isolation = executor_module._ExecutorIsolation(asyncio.new_event_loop())
    isolation.start()
    instance = object.__new__(executor_module.XinferenceDistributedExecutorV1)
    instance._isolation = isolation
    instance._loop = isolation.loop
    instance._is_shutdown = False
    instance.workers = [Worker(), Worker()]
    instance.parallel_config = SimpleNamespace(
        world_size=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
        prefill_context_parallel_size=1,
    )
    instance.scheduler_config = SimpleNamespace(async_scheduling=False)
    instance.vllm_config = SimpleNamespace()
    instance.kv_output_aggregator = None
    instance.ec_output_aggregator = None
    yield instance
    instance.shutdown()
    if not isolation.loop.is_closed():
        isolation.stop()
        isolation.loop.close()


@pytest.mark.parametrize("tp,pp", [(1, 2), (2, 2), (4, 3), (2, 1)])
def test_output_rank_is_first_tp_rank_of_last_stage(executor, tp, pp):
    executor.parallel_config.tensor_parallel_size = tp
    executor.parallel_config.pipeline_parallel_size = pp
    executor.parallel_config.world_size = tp * pp

    assert executor._get_output_rank() == (pp - 1) * tp


def test_output_rank_includes_prefill_context_parallelism(executor):
    executor.parallel_config.tensor_parallel_size = 2
    executor.parallel_config.pipeline_parallel_size = 3
    executor.parallel_config.prefill_context_parallel_size = 2
    executor.parallel_config.world_size = 12

    assert executor._get_output_rank() == 8


@pytest.mark.parametrize(
    "pp,async_scheduling,expected",
    [(3, False, 3), (2, True, 2), (1, True, 2), (1, False, 1)],
)
def test_pipeline_has_enough_batches_to_fill_stages(
    executor, pp, async_scheduling, expected
):
    executor.parallel_config.pipeline_parallel_size = pp
    executor.scheduler_config.async_scheduling = async_scheduling

    assert executor.max_concurrent_batches == expected


def test_batch_count_uses_new_vllm_config_when_available(executor):
    executor.vllm_config.max_concurrent_batches = 4

    assert executor.max_concurrent_batches == 4


def test_collective_rpc_is_non_blocking_and_keeps_rank_order(executor):
    result = executor.collective_rpc(
        "method", args=("argument",), kwargs={"option": True}, non_block=True
    )

    assert isinstance(result, Future)
    assert not result.done()
    assert all(
        worker.calls == [("method", ("argument",), {"option": True})]
        for worker in executor.workers
    )
    executor.workers[1].output.set_result("rank-1")
    executor.workers[0].output.set_result("rank-0")
    assert result.result(timeout=2) == ["rank-0", "rank-1"]


@pytest.mark.parametrize("non_block", [False, True])
def test_worker_kwargs_do_not_collide_with_rpc_controls(executor, non_block):
    kwargs = {
        "timeout": "worker timeout",
        "non_block": "worker non_block",
        "output_rank": "worker output_rank",
        "aggregate_output": "worker aggregate_output",
        "max_concurrent_workers": "worker concurrency",
        "async_run_tensor_parallel_workers_only": "worker selection",
        "args": "worker args",
        "kwargs": "worker kwargs",
    }
    for rank, worker in enumerate(executor.workers):
        worker.output.set_result(rank)
    result = executor.collective_rpc(
        "method", timeout=1, args=("input",), kwargs=kwargs, non_block=non_block
    )
    if non_block:
        result = result.result(timeout=2)
    assert result == [0, 1]
    assert all(
        worker.calls == [("method", ("input",), kwargs)] for worker in executor.workers
    )


@pytest.mark.parametrize("method", ["execute_model", "sample_tokens"])
def test_non_blocking_output_waits_for_all_stages(executor, method):
    result = getattr(executor, method)("input", non_block=True)
    executor.workers[1].output.set_result("tokens")

    assert not result.done()
    executor.workers[0].output.set_result(None)
    assert result.result(timeout=2) == "tokens"


@pytest.mark.parametrize("method", ["execute_model", "sample_tokens"])
def test_failure_on_non_output_rank_reaches_engine(executor, method):
    result = getattr(executor, method)(None, non_block=True)
    executor.workers[0].output.set_exception(RuntimeError("first stage failed"))

    # The last stage can be stuck waiting for activation tensors from stage 0.
    with pytest.raises(RuntimeError, match="first stage failed"):
        result.result(timeout=2)


@pytest.mark.parametrize("non_block", [False, True])
def test_collective_rpc_timeout_is_applied(executor, non_block):
    with pytest.raises(TimeoutError, match="method"):
        result = executor.collective_rpc("method", timeout=0.01, non_block=non_block)
        if non_block:
            result.result(timeout=2)


@pytest.mark.parametrize("method", ["execute_model", "sample_tokens"])
def test_model_execution_uses_vllm_timeout(executor_module, executor, method):
    executor_module.envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS = 0.01

    with pytest.raises(TimeoutError, match=method):
        getattr(executor, method)(None, non_block=True).result(timeout=2)


@pytest.mark.parametrize("method", ["execute_model", "sample_tokens"])
@pytest.mark.parametrize("tp,pp", [(1, 2), (2, 2), (1, 4)])
def test_blocking_output_comes_from_last_stage(executor, method, tp, pp):
    executor.parallel_config.tensor_parallel_size = tp
    executor.parallel_config.pipeline_parallel_size = pp
    executor.parallel_config.world_size = tp * pp
    executor.workers = [Worker() for _ in range(tp * pp)]
    for rank, worker in enumerate(executor.workers):
        worker.output.set_result("tokens" if rank == (pp - 1) * tp else None)

    assert getattr(executor, method)(None) == "tokens"


def test_draft_tokens_come_from_last_stage(executor):
    executor.workers[0].output.set_result(None)
    executor.workers[1].output.set_result([3, 4])

    assert executor.take_draft_token_ids() == [3, 4]


def test_check_health_queries_workers(executor):
    for worker in executor.workers:
        worker.output.set_result(None)
    executor.check_health()

    assert all(worker.calls[0][0] == "check_health" for worker in executor.workers)


def test_shutdown_stops_isolation_and_is_idempotent(executor):
    executor.shutdown()
    executor.shutdown()

    assert all(worker.killed for worker in executor.workers)
    assert executor._loop.is_closed()


def test_shutdown_shares_one_deadline_for_unresponsive_kills(
    executor_module, executor, monkeypatch
):
    monkeypatch.setattr(executor_module, "_SHUTDOWN_TIMEOUT_SECONDS", 0.05)
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        executor_module, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    timeouts = []

    class UnresponsiveKill(Future):
        def result(self, timeout=None):
            timeouts.append(timeout)
            # Consume the budget deterministically: Windows waits can return
            # just before the monotonic deadline despite timing out.
            clock.now += timeout
            raise TimeoutError()

    futures = [UnresponsiveKill() for _ in range(10)]
    executor.workers = [SimpleNamespace(kill=lambda fut=fut: fut) for fut in futures]
    started = time.monotonic()
    executor.shutdown()
    assert time.monotonic() - started < 0.5
    assert timeouts == [0.05] + [0] * 9
    assert not any(fut.done() for fut in futures)
    executor._isolation._thread.join(timeout=1)
    assert executor._loop.is_closed()


def test_shutdown_does_not_wait_for_stuck_cancellation(
    executor_module, executor, monkeypatch
):
    monkeypatch.setattr(executor_module, "_SHUTDOWN_TIMEOUT_SECONDS", 0.05)
    started = threading.Event()
    cancelling = threading.Event()
    release = threading.Event()

    async def unresponsive_rpc():
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelling.set()
            release.wait()

    result = asyncio.run_coroutine_threadsafe(unresponsive_rpc(), executor._loop)
    try:
        assert started.wait(timeout=1)
        shutdown_started = time.monotonic()
        executor.shutdown()
        assert time.monotonic() - shutdown_started < 0.5
        assert cancelling.wait(timeout=1)
        assert executor._isolation._thread.is_alive()
        assert not executor._loop.is_closed()
    finally:
        release.set()
        executor._isolation._thread.join(timeout=1)
    assert result.result(timeout=1) is None
    assert executor._loop.is_closed()


def test_output_aggregates_connector_metadata_from_every_rank(executor):
    calls = []

    def aggregate(outputs, output_rank):
        calls.append((outputs, output_rank))
        return "merged"

    executor.kv_output_aggregator = SimpleNamespace(aggregate=aggregate)
    executor.workers[0].output.set_result("rank-0-metadata")
    executor.workers[1].output.set_result("tokens")

    assert executor.sample_tokens() == "merged"
    assert calls == [(["rank-0-metadata", "tokens"], 1)]


@pytest.mark.parametrize(
    "world_size,n_worker,addresses", [(4, 2, 2), (3, 2, 3), (2, 0, 2)]
)
def test_invalid_placement_is_rejected_before_actor_creation(
    executor_module, monkeypatch, world_size, n_worker, addresses
):
    created = []

    async def create_actor(*args, **kwargs):
        created.append(kwargs)

    monkeypatch.setattr(executor_module.xo, "create_actor", create_actor)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(world_size=world_size, tensor_parallel_size=1),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )
    with pytest.raises(ValueError) as exc_info:
        executor_module.XinferenceDistributedExecutorV1(
            config, [f"127.0.0.1:{1000 + i}" for i in range(addresses)], n_worker
        )
    assert not created
    assert str(world_size) in str(exc_info.value)
    assert str(n_worker if addresses == world_size else addresses) in str(
        exc_info.value
    )


@pytest.mark.parametrize(
    "n_worker,world_size,tp",
    [(1, 1, 1), (1, 2, 2), (1, 2, 1), (2, 2, 1), (2, 4, 2), (4, 4, 1), (2, 8, 2)],
)
def test_init_executor_maps_ranks_and_propagates_environment(
    executor_module, monkeypatch, n_worker, world_size, tp
):
    created = []
    refs = {}
    destroyed = []
    env = {
        "VLLM_USE_V1": "1",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "VLLM_TRACE_FUNCTION": "1",
    }
    for name in ("TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("XINFERENCE_TEST_UNRELATED", "not forwarded")
    monkeypatch.setattr(executor_module, "get_next_port", lambda: 54321)
    monkeypatch.setattr(
        executor_module,
        "get_distributed_init_method",
        lambda host, port: f"tcp://{host}:{port}",
    )

    class Actor:
        def __init__(self, rank):
            self.rank = rank
            self.calls = []

        async def execute_method(self, method, *args, **kwargs):
            self.calls.append((method, args, kwargs))

    async def create_actor(cls, rpc_rank, address, uid):
        created.append((cls, rpc_rank, address, uid))
        await asyncio.sleep((world_size - rpc_rank) * 0.001)
        refs[rpc_rank] = Actor(rpc_rank)
        return refs[rpc_rank]

    async def destroy_actor(ref):
        destroyed.append(ref.rank)

    monkeypatch.setattr(executor_module.xo, "create_actor", create_actor)
    monkeypatch.setattr(executor_module.xo, "destroy_actor", destroy_actor)
    addresses = [
        f"10.0.0.{rank // (world_size // n_worker) + 1}:{1000 + rank}"
        for rank in range(world_size)
    ]
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=world_size,
            tensor_parallel_size=tp,
            max_parallel_loading_workers=None,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )
    instance = executor_module.XinferenceDistributedExecutorV1(
        config, addresses, n_worker
    )
    try:
        assert created == [
            (executor_module.WorkerActor, rank, address, f"VllmWorker_{rank}")
            for rank, address in enumerate(addresses)
        ]
        assert [worker._worker_actor_ref.rank for worker in instance.workers] == list(
            range(world_size)
        )
        assert instance._env_vars_for_all_workers == [env] * world_size
        assert (
            len({id(values) for values in instance._env_vars_for_all_workers})
            == world_size
        )
        for rank in range(world_size):
            calls = refs[rank].calls
            assert [call[0] for call in calls] == [
                "update_environment_variables",
                "init_worker",
                "init_device",
                "load_model",
            ]
            assert calls[0][1] == ([env] * world_size,)
            assert calls[1][1][0][rank] == dict(
                vllm_config=config,
                local_rank=rank % (world_size // n_worker),
                rank=rank,
                distributed_init_method="tcp://10.0.0.1:54321",
                is_driver_worker=rank % tp == 0,
            )
            assert all(call[2] == {} for call in calls)
        assert [
            worker._worker_actor_ref.rank for worker in instance.tp_driver_workers
        ] == [rank for rank in range(1, world_size) if rank % tp == 0]
        assert [
            worker._worker_actor_ref.rank for worker in instance.non_driver_workers
        ] == [rank for rank in range(1, world_size) if rank % tp != 0]
    finally:
        instance.shutdown()
    assert sorted(destroyed) == list(range(world_size))


def test_failed_init_destroys_all_created_actors(executor_module, monkeypatch):
    destroyed = []

    async def create_actor(*args, rpc_rank, **kwargs):
        if rpc_rank == 1:
            raise RuntimeError("rank creation failed")
        return rpc_rank

    async def destroy_actor(ref):
        destroyed.append(ref)

    monkeypatch.setattr(executor_module.xo, "create_actor", create_actor)
    monkeypatch.setattr(executor_module.xo, "destroy_actor", destroy_actor)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(world_size=3, tensor_parallel_size=1),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )
    with pytest.raises(RuntimeError, match="rank creation failed"):
        executor_module.XinferenceDistributedExecutorV1(
            config, [f"127.0.0.1:{1000 + i}" for i in range(3)], 1
        )
    assert sorted(destroyed) == [0, 2]


@pytest.mark.parametrize("method", ["execute_model", "sample_tokens"])
@pytest.mark.parametrize("kind", ["none", "sync", "async"])
def test_worker_resolves_output_before_serialization(executor_module, method, kind):
    from ..distributed_worker_actor import WorkerActor

    output = SimpleNamespace(tokens=[1, 2])
    if kind == "none":
        raw = None
        expected = None
    elif kind == "sync":
        raw = expected = output
    else:
        raw = SimpleNamespace(get_output=lambda: output)
        expected = output
    worker = WorkerActor.__new__(WorkerActor)
    worker._worker = SimpleNamespace(**{method: lambda *args: raw})

    assert worker.execute_method(method, None) is expected
