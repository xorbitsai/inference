# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import importlib.util
import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

from .....isolation import Isolation


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
    isolation = Isolation(asyncio.new_event_loop())
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
    with pytest.raises(ValueError):
        executor_module.XinferenceDistributedExecutorV1(
            config, [f"127.0.0.1:{1000 + i}" for i in range(addresses)], n_worker
        )
    assert not created


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
