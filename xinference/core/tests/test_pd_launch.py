# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ..model import ModelNotReadyError
from ..replica_config import DeviceConfig, ReplicaConfig
from ..supervisor import SupervisorActor


@pytest.fixture
async def launch_runtime(monkeypatch):
    supervisor = SupervisorActor()
    supervisor.address = "supervisor:9999"
    supervisor._status_guard_ref = AsyncMock()
    supervisor._block_tracker_mapping = {}
    supervisor._collective_manager_mapping = {}
    workers = []
    for i in range(2):
        worker = MagicMock(address=f"worker-{i}:1234")
        worker.launch_rank0_model = AsyncMock(return_value=("worker-0:4321", 9876))
        worker.launch_builtin_model = AsyncMock(return_value=f"worker-{i}:5678")
        worker.wait_for_load = AsyncMock()
        worker.start_transfer_for_vllm = AsyncMock()
        worker.get_model = AsyncMock(return_value=MagicMock())
        worker.terminate_model = AsyncMock()
        workers.append(worker)
    supervisor._worker_address_to_worker = {w.address: w for w in workers}
    supervisor._resolve_replica_config = AsyncMock(
        return_value=([(w, [0], 1) for w in workers], {0: "p", 1: "d"})
    )
    supervisor._resolve_download_hub_from_workers = AsyncMock(
        return_value="huggingface"
    )
    actors = {}

    async def create_actor(cls, *args, **kwargs):
        ref = AsyncMock(address=kwargs["address"], uid=kwargs["uid"])
        actors[cls.__name__] = ref
        return ref

    monkeypatch.setattr("xinference.core.supervisor.xo.create_actor", create_actor)
    destroy = AsyncMock()
    monkeypatch.setattr("xinference.core.supervisor.xo.destroy_actor", destroy)
    return supervisor, workers, actors, destroy


def launch_kwargs():
    return dict(
        model_uid="pd",
        model_name="test",
        model_size_in_billions=1,
        model_format="pytorch",
        quantization=None,
        model_engine="vLLM",
        model_type="LLM",
        replica=2,
        replica_config=[
            ReplicaConfig(
                role=role, devices=[DeviceConfig(worker_ip=f"worker-{i}:1234")]
            )
            for i, role in enumerate(["prefill", "decode"])
        ],
    )


@pytest.mark.asyncio
async def test_pd_launch_routes_and_terminates(launch_runtime):
    supervisor, workers, actors, destroy = launch_runtime
    assert await supervisor.launch_builtin_model(**launch_kwargs()) == "pd"
    assert await supervisor.get_model("pd") is actors["PDModelActor"]
    for i, worker in enumerate(workers):
        config = worker.launch_builtin_model.call_args.kwargs["xavier_config"]
        assert config["rank"] == i + 1
        assert config["world_size"] == 3
        assert config["role"] == ["prefill", "decode"][i]
        assert config["store_address"] == "worker-0"
        assert config["store_port"] == 9876
        assert worker.start_transfer_for_vllm.await_count == (2 if i == 0 else 1)
    actors["PDModelActor"].add_prefill_actor.assert_awaited_once_with(
        "pd-rep0", workers[0].get_model.return_value
    )
    actors["PDModelActor"].add_decode_actor.assert_awaited_once_with(
        "pd-rep1", workers[1].get_model.return_value
    )
    await supervisor.terminate_model("pd")
    assert not supervisor._pd_model_mapping
    assert not supervisor._pd_roles
    assert not supervisor._block_tracker_mapping
    assert not supervisor._collective_manager_mapping
    assert not supervisor._replica_model_uid_to_worker
    assert destroy.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["load", "transfer", "register"])
async def test_failed_launch_rolls_back(launch_runtime, stage):
    supervisor, workers, actors, destroy = launch_runtime
    if stage == "load":
        workers[1].wait_for_load.side_effect = RuntimeError("boom")
    elif stage == "transfer":
        workers[1].start_transfer_for_vllm.side_effect = RuntimeError("boom")
    else:
        workers[1].get_model.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        await supervisor.launch_builtin_model(**launch_kwargs())
    assert not supervisor._pd_model_mapping
    assert not supervisor._pd_roles
    assert not supervisor._model_uid_to_replica_info
    assert not supervisor._replica_model_uid_to_worker
    assert not supervisor._block_tracker_mapping
    assert not supervisor._collective_manager_mapping
    workers[1].terminate_model.assert_awaited_once()


@pytest.mark.asyncio
async def test_pd_route_waits_for_transfers(launch_runtime):
    supervisor, workers, actors, destroy = launch_runtime
    entered, proceed = asyncio.Event(), asyncio.Event()

    async def wait(*args):
        entered.set()
        await proceed.wait()

    workers[1].start_transfer_for_vllm.side_effect = wait
    launch = asyncio.create_task(supervisor.launch_builtin_model(**launch_kwargs()))
    await entered.wait()
    try:
        with pytest.raises(ModelNotReadyError):
            await supervisor.get_model("pd")
    finally:
        proceed.set()
        await launch
    assert await supervisor.get_model("pd") is actors["PDModelActor"]


@pytest.mark.asyncio
async def test_invalid_pd_launch_has_no_side_effects(launch_runtime):
    supervisor, workers, actors, destroy = launch_runtime
    kwargs = launch_kwargs()
    kwargs["replica_config"][1] = kwargs["replica_config"][0]
    with pytest.raises(ValueError, match="both prefill and decode"):
        await supervisor.launch_builtin_model(**kwargs)
    assert not actors
    assert not supervisor._model_uid_to_replica_info
    for worker in workers:
        worker.launch_builtin_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovery_refreshes_pd_router(launch_runtime):
    supervisor, workers, actors, destroy = launch_runtime
    await supervisor.launch_builtin_model(**launch_kwargs())
    await supervisor.call_collective_manager("pd", "unregister_rank", 1)
    actors["PDModelActor"].remove_prefill_actor.assert_awaited_once_with("pd-rep0")
    replacement = MagicMock()
    await supervisor.register_pd_replica("pd", "pd-rep0", replacement)
    actors["PDModelActor"].add_prefill_actor.assert_awaited_with("pd-rep0", replacement)
