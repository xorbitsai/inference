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
        ref.constructor_kwargs = kwargs
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
@pytest.mark.parametrize("pd", [False, True])
@pytest.mark.parametrize("quantization", [None, "none", "fp16", "bf16"])
async def test_mlx_xavier_launch_and_cleanup(launch_runtime, pd, quantization):
    supervisor, workers, actors, destroy = launch_runtime
    kwargs = launch_kwargs()
    kwargs.update(
        model_engine="MLX",
        model_format="mlx",
        quantization=quantization,
        xavier_cache_bytes=123456,
    )
    if not pd:
        kwargs["enable_xavier"] = True
        for replica in kwargs["replica_config"]:
            replica.role = "hybrid"
    await supervisor.launch_builtin_model(**kwargs)
    assert set(actors) == (
        {"XavierBytesCacheActor", "PDModelActor"} if pd else {"XavierBytesCacheActor"}
    )
    assert (
        actors["XavierBytesCacheActor"].constructor_kwargs["capacity_bytes"] == 123456
    )
    for role, worker in zip(("prefill", "decode"), workers):
        launch = worker.launch_builtin_model.call_args.kwargs
        config = {"address": supervisor.address, "uid": "xavier-cache-pd"}
        if pd:
            config["role"] = role
        assert launch["_xavier_cache_config"] == config
        assert launch["xavier_config"] is None
        assert "_nixl_config" not in launch
        worker.launch_rank0_model.assert_not_awaited()
        worker.start_transfer_for_vllm.assert_not_awaited()
    await supervisor.terminate_model("pd")
    assert not supervisor._xavier_cache_mapping and not supervisor._pd_model_mapping
    assert destroy.await_count == (2 if pd else 1)


@pytest.mark.asyncio
async def test_single_mlx_replica_disables_shared_xavier(launch_runtime, caplog):
    supervisor, workers, actors, _ = launch_runtime
    supervisor._resolve_replica_config.return_value = ([(workers[0], [0], 1)], {0: "p"})
    kwargs = launch_kwargs()
    kwargs.update(model_engine="MLX", model_format="mlx", enable_xavier=True, replica=1)
    kwargs["replica_config"] = kwargs["replica_config"][:1]
    kwargs["replica_config"][0].role = None
    await supervisor.launch_builtin_model(**kwargs)
    assert "replica<=1" in caplog.text
    assert not actors and not supervisor._xavier_cache_mapping
    assert (
        "_xavier_cache_config" not in workers[0].launch_builtin_model.call_args.kwargs
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["prefill", "decode"])
async def test_mlx_pd_recovery_keeps_bytes_cache_and_registers_replacement(role):
    import xoscar as xo

    from ...model.llm.xavier.backends.bytes.storage import XavierBytesCacheActor
    from ...model.llm.xavier.tests.test_bytes_storage import contract
    from ..worker import WorkerActor

    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        cache = await xo.create_actor(
            XavierBytesCacheActor, address=pool.external_address, uid="bytes-cache"
        )
        namespace = (await cache.configure(contract().to_dict()))["namespace"]
        worker = MagicMock()
        supervisor = AsyncMock()
        worker.get_supervisor_ref = AsyncMock(return_value=supervisor)
        worker.launch_builtin_model = AsyncMock(return_value="replacement:1234")
        worker.wait_for_load = AsyncMock()
        replacement = MagicMock()
        worker._model_uid_to_model = {"pd-rep0": replacement}
        config = dict(role=role, address=cache.address, uid=cache.uid)
        await WorkerActor.recover_model(
            worker, dict(model_uid="pd-rep0", _xavier_cache_config=config)
        )
        worker.launch_builtin_model.assert_awaited_once_with(
            model_uid="pd-rep0", _xavier_cache_config=config
        )
        worker.wait_for_load.assert_awaited_once_with("pd-rep0")
        supervisor.unregister_pd_replica.assert_awaited_once_with("pd", "pd-rep0")
        supervisor.register_pd_replica.assert_awaited_once_with(
            "pd", "pd-rep0", replacement
        )
        assert (await cache.get_stats())["namespace"] == namespace


@pytest.mark.asyncio
async def test_mlx_failed_launch_cleans_bytes_actor(launch_runtime):
    supervisor, workers, actors, destroy = launch_runtime
    kwargs = launch_kwargs()
    kwargs.update(model_engine="MLX", model_format="mlx")
    workers[1].wait_for_load.side_effect = RuntimeError("bad model")
    with pytest.raises(RuntimeError, match="bad model"):
        await supervisor.launch_builtin_model(**kwargs)
    assert not supervisor._xavier_cache_mapping
    assert not supervisor._model_uid_to_replica_info
    destroy.assert_awaited_once_with(actors["XavierBytesCacheActor"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"transfer_backend_type": "nixl"},
        {"model_format": "pytorch"},
        {"quantization": "4bit"},
        {"xavier_gpu_cache_bytes": 1024},
        {"xavier_cache_bytes": 0},
        {"xavier_cache_bytes": True},
    ],
)
async def test_mlx_invalid_launch_has_no_actors(launch_runtime, options):
    supervisor, workers, actors, _ = launch_runtime
    kwargs = launch_kwargs()
    kwargs.update(model_engine="MLX", model_format="mlx")
    kwargs.update(options)
    with pytest.raises(ValueError):
        await supervisor.launch_builtin_model(**kwargs)
    assert not actors and not supervisor._model_uid_to_replica_info
    for worker in workers:
        worker.launch_builtin_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_sglang_xavier_uses_hicache_without_vllm_collective(launch_runtime):
    supervisor, workers, actors, destroy = launch_runtime
    kwargs = launch_kwargs()
    kwargs.update(model_engine="SGLang", enable_xavier=True, xavier_cache_bytes=1024)
    for replica in kwargs["replica_config"]:
        replica.role = None
    await supervisor.launch_builtin_model(**kwargs)
    assert set(actors) == {"XavierCacheActor"}
    for worker in workers:
        launch = worker.launch_builtin_model.call_args.kwargs
        assert launch["xavier_config"] is None
        assert launch["_xavier_cache_config"] == {
            "address": supervisor.address,
            "uid": "xavier-cache-pd",
        }
        worker.start_transfer_for_vllm.assert_not_awaited()
        worker.launch_rank0_model.assert_not_awaited()
    await supervisor.terminate_model("pd")
    assert not supervisor._xavier_cache_mapping
    destroy.assert_awaited_once_with(actors["XavierCacheActor"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_format,quantization", [("pytorch", None), ("ggufv2", "4bit")]
)
async def test_single_sglang_replica_disables_shared_xavier(
    launch_runtime, model_format, quantization
):
    supervisor, workers, actors, _ = launch_runtime
    supervisor._resolve_replica_config.return_value = ([(workers[0], [0], 1)], {0: "p"})
    kwargs = launch_kwargs()
    kwargs.update(
        model_engine="SGLang",
        enable_xavier=True,
        replica=1,
        model_format=model_format,
        quantization=quantization,
    )
    kwargs["replica_config"] = kwargs["replica_config"][:1]
    kwargs["replica_config"][0].role = None
    await supervisor.launch_builtin_model(**kwargs)
    assert not actors and not supervisor._xavier_cache_mapping
    launch = workers[0].launch_builtin_model.call_args.kwargs
    assert "_xavier_cache_config" not in launch
    assert launch["xavier_config"] is None
    workers[1].launch_builtin_model.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["2001:db8::1", "::1"])
async def test_sglang_ipv6_host_survives_supervisor_engine_and_gpu_pool(
    launch_runtime, monkeypatch, host
):
    import importlib.metadata
    import importlib.util

    from ...model.llm.xavier.transport import get_transport_host, gpu_pool_options

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.11.1")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    supervisor, workers, _, _ = launch_runtime
    workers[0].address = f"tcp://[{host}]:1234"
    supervisor._worker_address_to_worker = {w.address: w for w in workers}
    kwargs = launch_kwargs()
    kwargs.update(model_engine="SGLang")
    kwargs["replica_config"][0].devices[0].worker_ip = workers[0].address
    await supervisor.launch_builtin_model(**kwargs)
    config = workers[0].launch_builtin_model.call_args.kwargs["_xavier_cache_config"]
    assert config["host"] == host
    engine_host = get_transport_host(config["host"])
    assert engine_host == host
    assert gpu_pool_options(engine_host, {}) == {
        "external_address": f"nixl://[{host}]:0"
    }


@pytest.mark.asyncio
async def test_sglang_xavier_failed_launch_cleans_cache_actor(launch_runtime):
    supervisor, workers, actors, destroy = launch_runtime
    kwargs = launch_kwargs()
    kwargs.update(model_engine="SGLang", enable_xavier=True)
    for replica in kwargs["replica_config"]:
        replica.role = None
    workers[1].wait_for_load.side_effect = RuntimeError("bad model")
    with pytest.raises(RuntimeError, match="bad model"):
        await supervisor.launch_builtin_model(**kwargs)
    assert not supervisor._xavier_cache_mapping
    destroy.assert_awaited_once_with(actors["XavierCacheActor"])


@pytest.mark.asyncio
async def test_sglang_pd_routes_roles_and_cleans_cache(launch_runtime):
    supervisor, workers, actors, destroy = launch_runtime
    kwargs = launch_kwargs()
    kwargs["model_engine"] = "SGLang"
    await supervisor.launch_builtin_model(**kwargs)
    assert set(actors) == {"XavierPDDirectory", "PDModelActor"}
    for rank, (role, worker) in enumerate(zip(("prefill", "decode"), workers), 1):
        launch = worker.launch_builtin_model.call_args.kwargs
        assert launch["_xavier_cache_config"]["role"] == role
        assert launch["_xavier_cache_config"]["rank"] == rank
        assert launch["_xavier_cache_config"]["host"] == f"worker-{rank - 1}"
        assert launch["xavier_config"] is None
        worker.launch_rank0_model.assert_not_awaited()
    assert await supervisor.get_model("pd") is actors["PDModelActor"]
    await supervisor.terminate_model("pd")
    assert not supervisor._xavier_cache_mapping
    assert not supervisor._pd_model_mapping
    assert destroy.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("option", ["xavier_cache_bytes", "xavier_gpu_cache_bytes"])
async def test_sglang_gpu_pd_rejects_cpu_cache_and_retained_history(
    launch_runtime, option
):
    supervisor, _, actors, _ = launch_runtime
    kwargs = launch_kwargs()
    kwargs.update(model_engine="SGLang", **{option: 1024})
    with pytest.raises(ValueError):
        await supervisor.launch_builtin_model(**kwargs)
    assert not actors


@pytest.mark.asyncio
async def test_sglang_native_nixl_launch_uses_same_pd_route(launch_runtime):
    supervisor, workers, actors, destroy = launch_runtime
    kwargs = launch_kwargs()
    kwargs.update(model_engine="SGLang", transfer_backend_type="nixl")
    await supervisor.launch_builtin_model(**kwargs)
    assert set(actors) == {"PDModelActor"}
    assert actors["PDModelActor"].constructor_kwargs["model_engine"] == "SGLang"
    for index, (role, worker) in enumerate(zip(("prefill", "decode"), workers)):
        launch = worker.launch_builtin_model.call_args.kwargs
        assert launch["_nixl_config"] == {"role": role, "host": f"worker-{index}"}
        assert "_xavier_cache_config" not in launch
        assert launch["xavier_config"] is None
        worker.launch_rank0_model.assert_not_awaited()
    await supervisor.terminate_model("pd")
    destroy.assert_awaited_once_with(actors["PDModelActor"])


@pytest.mark.asyncio
async def test_pd_launch_routes_and_terminates(launch_runtime):
    supervisor, workers, actors, destroy = launch_runtime
    assert await supervisor.launch_builtin_model(**launch_kwargs()) == "pd"
    assert await supervisor.get_model("pd") is actors["PDModelActor"]
    assert actors["PDModelActor"].constructor_kwargs["transport_backend"] == "xavier"
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


@pytest.mark.asyncio
async def test_nixl_launch_skips_xavier_collectives(launch_runtime):
    supervisor, workers, actors, destroy = launch_runtime
    await supervisor.launch_builtin_model(
        **launch_kwargs(), vllm_transfer_backend_type="nixl"
    )
    assert set(actors) == {"PDModelActor"}
    assert actors["PDModelActor"].constructor_kwargs["transport_backend"] == "nixl"
    for worker, role in zip(workers, ("prefill", "decode")):
        kwargs = worker.launch_builtin_model.call_args.kwargs
        assert kwargs["xavier_config"] is None
        assert kwargs["_nixl_config"] == {"role": role}
        worker.launch_rank0_model.assert_not_awaited()
        worker.start_transfer_for_vllm.assert_not_awaited()
    await supervisor.terminate_model("pd")
    assert not supervisor._pd_model_mapping
    assert not supervisor._replica_model_uid_to_worker
    assert destroy.await_count == 1


@pytest.mark.asyncio
async def test_nixl_worker_recovery_refreshes_route_without_collectives():
    from ..worker import WorkerActor

    worker = MagicMock()
    supervisor = AsyncMock()
    worker.get_supervisor_ref = AsyncMock(return_value=supervisor)
    worker.launch_builtin_model = AsyncMock(return_value="new:1234")
    worker.wait_for_load = AsyncMock()
    replacement = MagicMock()
    worker._model_uid_to_model = {"pd-rep0": replacement}
    await WorkerActor.recover_model(
        worker, {"model_uid": "pd-rep0", "_nixl_config": {"role": "prefill"}}
    )
    supervisor.unregister_pd_replica.assert_awaited_once_with("pd", "pd-rep0")
    supervisor.register_pd_replica.assert_awaited_once_with(
        "pd", "pd-rep0", replacement
    )
    supervisor.call_collective_manager.assert_not_awaited()


@pytest.mark.asyncio
async def test_nixl_rejects_unmanaged_scale_up(launch_runtime):
    supervisor, workers, actors, destroy = launch_runtime
    await supervisor.launch_builtin_model(
        **launch_kwargs(), vllm_transfer_backend_type="nixl"
    )
    with pytest.raises(ValueError, match="PD topology"):
        await supervisor._add_model_replica("pd")
    assert workers[0].launch_builtin_model.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "replica_config",
    [None, [ReplicaConfig(role="hybrid"), ReplicaConfig(role="hybrid")]],
)
async def test_nixl_requires_explicit_pd_roles(launch_runtime, replica_config):
    supervisor, workers, actors, destroy = launch_runtime
    kwargs = launch_kwargs()
    kwargs["replica_config"] = replica_config
    with pytest.raises(ValueError, match="NIXL requires explicit prefill and decode"):
        await supervisor.launch_builtin_model(
            **kwargs, vllm_transfer_backend_type="nixl"
        )
    assert not actors
    assert not supervisor._model_uid_to_replica_info
    for worker in workers:
        worker.launch_builtin_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_nixl_startup_replay_is_skipped_and_removed(tmp_path):
    import json

    from ..worker import WorkerActor

    class Worker:
        _load_persisted_launch_args = WorkerActor._load_persisted_launch_args
        _persist_launch_args = WorkerActor._persist_launch_args

        def _get_recovery_file_path(self):
            return str(tmp_path / "models.json")

    worker = Worker()
    worker._supervisor_ref = AsyncMock()
    worker._supervisor_ref.describe_model.return_value = {"model_name": "still-running"}
    worker._model_uid_to_launch_args = {}
    worker.launch_builtin_model = AsyncMock()
    worker.wait_for_load = AsyncMock()
    (tmp_path / "models.json").write_text(
        json.dumps(
            {"pd-rep0": {"model_uid": "pd-rep0", "_nixl_config": {"role": "prefill"}}}
        )
    )
    await WorkerActor._try_recover_models(worker)
    worker.launch_builtin_model.assert_not_awaited()
    worker.wait_for_load.assert_not_awaited()
    assert json.loads((tmp_path / "models.json").read_text()) == {}


def test_registration_snapshot_excludes_nixl():
    from ..worker import WorkerActor

    worker = MagicMock()
    worker._model_uid_to_model_spec = {"pd-rep0": {}, "regular-rep0": {}}
    worker._model_uid_to_launch_args = {
        "pd-rep0": {"_nixl_config": {"role": "prefill"}},
        "regular-rep0": {},
    }
    snapshots = WorkerActor._get_running_replica_states(worker)
    assert [item["replica_model_uid"] for item in snapshots] == ["regular-rep0"]


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [0, 8388608])
async def test_gpu_budget_reaches_each_xavier_replica(launch_runtime, budget):
    supervisor, workers, actors, destroy = launch_runtime
    await supervisor.launch_builtin_model(
        **launch_kwargs(), xavier_gpu_cache_bytes=budget
    )
    for worker in workers:
        kwargs = worker.launch_builtin_model.call_args.kwargs
        assert kwargs["xavier_config"]["gpu_cache_bytes"] == budget
        assert "xavier_gpu_cache_bytes" not in kwargs


@pytest.mark.asyncio
async def test_gpu_budget_rejects_native_backend_before_actor_creation(launch_runtime):
    supervisor, workers, actors, destroy = launch_runtime
    with pytest.raises(ValueError, match="requires Xavier"):
        await supervisor.launch_builtin_model(
            **launch_kwargs(),
            xavier_gpu_cache_bytes=1,
            vllm_transfer_backend_type="nixl",
        )
    assert not actors


@pytest.mark.asyncio
async def test_gpu_pool_options_reach_actual_subpool(monkeypatch):
    import importlib.metadata
    import importlib.util
    from types import MethodType, SimpleNamespace

    from ...model.llm.vllm.xavier.transport import gpu_pool_options
    from ..worker import WorkerActor

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.11.1")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    append = AsyncMock(return_value="nixl://10.0.0.1:2345")
    worker = SimpleNamespace(
        _main_pool=SimpleNamespace(append_sub_pool=append),
        _subpool_creation_lock=asyncio.Lock(),
        _ensure_subpool_monitor=AsyncMock(),
    )
    worker._append_sub_pool_protected = MethodType(
        WorkerActor._append_sub_pool_protected, worker
    )
    env = {}
    address = await WorkerActor._spawn_subpool(
        worker, "pd", env, [], **gpu_pool_options("10.0.0.1:1234", env)
    )
    assert address == "nixl://10.0.0.1:2345"
    append.assert_awaited_once_with(
        env=env, start_python=None, external_address="nixl://10.0.0.1:0"
    )
    assert env["UCX_MEMTYPE_CACHE"] == "n"
    worker._ensure_subpool_monitor.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [None, 0, 268435456])
async def test_gpu_pd_defaults_to_direct_handoff(launch_runtime, budget):
    supervisor, workers, actors, _ = launch_runtime
    await supervisor.launch_builtin_model(
        **launch_kwargs(), xavier_gpu_cache_bytes=budget
    )
    assert actors["PDModelActor"].constructor_kwargs["transport_backend"] == "xavier"
    for worker in workers:
        config = worker.launch_builtin_model.call_args.kwargs
        assert config["xavier_config"]["gpu_cache_bytes"] == (
            268435456 if budget is None else budget
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["prefill", "decode"])
async def test_sglang_recovery_replaces_registered_gpu_peer_and_invalidates_rooms(role):
    import xoscar as xo

    from ...model.llm.sglang.xavier.directory import XavierPDDirectory
    from ..worker import WorkerActor

    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        directory = await xo.create_actor(
            XavierPDDirectory, address=pool.external_address, uid="directory"
        )
        await directory.configure("ns")
        await directory.register_peer(0, "old:1234")
        await directory.prepare(1, "ns", "prompt", role, 0)
        await directory.prepare(2, "ns", "other", role, 1)
        with pytest.raises(ValueError, match="restarted"):
            await directory.register_peer(0, "new:1234")
        worker = MagicMock()
        supervisor = AsyncMock()
        worker.get_supervisor_ref = AsyncMock(return_value=supervisor)

        async def launch(**kwargs):
            assert (await directory.get_stats())["peers"] == {}
            with pytest.raises(RuntimeError, match="cancelled"):
                await directory.source(1)
            assert await directory.source(2) is None
            await directory.register_peer(0, "new:1234")
            return "new:1234"

        worker.launch_builtin_model = AsyncMock(side_effect=launch)
        worker.wait_for_load = AsyncMock()
        replacement = MagicMock()
        worker._model_uid_to_model = {"pd-rep0": replacement}
        await WorkerActor.recover_model(
            worker,
            {
                "model_uid": "pd-rep0",
                "_xavier_cache_config": {
                    "role": role,
                    "rank": 0,
                    "address": directory.address,
                    "uid": directory.uid,
                },
            },
        )
        supervisor.unregister_pd_replica.assert_awaited_once_with("pd", "pd-rep0")
        supervisor.register_pd_replica.assert_awaited_once_with(
            "pd", "pd-rep0", replacement
        )
        assert not await directory.unregister_peer(0, "old:1234")
        assert (await directory.get_stats())["peers"] == {0: "new:1234"}


@pytest.mark.asyncio
async def test_sglang_xavier_launch_accepts_default_model_format(launch_runtime):
    supervisor, workers, actors, _ = launch_runtime
    kwargs = launch_kwargs()
    kwargs.update(model_engine="SGLang", enable_xavier=True, model_format=None)
    await supervisor.launch_builtin_model(**kwargs)
    assert "XavierPDDirectory" in actors
    for worker in workers:
        assert worker.launch_builtin_model.call_args.kwargs["model_format"] is None


def test_registration_snapshot_includes_hicache_but_excludes_gpu_pd():
    from ..worker import WorkerActor

    worker = MagicMock()
    worker._model_uid_to_model_spec = {"cache-rep0": {}, "pd-rep0": {}}
    worker._model_uid_to_launch_args = {
        "cache-rep0": {"_xavier_cache_config": {"address": "s", "uid": "cache"}},
        "pd-rep0": {"_xavier_cache_config": {"role": "decode"}},
    }
    assert [
        item["replica_model_uid"]
        for item in WorkerActor._get_running_replica_states(worker)
    ] == ["cache-rep0"]


@pytest.mark.asyncio
async def test_hicache_startup_replay_is_preserved(tmp_path):
    import json

    from ..worker import WorkerActor

    class Worker:
        _load_persisted_launch_args = WorkerActor._load_persisted_launch_args
        _persist_launch_args = WorkerActor._persist_launch_args

        def _get_recovery_file_path(self):
            return str(tmp_path / "models.json")

    worker = Worker()
    worker._supervisor_ref = AsyncMock()
    worker._supervisor_ref.describe_model.return_value = {"model_name": "cache"}
    worker._model_uid_to_launch_args = {}
    worker.launch_builtin_model = AsyncMock()
    worker.wait_for_load = AsyncMock()
    config = {"address": "supervisor:1234", "uid": "cache"}
    (tmp_path / "models.json").write_text(
        json.dumps(
            {"cache-rep0": {"model_uid": "cache-rep0", "_xavier_cache_config": config}}
        )
    )
    await WorkerActor._try_recover_models(worker)
    assert (
        worker.launch_builtin_model.call_args.kwargs["_xavier_cache_config"] == config
    )
    worker.wait_for_load.assert_awaited_once_with("cache-rep0")


@pytest.mark.asyncio
async def test_failed_cache_actor_cleanup_emits_warning(launch_runtime, caplog):
    import logging

    supervisor, _, _, destroy = launch_runtime
    kwargs = launch_kwargs()
    kwargs.update(model_engine="SGLang", enable_xavier=True)
    await supervisor.launch_builtin_model(**kwargs)
    destroy.side_effect = RuntimeError("actor cleanup failed")
    with caplog.at_level(logging.WARNING):
        await supervisor.terminate_model("pd")
    assert "Destroy Xavier cache failed for pd" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("engines", [("vLLM", "SGLang"), ("SGLang", "vLLM")])
@pytest.mark.parametrize("default_engine", ["vLLM", "SGLang", "MLX"])
async def test_cross_engine_pd_launches_one_adapter_per_replica(
    launch_runtime, engines, default_engine
):
    supervisor, workers, actors, _ = launch_runtime
    kwargs = launch_kwargs()
    kwargs["model_engine"] = default_engine
    for index, (cfg, engine) in enumerate(zip(kwargs["replica_config"], engines)):
        cfg.model_engine = engine
        cfg.engine_config = {"engine_option": index}
    await supervisor.launch_builtin_model(**kwargs)
    assert set(actors) == {"XavierPDDirectory", "PDModelActor"}
    assert actors["PDModelActor"].constructor_kwargs["model_engine"] == "heterogeneous"
    for index, worker in enumerate(workers):
        launch = worker.launch_builtin_model.call_args.kwargs
        assert launch["model_engine"] == engines[index]
        assert launch["engine_option"] == index
        assert launch["xavier_config"] is None
        assert launch["_xavier_cache_config"]["heterogeneous"] is True
        assert launch["_xavier_cache_config"]["role"] == ("prefill", "decode")[index]
        worker.launch_rank0_model.assert_not_awaited()
        worker.start_transfer_for_vllm.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("engines", [("MLX", "vLLM"), ("SGLang", "MLX")])
async def test_cross_engine_mlx_is_rejected_before_allocating_actors(
    launch_runtime, engines
):
    supervisor, workers, actors, _ = launch_runtime
    kwargs = launch_kwargs()
    for cfg, engine in zip(kwargs["replica_config"], engines):
        cfg.model_engine = engine
    with pytest.raises(ValueError, match="vLLM and SGLang replicas"):
        await supervisor.launch_builtin_model(**kwargs)
    assert not actors
    for worker in workers:
        worker.launch_builtin_model.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key",
    [
        "model_engine",
        "xavier_config",
        "n_gpu",
        "gpu_idx",
        "envs",
        "_xavier_cache_config",
        "_nixl_config",
        "quantization",
    ],
)
async def test_pd_reserved_engine_options_fail_before_allocating_actors(
    launch_runtime, key
):
    supervisor, workers, actors, _ = launch_runtime
    kwargs = launch_kwargs()
    kwargs["replica_config"][0].engine_config = {key: None}
    with pytest.raises(ValueError, match="Reserved replica engine_config keys"):
        await supervisor.launch_builtin_model(**kwargs)
    assert not actors
    for worker in workers:
        worker.launch_builtin_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_per_replica_engine_override_preserves_homogeneous_mlx_pd(launch_runtime):
    supervisor, workers, actors, _ = launch_runtime
    kwargs = launch_kwargs()
    kwargs["model_format"] = "mlx"
    for cfg in kwargs["replica_config"]:
        cfg.model_engine = "MLX"
    await supervisor.launch_builtin_model(**kwargs)
    assert set(actors) == {"XavierBytesCacheActor", "PDModelActor"}
    for worker in workers:
        launch = worker.launch_builtin_model.call_args.kwargs
        assert launch["model_engine"] == "MLX"
        assert "rank" not in launch["_xavier_cache_config"]


@pytest.mark.asyncio
async def test_cross_engine_rejects_native_transport_before_launch(launch_runtime):
    supervisor, workers, actors, _ = launch_runtime
    kwargs = launch_kwargs()
    kwargs["replica_config"][1].model_engine = "SGLang"
    kwargs["transfer_backend_type"] = "nixl"
    with pytest.raises(ValueError, match="Xavier GPU transport"):
        await supervisor.launch_builtin_model(**kwargs)
    assert not actors
    for worker in workers:
        worker.launch_builtin_model.assert_not_awaited()
