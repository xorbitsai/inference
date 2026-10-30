# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# CI: linux

import asyncio
import itertools
import random
from unittest.mock import AsyncMock, MagicMock

import pytest

from xinference.core.launch_strategy import IdleFirstLaunchStrategy
from xinference.core.utils import assign_replica_gpu, build_replica_model_uid


class DummyRef:
    def __init__(self, address: str):
        self.address = address


class DummyWorkerRef:
    def __init__(
        self,
        address: str,
        model_count: int,
        launched: list,
        resolved_download_hub: str = "huggingface",
    ):
        self.address = address
        self._model_count = model_count
        self._launched = launched
        self._resolved_download_hub = resolved_download_hub
        self.resolve_calls: list[tuple] = []
        self.launch_kwargs: list[dict] = []

    async def get_model_count(self) -> int:
        return self._model_count

    async def get_gpu_allocation_status(self):
        return None

    async def launch_builtin_model(self, *args, **kwargs):
        self._launched.append(self.address)
        self.launch_kwargs.append(kwargs)
        if kwargs.get("shard") == 0:
            return "subpool", {"driver": "info"}
        return "subpool"

    async def resolve_download_hub(self, download_hub, model_path):
        self.resolve_calls.append((download_hub, model_path))
        return self._resolved_download_hub

    async def wait_for_load(self, model_uid: str):
        return None


class DummyStatusGuard:
    def __init__(self):
        self.instance_info = {}
        self.replica_counts = {}
        self.updates = []

    async def set_instance_info(self, model_uid: str, instance_info):
        self.instance_info[model_uid] = instance_info
        self.replica_counts[model_uid] = instance_info.replica

    async def update_instance_info(self, model_uid: str, updates: dict):
        self.updates.append((model_uid, updates))
        if "replica" in updates:
            self.replica_counts[model_uid] = updates["replica"]
        return None

    async def update_replica_status(
        self, model_uid: str, replica_id: int, updates: dict
    ):
        return None

    async def remove_replica_status(self, model_uid: str, replica_id: int):
        self.replica_counts[model_uid] -= 1
        return self.replica_counts[model_uid]


class DummySupervisor:
    def __init__(self, workers):
        from xinference.core.supervisor import SupervisorActor

        self._worker_address_to_worker = workers
        self._model_uid_to_replica_info = {}
        self._replica_model_uid_to_worker = {}
        self._replica_model_uid_to_worker_shards = {}
        self._initial_model_launches = {}
        self._async_launch_tasks = set()
        self._status_guard_ref = DummyStatusGuard()
        self._unexpected_down_replicas = {}
        self._worker_status = {}
        self._workers_launching = {}
        self._autostart_store_lock = asyncio.Lock()
        self._launch_history_store = MagicMock()
        self._collective_manager_mapping = {}
        self._block_tracker_mapping = {}
        self._build_replica_info = SupervisorActor._build_replica_info
        self._choose_worker = SupervisorActor._choose_worker.__get__(self)
        self._clear_unexpected_down_replicas = (
            SupervisorActor._clear_unexpected_down_replicas.__get__(self)
        )
        self._launch_builtin_sharded_model = (
            SupervisorActor._launch_builtin_sharded_model.__get__(self)
        )
        self._resolve_download_hub_from_workers = (
            SupervisorActor._resolve_download_hub_from_workers
        )
        self.launch_builtin_model = SupervisorActor.launch_builtin_model.__get__(self)
        self._launch_builtin_model = SupervisorActor._launch_builtin_model.__get__(self)
        self._run_initial_model_launch = (
            SupervisorActor._run_initial_model_launch.__get__(self)
        )
        self._get_worker_host = SupervisorActor._get_worker_host
        self._get_worker_refs_by_ip = SupervisorActor._get_worker_refs_by_ip.__get__(
            self
        )
        self._resolve_worker_addresses = (
            SupervisorActor._resolve_worker_addresses.__get__(self)
        )
        # Fields used by _invalidate_list_models_debounce_cache (called from
        # _launch_builtin_sharded_model after model launch completes).
        self._list_models_result_cache = {}
        self._list_models_cache_version = 0

    def _invalidate_list_models_debounce_cache(self) -> None:
        self._list_models_result_cache = {}
        self._list_models_result_cache_time = 0.0
        self._list_models_cache_version += 1

    def is_local_deployment(self) -> bool:
        return False

    async def terminate_model(self, model_uid: str, suppress_exception: bool = False):
        return None


class ShardLifecycleWorker(DummyWorkerRef):
    def __init__(self, address):
        super().__init__(address, model_count=0, launched=[])
        self.models = set()
        self.error = None
        self.fail_uid = None
        self.terminated = []

    async def launch_builtin_model(self, model_uid, **kwargs):
        if model_uid in self.models:
            raise ValueError(f"{model_uid} is already running")
        if model_uid == self.fail_uid:
            raise self.error
        self.models.add(model_uid)
        return await super().launch_builtin_model(model_uid=model_uid, **kwargs)

    async def terminate_model(self, model_uid):
        self.models.remove(model_uid)
        self.terminated.append(model_uid)


def _shard_lifecycle_supervisor(n_worker):
    from xinference.core.supervisor import SupervisorActor

    workers = [ShardLifecycleWorker(f"w{i}:1000") for i in range(n_worker)]
    supervisor = DummySupervisor({w.address: w for w in workers})
    supervisor._model_replica_locks = {}
    supervisor._model_reload_tasks = {}
    supervisor._model_reload_status = {}
    supervisor._cleanup_distributed_actors = AsyncMock()
    supervisor._clear_replica_model_gpu_memory = MagicMock()
    for name in (
        "terminate_model",
        "_terminate_model",
        "_clear_model_routing",
        "_get_model_replica_lock",
        "_iter_active_replica_model_uids",
    ):
        setattr(supervisor, name, getattr(SupervisorActor, name).__get__(supervisor))
    return supervisor, workers


async def _launch_shard_lifecycle(supervisor, workers, replica):
    return await supervisor._launch_builtin_sharded_model(
        model_uid="sharded",
        model_name="demo",
        model_size_in_billions=None,
        model_format=None,
        quantization=None,
        model_engine=None,
        model_type="LLM",
        n_gpu=None,
        n_worker=len(workers),
        worker_ip=[w.address for w in workers],
        replica=replica,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "n_worker,fail_shard,replica", [(2, 1, 1), (3, 2, 1), (2, 0, 1), (2, 1, 2)]
)
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_sharded_launch_failure_rolls_back(
    n_worker, fail_shard, replica, error_type
):
    supervisor, workers = _shard_lifecycle_supervisor(n_worker)
    failing_worker = workers[fail_shard]
    failing_worker.fail_uid = build_replica_model_uid("sharded", replica - 1)
    failing_worker.error = error_type("injected shard launch failure")

    with pytest.raises(error_type) as exc:
        await _launch_shard_lifecycle(supervisor, workers, replica)
    # Python 3.10 may recreate cancellation exceptions across a Task boundary.
    # Cancellation propagation and rollback, rather than object identity, are
    # the contract; ordinary launch failures still preserve their exception.
    if error_type is not asyncio.CancelledError:
        assert exc.value is failing_worker.error
    for i, worker in enumerate(workers):
        expected = [build_replica_model_uid("sharded", r) for r in range(replica - 1)]
        if i < fail_shard:
            expected.append(failing_worker.fail_uid)
        assert sorted(worker.terminated) == sorted(expected)
        assert not worker.models
    assert not supervisor._replica_model_uid_to_worker
    assert not supervisor._model_uid_to_replica_info

    failing_worker.fail_uid = None
    assert await _launch_shard_lifecycle(supervisor, workers, replica) == "sharded"
    await supervisor.terminate_model("sharded")
    assert all(not worker.models for worker in workers)


@pytest.mark.asyncio
@pytest.mark.parametrize("replica", [1, 2])
async def test_sharded_launch_terminate_relaunch(replica):
    supervisor, workers = _shard_lifecycle_supervisor(2)
    for _ in range(2):
        assert await _launch_shard_lifecycle(supervisor, workers, replica) == "sharded"
        for r in range(replica):
            uid = build_replica_model_uid("sharded", r)
            assert all(uid in worker.models for worker in workers)
            assert supervisor._replica_model_uid_to_worker[uid] == workers
        await supervisor.terminate_model("sharded")
        assert all(not worker.models for worker in workers)
        assert not supervisor._replica_model_uid_to_worker
        assert not supervisor._model_uid_to_replica_info


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["delete", "restart"])
async def test_sharded_launch_pending_shard_lifecycle(action):
    from xinference.core.supervisor import SupervisorActor

    supervisor, workers = _shard_lifecycle_supervisor(2)
    supervisor._replica_model_uid_to_worker_shards = {}
    if action == "delete":
        supervisor._pd_model_mapping = {}
        supervisor._refresh_replica_scheduler = (
            SupervisorActor._refresh_replica_scheduler
        )
        supervisor._status_guard_ref.get_replica_statuses = AsyncMock(return_value=[])
        methods = ("terminate_model_replica", "_terminate_model_replica")
    else:
        supervisor._get_model_uid_and_replica_index = (
            SupervisorActor._get_model_uid_and_replica_index
        )
        methods = (
            "_rebuild_worker_replica_state",
            "_remove_worker_from_replica_mappings",
        )
    for name in methods:
        setattr(supervisor, name, getattr(SupervisorActor, name).__get__(supervisor))

    entered, release = asyncio.Event(), asyncio.Event()
    original_launch = workers[1].launch_builtin_model

    async def delayed_launch(model_uid, **kwargs):
        entered.set()
        await release.wait()
        return await original_launch(model_uid, **kwargs)

    workers[1].launch_builtin_model = delayed_launch
    task = asyncio.create_task(_launch_shard_lifecycle(supervisor, workers, 1))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if action == "delete":
            with pytest.raises(ValueError, match="Model not found.*sharded-rep0"):
                await supervisor.terminate_model_replica("sharded", 0)
        else:
            # A restarted worker has already lost its models; termination is harmless.
            workers[0].models.clear()

            async def terminate_absent_model(model_uid):
                workers[0].terminated.append(model_uid)

            workers[0].terminate_model = terminate_absent_model
            workers[0].wait_for_load = AsyncMock(
                side_effect=RuntimeError("injected load failure")
            )
            supervisor._rebuild_worker_replica_state(workers[0], [])
    finally:
        release.set()
        outcomes = await asyncio.wait_for(
            asyncio.gather(task, return_exceptions=True), 5
        )

    uid = build_replica_model_uid("sharded", 0)
    if action == "delete":
        assert outcomes == ["sharded"]
        assert all(uid in worker.models for worker in workers)
        assert supervisor._replica_model_uid_to_worker[uid] == workers
        await supervisor.terminate_model("sharded")
    else:
        assert isinstance(outcomes[0], RuntimeError)
        assert str(outcomes[0]) == "injected load failure"
        assert all(worker.terminated == [uid] for worker in workers)
    assert all(not worker.models for worker in workers)
    assert not supervisor._replica_model_uid_to_worker
    assert not supervisor._model_uid_to_replica_info


def test_worker_address_resolution_prefers_exact_worker_address():
    worker_1 = DummyRef("worker-1:30001")
    worker_2 = DummyRef("worker-1:30002")
    supervisor = DummySupervisor(
        {
            worker_1.address: worker_1,
            worker_2.address: worker_2,
        }
    )

    assert supervisor._get_worker_refs_by_ip("worker-1:30001") == [worker_1]
    assert set(supervisor._get_worker_refs_by_ip("worker-1")) == {
        worker_1,
        worker_2,
    }


def test_worker_address_resolution_supports_multiple_addresses():
    worker_1 = DummyRef("worker-1:30001")
    worker_2 = DummyRef("worker-2:30001")
    supervisor = DummySupervisor(
        {
            worker_1.address: worker_1,
            worker_2.address: worker_2,
        }
    )

    refs = supervisor._get_worker_refs_by_ip("worker-1:30001, worker-2:30001")

    assert refs == [worker_1, worker_2]


def test_worker_address_resolution_rejects_unknown_address():
    worker = DummyRef("worker-1:30001")
    supervisor = DummySupervisor({worker.address: worker})

    with pytest.raises(
        ValueError, match="Worker ip address worker-1:30002 is not in the cluster"
    ):
        supervisor._get_worker_refs_by_ip("worker-1:30002")


def test_assign_replica_gpu_single_slot_reused():
    # single gpu_idx with multiple replicas should be invalid
    with pytest.raises(ValueError):
        assign_replica_gpu(build_replica_model_uid("foo", 0), replica=6, gpu_idx=[1])


def test_assign_replica_gpu_slicing():
    # multiple gpu_idx are sliced by replica id
    assert assign_replica_gpu(
        build_replica_model_uid("foo", 0), replica=3, gpu_idx=[0, 1, 2]
    ) == [0]
    assert assign_replica_gpu(
        build_replica_model_uid("foo", 1), replica=3, gpu_idx=[0, 1, 2]
    ) == [1]
    assert assign_replica_gpu(
        build_replica_model_uid("foo", 2), replica=3, gpu_idx=[0, 1, 2]
    ) == [2]


def test_idle_first_prefers_empty_gpu(monkeypatch):
    random.seed(42)
    strategy = IdleFirstLaunchStrategy(worker_status={})
    worker = DummyRef("w1:1000")
    worker_candidates = [
        {
            "ref": worker,
            "count": 0,
            "alloc": {
                "total": [0, 1],
                "models": {0: ["m0"]},  # gpu0 has load, gpu1 empty
                "user_specified": {},
            },
        }
    ]
    ref, gpu_idx = strategy.select_worker(worker_candidates)
    assert ref is worker
    assert gpu_idx == [1]  # pick the empty gpu1


def test_idle_first_balances_with_reserve(monkeypatch):
    random.seed(0)
    strategy = IdleFirstLaunchStrategy(worker_status={})
    worker = DummyRef("w1:1000")
    candidates = [
        {
            "ref": worker,
            "count": 0,
            "alloc": {
                "total": [0, 1],
                "models": {},
                "user_specified": {},
            },
        }
    ]
    # first pick should choose one gpu, reserve it, second pick should choose the other
    _, gpu_idx1 = strategy.select_worker(candidates)
    _, gpu_idx2 = strategy.select_worker(candidates)
    assert set(gpu_idx1 + gpu_idx2) == {0, 1}
    assert gpu_idx1 != gpu_idx2


def test_idle_first_fallback_to_count_when_no_alloc():
    random.seed(0)
    strategy = IdleFirstLaunchStrategy(worker_status={})
    w1 = DummyRef("w1:1000")
    w2 = DummyRef("w2:1000")
    candidates = [
        {"ref": w1, "count": 1, "alloc": None},
        {"ref": w2, "count": 0, "alloc": None},
    ]
    ref, gpu_idx = strategy.select_worker(candidates)
    assert ref is w2  # least count
    assert gpu_idx is None  # let worker decide when no alloc info


def test_multi_worker_multi_gpu_even_distribution():
    random.seed(123)
    strategy = IdleFirstLaunchStrategy(worker_status={})
    w1 = DummyRef("w1:1000")
    w2 = DummyRef("w2:1000")
    candidates = [
        {
            "ref": w1,
            "count": 0,
            "alloc": {
                "total": [0, 1],
                "models": {},
                "user_specified": {},
            },
        },
        {
            "ref": w2,
            "count": 0,
            "alloc": {
                "total": [0, 1],
                "models": {},
                "user_specified": {},
            },
        },
    ]
    seen = []
    for _ in range(8):
        ref, gpu_idx = strategy.select_worker(candidates)
        seen.append((ref.address, gpu_idx[0] if gpu_idx else None))
    # Expect each worker 4 replicas, each gpu 2 replicas
    from collections import Counter

    worker_count = Counter([w for w, _ in seen])
    gpu_count = Counter([(w, g) for w, g in seen])
    assert worker_count[w1.address] == 4
    assert worker_count[w2.address] == 4
    assert gpu_count[(w1.address, 0)] == 2
    assert gpu_count[(w1.address, 1)] == 2
    assert gpu_count[(w2.address, 0)] == 2
    assert gpu_count[(w2.address, 1)] == 2


def test_cpu_fallback_no_gpu_alloc():
    random.seed(0)
    strategy = IdleFirstLaunchStrategy(worker_status={})
    w1 = DummyRef("w1:1000")
    w2 = DummyRef("w2:1000")
    # Simulate no GPU allocation info (e.g., CPU-only workers)
    candidates = [
        {"ref": w1, "count": 2, "alloc": None},
        {"ref": w2, "count": 1, "alloc": None},
    ]
    ref, gpu_idx = strategy.select_worker(candidates)
    assert ref is w2  # choose least-loaded worker
    assert gpu_idx is None  # let worker decide (CPU path)


def test_idle_first_multi_gpu_single_worker():
    random.seed(0)
    strategy = IdleFirstLaunchStrategy(worker_status={})
    worker = DummyRef("w1:1000")
    candidates = [
        {
            "ref": worker,
            "count": 0,
            "alloc": {
                "total": [0, 1],
                "models": {},
                "user_specified": {},
            },
        }
    ]
    ref, gpu_idx = strategy.select_worker(candidates, n_gpu=2)
    assert ref is worker
    assert set(gpu_idx) == {0, 1}


def test_idle_first_multi_gpu_two_workers():
    random.seed(0)
    strategy = IdleFirstLaunchStrategy(worker_status={})
    w1 = DummyRef("w1:1000")
    w2 = DummyRef("w2:1000")
    candidates = [
        {
            "ref": w1,
            "count": 0,
            "alloc": {
                "total": [0, 1],
                "models": {0: ["m0"], 1: ["m1"]},
                "user_specified": {},
            },
        },
        {
            "ref": w2,
            "count": 0,
            "alloc": {
                "total": [0, 1],
                "models": {},
                "user_specified": {},
            },
        },
    ]
    ref, gpu_idx = strategy.select_worker(candidates, n_gpu=2)
    assert ref is w2


@pytest.mark.asyncio
async def test_distributed_launch_avoids_same_worker_for_shards():
    launched = []
    worker1 = DummyWorkerRef("w1:1000", model_count=1, launched=launched)
    worker2 = DummyWorkerRef("w2:1000", model_count=0, launched=launched)
    supervisor = DummySupervisor({"w1:1000": worker1, "w2:1000": worker2})

    # One model already runs on worker1 (n_gpu=1). Launch a new model with n_worker=2.
    await supervisor._launch_builtin_sharded_model(
        model_uid="demo-model",
        model_name="demo",
        model_size_in_billions=None,
        model_format=None,
        quantization=None,
        model_engine=None,
        model_type="LLM",
        n_gpu=1,
        n_worker=2,
        worker_ip=["w1:1000", "w2:1000"],
        wait_ready=True,
    )

    assert set(launched) == {"w1:1000", "w2:1000"}
    assert len(launched) == 2


@pytest.mark.asyncio
async def test_distributed_launch_normalizes_string_n_worker():
    launched = []
    worker1 = DummyWorkerRef("w1:1000", model_count=1, launched=launched)
    worker2 = DummyWorkerRef("w2:1000", model_count=0, launched=launched)
    supervisor = DummySupervisor({"w1:1000": worker1, "w2:1000": worker2})

    await supervisor.launch_builtin_model(
        model_uid="demo-model",
        model_name="demo",
        model_size_in_billions=None,
        model_format=None,
        quantization=None,
        model_engine=None,
        model_type="LLM",
        n_gpu=1,
        n_worker="2",
        worker_ip=["w1:1000", "w2:1000"],
        wait_ready=True,
    )

    assert set(launched) == {"w1:1000", "w2:1000"}
    assert {
        call["n_worker"]
        for worker in (worker1, worker2)
        for call in worker.launch_kwargs
    } == {2}


@pytest.mark.asyncio
async def test_download_hub_resolved_from_selected_workers_before_fanout(monkeypatch):
    monkeypatch.setenv("XINFERENCE_MODEL_SRC", "huggingface")
    launched = []
    worker1 = DummyWorkerRef("w1:1000", 1, launched, resolved_download_hub="modelscope")
    worker2 = DummyWorkerRef("w2:1000", 0, launched, resolved_download_hub="modelscope")
    supervisor = DummySupervisor({"w1:1000": worker1, "w2:1000": worker2})

    await supervisor.launch_builtin_model(
        model_uid="demo-model",
        model_name="demo",
        model_size_in_billions=None,
        model_format=None,
        quantization=None,
        model_engine=None,
        model_type="LLM",
        n_gpu=1,
        n_worker=2,
        worker_ip=["w1:1000", "w2:1000"],
        wait_ready=True,
    )

    assert worker1.resolve_calls == [(None, None)]
    assert worker2.resolve_calls == [(None, None)]
    assert [
        call["download_hub"]
        for worker in (worker1, worker2)
        for call in worker.launch_kwargs
    ] == ["modelscope", "modelscope"]


@pytest.mark.asyncio
async def test_distributed_launch_rejects_worker_hub_disagreement_before_fanout():
    launched = []
    worker1 = DummyWorkerRef(
        "w1:1000", 1, launched, resolved_download_hub="huggingface"
    )
    worker2 = DummyWorkerRef("w2:1000", 0, launched, resolved_download_hub="modelscope")
    supervisor = DummySupervisor({"w1:1000": worker1, "w2:1000": worker2})

    with pytest.raises(
        ValueError, match="Selected workers resolved different download hubs"
    ):
        await supervisor.launch_builtin_model(
            model_uid="demo-model",
            model_name="demo",
            model_size_in_billions=None,
            model_format=None,
            quantization=None,
            model_engine=None,
            model_type="LLM",
            n_gpu=1,
            n_worker=2,
            worker_ip=["w1:1000", "w2:1000"],
            download_hub="auto",
            wait_ready=True,
        )

    assert launched == []


@pytest.mark.asyncio
async def test_replica_launch_rejects_worker_hub_disagreement_before_fanout():
    launched = []
    worker1 = DummyWorkerRef(
        "w1:1000", 0, launched, resolved_download_hub="huggingface"
    )
    worker2 = DummyWorkerRef("w2:1000", 0, launched, resolved_download_hub="modelscope")
    supervisor = DummySupervisor({"w1:1000": worker1, "w2:1000": worker2})

    with pytest.raises(
        ValueError, match="Selected workers resolved different download hubs"
    ):
        await supervisor.launch_builtin_model(
            model_uid="demo-model",
            model_name="demo",
            model_size_in_billions=None,
            model_format=None,
            quantization=None,
            model_engine=None,
            model_type="LLM",
            replica=2,
            n_gpu=None,
            wait_ready=True,
        )

    assert launched == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "worker_ip",
    ["w1,w2", ["w1,w2"]],
)
async def test_distributed_launch_resolves_comma_separated_worker_ips(worker_ip):
    launched = []
    worker1 = DummyWorkerRef("w1:1000", model_count=1, launched=launched)
    worker2 = DummyWorkerRef("w2:1000", model_count=0, launched=launched)
    supervisor = DummySupervisor({"w1:1000": worker1, "w2:1000": worker2})

    await supervisor._launch_builtin_sharded_model(
        model_uid="demo-model",
        model_name="demo",
        model_size_in_billions=None,
        model_format=None,
        quantization=None,
        model_engine=None,
        model_type="LLM",
        n_gpu=1,
        n_worker=2,
        worker_ip=worker_ip,
        wait_ready=True,
    )

    assert set(launched) == {"w1:1000", "w2:1000"}
    assert len(launched) == 2


@pytest.mark.asyncio
async def test_distributed_launch_deduplicates_worker_ips():
    launched = []
    worker = DummyWorkerRef("w1:1000", model_count=0, launched=launched)
    supervisor = DummySupervisor({"w1:1000": worker})

    with pytest.raises(
        ValueError,
        match="n_worker cannot be larger than the number of available workers",
    ):
        await supervisor._launch_builtin_sharded_model(
            model_uid="demo-model",
            model_name="demo",
            model_size_in_billions=None,
            model_format=None,
            quantization=None,
            model_engine=None,
            model_type="LLM",
            n_gpu=1,
            n_worker=2,
            worker_ip=["w1,w1:1000"],
            wait_ready=True,
        )

    assert launched == []


@pytest.mark.asyncio
async def test_terminate_model_replica_updates_active_replica_set():
    from xinference.core.supervisor import ReplicaInfo, SupervisorActor

    class DummyReplicaWorkerRef:
        def __init__(self, address: str):
            self.address = address
            self.terminated: list[str] = []

        async def terminate_model(self, model_uid: str):
            self.terminated.append(model_uid)

    class DummySupervisor:
        terminate_model_replica = SupervisorActor.terminate_model_replica
        _terminate_model_replica = SupervisorActor._terminate_model_replica
        _get_model_replica_lock = SupervisorActor._get_model_replica_lock
        _refresh_replica_scheduler = staticmethod(
            SupervisorActor._refresh_replica_scheduler
        )
        _clear_unexpected_down_replicas = (
            SupervisorActor._clear_unexpected_down_replicas
        )
        _clear_replica_model_gpu_memory = (
            SupervisorActor._clear_replica_model_gpu_memory
        )

        def __init__(self):
            self._model_replica_locks = {}
            self._model_uid_to_replica_info = {
                "demo-model": ReplicaInfo(
                    replica=3,
                    scheduler=itertools.cycle([0, 1, 2]),
                    active_replica_ids=[0, 1, 2],
                )
            }
            self._replica_model_uid_to_worker = {
                build_replica_model_uid(
                    "demo-model", replica_id
                ): DummyReplicaWorkerRef(f"w{replica_id}:1000")
                for replica_id in range(3)
            }
            self._replica_model_uid_to_worker_shards = {}
            self._status_guard_ref = DummyStatusGuard()
            self._status_guard_ref.replica_counts["demo-model"] = 3
            self._pd_model_mapping = {}
            self._collective_manager_mapping = {}
            self._block_tracker_mapping = {}
            self._unexpected_down_replicas = {}
            self._worker_model_gpu_memory = {
                "worker:1000": {
                    build_replica_model_uid("demo-model", replica_id): {
                        replica_id: (replica_id + 1) * 100
                    }
                    for replica_id in range(3)
                }
            }
            self._worker_model_gpu_memory_update_time = {"worker:1000": 0.0}

        def _invalidate_list_models_debounce_cache(self):
            pass

    supervisor = DummySupervisor()
    remaining = await supervisor.terminate_model_replica("demo-model", 1)

    assert remaining == 2
    assert supervisor._model_uid_to_replica_info["demo-model"].replica == 2
    assert supervisor._model_uid_to_replica_info["demo-model"].active_replica_ids == [
        0,
        2,
    ]
    assert (
        build_replica_model_uid("demo-model", 1)
        not in supervisor._replica_model_uid_to_worker
    )
    assert next(supervisor._model_uid_to_replica_info["demo-model"].scheduler) == 0
    assert next(supervisor._model_uid_to_replica_info["demo-model"].scheduler) == 2
    cached_gpu_memory = supervisor._worker_model_gpu_memory["worker:1000"]
    assert build_replica_model_uid("demo-model", 0) in cached_gpu_memory
    assert build_replica_model_uid("demo-model", 1) not in cached_gpu_memory
    assert build_replica_model_uid("demo-model", 2) in cached_gpu_memory
    assert "worker:1000" in supervisor._worker_model_gpu_memory_update_time
    assert supervisor._status_guard_ref.updates[-1] == (
        "demo-model",
        {"replica": 2, "status": "READY"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["vLLM", "SGLang"])
@pytest.mark.parametrize("replica,n_worker", [(1, 1), (2, 1), (1, 2)])
@pytest.mark.parametrize("local", [False, True])
async def test_auto_weight_cache_respects_launch_topology(
    engine, replica, n_worker, local
):
    workers = [
        DummyWorkerRef(f"w{i}:1000", model_count=0, launched=[]) for i in range(2)
    ]
    supervisor = DummySupervisor({w.address: w for w in workers})
    supervisor.is_local_deployment = lambda: local
    await supervisor.launch_builtin_model(
        model_uid="demo-model",
        model_name="demo",
        model_size_in_billions=None,
        model_format="pytorch",
        quantization=None,
        model_engine=engine,
        model_type="LLM",
        n_gpu=1,
        replica=replica,
        n_worker=n_worker,
        wait_ready=True,
        worker_ip=[w.address for w in workers],
    )
    launches = [call for worker in workers for call in worker.launch_kwargs]
    assert launches
    expected = None if replica == 1 and (local or n_worker == 1) else False
    assert all(call.get("enable_weight_cache") is expected for call in launches)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", ["embedding", "rerank"])
@pytest.mark.parametrize("cache_kwargs", [{}, {"enable_weight_cache": None}])
async def test_auto_cache_option_is_not_forwarded_to_non_llm(model_type, cache_kwargs):
    worker = DummyWorkerRef("worker:9978", model_count=0, launched=[])
    supervisor = DummySupervisor({worker.address: worker})
    await supervisor.launch_builtin_model(
        model_uid="demo-model",
        model_name="demo",
        model_type=model_type,
        model_engine="vLLM",
        model_size_in_billions=None,
        model_format="pytorch",
        quantization=None,
        n_gpu=1,
        **cache_kwargs,
    )
    assert len(worker.launch_kwargs) == 1
    assert "enable_weight_cache" not in worker.launch_kwargs[0]
