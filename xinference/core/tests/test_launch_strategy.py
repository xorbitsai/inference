# Copyright 2022-2026 XProbe Inc.
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

import itertools
import random

import pytest

from xinference.core.launch_strategy import IdleFirstLaunchStrategy
from xinference.core.utils import assign_replica_gpu


class DummyRef:
    def __init__(self, address: str):
        self.address = address


class DummyWorkerRef:
    def __init__(self, address: str, model_count: int, launched: list):
        self.address = address
        self._model_count = model_count
        self._launched = launched

    async def get_model_count(self) -> int:
        return self._model_count

    async def launch_builtin_model(self, *args, **kwargs):
        self._launched.append(self.address)
        if kwargs.get("shard") == 0:
            return "subpool", {"driver": "info"}
        return "subpool"

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

    async def remove_replica_status(self, model_uid: str, replica_id: int):
        self.replica_counts[model_uid] -= 1
        return self.replica_counts[model_uid]


def test_assign_replica_gpu_single_slot_reused():
    # single gpu_idx with multiple replicas should be invalid
    with pytest.raises(ValueError):
        assign_replica_gpu("foo-0", replica=6, gpu_idx=[1])


def test_assign_replica_gpu_slicing():
    # multiple gpu_idx are sliced by replica id
    assert assign_replica_gpu("foo-0", replica=3, gpu_idx=[0, 1, 2]) == [0]
    assert assign_replica_gpu("foo-1", replica=3, gpu_idx=[0, 1, 2]) == [1]
    assert assign_replica_gpu("foo-2", replica=3, gpu_idx=[0, 1, 2]) == [2]


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
    from xinference.core.supervisor import SupervisorActor

    class DummySupervisor:
        _build_replica_info = staticmethod(SupervisorActor._build_replica_info)
        _choose_worker = SupervisorActor._choose_worker
        _launch_builtin_sharded_model = SupervisorActor._launch_builtin_sharded_model

        def __init__(self, workers):
            self._worker_address_to_worker = workers
            self._model_uid_to_replica_info = {}
            self._replica_model_uid_to_worker = {}
            self._status_guard_ref = DummyStatusGuard()

        def _gen_model_uid(self, model_name: str) -> str:
            return f"{model_name}-uid"

        async def terminate_model(
            self, model_uid: str, suppress_exception: bool = False
        ):
            return None

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
        _refresh_replica_scheduler = staticmethod(
            SupervisorActor._refresh_replica_scheduler
        )

        def __init__(self):
            self._model_uid_to_replica_info = {
                "demo-model": ReplicaInfo(
                    replica=3,
                    scheduler=itertools.cycle([0, 1, 2]),
                    active_replica_ids=[0, 1, 2],
                )
            }
            self._replica_model_uid_to_worker = {
                "demo-model-0": DummyReplicaWorkerRef("w0:1000"),
                "demo-model-1": DummyReplicaWorkerRef("w1:1000"),
                "demo-model-2": DummyReplicaWorkerRef("w2:1000"),
            }
            self._status_guard_ref = DummyStatusGuard()
            self._status_guard_ref.replica_counts["demo-model"] = 3
            self._collective_manager_mapping = {}
            self._block_tracker_mapping = {}

    supervisor = DummySupervisor()
    remaining = await supervisor.terminate_model_replica("demo-model", 1)

    assert remaining == 2
    assert supervisor._model_uid_to_replica_info["demo-model"].replica == 2
    assert supervisor._model_uid_to_replica_info["demo-model"].active_replica_ids == [
        0,
        2,
    ]
    assert "demo-model-1" not in supervisor._replica_model_uid_to_worker
    assert next(supervisor._model_uid_to_replica_info["demo-model"].scheduler) == 0
    assert next(supervisor._model_uid_to_replica_info["demo-model"].scheduler) == 2
    assert supervisor._status_guard_ref.updates[-1] == (
        "demo-model",
        {"replica": 2, "status": "READY"},
    )
