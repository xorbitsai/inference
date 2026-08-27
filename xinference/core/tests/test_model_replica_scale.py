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

import asyncio
import itertools
from types import SimpleNamespace

import pytest

from ..replica_config import DeviceConfig, ReplicaConfig
from ..supervisor import ReplicaInfo, SupervisorActor
from ..utils import build_replica_model_uid


class _FakeStatusGuard:
    def __init__(self, model_uid: str, replica_uid: str = "demo-0"):
        self._statuses = {
            model_uid: [
                SimpleNamespace(
                    replica_id=0,
                    replica_model_uid=build_replica_model_uid(model_uid, 0),
                    worker_address="worker-0:9978",
                    status="READY",
                    created_ts=1,
                    error_message=None,
                    replica_uid=replica_uid,
                    gpu_idx=None,
                )
            ]
        }
        self.instance_updates: list = []

    async def get_replica_statuses(self, model_uid: str):
        return list(self._statuses.get(model_uid, []))

    async def update_replica_status(self, model_uid: str, replica_id: int, updates):
        statuses = self._statuses.setdefault(model_uid, [])
        for status in statuses:
            if status.replica_id == replica_id:
                for key, value in updates.items():
                    setattr(status, key, value)
                return
        statuses.append(
            SimpleNamespace(
                replica_id=replica_id,
                replica_model_uid=updates.get("replica_model_uid", ""),
                worker_address=updates.get("worker_address", ""),
                status=updates.get("status", "CREATING"),
                created_ts=updates.get("created_ts", 0),
                error_message=updates.get("error_message"),
                replica_uid=updates.get("replica_uid"),
                gpu_idx=updates.get("gpu_idx"),
            )
        )

    async def update_instance_info(self, model_uid: str, updates):
        self.instance_updates.append((model_uid, updates))

    async def remove_replica_status(self, model_uid: str, replica_id: int):
        statuses = self._statuses.get(model_uid, [])
        self._statuses[model_uid] = [
            status for status in statuses if status.replica_id != replica_id
        ]
        return len(self._statuses[model_uid])


class _FakeScaleWorker:
    def __init__(
        self,
        address: str,
        launch_args,
        *,
        model_count: int = 0,
        total=(0,),
        models=None,
        allow_share: bool = True,
        launch_delay: float = 0,
    ):
        self.address = address
        self.launch_args = dict(launch_args)
        self.model_count = model_count
        self.total = list(total)
        self.models = models or {}
        self.allow_share = allow_share
        self.launch_delay = launch_delay
        self.launch_started = None
        self.continue_launch = None
        self.launches: list = []
        self.terminations: list[str] = []

    async def get_launch_args(self, model_uid: str):
        return dict(self.launch_args)

    async def get_model_count(self):
        return self.model_count

    async def get_gpu_allocation_status(self):
        return {
            "total": list(self.total),
            "models": self.models,
            "user_specified": {},
            "allow_multi_replica_per_gpu": self.allow_share,
        }

    async def launch_builtin_model(self, **kwargs):
        self.launches.append(kwargs)
        if self.launch_started is not None:
            self.launch_started.set()
        if self.continue_launch is not None:
            await self.continue_launch.wait()
        elif self.launch_delay:
            await asyncio.sleep(self.launch_delay)
        return "subpool"

    async def wait_for_load(self, model_uid: str):
        return None

    async def terminate_model(self, model_uid: str, **kwargs):
        self.terminations.append(model_uid)


def _make_supervisor(workers, launch_args, replica_uid="demo-0"):
    model_uid = "demo"
    supervisor = SupervisorActor()
    supervisor.address = "supervisor:9999"
    supervisor._worker_address_to_worker = {
        worker.address: worker for worker in workers
    }
    supervisor._worker_status = {}
    supervisor._model_uid_to_replica_info = {
        model_uid: ReplicaInfo(
            replica=1,
            scheduler=itertools.cycle([0]),
            active_replica_ids=[0],
        )
    }
    supervisor._replica_model_uid_to_worker = {
        build_replica_model_uid(model_uid, 0): workers[0]
    }
    workers[0].launch_args = dict(launch_args)
    supervisor._status_guard_ref = _FakeStatusGuard(model_uid, replica_uid)
    supervisor._collective_manager_mapping = {}
    supervisor._block_tracker_mapping = {}
    return supervisor


@pytest.mark.asyncio
async def test_concurrent_scale_up_requests_get_distinct_replica_ids():
    launch_args = {"n_gpu": "auto", "gpu_idx": None, "n_worker": 1}
    worker = _FakeScaleWorker(
        "worker-0:9978",
        launch_args,
        total=(0, 1),
        launch_delay=0.01,
    )
    supervisor = _make_supervisor([worker], launch_args)

    results = await asyncio.gather(
        supervisor.add_model_replica("demo"),
        supervisor.add_model_replica("demo"),
    )

    assert [result["replica_id"] for result in results] == [1, 2]
    assert supervisor._model_uid_to_replica_info["demo"].active_replica_ids == [
        0,
        1,
        2,
    ]
    assert [launch["model_uid"] for launch in worker.launches] == [
        build_replica_model_uid("demo", 1),
        build_replica_model_uid("demo", 2),
    ]


@pytest.mark.asyncio
async def test_scale_up_multiple_replicas_overrides_engine_and_device():
    launch_args = {
        "model_engine": "transformers",
        "n_gpu": "auto",
        "gpu_idx": None,
        "n_worker": 1,
    }
    worker = _FakeScaleWorker("worker-0:9978", launch_args, total=(0, 1))
    supervisor = _make_supervisor([worker], launch_args)

    results = await supervisor.add_model_replicas(
        "demo", replica=2, model_engine="vllm", n_gpu=0
    )

    assert [result["replica_id"] for result in results] == [1, 2]
    assert [launch["model_engine"] for launch in worker.launches] == ["vllm", "vllm"]
    assert [launch["n_gpu"] for launch in worker.launches] == [0, 0]


@pytest.mark.asyncio
async def test_scale_up_multiple_replicas_rolls_back_partial_failure():
    launch_args = {"n_gpu": 0, "gpu_idx": None, "n_worker": 1}
    worker = _FakeScaleWorker("worker-0:9978", launch_args)
    supervisor = _make_supervisor([worker], launch_args)
    original_launch = worker.launch_builtin_model
    launch_count = 0

    async def fail_second_launch(**kwargs):
        nonlocal launch_count
        launch_count += 1
        if launch_count == 2:
            raise RuntimeError("boom")
        await original_launch(**kwargs)

    worker.launch_builtin_model = fail_second_launch

    with pytest.raises(RuntimeError, match="boom"):
        await supervisor.add_model_replicas("demo", replica=2)

    assert supervisor._model_uid_to_replica_info["demo"].active_replica_ids == [0]
    assert build_replica_model_uid("demo", 1) in worker.terminations


@pytest.mark.asyncio
async def test_whole_model_termination_waits_for_scale_up_and_leaves_no_orphan():
    launch_args = {"n_gpu": "auto", "gpu_idx": None, "n_worker": 1}
    worker = _FakeScaleWorker("worker-0:9978", launch_args, total=(0, 1))
    worker.launch_started = asyncio.Event()
    worker.continue_launch = asyncio.Event()
    supervisor = _make_supervisor([worker], launch_args)

    scale_task = asyncio.create_task(supervisor.add_model_replica("demo"))
    await asyncio.wait_for(worker.launch_started.wait(), timeout=1)

    terminate_task = asyncio.create_task(supervisor.terminate_model("demo"))
    await asyncio.sleep(0)
    assert not terminate_task.done()
    assert worker.terminations == []

    worker.continue_launch.set()
    scale_result, _ = await asyncio.gather(scale_task, terminate_task)

    assert scale_result["replica_id"] == 1
    assert set(worker.terminations) == {
        build_replica_model_uid("demo", 0),
        build_replica_model_uid("demo", 1),
    }
    assert "demo" not in supervisor._model_uid_to_replica_info
    assert supervisor._replica_model_uid_to_worker == {}


@pytest.mark.asyncio
async def test_scale_up_default_replica_uid_uses_new_replica_id():
    launch_args = {"n_gpu": "auto", "gpu_idx": None, "n_worker": 1}
    worker = _FakeScaleWorker("worker-0:9978", launch_args)
    supervisor = _make_supervisor([worker], launch_args)
    config = ReplicaConfig(
        devices=[DeviceConfig(worker_ip=worker.address, n_gpu="auto")]
    )

    await supervisor.add_model_replica("demo", config)

    statuses = await supervisor._status_guard_ref.get_replica_statuses("demo")
    assert statuses[-1].replica_uid == "demo-1"


@pytest.mark.asyncio
async def test_scale_up_rejects_duplicate_replica_uid():
    launch_args = {"n_gpu": "auto", "gpu_idx": None, "n_worker": 1}
    worker = _FakeScaleWorker("worker-0:9978", launch_args)
    supervisor = _make_supervisor([worker], launch_args, replica_uid="secondary")
    config = ReplicaConfig(
        replica_uid="secondary",
        devices=[DeviceConfig(worker_ip=worker.address, n_gpu="auto")],
    )

    with pytest.raises(ValueError, match="Replica uid already exists"):
        await supervisor.add_model_replica("demo", config)

    assert worker.launches == []


@pytest.mark.asyncio
async def test_scale_up_auto_selection_skips_worker_without_gpu_capacity():
    launch_args = {"n_gpu": 1, "gpu_idx": None, "n_worker": 1}
    full_worker = _FakeScaleWorker(
        "worker-0:9978",
        launch_args,
        model_count=0,
        total=(0,),
        models={0: ["existing"]},
        allow_share=False,
    )
    available_worker = _FakeScaleWorker(
        "worker-1:9978",
        launch_args,
        model_count=5,
        total=(0,),
        allow_share=False,
    )
    supervisor = _make_supervisor([full_worker, available_worker], launch_args)

    result = await supervisor.add_model_replica("demo")

    assert result["worker_address"] == available_worker.address
    assert full_worker.launches == []
    assert available_worker.launches[0]["gpu_idx"] == [0]


@pytest.mark.asyncio
async def test_scale_up_preserves_gpu_count_from_legacy_explicit_gpu_idx():
    launch_args = {"n_gpu": "auto", "gpu_idx": [0, 1], "n_worker": 1}
    full_worker = _FakeScaleWorker(
        "worker-0:9978",
        launch_args,
        model_count=0,
        total=(0, 1),
        models={0: ["demo"], 1: ["demo"]},
        allow_share=False,
    )
    available_worker = _FakeScaleWorker(
        "worker-1:9978",
        launch_args,
        model_count=1,
        total=(2, 3),
        allow_share=False,
    )
    supervisor = _make_supervisor([full_worker, available_worker], launch_args)

    await supervisor.add_model_replica("demo")

    assert full_worker.launches == []
    assert available_worker.launches[0]["n_gpu"] == 2
    assert available_worker.launches[0]["gpu_idx"] == [2, 3]
