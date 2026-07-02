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
from typing import Any, List, Optional, Tuple, Union

import pytest
import pytest_asyncio
import xoscar as xo
from xoscar import MainActorPoolType, create_actor_pool, get_pool_config

from ...model.core import VirtualEnvSettings
from ..status_guard import InstanceInfo, LaunchStatus, ReplicaStatus
from ..supervisor import ReplicaInfo, SupervisorActor
from ..utils import merge_virtual_env_packages
from ..worker import WorkerActor


class MockWorkerActor(WorkerActor):
    def __init__(
        self,
        supervisor_address: str,
        main_pool: MainActorPoolType,
        cuda_devices: List[int],
        supervisor_endpoint: Optional[str] = None,
    ):
        super().__init__(
            supervisor_address=supervisor_address,
            supervisor_endpoint=supervisor_endpoint,
            main_pool=main_pool,
            gpu_devices=cuda_devices,
        )

    async def __post_create__(self):
        pass

    async def __pre_destroy__(self):
        pass

    def get_gpu_to_model_uid(self):
        return self._gpu_to_model_uids

    def get_user_specified_gpu_to_model_uids(self):
        return self._user_specified_gpu_to_model_uids

    def set_allow_multi_replica_per_gpu(self, allow: bool):
        self._allow_multi_replica_per_gpu = allow

    def get_launch_semaphore_value(self):
        return self._launch_semaphore._value

    def get_launch_active_count(self):
        return self._launch_active

    def get_launch_waiting_count(self):
        return self._launch_waiting

    async def is_model_vllm_backend(self, model_uid):
        if model_uid.startswith("embedding") or model_uid.startswith("rerank"):
            return False
        if model_uid.startswith("normal_"):
            return False
        if model_uid.startswith("vllm_"):
            return True
        for _dev, model_uids in self._gpu_to_model_uids.items():
            if model_uid in model_uids:
                return True
        return False

    async def launch_builtin_model(
        self,
        model_uid: str,
        model_name: str,
        model_size_in_billions: Optional[int],
        model_format: Optional[str],
        quantization: Optional[str],
        model_engine: Optional[str] = None,
        model_type: str = "LLM",
        n_gpu: Optional[Union[int, str]] = None,
        n_worker: Optional[int] = 1,
        shard: Optional[int] = 0,
        gpu_idx: Optional[Union[int, List[int]]] = None,
        **kwargs,
    ):
        subpool_address, devices = await self._create_subpool(
            model_uid,
            model_type,
            n_gpu=n_gpu,
            gpu_idx=gpu_idx,  # type: ignore
        )
        self._model_uid_to_model[model_uid] = DummyActorRef(subpool_address)
        self._model_uid_to_model_spec[model_uid] = {
            "model_uid": model_uid,
            "model_name": model_name,
            "model_type": model_type,
            "model_engine": model_engine,
        }
        self._model_uid_to_launch_args[model_uid] = {
            "model_uid": model_uid,
            "model_name": model_name,
            "model_size_in_billions": model_size_in_billions,
            "model_format": model_format,
            "quantization": quantization,
            "model_engine": model_engine,
            "model_type": model_type,
            "n_gpu": n_gpu,
            "n_worker": n_worker,
            "shard": shard,
            **kwargs,
        }
        self._model_uid_to_addr[model_uid] = subpool_address

    async def terminate_model(self, model_uid: str, is_model_die: bool = False):
        self.release_devices(model_uid)

        sub_pool_addr = self._model_uid_to_addr[model_uid]
        await self._main_pool.remove_sub_pool(sub_pool_addr)
        self._model_uid_to_model.pop(model_uid, None)
        self._model_uid_to_model_spec.pop(model_uid, None)
        self._model_uid_to_launch_args.pop(model_uid, None)
        del self._model_uid_to_addr[model_uid]

    # --- test helpers for report_status GPU attribution ---
    def set_supervisor_ref_for_test(self, ref):
        self._supervisor_ref = ref
        self._registered = True

    def set_gpu_attribution_tables_for_test(self, pid, subpool, total_devices):
        self._model_uid_to_pid = dict(pid)
        self._model_uid_to_subpool_pids = {k: set(v) for k, v in subpool.items()}
        self._total_gpu_devices = list(total_devices)


@pytest_asyncio.fixture
async def setup_pool():
    pool = await create_actor_pool(
        "test://127.0.0.1:" + str(xo.utils.get_next_port()),
        n_process=0,
    )
    async with pool:
        yield pool


@pytest.mark.asyncio
async def test_allocate_cuda_devices(setup_pool):
    pool = setup_pool
    addr = pool.external_address

    worker: xo.ActorRefType["MockWorkerActor"] = await xo.create_actor(  # type: ignore
        MockWorkerActor,
        address=addr,
        uid=WorkerActor.default_uid(),
        supervisor_address="test",
        main_pool=pool,
        cuda_devices=[i for i in range(8)],
    )

    devices = await worker.allocate_devices(model_uid="mock_model_1", n_gpu=1)
    assert devices == [0]

    devices = await worker.allocate_devices(model_uid="mock_model_2", n_gpu=4)
    assert devices == [1, 2, 3, 4]

    devices = await worker.allocate_devices(model_uid="mock_model_3", n_gpu=3)
    assert devices == [5, 6, 7]

    devices = await worker.allocate_devices(model_uid="mock_model_4", n_gpu=5)
    assert devices == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_terminate_model_flag(setup_pool):
    pool = setup_pool
    addr = pool.external_address

    worker: xo.ActorRefType["MockWorkerActor"] = await xo.create_actor(  # type: ignore
        MockWorkerActor,
        address=addr,
        uid=WorkerActor.default_uid(),
        supervisor_address="test",
        main_pool=pool,
        cuda_devices=[i for i in range(8)],
    )

    await worker.launch_builtin_model(
        "model_model_1", "mock_model_name", None, None, None, n_gpu=1
    )

    await worker.launch_builtin_model(
        "model_model_2", "mock_model_name", None, None, None, n_gpu=4
    )

    devices = await worker.allocate_devices(model_uid="model_model_3", n_gpu=3)
    assert devices == [5, 6, 7]
    await worker.release_devices(model_uid="model_model_3")

    await worker.launch_builtin_model(
        "model_model_3", "mock_model_name", None, None, None, n_gpu=3
    )

    with pytest.raises(KeyError):
        await worker.terminate_model("model_model_4")

    pool_config = (await get_pool_config(addr)).as_dict()
    assert len(pool_config["pools"]) == 4  # A main pool and 3 sub pools.

    await worker.terminate_model("model_model_2")
    pool_config = (await get_pool_config(addr)).as_dict()
    assert len(pool_config["pools"]) == 3  # A main pool and 2 sub pools.

    gpu_to_model_id = await worker.get_gpu_to_model_uid()
    for dev in devices:
        assert "model_model_3" in gpu_to_model_id[dev]
    await worker.terminate_model("model_model_3")

    gpu_to_model_id = await worker.get_gpu_to_model_uid()
    for dev in devices:
        assert dev not in gpu_to_model_id


def test_merge_virtual_env_packages_override_and_append():
    base_packages = [
        "transformers @ git+https://github.com/huggingface/transformers@abcdef",
        "accelerate>=0.34.2",
        "#system_numpy#",
    ]
    extra_packages = ["transformers==5.0.0.dev0", "numpy==2.1.0"]

    merged = merge_virtual_env_packages(base_packages, extra_packages)

    assert merged == [
        "transformers==5.0.0.dev0",  # user-specified overrides default
        "accelerate>=0.34.2",
        "#system_numpy#",
        "numpy==2.1.0",
    ]


class DummyVirtualEnvManager:
    def __init__(self):
        self.env_path = "/tmp/fake-virtualenv"
        self.calls = []

    def install_packages(self, packages, **kwargs):
        self.calls.append((packages, kwargs))


class DummySupervisorRef:
    def __init__(self, fail_report_status_times: int = 0):
        self.fail_report_status_times = fail_report_status_times
        self.add_worker_calls: List[Tuple[str, List[dict], List[str]]] = []
        self.report_worker_status_calls: List[Tuple[str, Any]] = []

    async def add_worker(
        self,
        worker_address: str,
        replica_states=None,
        replica_model_uids=None,
    ):
        self.add_worker_calls.append(
            (
                worker_address,
                list(replica_states or []),
                list(replica_model_uids or []),
            )
        )

    async def report_worker_status(self, worker_address: str, status):
        if self.fail_report_status_times > 0:
            self.fail_report_status_times -= 1
            raise RuntimeError("stale supervisor")
        self.report_worker_status_calls.append((worker_address, status))

    async def record_model_version(self, model_version_infos, worker_address: str):
        return None


class DummyActorRef:
    def __init__(self, address: str):
        self.address = address


class DummyReplicaWorkerRef(DummyActorRef):
    def __init__(self, address: str, models=None):
        super().__init__(address)
        self._models = models or {}

    async def list_models(self):
        return dict(self._models)

    async def describe_model(self, model_uid: str):
        return dict(self._models[model_uid])

    async def get_model_status(self, model_uid: str):
        return {"model_uid": model_uid, "worker_address": self.address}

    async def get_model(self, model_uid: str):
        return {"model_uid": model_uid, "worker_address": self.address}


class DummyStatusGuardRef:
    def __init__(self):
        self.instance_infos = {}

    async def set_instance_info(self, model_uid: str, instance_info: InstanceInfo):
        self.instance_infos[model_uid] = instance_info

    async def get_instance_info(self, model_name=None, model_uid=None):
        if model_uid is not None:
            info = self.instance_infos.get(model_uid)
            return [info] if info is not None else []
        infos = list(self.instance_infos.values())
        if model_name is None:
            return infos
        return [info for info in infos if info.model_name == model_name]

    async def update_instance_info(self, model_uid: str, updates: dict):
        self.instance_infos[model_uid].update(**updates)

    async def update_replica_status(
        self, model_uid: str, replica_id: int, status_update: dict
    ):
        info = self.instance_infos.get(model_uid)
        if info is None:
            return
        for replica_status in info.replica_statuses:
            if replica_status.replica_id == replica_id:
                for key, value in status_update.items():
                    setattr(replica_status, key, value)
                return
        info.replica_statuses.append(
            ReplicaStatus(
                replica_id=replica_id,
                replica_model_uid=status_update.get("replica_model_uid", ""),
                worker_address=status_update.get("worker_address", ""),
                status=status_update.get("status", LaunchStatus.CREATING.name),
                created_ts=status_update.get("created_ts", 0),
                error_message=status_update.get("error_message"),
            )
        )

    async def get_replica_statuses(self, model_uid: str):
        info = self.instance_infos.get(model_uid)
        return [] if info is None else info.replica_statuses


def test_prepare_virtual_env_injects_engine_vars():
    manager = DummyVirtualEnvManager()
    settings = VirtualEnvSettings(packages=["pkgA==1.0.0"], inherit_pip_config=False)
    WorkerActor._prepare_virtual_env(
        manager,
        settings,
        ["pkgB==2.0.0"],
        model_engine="vllm",
    )

    assert len(manager.calls) == 1
    packages, kwargs = manager.calls[0]
    assert packages == ["pkgA==1.0.0", "pkgB==2.0.0"]
    assert kwargs["engine"] == "vllm"
    assert kwargs["model_engine"] == "vllm"


@pytest.mark.asyncio
async def test_worker_report_status_reconnects_and_replays_running_models(
    setup_pool, monkeypatch
):
    pool = setup_pool
    addr = pool.external_address

    worker: xo.ActorRefType["MockWorkerActor"] = await xo.create_actor(  # type: ignore
        MockWorkerActor,
        address=addr,
        uid=WorkerActor.default_uid(),
        supervisor_address="test://supervisor",
        supervisor_endpoint=None,
        main_pool=pool,
        cuda_devices=[0],
    )

    await worker.launch_builtin_model(
        "model-a-0", "mock_model_name", None, None, None, n_gpu=1
    )
    await worker.launch_builtin_model(
        "model-b-1", "mock_model_name", None, None, None, n_gpu=None
    )

    first_supervisor = DummySupervisorRef(fail_report_status_times=1)
    second_supervisor = DummySupervisorRef()
    refs = [first_supervisor, second_supervisor]
    monkeypatch.setattr("xinference.core.worker.time.time", lambda: 1710000000)

    async def fake_get_supervisor_ref(self, add_worker=True):
        if self._supervisor_ref is None:
            self._supervisor_ref = refs.pop(0)
        if add_worker:
            await self._supervisor_ref.add_worker(
                self.address,
                replica_states=self._get_running_replica_states(),
            )
        return self._supervisor_ref

    monkeypatch.setattr(WorkerActor, "get_supervisor_ref", fake_get_supervisor_ref)
    monkeypatch.setattr(
        "xinference.core.worker.gather_node_info", lambda: {"cpu": "ok"}
    )

    await worker.report_status()

    assert first_supervisor.report_worker_status_calls == []
    assert second_supervisor.add_worker_calls == [
        (
            addr,
            [
                {
                    "replica_model_uid": "model-a-0",
                    "n_worker": 1,
                    "shard": 0,
                    "model_uid": "model-a",
                    "model_name": "mock_model_name",
                    "model_version": None,
                    "model_ability": [],
                    "status": "READY",
                    "created_ts": 1710000000,
                    "instance_created_ts": 1710000000,
                },
                {
                    "replica_model_uid": "model-b-1",
                    "n_worker": 1,
                    "shard": 0,
                    "model_uid": "model-b",
                    "model_name": "mock_model_name",
                    "model_version": None,
                    "model_ability": [],
                    "status": "READY",
                    "created_ts": 1710000000,
                    "instance_created_ts": 1710000000,
                },
            ],
            [],
        )
    ]
    assert second_supervisor.report_worker_status_calls == [(addr, {"cpu": "ok"})]


@pytest.mark.asyncio
async def test_worker_report_status_refreshes_supervisor_internal_address_on_reconnect(
    setup_pool, monkeypatch
):
    pool = setup_pool
    addr = pool.external_address

    worker: xo.ActorRefType["MockWorkerActor"] = await xo.create_actor(  # type: ignore
        MockWorkerActor,
        address=addr,
        uid=WorkerActor.default_uid(),
        supervisor_address="test://stale-supervisor",
        supervisor_endpoint="http://supervisor-endpoint",
        main_pool=pool,
        cuda_devices=[0],
    )

    await worker.launch_builtin_model(
        "model-a-0", "mock_model_name", None, None, None, n_gpu=1
    )

    refreshed_supervisor = DummySupervisorRef()
    refresh_calls = []

    class DummyRESTfulClient:
        def __init__(self, base_url):
            self.base_url = base_url

        def _get_supervisor_internal_address(self):
            refresh_calls.append(self.base_url)
            return "test://fresh-supervisor"

    async def fake_actor_ref(address, uid):
        if address == "test://stale-supervisor":
            raise RuntimeError("stale supervisor address")
        if address == "test://fresh-supervisor":
            return refreshed_supervisor
        return DummyActorRef(address)

    monkeypatch.setattr(xo, "actor_ref", fake_actor_ref)
    monkeypatch.setattr(
        "xinference.core.worker.RESTfulClient",
        DummyRESTfulClient,
        raising=False,
    )
    monkeypatch.setattr(
        "xinference.core.worker.gather_node_info", lambda: {"cpu": "ok"}
    )

    await worker.report_status()

    assert refresh_calls == ["http://supervisor-endpoint"]
    assert len(refreshed_supervisor.add_worker_calls) == 1
    worker_address, replica_states, replica_model_uids = (
        refreshed_supervisor.add_worker_calls[0]
    )
    assert worker_address == addr
    assert replica_model_uids == []
    assert len(replica_states) == 1
    assert replica_states[0]["replica_model_uid"] == "model-a-0"
    assert replica_states[0]["n_worker"] == 1
    assert replica_states[0]["shard"] == 0
    assert refreshed_supervisor.report_worker_status_calls == [(addr, {"cpu": "ok"})]


@pytest.mark.asyncio
async def test_worker_report_status_does_not_refresh_address_when_connection_is_healthy(
    setup_pool, monkeypatch
):
    pool = setup_pool
    addr = pool.external_address

    worker: xo.ActorRefType["MockWorkerActor"] = await xo.create_actor(  # type: ignore
        MockWorkerActor,
        address=addr,
        uid=WorkerActor.default_uid(),
        supervisor_address="test://healthy-supervisor",
        supervisor_endpoint="http://supervisor-endpoint",
        main_pool=pool,
        cuda_devices=[0],
    )

    healthy_supervisor = DummySupervisorRef()
    refresh_calls = []

    class DummyRESTfulClient:
        def __init__(self, base_url):
            self.base_url = base_url

        def _get_supervisor_internal_address(self):
            refresh_calls.append(self.base_url)
            return "test://fresh-supervisor"

    async def fake_actor_ref(address, uid):
        if address == "test://healthy-supervisor":
            return healthy_supervisor
        return DummyActorRef(address)

    monkeypatch.setattr(xo, "actor_ref", fake_actor_ref)
    monkeypatch.setattr(
        "xinference.core.worker.RESTfulClient",
        DummyRESTfulClient,
        raising=False,
    )
    monkeypatch.setattr(
        "xinference.core.worker.gather_node_info", lambda: {"cpu": "ok"}
    )

    await worker.report_status()

    assert refresh_calls == []
    assert healthy_supervisor.report_worker_status_calls == [(addr, {"cpu": "ok"})]


@pytest.mark.asyncio
async def test_supervisor_add_worker_idempotent_rebuilds_replica_state(monkeypatch):
    supervisor = SupervisorActor()
    supervisor._status_guard_ref = DummyStatusGuardRef()
    worker_ref = DummyReplicaWorkerRef(
        "worker-1",
        models={
            "model-a-0": {"model_uid": "model-a-0", "address": "worker-1"},
            "model-a-1": {"model_uid": "model-a-1", "address": "worker-1"},
        },
    )

    async def fake_actor_ref(address, uid):
        assert address == "worker-1"
        return worker_ref

    monkeypatch.setattr(xo, "actor_ref", fake_actor_ref)

    replica_states = [
        {"replica_model_uid": "model-a-0", "n_worker": 1, "shard": 0},
        {"replica_model_uid": "model-a-1", "n_worker": 1, "shard": 0},
        {"replica_model_uid": "model-a-rank0", "n_worker": 1, "shard": 0},
    ]

    await supervisor.add_worker("worker-1", replica_states=replica_states)
    await supervisor.add_worker("worker-1", replica_states=replica_states)

    replica_info = supervisor._model_uid_to_replica_info["model-a"]
    assert replica_info.replica == 2
    assert supervisor._replica_model_uid_to_worker["model-a-0"] is worker_ref
    assert supervisor._replica_model_uid_to_worker["model-a-1"] is worker_ref
    assert "model-a-rank0" not in supervisor._replica_model_uid_to_worker
    assert replica_info.replica_to_worker_refs[0] == [worker_ref]
    assert replica_info.replica_to_worker_refs[1] == [worker_ref]


@pytest.mark.asyncio
async def test_supervisor_report_worker_status_rejects_unregistered_worker():
    """report_worker_status must reject a worker absent from the registry
    (e.g. after a supervisor restart) instead of fabricating a _worker_status
    entry, so the worker is pushed into its reconnect/add_worker path and the
    registry (and workers_total) self-heals."""
    from ..supervisor import WorkerNotRegisteredError

    supervisor = SupervisorActor()
    assert "ghost-worker" not in supervisor._worker_address_to_worker

    with pytest.raises(WorkerNotRegisteredError):
        await supervisor.report_worker_status("ghost-worker", {"cpu": "ok"})

    # No stale status entry should have been created.
    assert "ghost-worker" not in supervisor._worker_status


@pytest.mark.asyncio
async def test_supervisor_report_worker_status_accepts_registered_worker():
    """A worker present in the registry reports normally and its status is
    recorded."""
    supervisor = SupervisorActor()
    supervisor._worker_address_to_worker["worker-1"] = object()

    await supervisor.report_worker_status("worker-1", {"cpu": "ok"})

    assert "worker-1" in supervisor._worker_status
    assert supervisor._worker_status["worker-1"].status == {"cpu": "ok"}


@pytest.mark.asyncio
async def test_supervisor_add_worker_preserves_sharded_replicas_on_replay(monkeypatch):
    supervisor = SupervisorActor()
    supervisor._status_guard_ref = DummyStatusGuardRef()
    shard0 = DummyReplicaWorkerRef(
        "worker-0",
        models={"model-s-0": {"model_uid": "model-s-0", "address": "worker-0"}},
    )
    shard1 = DummyReplicaWorkerRef(
        "worker-1",
        models={"model-s-0": {"model_uid": "model-s-0", "address": "worker-1"}},
    )

    supervisor._worker_address_to_worker = {
        "worker-0": shard0,
        "worker-1": shard1,
    }
    supervisor._replica_model_uid_to_worker = {"model-s-0": (shard0, shard1)}
    supervisor._model_uid_to_replica_info = {
        "model-s": ReplicaInfo(replica=1, scheduler=itertools.cycle(range(1)))
    }
    supervisor._model_uid_to_replica_info["model-s"].replica_to_worker_refs[0].extend(
        [shard0, shard1]
    )

    async def fake_actor_ref(address, uid):
        if address == "worker-0":
            return shard0
        if address == "worker-1":
            return shard1
        raise AssertionError(address)

    monkeypatch.setattr(xo, "actor_ref", fake_actor_ref)

    await supervisor.add_worker(
        "worker-1",
        replica_states=[{"replica_model_uid": "model-s-0", "n_worker": 2, "shard": 1}],
    )

    worker_refs = supervisor._replica_model_uid_to_worker["model-s-0"]
    assert isinstance(worker_refs, tuple)
    assert worker_refs == (shard0, shard1)
    assert supervisor._model_uid_to_replica_info["model-s"].replica_to_worker_refs[
        0
    ] == [
        shard0,
        shard1,
    ]
    assert (
        await supervisor._status_guard_ref.get_instance_info(model_uid="model-s") == []
    )


@pytest.mark.asyncio
async def test_supervisor_add_worker_rebuilds_sharded_replica_order(monkeypatch):
    supervisor = SupervisorActor()
    supervisor._status_guard_ref = DummyStatusGuardRef()
    shard0 = DummyReplicaWorkerRef(
        "worker-0",
        models={"model-s-0": {"model_uid": "model-s-0", "address": "worker-0"}},
    )
    shard1 = DummyReplicaWorkerRef(
        "worker-1",
        models={"model-s-0": {"model_uid": "model-s-0", "address": "worker-1"}},
    )

    async def fake_actor_ref(address, uid):
        if address == "worker-0":
            return shard0
        if address == "worker-1":
            return shard1
        raise AssertionError(address)

    monkeypatch.setattr(xo, "actor_ref", fake_actor_ref)

    await supervisor.add_worker(
        "worker-1",
        replica_states=[{"replica_model_uid": "model-s-0", "n_worker": 2, "shard": 1}],
    )
    await supervisor.add_worker(
        "worker-0",
        replica_states=[{"replica_model_uid": "model-s-0", "n_worker": 2, "shard": 0}],
    )

    worker_refs = supervisor._replica_model_uid_to_worker["model-s-0"]
    assert isinstance(worker_refs, tuple)
    assert worker_refs == (shard0, shard1)
    assert supervisor._model_uid_to_replica_info["model-s"].replica_to_worker_refs[
        0
    ] == [
        shard0,
        shard1,
    ]
    assert (
        await supervisor._status_guard_ref.get_instance_info(model_uid="model-s") == []
    )


@pytest.mark.asyncio
async def test_supervisor_add_worker_rebuilds_replica_details_after_reconnect(
    monkeypatch,
):
    supervisor = SupervisorActor()
    supervisor._status_guard_ref = DummyStatusGuardRef()
    worker_ref = DummyReplicaWorkerRef(
        "worker-1",
        models={
            "model-a-0": {"model_uid": "model-a-0", "address": "worker-1"},
            "model-a-1": {"model_uid": "model-a-1", "address": "worker-1"},
        },
    )

    async def fake_actor_ref(address, uid):
        assert address == "worker-1"
        return worker_ref

    monkeypatch.setattr(xo, "actor_ref", fake_actor_ref)

    replica_states = [
        {
            "replica_model_uid": "model-a-0",
            "n_worker": 1,
            "shard": 0,
            "model_uid": "model-a",
            "model_name": "SenseVoiceSmall",
            "model_version": None,
            "model_ability": ["audio"],
            "status": LaunchStatus.READY.name,
            "created_ts": 1710000001,
            "instance_created_ts": 1710000001,
        },
        {
            "replica_model_uid": "model-a-1",
            "n_worker": 1,
            "shard": 0,
            "model_uid": "model-a",
            "model_name": "SenseVoiceSmall",
            "model_version": None,
            "model_ability": ["audio"],
            "status": LaunchStatus.READY.name,
            "created_ts": 1710000002,
            "instance_created_ts": 1710000001,
        },
    ]

    await supervisor.add_worker("worker-1", replica_states=replica_states)

    instance_infos = await supervisor._status_guard_ref.get_instance_info(
        model_uid="model-a"
    )
    assert len(instance_infos) == 1
    instance_info = instance_infos[0]
    assert instance_info.model_name == "SenseVoiceSmall"
    assert instance_info.model_uid == "model-a"
    assert instance_info.model_ability == ["audio"]
    assert instance_info.replica == 2
    assert instance_info.status == LaunchStatus.READY.name
    assert instance_info.instance_created_ts == 1710000001

    replica_statuses = await supervisor._status_guard_ref.get_replica_statuses(
        "model-a"
    )
    assert len(replica_statuses) == 2
    assert [status.replica_id for status in replica_statuses] == [0, 1]
    assert [status.replica_model_uid for status in replica_statuses] == [
        "model-a-0",
        "model-a-1",
    ]
    assert [status.worker_address for status in replica_statuses] == [
        "worker-1",
        "worker-1",
    ]
    assert [status.status for status in replica_statuses] == [
        LaunchStatus.READY.name,
        LaunchStatus.READY.name,
    ]


def test_prepare_virtual_env_without_engine_vars():
    manager = DummyVirtualEnvManager()
    settings = VirtualEnvSettings(packages=["pkgA==1.0.0"], inherit_pip_config=False)
    WorkerActor._prepare_virtual_env(
        manager,
        settings,
        ["pkgB==2.0.0"],
        model_engine=None,
    )

    assert len(manager.calls) == 1
    _, kwargs = manager.calls[0]
    assert "engine" not in kwargs
    assert "model_engine" not in kwargs


def test_prepare_virtual_env_inherit_pip_config(monkeypatch):
    manager = DummyVirtualEnvManager()
    settings = VirtualEnvSettings(packages=["pkgA==1.0.0"], inherit_pip_config=True)

    monkeypatch.setattr(
        "xinference.core.worker.get_pip_config_args",
        lambda: {"index_url": "https://example.invalid/simple"},
    )

    WorkerActor._prepare_virtual_env(
        manager,
        settings,
        None,
        model_engine="mlx",
    )

    assert len(manager.calls) == 1
    _, kwargs = manager.calls[0]
    assert kwargs["index_url"] == "https://example.invalid/simple"


def test_prepare_virtual_env_keeps_system_markers():
    manager = DummyVirtualEnvManager()
    settings = VirtualEnvSettings(
        packages=["pkgA==1.0.0", "#system_torch#", "#system_numpy#"],
        inherit_pip_config=False,
    )

    WorkerActor._prepare_virtual_env(
        manager,
        settings,
        ["pkgB==2.0.0", "#system_torchaudio#"],
        model_engine=None,
    )

    assert len(manager.calls) == 1
    packages, _ = manager.calls[0]
    assert packages == [
        "pkgA==1.0.0",
        "#system_torch#",
        "#system_numpy#",
        "pkgB==2.0.0",
        "#system_torchaudio#",
    ]


def test_prepare_virtual_env_expands_engine_dependencies_before_user_override():
    manager = DummyVirtualEnvManager()
    settings = VirtualEnvSettings(
        packages=['#vllm_dependencies# ; #engine# == "vllm"'],
        inherit_pip_config=False,
    )

    WorkerActor._prepare_virtual_env(
        manager,
        settings,
        ["vllm==0.10.2"],
        model_engine="vllm",
    )

    assert len(manager.calls) == 1
    packages, _ = manager.calls[0]
    assert packages.count("vllm==0.10.2") == 1
    assert "vllm>=0.11.2" not in packages


@pytest.mark.asyncio
async def test_launch_embedding_model(setup_pool):
    pool = setup_pool
    addr = pool.external_address

    worker: xo.ActorRefType["MockWorkerActor"] = await xo.create_actor(  # type: ignore
        MockWorkerActor,
        address=addr,
        uid=WorkerActor.default_uid(),
        supervisor_address="test",
        main_pool=pool,
        cuda_devices=[i for i in range(4)],
    )

    # test embedding device candidates 1
    await worker.launch_builtin_model(
        "model_model_1", "mock_model_name", None, None, None, n_gpu=3
    )
    gpu_to_model_id = await worker.get_gpu_to_model_uid()
    assert len(gpu_to_model_id) == 3

    await worker.launch_builtin_model(
        "model_model_2", "mock_model_name", None, None, None, "embedding", n_gpu=1
    )

    gpu_to_model_id = await worker.get_gpu_to_model_uid()
    assert "model_model_2" in gpu_to_model_id[3]

    # test terminate LLM model, then launch embedding model
    await worker.terminate_model("model_model_1")
    await worker.launch_builtin_model(
        "model_model_3", "mock_model_name", None, None, None, "embedding", n_gpu=1
    )
    gpu_to_model_id = await worker.get_gpu_to_model_uid()
    assert "model_model_3" in gpu_to_model_id[0]

    await worker.terminate_model("model_model_2")
    await worker.terminate_model("model_model_3")
    gpu_to_model_id = await worker.get_gpu_to_model_uid()
    assert 0 not in gpu_to_model_id
    assert 3 not in gpu_to_model_id

    # test embedding device candidates 2
    await worker.launch_builtin_model(
        "model_model_1", "mock_model_name", None, None, None, n_gpu=2
    )

    await worker.launch_builtin_model(
        "model_model_2", "mock_model_name", None, None, None, "embedding", n_gpu=1
    )

    await worker.launch_builtin_model(
        "model_model_3", "mock_model_name", None, None, None, "embedding", n_gpu=1
    )
    gpu_to_model_id = await worker.get_gpu_to_model_uid()
    assert "model_model_2" in gpu_to_model_id[2]
    assert "model_model_3" in gpu_to_model_id[3]

    await worker.launch_builtin_model(
        "model_model_4", "mock_model_name", None, None, None, "embedding", n_gpu=1
    )
    gpu_to_model_id = await worker.get_gpu_to_model_uid()
    assert "model_model_4" in gpu_to_model_id[0]

    for i in range(1, 5):
        await worker.terminate_model(f"model_model_{i}")
    gpu_to_model_id = await worker.get_gpu_to_model_uid()
    assert len(gpu_to_model_id) == 0


@pytest.mark.asyncio
async def test_launch_model_with_gpu_idx(setup_pool):
    pool = setup_pool
    addr = pool.external_address

    worker: xo.ActorRefType["MockWorkerActor"] = await xo.create_actor(  # type: ignore
        MockWorkerActor,
        address=addr,
        uid=WorkerActor.default_uid(),
        supervisor_address="test",
        main_pool=pool,
        cuda_devices=[i for i in range(4)],
    )
    assert (await xo.actor_ref(addr, WorkerActor.default_uid())).uid == b"worker"

    # test normal model
    await worker.launch_builtin_model(
        "normal_model_model_1", "mock_model_name", None, None, None, "LLM", n_gpu=1
    )
    llm_info = await worker.get_gpu_to_model_uid()
    assert len(llm_info) == 1
    assert 0 in llm_info

    await worker.launch_builtin_model(
        "model_model_2", "mock_model_name", None, None, None, "LLM", gpu_idx=[0]
    )
    llm_info = await worker.get_gpu_to_model_uid()
    assert len(llm_info) == 1
    assert 0 in llm_info

    user_specified_info = await worker.get_user_specified_gpu_to_model_uids()
    assert len(user_specified_info) == 1
    assert 0 in user_specified_info
    assert len(user_specified_info[0]) == 1
    assert list(user_specified_info[0])[0][0] == "model_model_2"
    assert list(user_specified_info[0])[0][1] == "LLM"

    # test vllm model
    await worker.launch_builtin_model(
        "vllm_model_model_3", "mock_model_name", None, None, None, "LLM", n_gpu=1
    )
    llm_info = await worker.get_gpu_to_model_uid()
    assert len(llm_info) == 2
    assert 0 in llm_info
    assert 1 in llm_info

    # Force single-replica mode to verify conflict handling on occupied GPU.
    await worker.set_allow_multi_replica_per_gpu(False)
    with pytest.raises(RuntimeError):
        await worker.launch_builtin_model(
            "model_model_4", "mock_model_name", None, None, None, "LLM", gpu_idx=[1]
        )
    await worker.set_allow_multi_replica_per_gpu(True)

    await worker.launch_builtin_model(
        "model_model_4", "mock_model_name", None, None, None, "LLM", gpu_idx=[2]
    )
    llm_info = await worker.get_gpu_to_model_uid()
    assert len(llm_info) == 2
    assert 0 in llm_info
    assert 1 in llm_info

    user_specified_info = await worker.get_user_specified_gpu_to_model_uids()
    assert len(user_specified_info) == 2
    assert 0 in user_specified_info
    assert 2 in user_specified_info
    assert len(user_specified_info[2]) == 1
    assert list(user_specified_info[2])[0][0] == "model_model_4"
    assert list(user_specified_info[2])[0][1] == "LLM"

    # then launch a LLM without gpu_idx
    await worker.launch_builtin_model(
        "normal_model_model_5", "mock_model_name", None, None, None, "LLM", n_gpu=1
    )
    llm_info = await worker.get_gpu_to_model_uid()
    assert len(llm_info) == 3
    assert 0 in llm_info
    assert 1 in llm_info
    assert 3 in llm_info

    # launch without gpu_idx again, should reuse the least loaded GPU
    await worker.launch_builtin_model(
        "normal_model_model_6", "mock_model_name", None, None, None, "LLM", n_gpu=1
    )
    llm_info = await worker.get_gpu_to_model_uid()
    assert len(llm_info) == 4

    #  test terminate and cleanup
    await worker.terminate_model("normal_model_model_1")
    await worker.terminate_model("model_model_2")
    await worker.terminate_model("vllm_model_model_3")
    await worker.terminate_model("model_model_4")
    await worker.terminate_model("normal_model_model_5")
    await worker.terminate_model("normal_model_model_6")

    llm_info = await worker.get_gpu_to_model_uid()
    assert len(llm_info) == 0

    user_specified_info = await worker.get_user_specified_gpu_to_model_uids()
    for idx, model_infos in user_specified_info.items():
        assert len(model_infos) == 0

    # next, test with embedding models
    await worker.launch_builtin_model(
        "embedding_1", "mock_model_name", None, None, None, "embedding", n_gpu=1
    )
    llm_info = await worker.get_gpu_to_model_uid()
    assert len(llm_info) == 1
    assert 0 in llm_info

    await worker.launch_builtin_model(
        "vllm_mock_model_2", "mock_model_name", None, None, None, "LLM", gpu_idx=[0]
    )
    llm_info = await worker.get_gpu_to_model_uid()
    assert len(llm_info) == 1
    assert 0 in llm_info

    user_specified_info = await worker.get_user_specified_gpu_to_model_uids()
    assert len(user_specified_info[0]) == 1
    assert list(user_specified_info[0])[0][0] == "vllm_mock_model_2"
    assert list(user_specified_info[0])[0][1] == "LLM"

    # launch should reuse all GPUs starting from the least loaded ones
    await worker.launch_builtin_model(
        "normal_mock_model_3", "mock_model_name", None, None, None, "LLM", n_gpu=4
    )

    await worker.launch_builtin_model(
        "embedding_3", "mock_model_name", None, None, None, "embedding", n_gpu=1
    )
    await worker.launch_builtin_model(
        "rerank_4", "mock_model_name", None, None, None, "rerank", gpu_idx=[0]
    )
    await worker.launch_builtin_model(
        "embedding_5", "mock_model_name", None, None, None, "embedding", n_gpu=1
    )
    await worker.launch_builtin_model(
        "rerank_6", "mock_model_name", None, None, None, "rerank", n_gpu=1
    )
    await worker.launch_builtin_model(
        "rerank_7", "mock_model_name", None, None, None, "rerank", n_gpu=1
    )
    user_specified_info = await worker.get_user_specified_gpu_to_model_uids()
    llm_info = await worker.get_gpu_to_model_uid()
    all_models = set().union(*llm_info.values())
    assert {
        "embedding_1",
        "embedding_3",
        "embedding_5",
        "rerank_6",
        "rerank_7",
    } <= all_models
    # GPU0 has user-specified vLLM plus rerank_4, so two entries expected there
    assert len(user_specified_info[0]) == 2

    # cleanup
    await worker.terminate_model("embedding_1")
    await worker.terminate_model("vllm_mock_model_2")
    await worker.terminate_model("normal_mock_model_3")
    await worker.terminate_model("embedding_3")
    await worker.terminate_model("rerank_4")
    await worker.terminate_model("embedding_5")
    await worker.terminate_model("rerank_6")
    await worker.terminate_model("rerank_7")

    user_specified_info = await worker.get_user_specified_gpu_to_model_uids()
    llm_info = await worker.get_gpu_to_model_uid()
    for info in [llm_info, user_specified_info]:
        for dev, details in info.items():
            assert len(details) == 0


class _FakeChild:
    def __init__(self, pid: int):
        self.pid = pid


class _FakeProcess:
    """Stand-in for psutil.Process exposing a preset recursive children map."""

    _children_map: dict = {}

    def __init__(self, pid: int):
        self._pid = pid

    def children(self, recursive: bool = False):
        return [_FakeChild(p) for p in self._children_map.get(self._pid, [])]


async def _make_gpu_worker(pool, cuda_devices=(0, 1)):
    worker: xo.ActorRefType["MockWorkerActor"] = await xo.create_actor(  # type: ignore
        MockWorkerActor,
        address=pool.external_address,
        uid=WorkerActor.default_uid(),
        supervisor_address="test",
        main_pool=pool,
        cuda_devices=list(cuda_devices),
    )
    sup = DummySupervisorRef()
    await worker.set_supervisor_ref_for_test(sup)
    return worker, sup


def _patch_gpu_sources(monkeypatch, gpu_mem, children_map=None, calls=None):
    import psutil

    def _fake_get_per_process_gpu_memory():
        if calls is not None:
            calls.append("called")
        return gpu_mem

    monkeypatch.setattr("xinference.core.worker.gather_node_info", lambda: {})
    monkeypatch.setattr(
        "xinference.device_utils.get_per_process_gpu_memory",
        _fake_get_per_process_gpu_memory,
    )
    _FakeProcess._children_map = children_map or {}
    monkeypatch.setattr(psutil, "Process", _FakeProcess)


@pytest.mark.asyncio
async def test_report_status_attributes_gpu_by_subpool_pids(setup_pool, monkeypatch):
    # Scenario 4: multi-card multi-replica LLM. Each replica's GPU holders are
    # its registered secondary sub-pool PIDs; same card shared, no cross/double.
    worker, sup = await _make_gpu_worker(setup_pool)
    await worker.set_gpu_attribution_tables_for_test(
        pid={}, subpool={"A": {100, 101}, "B": {200, 201}}, total_devices=[0, 1]
    )
    _patch_gpu_sources(
        monkeypatch,
        gpu_mem={
            100: {0: 1000},
            101: {1: 1000},
            200: {0: 500},
            201: {1: 500},
        },
    )

    await worker.report_status()

    status = sup.report_worker_status_calls[-1][1]
    assert status["model_gpu_memory"]["A"] == {0: 1000, 1: 1000}
    assert status["model_gpu_memory"]["B"] == {0: 500, 1: 500}


@pytest.mark.asyncio
async def test_report_status_attributes_gpu_via_recursive_children(
    setup_pool, monkeypatch
):
    # Scenario 1/2: single-card vLLM. GPU is held by the forked EngineCore, a
    # recursive child of the primary sub-pool PID -- not registered directly.
    worker, sup = await _make_gpu_worker(setup_pool, cuda_devices=[0])
    await worker.set_gpu_attribution_tables_for_test(
        pid={}, subpool={"A": {100}}, total_devices=[0]
    )
    _patch_gpu_sources(
        monkeypatch,
        gpu_mem={555: {0: 2000}},  # only the EngineCore holds GPU memory
        children_map={100: [555]},
    )

    await worker.report_status()

    status = sup.report_worker_status_calls[-1][1]
    assert status["model_gpu_memory"]["A"] == {0: 2000}


@pytest.mark.asyncio
async def test_report_status_replicas_do_not_cross_or_double_count(
    setup_pool, monkeypatch
):
    # Same-card concurrent replicas: disjoint PID sets attribute independently;
    # a PID is counted once within a replica even if it appears via own_pid too.
    worker, sup = await _make_gpu_worker(setup_pool, cuda_devices=[0])
    await worker.set_gpu_attribution_tables_for_test(
        pid={"A": 100},  # own_pid overlaps subpool set -> must not double count
        subpool={"A": {100}, "B": {200}},
        total_devices=[0],
    )
    _patch_gpu_sources(
        monkeypatch,
        gpu_mem={100: {0: 700}, 200: {0: 300}},
    )

    await worker.report_status()

    status = sup.report_worker_status_calls[-1][1]
    assert status["model_gpu_memory"]["A"] == {0: 700}
    assert status["model_gpu_memory"]["B"] == {0: 300}


@pytest.mark.asyncio
async def test_report_status_skips_gpu_collection_when_cpu_only(
    setup_pool, monkeypatch
):
    # No GPU devices -> the guard skips NVML entirely; no model_gpu_memory key.
    worker, sup = await _make_gpu_worker(setup_pool, cuda_devices=[])
    await worker.set_gpu_attribution_tables_for_test(
        pid={"A": 100}, subpool={"A": {100}}, total_devices=[]
    )
    calls: list = []
    _patch_gpu_sources(monkeypatch, gpu_mem={100: {0: 1000}}, calls=calls)

    await worker.report_status()

    status = sup.report_worker_status_calls[-1][1]
    assert "model_gpu_memory" not in status
    assert calls == []  # NVML never queried on CPU-only worker


@pytest.mark.asyncio
async def test_launch_semaphore_default(setup_pool):
    pool = setup_pool
    addr = pool.external_address

    worker: xo.ActorRefType["MockWorkerActor"] = await xo.create_actor(  # type: ignore
        MockWorkerActor,
        address=addr,
        uid=WorkerActor.default_uid(),
        supervisor_address="test",
        main_pool=pool,
        cuda_devices=[0],
    )

    # Semaphore initialized in __init__, default value 5
    assert await worker.get_launch_semaphore_value() == 5
    assert await worker.get_launch_active_count() == 0
    assert await worker.get_launch_waiting_count() == 0


@pytest.mark.asyncio
async def test_launch_semaphore_env_override(setup_pool, monkeypatch):
    monkeypatch.setattr("xinference.core.worker.XINFERENCE_MAX_CONCURRENT_LAUNCHES", 2)

    pool = setup_pool
    addr = pool.external_address

    worker: xo.ActorRefType["MockWorkerActor"] = await xo.create_actor(  # type: ignore
        MockWorkerActor,
        address=addr,
        uid=WorkerActor.default_uid(),
        supervisor_address="test",
        main_pool=pool,
        cuda_devices=[0],
    )

    assert await worker.get_launch_semaphore_value() == 2


@pytest.mark.asyncio
async def test_launch_semaphore_concurrency():
    """Verify semaphore correctly limits concurrent launches."""
    max_concurrent = 2
    peak_concurrent = 0
    current_concurrent = 0

    sem = asyncio.Semaphore(max_concurrent)

    async def simulate_launch():
        nonlocal peak_concurrent, current_concurrent
        async with sem:
            current_concurrent += 1
            if current_concurrent > peak_concurrent:
                peak_concurrent = current_concurrent
            await asyncio.sleep(0.05)
            current_concurrent -= 1

    # Launch 6 concurrent tasks with semaphore limited to 2
    await asyncio.gather(*[simulate_launch() for _ in range(6)])

    assert peak_concurrent == max_concurrent


class _RecordingWorkerRef(DummyActorRef):
    """Worker ref that records terminate_model calls for rank0 eviction tests."""

    def __init__(self, address: str):
        super().__init__(address)
        self.terminated: List[str] = []

    async def terminate_model(self, model_uid: str):
        self.terminated.append(model_uid)


@pytest.mark.asyncio
async def test_cleanup_distributed_actors_terminates_rank0():
    """rank0 lives in its own subpool that a regular replica's OOM never
    terminates, so _cleanup_distributed_actors(terminate_rank0_on_worker=True)
    must RPC the worker to terminate it -- not merely drop the supervisor
    mapping (which would leak the rank0 actor/subpool)."""
    supervisor = SupervisorActor()
    rank0_ref = _RecordingWorkerRef("worker-1")
    supervisor._replica_model_uid_to_worker = {"model-x-rank0": rank0_ref}
    supervisor._collective_manager_mapping = {}
    supervisor._block_tracker_mapping = {}

    await supervisor._cleanup_distributed_actors(
        "model-x", terminate_rank0_on_worker=True
    )

    assert rank0_ref.terminated == ["model-x-rank0"]
    assert "model-x-rank0" not in supervisor._replica_model_uid_to_worker


@pytest.mark.asyncio
async def test_cleanup_distributed_actors_skips_rank0_rpc_when_false():
    """With terminate_rank0_on_worker=False the supervisor mapping is dropped
    but no terminate RPC is issued (the graceful caller already knows rank0 is
    gone)."""
    supervisor = SupervisorActor()
    rank0_ref = _RecordingWorkerRef("worker-1")
    supervisor._replica_model_uid_to_worker = {"model-x-rank0": rank0_ref}
    supervisor._collective_manager_mapping = {}
    supervisor._block_tracker_mapping = {}

    await supervisor._cleanup_distributed_actors(
        "model-x", terminate_rank0_on_worker=False
    )

    assert rank0_ref.terminated == []
    assert "model-x-rank0" not in supervisor._replica_model_uid_to_worker


@pytest.mark.asyncio
async def test_mark_replica_dead_last_replica_terminates_rank0():
    """End to end: when the single (last) replica of a Xavier model exhausts
    auto-recover, mark_replica_dead's last-replica branch must terminate the
    separate rank0 actor, drop its mapping, and keep the failure marker lit."""

    class _Info:
        model_name = "m"

    class _StatusGuard:
        async def get_instance_info(self, model_name=None, model_uid=None):
            return [_Info()]

        async def remove_replica_status(self, model_uid: str, replica_id: int):
            return 0  # last replica gone

        async def update_instance_info(self, model_uid: str, updates: dict):
            pass

    supervisor = SupervisorActor()
    supervisor._status_guard_ref = _StatusGuard()
    supervisor._collective_manager_mapping = {}
    supervisor._block_tracker_mapping = {}

    replica_ref = _RecordingWorkerRef("worker-1")
    rank0_ref = _RecordingWorkerRef("worker-1")
    replica_info = ReplicaInfo(replica=1, scheduler=itertools.cycle(range(1)))
    replica_info.active_replica_ids.append(0)
    replica_info.replica_to_worker_refs[0].append(replica_ref)
    supervisor._model_uid_to_replica_info = {"model-x": replica_info}
    supervisor._replica_model_uid_to_worker = {
        "model-x-0": replica_ref,
        "model-x-rank0": rank0_ref,
    }

    await supervisor.mark_replica_dead("model-x-0")

    # rank0 terminated on the worker and supervisor mapping dropped.
    assert rank0_ref.terminated == ["model-x-rank0"]
    assert "model-x-rank0" not in supervisor._replica_model_uid_to_worker
    # Dead replica evicted and model taken offline.
    assert "model-x" not in supervisor._model_uid_to_replica_info
    assert "model-x-0" not in supervisor._replica_model_uid_to_worker
    # Failure gauge marker stays lit (mark_replica_dead must not clear it).
    assert ("model-x", 0) in supervisor._unexpected_down_replicas


@pytest.mark.asyncio
async def test_recover_model_pops_launch_ts_from_kwargs():
    """launch_ts is an internal timestamp stamped onto the launch snapshot at
    launch_builtin_model entry. recover_model splats the snapshot back via
    launch_builtin_model(**launch_args), which would inject launch_ts into the
    model's self._kwargs. Models that forward the full self._kwargs into a
    strict constructor (e.g. jina-reranker-v3 -> AutoModelForCausalLM.from_pretrained)
    then crash with TypeError. recover_model must pop launch_ts before the splat,
    mirroring the existing cleanup in recover_models_on_startup."""

    # Use a minimal mock object to avoid WorkerActor initialization complexity
    class _MockWorker:
        async def launch_builtin_model(self, **kwargs):
            self._captured_kwargs = kwargs
            return "mock-subpool-address"

        async def get_supervisor_ref(self, add_worker=False):
            return None

    worker = _MockWorker()

    # Simulate a cached launch snapshot with launch_ts (the internal timestamp)
    launch_args = {
        "model_uid": "test-model-0",
        "model_name": "test-model",
        "launch_ts": 1780900592,  # Internal timestamp, not a model param
        "some_param": "value",
    }
    original_launch_args = dict(launch_args)

    # Mock parse_replica_model_uid to avoid dependency
    def mock_parse(uid):
        return ("test-model", 0)

    import xinference.core.worker as worker_module

    original_parse = worker_module.parse_replica_model_uid
    worker_module.parse_replica_model_uid = mock_parse

    try:
        # Call recover_model directly (it's a standalone async method)
        await WorkerActor.recover_model(worker, launch_args)

        # Assert: launch_builtin_model was called without launch_ts
        assert hasattr(worker, "_captured_kwargs")
        assert "launch_ts" not in worker._captured_kwargs
        assert worker._captured_kwargs["model_uid"] == "test-model-0"
        assert worker._captured_kwargs["some_param"] == "value"

        # Assert: original launch_args still has launch_ts (copy isolation)
        assert launch_args == original_launch_args
        assert launch_args["launch_ts"] == 1780900592
    finally:
        worker_module.parse_replica_model_uid = original_parse
