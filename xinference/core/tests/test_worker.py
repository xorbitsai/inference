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
