# Copyright 2022-2023 XProbe Inc.
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
from typing import List, Optional

import pytest
import pytest_asyncio
import xoscar as xo
from xoscar import MainActorPoolType, create_actor_pool, get_pool_config

from ..worker import WorkerActor


class MockWorkerActor(WorkerActor):
    def __init__(
        self,
        supervisor_address: str,
        main_pool: MainActorPoolType,
        cuda_devices: List[int],
    ):
        super().__init__(supervisor_address, main_pool, cuda_devices)

    async def __post_create__(self):
        pass

    async def __pre_destroy__(self):
        pass

    def get_gpu_to_model_uids(self):
        return self._gpu_to_model_uids

    async def launch_builtin_model(
        self,
        model_uid: str,
        model_name: str,
        model_size_in_billions: Optional[int],
        model_format: Optional[str],
        quantization: Optional[str],
        model_type: str = "LLM",
        n_gpu: Optional[int] = None,
        **kwargs,
    ):
        subpool_address, devices = await self._choose_subpool(model_uid, n_gpu=n_gpu)
        for dev in devices:
            self._gpu_to_model_uids[int(dev)].add(model_uid)
        self._model_uid_to_addr[model_uid] = subpool_address

    async def terminate_model(self, model_uid: str):
        for device in self._gpu_to_model_uids:
            if model_uid in self._gpu_to_model_uids[device]:
                self._gpu_to_model_uids[device].remove(model_uid)

        sub_pool_addr = self._model_uid_to_addr[model_uid]
        await self._main_pool.remove_sub_pool(sub_pool_addr)
        del self._model_uid_to_addr[model_uid]


@pytest_asyncio.fixture
async def setup_pool():
    pool = await create_actor_pool(
        f"test://127.0.0.1:{xo.utils.get_next_port()}", n_process=0
    )
    async with pool:
        yield pool


@pytest.mark.asyncio
async def test_allocate_cuda_devices(setup_pool):
    pool = setup_pool
    addr = pool.external_address

    worker: xo.ActorRefType["MockWorkerActor"] = await xo.create_actor(
        MockWorkerActor,
        address=addr,
        uid=WorkerActor.uid(),
        supervisor_address="test",
        main_pool=pool,
        cuda_devices=[i for i in range(8)],
    )

    devices = await worker.allocate_devices(1)
    await worker.launch_builtin_model("x1", "x1", None, None, None, n_gpu=1)
    assert devices == [0]

    devices = await worker.allocate_devices(4)
    await worker.launch_builtin_model("x2", "x2", None, None, None, n_gpu=4)
    assert devices == [1, 2, 3, 4]

    devices = await worker.allocate_devices(4)
    await worker.launch_builtin_model("x3", "x3", None, None, None, n_gpu=4)
    assert devices == [0, 5, 6, 7]

    devices = await worker.allocate_devices(5)
    await worker.launch_builtin_model("x4", "x4", None, None, None, n_gpu=5)
    assert devices == [1, 2, 3, 4, 5]

    devices = await worker.allocate_devices(2)
    await worker.launch_builtin_model("x5", "x5", None, None, None, n_gpu=2)
    assert devices == [6, 7]

    devices = await worker.allocate_devices(1)
    await worker.launch_builtin_model("x6", "x6", None, None, None, n_gpu=1)
    assert devices == [0]

    devices = await worker.allocate_devices(1)
    await worker.launch_builtin_model("x7", "x7", None, None, None, n_gpu=1)
    assert devices == [1]

    pool_config = (await get_pool_config(addr)).as_dict()
    assert len(pool_config["pools"]) == 8


@pytest.mark.asyncio
async def test_terminate_model_flag(setup_pool):
    pool = setup_pool
    addr = pool.external_address

    worker: xo.ActorRefType["MockWorkerActor"] = await xo.create_actor(
        MockWorkerActor,
        address=addr,
        uid=WorkerActor.uid(),
        supervisor_address="test",
        main_pool=pool,
        cuda_devices=[i for i in range(8)],
    )

    await worker.launch_builtin_model("x1", "x1", None, None, None, n_gpu=1)

    await worker.launch_builtin_model("x2", "x2", None, None, None, n_gpu=4)

    await worker.launch_builtin_model("x3", "x3", None, None, None, n_gpu=4)

    devices = await worker.allocate_devices(5)
    await worker.launch_builtin_model("x4", "x4", None, None, None, n_gpu=5)
    assert devices == [1, 2, 3, 4, 5]

    with pytest.raises(KeyError):
        await worker.terminate_model("x5")

    pool_config = (await get_pool_config(addr)).as_dict()
    assert len(pool_config["pools"]) == 5

    await worker.terminate_model("x2")
    pool_config = (await get_pool_config(addr)).as_dict()
    assert len(pool_config["pools"]) == 4

    gpu_to_model_ids = await worker.get_gpu_to_model_uids()
    for dev in devices:
        assert "x4" in gpu_to_model_ids[dev]
    await worker.terminate_model("x4")
    gpu_to_model_ids = await worker.get_gpu_to_model_uids()
    for dev in devices:
        assert "x4" not in gpu_to_model_ids[dev]
