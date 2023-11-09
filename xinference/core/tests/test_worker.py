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

    def get_gpu_to_model_uid(self):
        return self._gpu_to_model_uid

    def get_gpu_to_embedding_model_uids(self):
        return self._gpu_to_embedding_model_uids

    async def is_model_vllm_backend(self, model_uid):
        for _dev in self._gpu_to_model_uid:
            if model_uid == self._gpu_to_model_uid[_dev]:
                return True
        return False

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
        subpool_address, devices = await self._create_subpool(
            model_uid, model_type, n_gpu=n_gpu
        )
        self._model_uid_to_addr[model_uid] = subpool_address

    async def terminate_model(self, model_uid: str):
        self.release_devices(model_uid)

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

    devices = await worker.allocate_devices(model_uid="mock_model_1", n_gpu=1)
    assert devices == [0]

    devices = await worker.allocate_devices(model_uid="mock_model_2", n_gpu=4)
    assert devices == [1, 2, 3, 4]

    devices = await worker.allocate_devices(model_uid="mock_model_3", n_gpu=3)
    assert devices == [5, 6, 7]

    with pytest.raises(RuntimeError):
        await worker.allocate_devices(model_uid="mock_model_4", n_gpu=5)


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
        assert "model_model_3" == gpu_to_model_id[dev]
    await worker.terminate_model("model_model_3")

    gpu_to_model_id = await worker.get_gpu_to_model_uid()
    for dev in devices:
        assert dev not in gpu_to_model_id


@pytest.mark.asyncio
async def test_launch_embedding_model(setup_pool):
    pool = setup_pool
    addr = pool.external_address

    worker: xo.ActorRefType["MockWorkerActor"] = await xo.create_actor(
        MockWorkerActor,
        address=addr,
        uid=WorkerActor.uid(),
        supervisor_address="test",
        main_pool=pool,
        cuda_devices=[i for i in range(4)],
    )

    # test embedding device candidates 1
    await worker.launch_builtin_model(
        "model_model_1", "mock_model_name", None, None, None, n_gpu=3
    )
    embedding_info = await worker.get_gpu_to_embedding_model_uids()
    assert len(embedding_info) == 0

    await worker.launch_builtin_model(
        "model_model_2", "mock_model_name", None, None, None, "embedding", n_gpu=1
    )

    embedding_info = await worker.get_gpu_to_embedding_model_uids()
    assert 3 in embedding_info
    assert len(embedding_info[3]) == 1
    assert "model_model_2" in embedding_info[3]

    # test terminate LLM model, then launch embedding model
    await worker.terminate_model("model_model_1")
    await worker.launch_builtin_model(
        "model_model_3", "mock_model_name", None, None, None, "embedding", n_gpu=1
    )
    embedding_info = await worker.get_gpu_to_embedding_model_uids()
    assert 0 in embedding_info
    assert len(embedding_info[0]) == 1
    assert "model_model_3" in embedding_info[0]

    await worker.terminate_model("model_model_2")
    await worker.terminate_model("model_model_3")
    embedding_info = await worker.get_gpu_to_embedding_model_uids()
    assert len(embedding_info[0]) == 0
    assert len(embedding_info[3]) == 0

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
    assert 2 in embedding_info
    assert 3 in embedding_info
    assert len(embedding_info[2]) == 1
    assert len(embedding_info[3]) == 1
    assert "model_model_2" in embedding_info[2]
    assert "model_model_3" in embedding_info[3]

    await worker.launch_builtin_model(
        "model_model_4", "mock_model_name", None, None, None, "embedding", n_gpu=1
    )
    assert len(embedding_info[2]) == 2
    assert len(embedding_info[3]) == 1
    assert "model_model_2" in embedding_info[2]
    assert "model_model_4" in embedding_info[2]
    assert "model_model_3" in embedding_info[3]

    for i in range(1, 5):
        await worker.terminate_model(f"model_model_{i}")
    assert len(embedding_info[2]) == 0
    assert len(embedding_info[3]) == 0

    # test no slots
    for i in range(1, 5):
        await worker.launch_builtin_model(
            f"model_model_{i}", "mock_model_name", None, None, None, n_gpu=1
        )
    with pytest.raises(RuntimeError):
        await worker.launch_builtin_model(
            "model_model_5", "mock_model_name", None, None, None, "embedding", n_gpu=1
        )
    # launch CPU would work
    await worker.launch_builtin_model(
        "model_model_5", "mock_model_name", None, None, None, "embedding", n_gpu=None
    )
    for i in range(1, 6):
        await worker.terminate_model(f"model_model_{i}")
