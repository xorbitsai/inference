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
from typing import List, Optional, Union

import pytest
import pytest_asyncio
import xoscar as xo
from xoscar import MainActorPoolType, create_actor_pool, get_pool_config

from ..launch_strategy import IdleFirstLaunchStrategy
from ..utils import merge_virtual_env_packages
from ..worker import WorkerActor


class DeterministicIdleFirstLaunchStrategy(IdleFirstLaunchStrategy):
    def _select_emptiest_gpu(
        self,
        candidates,
        pending_gpu_counts,
        allocated_gpus,
    ):
        """
        Deterministic tie-breaking for tests so we do not rely on real GPU state.
        """
        if not candidates:
            return None

        scored = []
        for dev in candidates:
            available = self._gpu_memory_info.get(dev, {}).get("available", 0)
            penalty = pending_gpu_counts.get(dev, 0) + len(
                allocated_gpus.get(dev, set())
            )
            scored.append((dev, available - penalty))

        # Prefer higher available memory, then the lowest GPU index.
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[0][0]


class MockWorkerActor(WorkerActor):
    def __init__(
        self,
        supervisor_address: str,
        main_pool: MainActorPoolType,
        cuda_devices: List[int],
    ):
        super().__init__(supervisor_address, main_pool, cuda_devices)
        self._test_gpu_memory_info = {
            idx: {"total": 24000.0, "used": 0.0, "available": 24000.0}
            for idx in cuda_devices
        }

    def _create_launch_strategy_instance(self):
        return DeterministicIdleFirstLaunchStrategy(
            self._total_gpu_devices,
            gpu_memory_info=self._test_gpu_memory_info,
            model_spread_used_gpus=self._model_spread_used_gpus,
            active_model_counts=self._active_model_counts,
        )

    async def __post_create__(self):
        pass

    async def __pre_destroy__(self):
        pass

    def get_gpu_to_model_uid(self):
        return self._gpu_to_model_uid

    def get_gpu_to_embedding_model_uids(self):
        return self._gpu_to_embedding_model_uids

    def get_user_specified_gpu_to_model_uids(self):
        return self._user_specified_gpu_to_model_uids

    async def is_model_vllm_backend(self, model_uid):
        if model_uid.startswith("normal_"):
            return False
        if model_uid.startswith("vllm_"):
            return True
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
        gpu_idx: Optional[Union[int, List[int]]] = None,
        **kwargs,
    ):
        subpool_address, devices = await self._create_subpool(
            model_uid, model_type, n_gpu=n_gpu, gpu_idx=gpu_idx  # type: ignore
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

    devices = await worker.allocate_devices(model_uid="mock_model_2", n_gpu=2)
    assert devices == [1, 2]

    devices = await worker.allocate_devices(model_uid="mock_model_3", n_gpu=1)
    assert devices == [3]

    devices = await worker.allocate_devices(model_uid="mock_model_4", n_gpu=1)
    assert devices == [4]


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
    model3_devices = [
        dev for dev, uids in gpu_to_model_id.items() if "model_model_3" in uids
    ]
    assert model3_devices
    for dev in model3_devices:
        assert "model_model_3" in gpu_to_model_id[dev]
    await worker.terminate_model("model_model_3")

    gpu_to_model_id = await worker.get_gpu_to_model_uid()
    for dev in model3_devices:
        assert "model_model_3" not in gpu_to_model_id.get(dev, set())


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
    assert len(embedding_info[1]) == 0

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
    embedding_info = await worker.get_gpu_to_embedding_model_uids()
    assert 2 in embedding_info
    assert 3 in embedding_info
    assert len(embedding_info[2]) == 1
    assert len(embedding_info[3]) == 1
    assert "model_model_2" in embedding_info[2]
    assert "model_model_3" in embedding_info[3]

    await worker.launch_builtin_model(
        "model_model_4", "mock_model_name", None, None, None, "embedding", n_gpu=1
    )
    embedding_info = await worker.get_gpu_to_embedding_model_uids()
    assert len(embedding_info[0]) == 1
    assert len(embedding_info[2]) == 1
    assert len(embedding_info[3]) == 1
    assert "model_model_2" in embedding_info[2]
    assert "model_model_3" in embedding_info[3]
    assert "model_model_4" in embedding_info[0]

    for i in range(1, 5):
        await worker.terminate_model(f"model_model_{i}")
    embedding_info = await worker.get_gpu_to_embedding_model_uids()
    assert len(embedding_info[2]) == 0
    assert len(embedding_info[3]) == 0

    # test no slots
    for i in range(1, 5):
        await worker.launch_builtin_model(
            f"model_model_{i}", "mock_model_name", None, None, None, n_gpu=1
        )
    await worker.launch_builtin_model(
        "model_model_5", "mock_model_name", None, None, None, "embedding", n_gpu=1
    )
    # launch CPU would work
    await worker.launch_builtin_model(
        "model_model_5", "mock_model_name", None, None, None, "embedding", n_gpu=None
    )
    for i in range(1, 6):
        await worker.terminate_model(f"model_model_{i}")


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
    assert 0 in llm_info
    assert "normal_model_model_1" in llm_info[0]

    await worker.launch_builtin_model(
        "model_model_2", "mock_model_name", None, None, None, "LLM", gpu_idx=[0]
    )
    llm_info = await worker.get_gpu_to_model_uid()
    assert 0 in llm_info
    assert "model_model_2" in llm_info[0]

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
    vllm_gpu = next(
        dev for dev, uids in llm_info.items() if "vllm_model_model_3" in uids
    )
    assert vllm_gpu != 0

    with pytest.raises(RuntimeError):
        await worker.launch_builtin_model(
            "model_model_4",
            "mock_model_name",
            None,
            None,
            None,
            "LLM",
            gpu_idx=[vllm_gpu],
        )

    target_gpu = next(dev for dev in [1, 2, 3] if dev != vllm_gpu)
    await worker.launch_builtin_model(
        "model_model_4",
        "mock_model_name",
        None,
        None,
        None,
        "LLM",
        gpu_idx=[target_gpu],
    )
    llm_info = await worker.get_gpu_to_model_uid()
    assert target_gpu in llm_info
    assert "model_model_4" in llm_info[target_gpu]

    user_specified_info = await worker.get_user_specified_gpu_to_model_uids()
    assert len(user_specified_info) == 2
    assert 0 in user_specified_info
    assert target_gpu in user_specified_info
    assert len(user_specified_info[target_gpu]) == 1
    assert list(user_specified_info[target_gpu])[0][0] == "model_model_4"
    assert list(user_specified_info[target_gpu])[0][1] == "LLM"

    # then launch a LLM without gpu_idx
    await worker.launch_builtin_model(
        "normal_model_model_5", "mock_model_name", None, None, None, "LLM", n_gpu=1
    )
    llm_info = await worker.get_gpu_to_model_uid()
    assert 0 in llm_info

    # launch without gpu_idx again, error
    await worker.launch_builtin_model(
        "normal_model_model_6", "mock_model_name", None, None, None, "LLM", n_gpu=1
    )

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
    embedding_info = await worker.get_gpu_to_embedding_model_uids()
    assert len(embedding_info) == 1
    assert 0 in embedding_info

    await worker.launch_builtin_model(
        "vllm_mock_model_2", "mock_model_name", None, None, None, "LLM", gpu_idx=[0]
    )
    embedding_info = await worker.get_gpu_to_embedding_model_uids()
    assert len(embedding_info) == 1
    assert 0 in embedding_info

    user_specified_info = await worker.get_user_specified_gpu_to_model_uids()
    assert len(user_specified_info[0]) == 1
    assert list(user_specified_info[0])[0][0] == "vllm_mock_model_2"
    assert list(user_specified_info[0])[0][1] == "LLM"

    # never choose gpu 0 again
    devices = await worker.allocate_devices(model_uid="normal_mock_model_3", n_gpu=4)
    assert all(dev != 0 for dev in devices)

    # should be on gpu 1
    await worker.launch_builtin_model(
        "embedding_3", "mock_model_name", None, None, None, "embedding", n_gpu=1
    )
    # should be on gpu 0
    with pytest.raises(RuntimeError):
        await worker.launch_builtin_model(
            "rerank_4", "mock_model_name", None, None, None, "rerank", gpu_idx=[0]
        )
    # should be on gpu 2
    await worker.launch_builtin_model(
        "embedding_5", "mock_model_name", None, None, None, "embedding", n_gpu=1
    )
    # should be on gpu 3
    await worker.launch_builtin_model(
        "rerank_6", "mock_model_name", None, None, None, "rerank", n_gpu=1
    )
    # should be on gpu 1, due to there are the fewest models on it
    await worker.launch_builtin_model(
        "rerank_7", "mock_model_name", None, None, None, "rerank", n_gpu=1
    )
    embedding_info = await worker.get_gpu_to_embedding_model_uids()
    user_specified_info = await worker.get_user_specified_gpu_to_model_uids()
    rerank7_gpu = next(
        dev for dev, uids in embedding_info.items() if "rerank_7" in uids
    )
    assert rerank7_gpu != 0
    assert len(embedding_info[0]) == 1
    assert len(user_specified_info[0]) == 1
    assert len(user_specified_info[rerank7_gpu]) == 0

    # cleanup
    await worker.terminate_model("embedding_1")
    await worker.terminate_model("vllm_mock_model_2")
    await worker.terminate_model("embedding_3")
    await worker.terminate_model("embedding_5")
    await worker.terminate_model("rerank_6")
    await worker.terminate_model("rerank_7")

    embedding_info = await worker.get_gpu_to_embedding_model_uids()
    user_specified_info = await worker.get_user_specified_gpu_to_model_uids()
    for info in [embedding_info, user_specified_info]:
        for dev, details in info.items():
            assert len(details) == 0
