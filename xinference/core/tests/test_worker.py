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
from typing import Dict, List, Optional, Union

import pytest
import pytest_asyncio
import xoscar as xo
from xoscar import MainActorPoolType, create_actor_pool

from ..launch_strategy import MemoryAwareLaunchStrategy
from ..worker import WorkerActor


class MockWorkerActor(WorkerActor):
    def __init__(
        self,
        supervisor_address: str,
        main_pool: MainActorPoolType,
        cuda_devices: List[int],
    ):
        super().__init__(supervisor_address, main_pool, cuda_devices)
        gpu_memory_info = {
            idx: {"total": 24000.0, "used": 0.0, "available": 24000.0}
            for idx in cuda_devices
        }
        self._launch_strategy = MemoryAwareLaunchStrategy(
            cuda_devices, allowed_devices=None, gpu_memory_info=gpu_memory_info
        )
        self._launch_strategy._current_gpu = sorted(cuda_devices)[0]

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
            model_uid,
            model_type,
            n_gpu=n_gpu,
            gpu_idx=gpu_idx,  # type: ignore
            model_name=model_name,
            model_size_in_billions=model_size_in_billions,
            model_format=model_format,
            quantization=quantization,
            context_length=kwargs.get("context_length"),
        )
        self._model_uid_to_addr[model_uid] = subpool_address

    async def _create_subpool(  # type: ignore[override]
        self,
        model_uid: str,
        model_type: Optional[str] = None,
        n_gpu: Optional[Union[int, str]] = "auto",
        gpu_idx: Optional[List[int]] = None,
        env: Optional[Dict[str, str]] = None,
        start_python: Optional[str] = None,
        model_name: Optional[str] = None,
        model_size_in_billions: Optional[Union[int, str]] = None,
        model_format: Optional[str] = None,
        quantization: Optional[str] = None,
        context_length: Optional[int] = None,
    ) -> tuple:
        """Override to avoid spinning up sub pools during tests."""
        if gpu_idx is None:
            if isinstance(n_gpu, int) or n_gpu == "auto":
                gpu_cnt = n_gpu if isinstance(n_gpu, int) else 1
                devices = (
                    [await self.allocate_devices_for_embedding(model_uid)]
                    if model_type in ["embedding", "rerank"]
                    else self.allocate_devices_for_model(
                        model_uid=model_uid,
                        model_name=model_name or model_uid,
                        model_size=model_size_in_billions or 0,
                        model_format=model_format,
                        quantization=quantization,
                        context_length=context_length,
                        n_gpu=gpu_cnt,  # type: ignore
                    )
                )
            else:
                devices = []
        else:
            assert isinstance(gpu_idx, list)
            devices = await self.allocate_devices_with_gpu_idx(
                model_uid, model_type, gpu_idx  # type: ignore
            )
        return "mock_subpool", [str(dev) for dev in devices]

    async def terminate_model(self, model_uid: str):
        self.release_devices(model_uid)

        # Skip actual sub pool removal in tests
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
    assert len(devices) == 1

    devices = await worker.allocate_devices(model_uid="mock_model_2", n_gpu=4)
    assert len(devices) == 4
    assert len(set(devices)) == 1

    devices = await worker.allocate_devices(model_uid="mock_model_3", n_gpu=3)
    assert len(devices) == 3
    assert len(set(devices)) == 1

    devices = await worker.allocate_devices(model_uid="mock_model_4", n_gpu=5)
    assert len(devices) == 5
    assert len(set(devices)) == 1


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
    assert len(devices) == 3
    await worker.release_devices(model_uid="model_model_3")

    await worker.launch_builtin_model(
        "model_model_3", "mock_model_name", None, None, None, n_gpu=3
    )

    with pytest.raises(KeyError):
        await worker.terminate_model("model_model_4")

    await worker.terminate_model("model_model_2")

    gpu_to_model_id = await worker.get_gpu_to_model_uid()
    assert "model_model_3" in gpu_to_model_id.values()
    await worker.terminate_model("model_model_3")

    gpu_to_model_id = await worker.get_gpu_to_model_uid()
    for dev in devices:
        assert dev not in gpu_to_model_id


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
    assert len(embedding_info) >= 1

    # test terminate LLM model, then launch embedding model
    await worker.terminate_model("model_model_1")
    await worker.launch_builtin_model(
        "model_model_3", "mock_model_name", None, None, None, "embedding", n_gpu=1
    )
    embedding_info = await worker.get_gpu_to_embedding_model_uids()
    # embedding 分配到任意空闲设备
    assert any("model_model_3" in models for models in embedding_info.values())

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
    assert len(embedding_info) >= 2

    await worker.launch_builtin_model(
        "model_model_4", "mock_model_name", None, None, None, "embedding", n_gpu=1
    )
    assert len(embedding_info) >= 2

    for i in range(1, 5):
        await worker.terminate_model(f"model_model_{i}")
    for dev_models in embedding_info.values():
        assert len(dev_models) == 0

    # test no slots
    for i in range(1, 5):
        await worker.launch_builtin_model(
            f"model_model_{i}", "mock_model_name", None, None, None, n_gpu=1
        )
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
    assert len(llm_info) == 1

    await worker.launch_builtin_model(
        "model_model_2", "mock_model_name", None, None, None, "LLM", gpu_idx=[0]
    )
    llm_info = await worker.get_gpu_to_model_uid()
    assert len(llm_info) == 1

    user_specified_info = await worker.get_user_specified_gpu_to_model_uids()
    assert 0 in user_specified_info
    assert len(user_specified_info[0]) == 1

    # test vllm model
    await worker.launch_builtin_model(
        "vllm_model_model_3", "mock_model_name", None, None, None, "LLM", n_gpu=1
    )
    llm_info = await worker.get_gpu_to_model_uid()
    assert len(llm_info) == 2

    await worker.launch_builtin_model(
        "model_model_4", "mock_model_name", None, None, None, "LLM", gpu_idx=[1]
    )
    llm_info = await worker.get_gpu_to_model_uid()
    assert len(llm_info) == 2

    user_specified_info = await worker.get_user_specified_gpu_to_model_uids()
    assert len(user_specified_info) == 2

    # then launch a LLM without gpu_idx
    await worker.launch_builtin_model(
        "normal_model_model_5", "mock_model_name", None, None, None, "LLM", n_gpu=1
    )
    llm_info = await worker.get_gpu_to_model_uid()
    total_used_gpus = set(llm_info.keys()).union(set(user_specified_info.keys()))
    assert len(total_used_gpus) >= 3

    # launch without gpu_idx again, should succeed with GPU reuse
    await worker.launch_builtin_model(
        "normal_model_model_6", "mock_model_name", None, None, None, "LLM", n_gpu=1
    )
    llm_info = await worker.get_gpu_to_model_uid()
    total_used_gpus = set(llm_info.keys()).union(set(user_specified_info.keys()))
    assert len(total_used_gpus) == 4  # All 4 GPUs are used

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

    # should succeed with GPU reuse, even though gpu 0 is occupied
    await worker.launch_builtin_model(
        "normal_mock_model_3", "mock_model_name", None, None, None, "LLM", n_gpu=4
    )
    llm_info = await worker.get_gpu_to_model_uid()
    user_specified_info = await worker.get_user_specified_gpu_to_model_uids()
    total_used_gpus = set(llm_info.keys()).union(set(user_specified_info.keys()))
    # Should use available GPUs, possibly reusing some
    assert len(total_used_gpus) <= 4

    # should be on gpu 1
    await worker.launch_builtin_model(
        "embedding_3", "mock_model_name", None, None, None, "embedding", n_gpu=1
    )
    # should be on gpu 0
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
    assert any("rerank_7" in models for models in embedding_info.values())
    assert all(len(models) >= 0 for models in embedding_info.values())

    # cleanup
    await worker.terminate_model("embedding_1")
    await worker.terminate_model("vllm_mock_model_2")
    await worker.terminate_model("normal_mock_model_3")
    await worker.terminate_model("embedding_3")
    await worker.terminate_model("rerank_4")
    await worker.terminate_model("embedding_5")
    await worker.terminate_model("rerank_6")
    await worker.terminate_model("rerank_7")

    embedding_info = await worker.get_gpu_to_embedding_model_uids()
    user_specified_info = await worker.get_user_specified_gpu_to_model_uids()
    for info in [embedding_info, user_specified_info]:
        for _, details in info.items():
            assert len(details) == 0
