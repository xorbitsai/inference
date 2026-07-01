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

import asyncio
import logging
import sys
from functools import partial
from typing import Type

import pytest
import xoscar as xo
from xoscar.utils import lazy_import

from .....device_utils import gpu_count
from ...cache_manager import LLMCacheManager as CacheManager
from ...llm_family import BUILTIN_LLM_FAMILIES

vllm = lazy_import("vllm")


@pytest.fixture
async def actor_pool_context():
    logging.basicConfig(level=logging.DEBUG)
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        yield pool


@pytest.mark.skipif(sys.platform != "linux", reason="Run for linux only")
@pytest.mark.skipif(vllm is None, reason="vllm need to be installed")
@pytest.mark.skipif(gpu_count() < 2, reason="At lease 2 gpus required to test")
async def test_distributed_executor(actor_pool_context):
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine as _AsyncLLMEngine
    from vllm.executor.executor_base import ExecutorBase
    from vllm.sampling_params import SamplingParams

    from ..distributed_executor import XinferenceDistributedExecutor

    class AsyncLLMEngine(_AsyncLLMEngine):
        @classmethod
        def _get_executor_cls(cls, engine_config: VllmConfig) -> Type[ExecutorBase]:
            return partial(  # type: ignore
                XinferenceDistributedExecutor,
                pool_addresses=[worker_addr1, worker_addr2],
                n_worker=1,
                loop=loop,
            )

    pool = actor_pool_context

    worker_addr1 = await pool.append_sub_pool(env={"CUDA_VISIBLE_DEVICES": "0,1"})
    worker_addr2 = await pool.append_sub_pool(env={"CUDA_VISIBLE_DEVICES": "0,1"})
    loop = asyncio.get_running_loop()

    llm_family = next(
        f for f in BUILTIN_LLM_FAMILIES if f.model_name == "qwen2.5-instruct"
    )
    llm_family = llm_family.copy()
    spec = next(
        s
        for s in llm_family.model_specs
        if s.model_size_in_billions == 7 and s.model_format == "pytorch"
    )
    llm_family.model_specs = [spec]
    model_path = CacheManager(llm_family).cache()

    def load(tp: bool = True):
        if tp:
            kwargs = {"tensor_parallel_size": 2}
        else:
            kwargs = {"pipeline_parallel_size": 2}
        engine_args = AsyncEngineArgs(
            model=model_path, gpu_memory_utilization=0.8, enforce_eager=True, **kwargs
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        return engine

    engine = await asyncio.to_thread(load, tp=True)
    for _ in range(2):
        # test 2 rounds
        outputs = []
        async for output in engine.generate(
            "Hi", SamplingParams(max_tokens=1), None, None
        ):
            outputs.append(output)

        assert len(outputs) == 1

    await asyncio.to_thread(engine.engine.model_executor.shutdown)
