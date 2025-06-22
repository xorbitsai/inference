# Copyright 2022-2025 XProbe Inc.
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
import io
import json
import os
import platform
import sys
from unittest.mock import patch

import pytest
import pytest_asyncio
import xoscar as xo

from .....core.model import ModelActor
from ... import BUILTIN_LLM_FAMILIES
from ..core import MLXChatModel
from ..distributed_models.core import ReceiverActor


@pytest_asyncio.fixture
async def setup_pool():
    pool = await xo.create_actor_pool(
        f"test://127.0.0.1:{xo.utils.get_next_port()}", n_process=2
    )
    async with pool:
        yield pool


async def patched_send(self, data):
    # test actor pool will not do actual serialziatin
    # force to do it
    from xoscar.aio.file import AioFileObject
    from xoscar.serialization import AioSerializer

    fobj = io.BytesIO()
    buffers = await AioSerializer(data).run()
    f = AioFileObject(fobj)
    for b in buffers:
        await f.write(b)

    self._recv_queue.put_nowait(fobj.getvalue())


async def patched_recv(self):
    from xoscar.aio.file import AioFileObject
    from xoscar.serialization import AioDeserializer

    bytes = await self._recv_queue.get()
    fobj = io.BytesIO(bytes)
    f = AioFileObject(fobj)
    return await AioDeserializer(f).run()


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.processor() != "arm",
    reason="MLX only works for Apple silicon chip",
)
@pytest.mark.asyncio
@patch.object(ReceiverActor, "send", patched_send)
@patch.object(ReceiverActor, "recv", patched_recv)
async def test_distributed_mlx_model(setup_pool):
    from huggingface_hub import snapshot_download

    pool = setup_pool

    model_path = snapshot_download("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

    model_family = next(
        n for n in BUILTIN_LLM_FAMILIES if n.model_name == "qwen2.5-instruct"
    )
    model_spec = next(
        s
        for s in model_family.model_specs
        if s.model_format == "mlx"
        and s.model_size_in_billions == "0_5"
        and "4bit" in s.quantizations
    )

    addresses = pool

    class MockModelActor(ModelActor):
        def __init__(self, shard: int, addr: str, shard_0_addr: str = None):
            if shard == 0:
                driver_info = {"address": addr}
            else:
                driver_info = {"address": shard_0_addr}
            model = MLXChatModel(
                "qwen2.5-instruct-0",
                model_family,
                model_spec,
                "4bit",
                model_path,
                {
                    "address": addr,
                    "n_worker": 2,
                    "shard": shard,
                    "driver_info": driver_info,
                },
            )

            super().__init__(
                None,
                addr,
                model,
                "qwen2.5-instruct",
                n_worker=2,
                shard=shard,
                driver_info=driver_info,
            )

        async def record_metrics(self, *args, **kwargs):
            # patch method
            pass

    addresses = list(pool.sub_processes)
    shard0_ref = await xo.create_actor(
        MockModelActor,
        0,
        addresses[0],
        address=pool.external_address,
        uid="qwen2.5-instruct-0",
        allocate_strategy=xo.allocate_strategy.ProcessIndex(1),
    )
    assert shard0_ref.address == addresses[0]
    shard1_ref = await xo.create_actor(
        MockModelActor,
        1,
        addresses[1],
        shard_0_addr=shard0_ref.address,
        address=pool.external_address,
        uid="qwen2.5-instruct-0",
        allocate_strategy=xo.allocate_strategy.ProcessIndex(2),
    )
    assert shard0_ref.address != shard1_ref.address
    await asyncio.gather(shard0_ref.load(), shard1_ref.load())
    await asyncio.gather(shard0_ref.wait_for_load(), shard1_ref.wait_for_load())

    for stream in [False, True]:
        result = await shard0_ref.chat(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hello! How can I assist you today?"},
                {"role": "user", "content": "write a poem"},
            ],
            generate_config={"stream": stream, "max_tokens": 20},
        )
        if stream:
            async for chunk in result:
                assert chunk
        else:
            result = json.loads(result)
            assert result["choices"][0]["message"] is not None
