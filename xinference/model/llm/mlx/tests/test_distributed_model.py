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
import platform
import sys
from pathlib import Path

import pytest
import pytest_asyncio
import xoscar as xo


class ModelActor(xo.StatelessActor):
    def __init__(self, rank: int, model_uid: str, model_path: str):
        super().__init__()
        self.rank = rank
        self.model_uid = model_uid
        self.model_path = model_path
        self.loop = asyncio.get_running_loop()

    def set_rank_addresses(self, rank_addresses):
        self.rank_to_addresses = rank_addresses

    def _load(self):
        import mlx.core as mx
        from mlx_lm.utils import load_model, load_tokenizer

        from ..distributed_models.core import SafeKVCache
        from ..distributed_models.qwen2 import Model, ModelArgs

        get_class = lambda *_, **__: (Model, ModelArgs)

        self.model, config = load_model(
            Path(self.model_path), lazy=True, get_model_classes=get_class
        )
        model = self.model.model
        model.rank = self.rank
        model.world_size = 2
        model.model_uid = self.model_uid
        model.loop = self.loop
        model.address = self.address
        model.rank_to_addresses = self.rank_to_addresses

        model.prepare()
        model.pipeline()
        mx.eval(model.parameters())

        self.tokenizer = load_tokenizer(
            Path(self.model_path), {}, eos_token_ids=config.get("eos_token_id", None)
        )

        # Patch the model's make_cache method to use SafeKVCache
        def make_safe_cache():
            num_layers = len(model.layers)
            return [SafeKVCache() for _ in range(num_layers)]

        self.model.make_cache = make_safe_cache

    async def load(self):
        return await asyncio.to_thread(self._load)

    def _generate(self, prompt: str, **kwargs):
        from mlx_lm.generate import generate

        messages = [{"role": "user", "content": prompt}]
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

        return generate(self.model, self.tokenizer, prompt, **kwargs)

    async def generate(self, prompt: str, **kwargs):
        return await asyncio.to_thread(self._generate, prompt, **kwargs)


# xo.create_actor_pool/create_actor fork worker subprocesses without any
# built-in timeout. On a loaded CI runner this can hit a fork+GIL race and
# hang forever (seen hanging until pytest's own --timeout, tens of minutes
# later, with every thread blocked in os.waitpid). Wrap each call so a stuck
# fork fails fast and points at the actual stage instead of a generic
# pytest-timeout with no context.
_POOL_CREATE_TIMEOUT = 90
_ACTOR_CREATE_TIMEOUT = 60


@pytest_asyncio.fixture
async def setup_pool():
    pool = await asyncio.wait_for(
        xo.create_actor_pool(f"127.0.0.1:{xo.utils.get_next_port()}", n_process=2),
        timeout=_POOL_CREATE_TIMEOUT,
    )
    async with pool:
        yield pool


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.processor() != "arm",
    reason="MLX only works for Apple silicon chip",
)
def test_wire_roundtrip():
    """The inter-shard exchange must survive every dtype models ship with,
    including bfloat16, which numpy cannot represent directly."""
    import mlx.core as mx

    from ..distributed_models.core import _from_wire, _to_wire

    for dtype in (mx.bfloat16, mx.float16, mx.float32):
        arr = mx.random.normal((3, 5)).astype(dtype)
        restored = _from_wire(_to_wire(arr))
        assert restored.dtype == arr.dtype
        assert mx.array_equal(restored, arr).item()


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.processor() != "arm",
    reason="MLX only works for Apple silicon chip",
)
@pytest.mark.asyncio
async def test_distributed(setup_pool):
    from huggingface_hub import snapshot_download

    pool = setup_pool

    model_path = snapshot_download("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

    shard0 = await asyncio.wait_for(
        xo.create_actor(
            ModelActor,
            0,
            "qwen2.5-instruct",
            model_path,
            address=pool.external_address,
            allocate_strategy=xo.allocate_strategy.ProcessIndex(1),
            uid="model_0",
        ),
        timeout=_ACTOR_CREATE_TIMEOUT,
    )
    shard1 = await asyncio.wait_for(
        xo.create_actor(
            ModelActor,
            1,
            "qwen2.5-instruct",
            model_path,
            address=pool.external_address,
            allocate_strategy=xo.allocate_strategy.ProcessIndex(2),
            uid="model_1",
        ),
        timeout=_ACTOR_CREATE_TIMEOUT,
    )
    rank_addresses = {0: shard0.address, 1: shard1.address}
    await shard0.set_rank_addresses(rank_addresses)
    await shard1.set_rank_addresses(rank_addresses)

    await shard0.load()
    await shard1.load()

    t1 = asyncio.create_task(shard0.generate("hello", max_tokens=3))
    t2 = asyncio.create_task(shard1.generate("hello", max_tokens=3))
    await asyncio.sleep(0)
    # gather() surfaces the first shard failure immediately. Awaiting the
    # shards one by one deadlocks instead: when one shard dies mid-pipeline,
    # its peer blocks forever on the tensor exchange, and the exception is
    # never observed (seen as a 50-minute silent hang on CI).
    result, _ = await asyncio.gather(t1, t2)

    assert result is not None
