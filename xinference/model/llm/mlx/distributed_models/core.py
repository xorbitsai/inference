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
import logging
import os
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import xoscar as xo
from xoscar.utils import lazy_import

if TYPE_CHECKING:
    import mlx.core as mx
else:
    mx = lazy_import("mlx.core")
logger = logging.getLogger(__name__)

DEBUG_DISTRIBUTED_MLX = bool(int(os.getenv("XINFERENCE_DEBUG_DISTRIBUTED_MLX", "0")))


def _to_wire(result: "mx.array"):
    """Encode an mx.array for the inter-shard exchange.

    Must run on the compute thread: np.array() evaluates and copies there,
    on the stream that owns the pending graph. numpy has no bfloat16 and
    np.array() on a bfloat16 mx.array raises, so ship its raw bits as
    uint16 and let _from_wire restore the dtype.
    """
    if result.dtype == mx.bfloat16:
        return np.array(result.view(mx.uint16)), True
    return np.array(result), False


def _from_wire(payload) -> "mx.array":
    """Decode an inter-shard payload back into an mx.array.

    Must run on the compute thread so the new mx.array belongs to the
    stream that will evaluate it.
    """
    data, is_bf16 = payload
    result = mx.array(data)
    return result.view(mx.bfloat16) if is_bf16 else result


class ReceiverActor(xo.StatelessActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._recv_queue = asyncio.Queue()

    @classmethod
    def gen_uid(cls, uid: str, rank: int):
        return f"Receiver-{uid}-{rank}"

    async def send(self, data):
        # no need to use async function,
        # but make it more convenient to patch this function for test purpose
        # NOTE: the payload is a _to_wire() tuple and is queued untouched.
        # This coroutine runs on the event-loop thread; constructing an
        # mx.array here would bind it to this thread's stream, and
        # evaluating it later from the compute thread fails on mlx>=0.31
        # with 'There is no Stream(gpu, N) in current thread'.
        self._recv_queue.put_nowait(data)

    async def recv(self):
        return await self._recv_queue.get()


class DistributedModelMixin:
    rank: int
    world_size: int
    model_uid: Optional[str]
    address: Optional[str]
    _receiver_ref: Optional[xo.ActorRefType[ReceiverActor]]
    rank_to_addresses: Optional[Dict[int, str]]

    layers: list

    def __init__(self):
        self.rank = 0
        self.world_size = 1
        self.model_uid = None
        self.loop = None
        self.address = None
        # actor ref
        self._receiver_ref = None
        self.rank_to_addresses = None

    def prepare(self):
        coro = xo.create_actor(
            ReceiverActor,
            uid=ReceiverActor.gen_uid(self.model_uid, self.rank),
            address=self.address,
        )
        self._receiver_ref = asyncio.run_coroutine_threadsafe(coro, self.loop).result()
        if DEBUG_DISTRIBUTED_MLX:
            logger.debug("Finish preparing distributed env for rank %s", self.rank)

    def _send_stage_result(self, result: "mx.array"):
        assert self.rank > 0
        assert self.rank_to_addresses is not None
        assert self.model_uid is not None
        last_rank = self.rank - 1
        if DEBUG_DISTRIBUTED_MLX:
            logger.debug(
                "Start to send %s partial result to rank %d", self.model_uid, last_rank
            )

        # Encode on this (compute) thread: mx.array payloads must never
        # cross threads on mlx>=0.31.
        payload = _to_wire(result)

        async def send():
            receiver_ref = await xo.actor_ref(
                uid=ReceiverActor.gen_uid(self.model_uid, last_rank),
                address=self.rank_to_addresses[last_rank],
            )
            return await receiver_ref.send(payload)

        asyncio.run_coroutine_threadsafe(send(), self.loop).result()
        if DEBUG_DISTRIBUTED_MLX:
            logger.debug(
                "Finish send %s partial result to rank %d, shape %s",
                self.model_uid,
                last_rank,
                result.shape,
            )

    def _wait_prev_stage_result(self):
        if DEBUG_DISTRIBUTED_MLX:
            logger.debug("Wait for partial result from prev shard %d", self.rank + 1)
        coro = self._receiver_ref.recv()
        result = asyncio.run_coroutine_threadsafe(coro, self.loop).result()
        # Decode on this (compute) thread so the new mx.array belongs to
        # the stream that will evaluate it.
        result = _from_wire(result)
        if DEBUG_DISTRIBUTED_MLX:
            logger.debug(
                "Received partial result from prev shard %d, shape %s",
                self.rank + 1,
                result.shape,
            )
        return result

    def _broadcast_result(self, result: "mx.array"):
        if DEBUG_DISTRIBUTED_MLX:
            logger.debug("broadcast result from driver")

        # Same as _send_stage_result: encode on the compute thread.
        payload = _to_wire(result)

        async def broadcast(rank: int):
            assert self.model_uid is not None
            assert self.rank_to_addresses is not None

            receiver = await xo.actor_ref(
                uid=ReceiverActor.gen_uid(self.model_uid, rank),
                address=self.rank_to_addresses[rank],
            )
            await receiver.send(payload)

        async def broadcast_all():
            coros = []
            for rank in range(1, self.world_size):
                coros.append(broadcast(rank))
            await asyncio.gather(*coros)

        return asyncio.run_coroutine_threadsafe(broadcast_all(), self.loop).result()

    def _get_result(self) -> "mx.array":
        if DEBUG_DISTRIBUTED_MLX:
            logger.debug("Get result from broadcasted data on self receiver")
        assert self.model_uid is not None
        coro = xo.actor_ref(
            uid=ReceiverActor.gen_uid(self.model_uid, self.rank), address=self.address
        )
        ref = asyncio.run_coroutine_threadsafe(coro, self.loop).result()
        result = asyncio.run_coroutine_threadsafe(ref.recv(), loop=self.loop).result()
        # Decode on this (compute) thread.
        return _from_wire(result)

    def pipeline(self):
        pipeline_size, rank = self.world_size, self.rank
        layers_per_rank = len(self.layers) // pipeline_size
        extra = len(self.layers) - layers_per_rank * pipeline_size
        if self.rank < extra:
            layers_per_rank += 1
        self.start_idx = (pipeline_size - rank - 1) * layers_per_rank
        self.end_idx = self.start_idx + layers_per_rank
        self.layers = self.layers[: self.end_idx]
        self.layers[: self.start_idx] = [None] * self.start_idx
        self.num_layers = len(self.layers) - self.start_idx


class SafeKVCache:
    """
    A safe wrapper around mlx_lm's KVCache that handles None keys gracefully.
    This is needed because mlx_lm's generate function accesses cache.state
    before the cache is properly initialized.
    """

    def __init__(self):
        from mlx_lm.models.cache import KVCache

        self._cache = KVCache()

    @property
    def state(self):
        # Safe access to state property
        if self._cache.keys is None:
            return None, None
        if self._cache.offset == self._cache.keys.shape[2]:
            return self._cache.keys, self._cache.values
        else:
            return (
                self._cache.keys[..., : self._cache.offset, :],
                self._cache.values[..., : self._cache.offset, :],
            )

    @state.setter
    def state(self, v):
        # Safe setter for state property
        if v is None or v[0] is None:
            self._cache.keys = None
            self._cache.values = None
            self._cache.offset = 0
        else:
            self._cache.keys, self._cache.values = v
            self._cache.offset = self._cache.keys.shape[2]

    def __getattr__(self, name):
        # Delegate all other attributes and methods to the underlying cache
        return getattr(self._cache, name)
