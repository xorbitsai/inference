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
from functools import lru_cache
from queue import Queue
from typing import Dict, List, Optional, no_type_check

import torch
import xoscar as xo
from vllm.core.scheduler import Scheduler
from vllm.utils import TORCH_DTYPE_TO_NUMPY_DTYPE, Device
from vllm.worker.cache_engine import CacheEngine

from .collective import CollectiveRank

logger = logging.getLogger(__name__)


class BufferTransferMixin:
    def __init__(self):
        self.num_buffer: int = 0  # type: ignore
        self.buffers: List[torch.Tensor] = []  # type: ignore
        self.buffer_queue: Optional[Queue] = None  # type: ignore
        self.transfer_block_num = 0
        self.num_attn_layers = 0

    def init_buffer(
        self, num_buffer: int, buffer_shape, buffer_dtype, buffer_device, pin_memory
    ):
        # (transfer_block_num, num_attn_layers, 2, *kv_cache_shape[2:])

        if buffer_dtype is torch.bfloat16:
            buffer_dtype = torch.float16

        self.num_buffer = num_buffer
        self.transfer_block_num = buffer_shape[0]
        self.num_attn_layers = buffer_shape[1]

        self.buffers = [
            torch.zeros(
                size=buffer_shape,
                dtype=buffer_dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            for _ in range(self.num_buffer)
        ]

        self.buffer_queue = Queue()
        for i in range(self.num_buffer):
            self.buffer_queue.put_nowait(i)
        logger.debug(
            f"Init buffer done. "
            f"transfer_block_num: {self.transfer_block_num}, "
            f"num_buffer: {self.num_buffer}, "
            f"buffer_dtype: {buffer_dtype}, "
            f"buffer_shape: {buffer_shape}"
        )

    @no_type_check
    def get_buffer_index(self) -> int:
        return self.buffer_queue.get()

    @no_type_check
    def free_buffer_index(self, index: int) -> None:
        self.buffer_queue.put_nowait(index)

    def get_swap_buffer(self, index: int, num_blocks: int) -> torch.Tensor:
        buf = self.buffers[index]
        buffer = buf[:num_blocks].view(
            self.num_attn_layers, 2, num_blocks, *buf.shape[3:]
        )
        return buffer

    @lru_cache(maxsize=None)
    def get_gloo_dtype(self, input_dtype: torch.dtype):
        from xoscar.collective.common import TypeMappingGloo

        return TypeMappingGloo[TORCH_DTYPE_TO_NUMPY_DTYPE[input_dtype]]


class TransferActor(xo.StatelessActor, BufferTransferMixin, CollectiveRank):
    @classmethod
    def default_uid(cls):
        return f"vllm-transfer-actor"

    def __init__(
        self,
        rank: int,
        world_size: int,
        rank_address: str,
        store_address: str,
        store_port: int,
        world_addresses: List[str],
    ):
        super().__init__()
        CollectiveRank.__init__(
            self,
            rank,
            world_size,
            rank_address,
            store_address,
            store_port,
            world_addresses,
        )
        self._cache_engine: Optional[List[CacheEngine]] = None
        self._scheduler: Optional[List[Scheduler]] = None
        self._swap_stream = torch.cuda.Stream()

    async def __post_create__(self):
        self.init_rank()

    def setup(
        self,
        cache_engine: List[CacheEngine],
        scheduler: List[Scheduler],
        num_buffer: int,
        buffer_shape,
        buffer_dtype,
        buffer_device,
        pin_memory: bool,
    ):
        self._cache_engine = cache_engine
        self._scheduler = scheduler
        self.init_buffer(
            num_buffer, buffer_shape, buffer_dtype, buffer_device, pin_memory
        )

    async def __pre_destroy__(self):
        self._context.closeConnections()

    def _get_cache_engine(self, virtual_engine: int) -> CacheEngine:
        return self._cache_engine[virtual_engine]  # type: ignore

    @staticmethod
    def _get_swap_block_ids(src_to_dst: Dict[int, int], is_sender: bool) -> List[int]:
        return list(sorted([r if is_sender else l for r, l in src_to_dst.items()]))

    def _swap_out_to_buffer(
        self, cache_engine: CacheEngine, cpu_buf_index: int, block_ids: List[int]
    ) -> torch.Tensor:
        num_blocks = len(block_ids)
        src_to_dst = torch.tensor(
            [(block_num, idx) for idx, block_num in enumerate(block_ids)],
            device="cpu",
            dtype=torch.int64,
        ).view(-1, 2)
        cpu_buf = self.get_swap_buffer(cpu_buf_index, num_blocks)
        with torch.cuda.stream(self._swap_stream):
            for i in range(self.num_attn_layers):
                cache_engine.attn_backend.swap_blocks(
                    cache_engine.gpu_cache[i], cpu_buf[i], src_to_dst
                )
        torch.cuda.Stream.synchronize(self._swap_stream)
        return cpu_buf

    def _swap_in_from_buffer(
        self, cache_engine: CacheEngine, cpu_buf: torch.Tensor, block_ids: List[int]
    ) -> None:
        src_to_dst = torch.tensor(
            [(idx, block_num) for idx, block_num in enumerate(block_ids)],
            device="cpu",
            dtype=torch.int64,
        ).view(-1, 2)
        with torch.cuda.stream(self._swap_stream):
            for i in range(self.num_attn_layers):
                cache_engine.attn_backend.swap_blocks(
                    cpu_buf[i], cache_engine.gpu_cache[i], src_to_dst
                )
        torch.cuda.Stream.synchronize(self._swap_stream)

    def _incr_count_for_block_id(self, virtual_engine: int, block_ids: List[int]):
        """
        The reference count of the `block_id` involved in the transfer is incremented by 1
        to ensure it is not reclaimed.
        """
        scheduler = self._scheduler[virtual_engine]  # type: ignore
        gpu_allocator = scheduler.block_manager.block_allocator._allocators[Device.GPU]

        for _id in block_ids:
            gpu_allocator._refcounter.incr(_id)

    def _decr_count_for_block_id(self, virtual_engine: int, block_ids: List[int]):
        """
        After the transfer, the reference count is decremented by 1.
        """
        scheduler = self._scheduler[virtual_engine]  # type: ignore
        gpu_allocator = scheduler.block_manager.block_allocator._allocators[Device.GPU]

        for _id in block_ids:
            gpu_allocator._refcounter.decr(_id)

    async def do_send(
        self, virtual_engine: int, to_rank: int, src_to_dst: Dict[int, int]
    ):
        """
        Sending logic: GPU -> Buffer -> Gloo send.
        GPU -> Buffer is directly handled using the internal `swap_out` interface of vllm.
        """
        from xoscar.collective import xoscar_pygloo as xp

        cache_engine = self._get_cache_engine(virtual_engine)

        block_ids = self._get_swap_block_ids(src_to_dst, is_sender=True)
        self._incr_count_for_block_id(virtual_engine, block_ids)
        cpu_buf_index = self.get_buffer_index()
        total_blocks: int = len(block_ids)

        try:
            for start_idx in range(0, total_blocks, self.transfer_block_num):
                offset = min(self.transfer_block_num, total_blocks - start_idx)
                send_block_ids = block_ids[start_idx : start_idx + offset]
                sendbuf = self._swap_out_to_buffer(
                    cache_engine, cpu_buf_index, send_block_ids
                )
                assert sendbuf.is_contiguous()
                sendptr = sendbuf.numpy().ctypes.data
                data_size = sendbuf.numel()
                datatype = self.get_gloo_dtype(sendbuf.dtype)
                peer = to_rank
                xp.send(self._context, sendptr, data_size, datatype, peer)
        finally:
            self._decr_count_for_block_id(virtual_engine, block_ids)
            self.free_buffer_index(cpu_buf_index)

    async def do_recv(
        self, virtual_engine: int, from_rank: int, src_to_dst: Dict[int, int]
    ):
        """
        Receiving logic: Gloo recv -> Buffer -> GPU.
        Buffer -> GPU is directly handled using the internal `swap_in` interface of vllm.
        """
        from xoscar.collective import xoscar_pygloo as xp

        cache_engine = self._get_cache_engine(virtual_engine)

        block_ids = self._get_swap_block_ids(src_to_dst, is_sender=False)
        self._incr_count_for_block_id(virtual_engine, block_ids)
        total_blocks = len(block_ids)
        cpu_buf_index = self.get_buffer_index()

        try:
            for start_idx in range(0, total_blocks, self.transfer_block_num):
                offset = min(self.transfer_block_num, total_blocks - start_idx)
                recv_block_ids = block_ids[start_idx : start_idx + offset]
                recvbuf = self.get_swap_buffer(cpu_buf_index, len(recv_block_ids))
                assert recvbuf.is_contiguous()
                recvptr = recvbuf.numpy().ctypes.data
                data_size = recvbuf.numel()
                datatype = self.get_gloo_dtype(recvbuf.dtype)
                peer = from_rank
                xp.recv(self._context, recvptr, data_size, datatype, peer)

                self._swap_in_from_buffer(cache_engine, recvbuf, recv_block_ids)
        finally:
            self._decr_count_for_block_id(virtual_engine, block_ids)
            self.free_buffer_index(cpu_buf_index)

    async def recv(
        self, virtual_engine: int, from_rank: int, src_to_dst: Dict[int, int]
    ):
        """
        This is the external entry point for the call.
        The transfer logic is as follows:
        the receiver requests the sender to send the data directly to itself in a point-to-point manner.
        """
        from_address = self._world_addresses[from_rank]
        sender_ref = await xo.actor_ref(
            address=from_address, uid=f"{TransferActor.default_uid()}-{from_rank}"
        )
        await asyncio.gather(
            sender_ref.do_send(virtual_engine, self._rank, src_to_dst),
            self.do_recv(virtual_engine, from_rank, src_to_dst),
        )


class Rank0TransferActor(xo.StatelessActor, CollectiveRank):
    """
    The Rank 0 transfer actor is only used for constructing the collective communication world,
    so it only needs to inherit the `CollectiveWorld` class.
    """

    @classmethod
    def default_uid(cls):
        return f"vllm-transfer-actor"

    def __init__(
        self,
        rank: int,
        world_size: int,
        rank_address: str,
        store_address: str,
        store_port: int,
        world_addresses: List[str],
    ):
        CollectiveRank.__init__(
            self,
            rank,
            world_size,
            rank_address,
            store_address,
            store_port,
            world_addresses,
        )

    async def __post_create__(self):
        self.init_rank()
