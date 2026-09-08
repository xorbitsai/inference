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
from functools import lru_cache
from queue import Queue
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, no_type_check

import numpy as np
import torch
import xoscar as xo

from .collective import CollectiveRank
from .snapshot import KVSnapshotStore

try:
    from vllm.utils import TORCH_DTYPE_TO_NUMPY_DTYPE, Device
except Exception:  # pragma: no cover - vLLM 0.23+ no longer exposes v0 helpers.
    TORCH_DTYPE_TO_NUMPY_DTYPE = {
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
    }
    Device = None  # type: ignore

if TYPE_CHECKING:
    from vllm.core.scheduler import Scheduler
    from vllm.worker.cache_engine import CacheEngine
else:
    Scheduler = Any  # type: ignore
    CacheEngine = Any  # type: ignore

logger = logging.getLogger(__name__)

# gloo (xoscar collective) cannot transfer bfloat16 tensors, so bf16 KV must be
# moved as another dtype. Use float32 rather than float16: float16's 5-bit
# exponent overflows to +/-inf for KV values above 65504 (which occur in
# Qwen3.5's linear_attn state cache and attention outliers), silently
# corrupting the transferred cache and producing garbage generation. bf16 has
# the same 8-bit exponent as float32, so bf16 -> float32 -> bf16 is lossless.
XAVIER_BF16_TRANSPORT_DTYPE = torch.float32


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

        if buffer_dtype == torch.bfloat16:
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
        self._snapshot_store: Optional[KVSnapshotStore] = None
        self._layer_send_tasks_v1: Set[asyncio.Task[Any]] = set()
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
        for task in self._layer_send_tasks_v1:
            task.cancel()
        self._context.closeConnections()

    def _get_cache_engine(self, virtual_engine: int) -> CacheEngine:
        return self._cache_engine[virtual_engine]  # type: ignore

    def configure_snapshots_v1(self, capacity: int):
        from .snapshot import KVSnapshotStore

        if capacity <= 0:
            raise ValueError("Xavier snapshot capacity must be positive")
        if self._snapshot_store is None:
            self._snapshot_store = KVSnapshotStore(capacity)

    def stage_layer_blocks_v1(self, request_id, layer_name, block_ids, blocks):
        if blocks.dtype == torch.bfloat16:
            blocks = blocks.to(dtype=XAVIER_BF16_TRANSPORT_DTYPE)
        self._snapshot_store.stage(
            layer_name, block_ids, blocks.detach().cpu().contiguous()
        )
        logger.debug(
            "Stage Xavier V1 blocks: request=%s, rank=%s, layer=%s, blocks=%s",
            request_id,
            self._rank,
            layer_name,
            block_ids,
        )

    def publish_blocks_v1(self, keys, layers):
        available = self._snapshot_store.publish(keys, set(layers))
        evicted = list(self._snapshot_store.evicted)
        self._snapshot_store.evicted.clear()
        return available, evicted

    def reserve_blocks_v1(self, lease, keys):
        return self._snapshot_store is not None and self._snapshot_store.reserve(
            lease, keys
        )

    def release_blocks_v1(self, lease):
        if self._snapshot_store is not None:
            self._snapshot_store.release(lease)

    def release_consumer_leases_v1(self, rank):
        if self._snapshot_store is not None:
            self._snapshot_store.release_consumer(rank)

    async def reserve_remote_blocks_v1(self, lease, transfers):
        refs = []
        success = False
        try:
            for rank, mapping in transfers.items():
                ref = await xo.actor_ref(
                    address=self._world_addresses[rank],
                    uid=f"{TransferActor.default_uid()}-{rank}",
                )
                refs.append(ref)
                if not await ref.reserve_blocks_v1(lease, list(mapping)):
                    return False
            success = True
            return True
        except Exception:
            logger.debug(
                "Snapshot reservation failed; recomputing locally", exc_info=True
            )
            return False
        finally:
            if not success:
                await asyncio.gather(
                    *(ref.release_blocks_v1(lease) for ref in refs),
                    return_exceptions=True,
                )

    async def release_remote_blocks_v1(self, lease, transfers):
        async def release(rank):
            ref = await xo.actor_ref(
                address=self._world_addresses[rank],
                uid=f"{TransferActor.default_uid()}-{rank}",
            )
            await ref.release_blocks_v1(lease)

        await asyncio.gather(
            *(release(rank) for rank in transfers), return_exceptions=True
        )

    def _get_staged_layer_blocks_v1(self, layer_name, remote_block_ids):
        return self._snapshot_store.read(layer_name, remote_block_ids)

    def has_layer_blocks_v1(self, layer_name, remote_block_ids):
        return self._snapshot_store is not None and all(
            key in self._snapshot_store.ready
            and layer_name in self._snapshot_store.blocks[key]
            for key in remote_block_ids
        )

    def do_send_layer_blocks_v1(
        self, to_rank: int, layer_name: str, remote_block_ids: List[int]
    ):
        from xoscar.collective import xoscar_pygloo as xp

        sendbuf = self._get_staged_layer_blocks_v1(layer_name, remote_block_ids)
        assert sendbuf.is_contiguous()
        sendptr = sendbuf.numpy().ctypes.data
        data_size = sendbuf.numel()
        datatype = self.get_gloo_dtype(sendbuf.dtype)
        logger.debug(
            "Send Xavier V1 blocks: rank=%s, to_rank=%s, layer=%s, blocks=%s, "
            "shape=%s, dtype=%s",
            self._rank,
            to_rank,
            layer_name,
            remote_block_ids,
            tuple(sendbuf.shape),
            sendbuf.dtype,
        )
        xp.send(self._context, sendptr, data_size, datatype, to_rank)
        logger.debug(
            "Send Xavier V1 blocks done: rank=%s, to_rank=%s, layer=%s, " "blocks=%s",
            self._rank,
            to_rank,
            layer_name,
            remote_block_ids,
        )

    async def start_send_layer_blocks_v1(
        self, to_rank: int, layer_name: str, remote_block_ids: List[int]
    ) -> None:
        task = asyncio.create_task(
            asyncio.to_thread(
                self.do_send_layer_blocks_v1,
                to_rank,
                layer_name,
                remote_block_ids,
            )
        )
        self._layer_send_tasks_v1.add(task)

        def _on_done(fut):
            self._layer_send_tasks_v1.discard(fut)
            try:
                fut.result()
            except asyncio.CancelledError:
                logger.debug(
                    "Send Xavier V1 blocks cancelled: rank=%s, to_rank=%s, "
                    "layer=%s, blocks=%s",
                    self._rank,
                    to_rank,
                    layer_name,
                    remote_block_ids,
                )
            except Exception:
                logger.exception(
                    "Send Xavier V1 blocks failed: rank=%s, to_rank=%s, "
                    "layer=%s, blocks=%s",
                    self._rank,
                    to_rank,
                    layer_name,
                    remote_block_ids,
                )

        task.add_done_callback(_on_done)
        logger.debug(
            "Scheduled Xavier V1 block send: rank=%s, to_rank=%s, layer=%s, "
            "blocks=%s",
            self._rank,
            to_rank,
            layer_name,
            remote_block_ids,
        )

    def do_recv_layer_blocks_v1(
        self,
        from_rank: int,
        recv_shape: Tuple[int, ...],
        recv_dtype: torch.dtype,
    ) -> torch.Tensor:
        from xoscar.collective import xoscar_pygloo as xp

        if recv_dtype == torch.bfloat16:
            recv_dtype = XAVIER_BF16_TRANSPORT_DTYPE
        recvbuf = torch.empty(size=recv_shape, dtype=recv_dtype, device="cpu")
        assert recvbuf.is_contiguous()
        recvptr = recvbuf.numpy().ctypes.data
        data_size = recvbuf.numel()
        datatype = self.get_gloo_dtype(recvbuf.dtype)
        logger.debug(
            "Recv Xavier V1 blocks: rank=%s, from_rank=%s, shape=%s, dtype=%s",
            self._rank,
            from_rank,
            recv_shape,
            recvbuf.dtype,
        )
        xp.recv(self._context, recvptr, data_size, datatype, from_rank)
        logger.debug(
            "Recv Xavier V1 blocks done: rank=%s, from_rank=%s, shape=%s, " "dtype=%s",
            self._rank,
            from_rank,
            recv_shape,
            recvbuf.dtype,
        )
        return recvbuf

    async def read_layer_blocks_v1(
        self,
        from_rank: int,
        layer_name: str,
        src_to_dst: Dict[int, int],
        recv_shape: Tuple[int, ...],
        recv_dtype: torch.dtype,
    ) -> torch.Tensor:
        remote_block_ids = list(src_to_dst.keys())
        from_address = self._world_addresses[from_rank]
        sender_ref = await xo.actor_ref(
            address=from_address, uid=f"{TransferActor.default_uid()}-{from_rank}"
        )
        if not await sender_ref.has_layer_blocks_v1(layer_name, remote_block_ids):
            raise KeyError(
                "No staged Xavier V1 blocks on rank "
                f"{from_rank}: layer={layer_name!r}, blocks={remote_block_ids}"
            )
        await sender_ref.start_send_layer_blocks_v1(
            self._rank, layer_name, remote_block_ids
        )
        recvbuf = await asyncio.to_thread(
            self.do_recv_layer_blocks_v1, from_rank, recv_shape, recv_dtype
        )
        return recvbuf

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

    async def read_blocks(self, from_rank: int, src_to_dst: Dict[int, int]):
        """
        Receiving logic: Gloo recv -> Buffer -> GPU.
        Buffer -> GPU is directly handled using the internal `swap_in` interface of vllm.
        """
        from xoscar.collective import xoscar_pygloo as xp

        block_ids = self._get_swap_block_ids(src_to_dst, is_sender=False)
        total_blocks = len(block_ids)
        if not total_blocks:
            raise ValueError("Cannot receive an empty block mapping")
        cpu_buf_index = self.get_buffer_index()
        chunks = []
        try:
            for start_idx in range(0, total_blocks, self.transfer_block_num):
                offset = min(self.transfer_block_num, total_blocks - start_idx)
                recvbuf = self.get_swap_buffer(cpu_buf_index, offset)
                assert recvbuf.is_contiguous()
                recvptr = recvbuf.numpy().ctypes.data
                data_size = recvbuf.numel()
                datatype = self.get_gloo_dtype(recvbuf.dtype)
                xp.recv(self._context, recvptr, data_size, datatype, from_rank)
                # The next receive reuses this buffer. Preserve every chunk
                # until the caller can swap the complete request into its cache.
                chunks.append(recvbuf.clone() if total_blocks > offset else recvbuf)
            result = torch.cat(chunks, dim=2) if len(chunks) > 1 else chunks[0]
            return result, block_ids, cpu_buf_index
        except BaseException:
            self.free_buffer_index(cpu_buf_index)
            raise

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
