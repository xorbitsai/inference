import asyncio
from typing import Dict, List, Optional

import torch
import xoscar as xo
from vllm.core.scheduler import Scheduler
from vllm.utils import Device
from vllm.worker.cache_engine import CacheEngine


class TransferActor(xo.StatelessActor):
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
        self._rank = rank
        self._world_size = world_size
        self._store_address = store_address
        self._rank_address = rank_address
        self._store_port = store_port
        self._world_addresses = world_addresses
        self._context = None
        self._cache_engine: Optional[List[CacheEngine]] = None
        self._scheduler: Optional[List[Scheduler]] = None
        self._swap_stream = torch.cuda.Stream()

    async def __post_create__(self):
        from xoscar.collective import xoscar_pygloo as xp

        context = xp.rendezvous.Context(self._rank, self._world_size)

        attr = xp.transport.tcp.attr(self._rank_address.split(":")[0])
        dev = xp.transport.tcp.CreateDevice(attr)

        opt = xp.rendezvous.TCPStoreOptions()
        opt.port = self._store_port
        opt.numWorkers = self._world_size
        opt.isServer = self._rank == 0

        store = xp.rendezvous.TCPStore(self._store_address, opt)
        store = xp.rendezvous.PrefixStore(str(self._world_size), store)

        context.connectFullMesh(store, dev)
        self._context = context
        print(
            f"Rank {self._rank} connect successfully, world addresses: {self._world_addresses}"
        )

    def setup(self, cache_engine: List[CacheEngine], scheduler: List[Scheduler]):
        self._cache_engine = cache_engine
        self._scheduler = scheduler
        print(f"=====In transfer: {self._cache_engine}")

    @staticmethod
    def _gen_src_to_dst(block_ids: List[int]) -> torch.Tensor:
        return torch.tensor(
            [(x, x) for x in block_ids], device="cpu", dtype=torch.int64
        ).view(-1, 2)

    def _get_buf(self, virtual_engine: int, block_ids: List[int]):
        cache_engine = self._get_cache_engine(virtual_engine)
        cpu_cache = cache_engine.cpu_cache
        stack_res = []
        for i in range(cache_engine.num_attention_layers):
            stack_res.append(cpu_cache[i][:, block_ids, ...])
        return torch.stack(stack_res).contiguous()

    def _get_cache_engine(self, virtual_engine: int) -> CacheEngine:
        return self._cache_engine[virtual_engine]  # type: ignore

    @staticmethod
    def _get_swap_block_ids(src_to_dst: Dict[int, int], is_sender: bool) -> List[int]:
        return list(sorted([r if is_sender else l for r, l in src_to_dst.items()]))

    def _incr_count_for_block_id(self, virtual_engine: int, block_ids: List[int]):
        scheduler = self._scheduler[virtual_engine]  # type: ignore
        gpu_allocator = scheduler.block_manager.block_allocator._allocators[Device.GPU]

        for _id in block_ids:
            gpu_allocator._refcounter.incr(_id)
            # print(f"========Block id: {_id} ref count: {ref_count_gpu}")

    def _decr_count_for_block_id(self, virtual_engine: int, block_ids: List[int]):
        scheduler = self._scheduler[virtual_engine]  # type: ignore
        gpu_allocator = scheduler.block_manager.block_allocator._allocators[Device.GPU]

        for _id in block_ids:
            gpu_allocator._refcounter.decr(_id)
            # print(f"========Block id: {_id} ref count: {ref_count_gpu}")

    async def do_send(
        self, virtual_engine: int, to_rank: int, src_to_dst: Dict[int, int]
    ):
        from xoscar.collective import xoscar_pygloo as xp

        # from xoscar.collective.common import TypeMappingGloo

        cache_engine = self._get_cache_engine(virtual_engine)

        block_ids = self._get_swap_block_ids(src_to_dst, is_sender=True)
        _src_to_dst = self._gen_src_to_dst(block_ids)
        self._incr_count_for_block_id(virtual_engine, block_ids)

        try:
            print(f"========before swap_out in rank {self._rank}")
            with torch.cuda.stream(self._swap_stream):
                cache_engine.swap_out(_src_to_dst)
            torch.cuda.Stream.synchronize(self._swap_stream)
            print(f"========swap_out in rank {self._rank} successfully")
            sendbuf = self._get_buf(virtual_engine, block_ids)
            sendptr = sendbuf.numpy().ctypes.data
            data_size = sendbuf.numel()
            datatype = xp.GlooDataType_t.glooFloat16
            peer = to_rank
            xp.send(self._context, sendptr, data_size, datatype, peer)
        finally:
            self._decr_count_for_block_id(virtual_engine, block_ids)
        print(f"Send to {to_rank} successfully")

    async def do_recv(
        self, virtual_engine: int, from_rank: int, src_to_dst: Dict[int, int]
    ):
        from xoscar.collective import xoscar_pygloo as xp

        # from xoscar.collective.common import TypeMappingGloo

        cache_engine = self._get_cache_engine(virtual_engine)
        cpu_cache = cache_engine.cpu_cache

        block_ids = self._get_swap_block_ids(src_to_dst, is_sender=False)
        _src_to_dst = self._gen_src_to_dst(block_ids)
        self._incr_count_for_block_id(virtual_engine, block_ids)

        try:
            recvbuf = self._get_buf(virtual_engine, block_ids)
            recvptr = recvbuf.numpy().ctypes.data
            data_size = recvbuf.numel()
            datatype = xp.GlooDataType_t.glooFloat16
            peer = from_rank
            xp.recv(self._context, recvptr, data_size, datatype, peer)

            for i in range(cache_engine.num_attention_layers):
                cpu_cache[i][:, block_ids, ...] = recvbuf[i, ...]
            print(f"Recv from {from_rank} successfully")

            with torch.cuda.stream(self._swap_stream):
                cache_engine.swap_in(_src_to_dst)
            torch.cuda.Stream.synchronize(self._swap_stream)
        finally:
            self._decr_count_for_block_id(virtual_engine, block_ids)
        print(f"swap_in in rank {self._rank} successfully")

    async def recv(
        self, virtual_engine: int, from_address: str, src_to_dst: Dict[int, int]
    ):
        rank = self._world_addresses.index(from_address)
        sender_ref = await xo.actor_ref(
            address=from_address, uid=f"{TransferActor.default_uid()}-{rank}"
        )
        await asyncio.gather(
            sender_ref.do_send(virtual_engine, self._rank, src_to_dst),
            self.do_recv(virtual_engine, rank, src_to_dst),
        )
