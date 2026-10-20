# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""TRT-owned KV slots transferred through Xavier's GPU NIXL slabs."""

import asyncio
import ctypes
import os
import sys
import time
from types import SimpleNamespace

import torch
import xoscar as xo

from ..xavier.backends.torch.direct_handoff import DirectGPUTransfer
from ..xavier.backends.torch.gpu_transfer import GPUTransferMixin, finish_before_cancel


class _CUDABytes:
    def __init__(self, pointer, size):
        self.__cuda_array_interface__ = dict(
            shape=(size,), typestr="|u1", data=(pointer, False), version=3
        )


def export_pool(cache):
    from cuda.bindings import driver, runtime

    result, base, size = driver.cuMemGetAddressRange(cache.data_ptr())
    if int(result) != 0:
        raise RuntimeError(f"Cannot locate TRT KV allocation: {result}")
    offset = cache.data_ptr() - int(base)
    if offset < 0 or offset + cache.numel() * cache.element_size() > size:
        raise ValueError("TRT KV view exceeds its CUDA allocation")
    result, handle = runtime.cudaIpcGetMemHandle(int(base))
    if int(result) != 0:
        raise RuntimeError(f"Cannot export TRT KV allocation: {result}")
    # cudaIpcMemHandle_t has a fixed, 64-byte CUDA ABI. This exports the C++
    # allocation without asking Torch's caching allocator to own the pointer.
    return dict(
        handle=ctypes.string_at(handle.getPtr(), 64),
        offset=offset,
        shape=tuple(cache.shape),
    )


class TRTGPUActor(GPUTransferMixin, xo.StatelessActor):
    @classmethod
    def default_uid(cls):
        return "trt-xavier-transfer"

    def __init__(self, descriptor, device, contract, directory, rank):
        super().__init__()
        self.descriptor, self.device = descriptor, device
        self.contract, self.directory, self.rank = contract, directory, rank
        self._world_addresses: dict = {}
        self.sources: dict = {}
        self.aborted: dict = {}
        self.draining = False
        self._ipc_pointer = None
        self._drained_finished = set()
        self._drain_task = None

    async def __post_create__(self):
        if "tensorrt_llm" in sys.modules:
            raise RuntimeError(
                "TRT GPU transfer workers must not load TRT runtime libraries"
            )
        from cuda.bindings import runtime

        torch.cuda.set_device(self.device)
        handle = runtime.cudaIpcMemHandle_t()
        raw = self.descriptor["handle"]
        if len(raw) != 64:
            raise ValueError("Invalid TRT CUDA IPC handle")
        ctypes.memmove(handle.getPtr(), raw, 64)
        result, pointer = runtime.cudaIpcOpenMemHandle(handle, 1)
        if int(result) != 0:
            raise RuntimeError(f"Cannot import TRT KV allocation: {result}")
        self._ipc_pointer = pointer
        try:
            shape = self.descriptor["shape"]
            nbytes = 2
            for size in shape:
                nbytes *= size
            cache = (
                torch.as_tensor(
                    _CUDABytes(int(pointer) + self.descriptor["offset"], nbytes),
                    device=f"cuda:{self.device}",
                )
                .view(torch.bfloat16)
                .reshape(shape)
            )
            self._snapshot_store = SimpleNamespace(capacity=len(cache))
            self._gpu_transfer = DirectGPUTransfer(
                self, {"kv": cache}, 0, slab_bytes=64 * 1024**2
            )
            page_bytes = cache[0].numel() * cache.element_size()
            self._gpu_transfer.add_slab_views(
                range(page_bytes, self._gpu_transfer.slab_bytes, page_bytes)
            )
        except BaseException:
            self._close_import()
            raise

    async def __pre_destroy__(self):
        await self.drain()
        self.sources.clear()

    async def publish(self, room, request_id, blocks):
        if self.draining:
            raise RuntimeError("TRT Xavier GPU transfer is shutting down")
        info = await self.directory.request_info(room)
        count = (max(info["prompt_tokens"] - 1, 0) + 63) // 64
        if (
            room in self.aborted
            or room in self.sources
            or len(blocks) != count
            or len(set(blocks)) != count
        ):
            raise ValueError("Invalid TRT Xavier source KV slots")
        ticket = str(room)
        if blocks:
            self._gpu_transfer.register_direct(ticket, request_id, blocks)
        self.sources[room] = ticket
        await self.directory.publish_source(
            room, dict(address=self.address, rank=self.rank, blocks=blocks)
        )

    async def abort(self, room):
        self.aborted[room] = time.monotonic() + 300
        ticket = self.sources.get(room)
        if ticket is not None:
            self._gpu_transfer.release_direct(ticket)
            # A cancelled producer cannot return slots while a peer's gather
            # still reads them. The shared transport drains its NIXL operations.
            while ticket in self._gpu_transfer.direct_requests:
                await asyncio.sleep(0.001)
            self.sources.pop(room, None)

    def poll_direct_gpu_v1(self):
        if self.draining:
            finished, self._drained_finished = self._drained_finished, set()
            return finished
        self.aborted = {
            room: deadline
            for room, deadline in self.aborted.items()
            if deadline > time.monotonic()
        }
        finished = self._gpu_transfer.poll_direct()
        for room, ticket in list(self.sources.items()):
            if ticket not in self._gpu_transfer.direct_requests:
                self.sources.pop(room, None)
        return finished

    async def load(self, room, destinations):
        if self.draining:
            raise RuntimeError("TRT Xavier GPU transfer is shutting down")
        return await self._gpu_transfer.run(self._load, room, destinations)

    async def _load(self, room, destinations):
        info = await self.directory.request_info(room)
        source = await self.directory.wait_source(room)
        blocks = source["blocks"]
        count = (max(info["prompt_tokens"] - 1, 0) + 63) // 64
        if len(blocks) != count or len(destinations) != count:
            raise ValueError("TRT Xavier source and destination page counts differ")
        mapping = dict(zip(blocks, destinations))
        if len(mapping) != count or len(set(destinations)) != count:
            raise ValueError("Duplicate TRT Xavier KV slots")
        rank = source["rank"]
        self._world_addresses[rank] = source["address"]
        sender = await xo.actor_ref(
            address=source["address"], uid=f"{self.default_uid()}-{rank}"
        )
        if mapping:
            ticket = str(room)
            if not await sender.claim_direct_gpu_v1(ticket):
                raise RuntimeError("TRT Xavier GPU source is unavailable")
            invalid = await self._gpu_transfer.load_direct(
                [{rank: {"kv": mapping}}], [ticket]
            )
            if invalid:
                raise RuntimeError("TRT Xavier GPU handoff did not complete")
        cache = self._gpu_transfer.caches["kv"]
        nbytes = count * cache[0].numel() * cache.element_size()
        await self.directory.complete(room, nbytes, max(info["prompt_tokens"] - 1, 0))
        return nbytes

    async def drain(self):
        if self._drain_task is None:
            self._drain_task = asyncio.create_task(self._drain())
        return await finish_before_cancel(self._drain_task)

    async def _drain(self):
        self.draining = True
        for ticket in list(self._gpu_transfer.direct_requests):
            self._gpu_transfer.release_direct(ticket)
        if self._gpu_transfer.tasks:
            await asyncio.gather(
                *list(self._gpu_transfer.tasks), return_exceptions=True
            )
        self._drained_finished.update(self._gpu_transfer.poll_direct())
        await self._gpu_transfer.close()
        # Close every importer before the owning TRT process frees C++ memory.
        self._close_import()

    def _close_import(self):
        if self._ipc_pointer is not None:
            from cuda.bindings import runtime

            (result,) = runtime.cudaIpcCloseMemHandle(self._ipc_pointer)
            if int(result) != 0:
                raise RuntimeError(f"Cannot close TRT KV import: {result}")
            self._ipc_pointer = None

    def get_stats(self):
        return dict(
            self._gpu_transfer.metrics,
            transfer_pid=os.getpid(),
            active_transfers=len(self._gpu_transfer.direct_requests),
        )


class GPUService:
    """Retain the C++ pool wrapper until its independent IPC importer closes."""

    def __init__(self, client, cache):
        self.client, self.cache = client, cache
        self.pool = self.actor = None
        try:
            client.call(self._start())
        except BaseException:
            self.close()
            raise

    async def _start(self):
        from xoscar.backends.allocate_strategy import ProcessIndex

        from ..xavier.transport import gpu_pool_options

        options = gpu_pool_options(self.client.config["host"], os.environ)
        torch.cuda.synchronize(self.cache.device)
        descriptor = export_pool(self.cache)
        self.pool = await xo.create_actor_pool(
            options["external_address"],
            n_process=1,
            modules=["xinference.core._trt_transfer_worker"],
            subprocess_start_method="spawn",
            auto_recover=False,
        )
        await self.pool.start()
        rank = self.client.config["rank"]
        self.actor = await xo.create_actor(
            TRTGPUActor,
            descriptor,
            self.cache.device.index,
            self.client.contract,
            await self.client.directory(),
            rank,
            allocate_strategy=ProcessIndex(1),
            address=self.pool.external_address,
            uid=f"{TRTGPUActor.default_uid()}-{rank}",
        )
        await (await self.client.directory()).register_peer(rank, self.actor.address)

    def rpc(self, method, *args):
        return self.client.call(getattr(self.actor, method)(*args))

    def close(self):
        async def stop():
            try:
                if self.actor is not None:
                    try:
                        await (await self.client.directory()).unregister_peer(
                            self.client.config["rank"], self.actor.address
                        )
                    finally:
                        await xo.destroy_actor(self.actor)
            finally:
                if self.pool is not None:
                    await self.pool.stop()

        self.client.call(stop())
        self.cache = None


_service = None


def register_service(client, cache):
    global _service
    close_service()
    _service = GPUService(client, cache)


def service():
    if _service is None:
        raise RuntimeError("TRT Xavier GPU KV pool is not registered")
    return _service


def close_service():
    global _service
    if _service is not None:
        _service.close()
        _service = None
