# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""SGLang P/D slots served by Xavier's shared GPU-to-GPU transfer layer."""

import asyncio
import atexit
import ctypes
import json
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch
import xoscar as xo
from sglang.srt.disaggregation.base.conn import (
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVPoll,
    KVTransferDestination,
    KVTransferMetric,
)

from ...xavier.backends.torch.direct_handoff import DirectGPUTransfer
from ...xavier.backends.torch.gpu_transfer import finish_before_cancel
from ...xavier.contract import KVCacheContract, fingerprint_metadata

GPU_CONFIG_ENV = "XINFERENCE_SGLANG_XAVIER_GPU_CONFIG"


class _CUDABytes:
    def __init__(self, pointer, size):
        self.__cuda_array_interface__ = dict(
            shape=(size,),
            typestr="|u1",
            data=(pointer, False),
            version=3,
        )


def _buffers(args, contract):
    """Borrow live engine allocations; the native scheduler retains ownership."""
    if os.environ.get("SGLANG_MOONCAKE_CUSTOM_MEM_POOL") == "INTRA_NODE_NVLINK":
        raise ValueError("SGLang Xavier requires CPU first-token metadata buffers")
    if (
        len(args.kv_data_ptrs) != 2 * contract.num_layers
        or len(args.kv_data_lens) != len(args.kv_data_ptrs)
        or len(args.kv_item_lens) != len(args.kv_data_ptrs)
        or any(size != contract.layer_nbytes // 2 for size in args.kv_item_lens)
        or args.page_size != contract.block_size
        # SGLang retains the spelling "auto" in KVArgs after resolving the
        # actual pool to FP16. The launch contract requires FP16 weights and
        # auto/FP16 KV; page byte geometry is verified independently here.
        or args.kv_cache_dtype_str not in ("auto", "float16", "torch.float16")
        or args.state_types
        or args.num_draft_entries
    ):
        raise ValueError(
            "SGLang GPU KV buffers differ from Xavier's FP16 contract: "
            f"buffers={len(args.kv_data_ptrs)}, item_bytes={args.kv_item_lens}, "
            f"page_size={args.page_size}, dtype={args.kv_cache_dtype_str}, "
            f"state_types={args.state_types}, draft_entries={args.num_draft_entries}, "
            f"expected_layers={contract.num_layers}, expected_item_bytes={contract.layer_nbytes // 2}"
        )
    caches = {}
    for i, (pointer, size, item) in enumerate(
        zip(args.kv_data_ptrs, args.kv_data_lens, args.kv_item_lens)
    ):
        if pointer <= 0 or size <= 0 or size % item:
            raise ValueError("Unaligned SGLang GPU KV buffer")
        caches[str(i)] = torch.as_tensor(
            _CUDABytes(pointer, size), device=f"cuda:{args.gpu_id}"
        ).reshape(-1, item)
    if (
        not args.aux_data_ptrs
        or len(args.aux_data_ptrs) != len(args.aux_data_lens)
        or len(args.aux_data_ptrs) != len(args.aux_item_lens)
        or any(
            ptr <= 0 or item <= 0 or size <= 0 or size % item
            for ptr, size, item in zip(
                args.aux_data_ptrs, args.aux_data_lens, args.aux_item_lens
            )
        )
        or len({len(cache) for cache in caches.values()}) != 1
    ):
        raise ValueError("Invalid SGLang KV or first-token metadata geometry")
    aux = [
        torch.from_numpy(
            np.ctypeslib.as_array((ctypes.c_ubyte * size).from_address(ptr))
        ).reshape(-1, item)
        for ptr, size, item in zip(
            args.aux_data_ptrs, args.aux_data_lens, args.aux_item_lens
        )
    ]
    return caches, aux


class XavierGPUActor(xo.StatelessActor):
    @classmethod
    def default_uid(cls):
        return "sglang-xavier-transfer"

    def __init__(self, args, contract, directory, namespace, rank):
        super().__init__()
        self.args, self.contract = args, contract
        self.directory, self.namespace, self.rank = directory, namespace, rank
        self._world_addresses = {}
        self.rooms = {}
        self.tasks = {}

    async def __post_create__(self):
        self.caches, self.aux = _buffers(self.args, self.contract)
        self._snapshot_store = SimpleNamespace(
            capacity=len(next(iter(self.caches.values())))
        )
        self.transfer = DirectGPUTransfer(self, self.caches, 0)

    async def __pre_destroy__(self):
        for room in list(self.tasks):
            await self.abort(room)
        await self.transfer.close()

    async def open(self, room):
        if room in self.rooms:
            raise ValueError("Duplicate SGLang Xavier GPU room")
        self.rooms[room] = dict(chunks=[], released=set(), total=None, sent=0)
        await self.directory.publish_source(
            room, dict(address=self.address, rank=self.rank)
        )

    def init(self, room, count, aux_index):
        state = self.rooms[room]
        state.update(total=count, aux_index=aux_index)

    async def add_chunk(self, room, pages):
        state = self.rooms[room]
        ticket = f"{room}:{len(state['chunks'])}"
        self.transfer.register_direct(ticket, ticket, pages)
        state["sent"] += len(pages)
        final = state["sent"] == state["total"]
        if state["sent"] > state["total"]:
            raise ValueError("SGLang Xavier sent too many KV pages")
        aux = (
            [bytes(buf[state["aux_index"]].numpy()) for buf in self.aux]
            if final
            else None
        )
        # Scheduler-owned source slots remain pinned until every chunk is read.
        state["chunks"].append(dict(ticket=ticket, pages=pages, final=final, aux=aux))

    def chunk(self, room, index):
        state = self.rooms.get(room)
        if state is None or state.get("aborted"):
            raise RuntimeError("SGLang Xavier producer was cancelled")
        return state["chunks"][index] if index < len(state["chunks"]) else None

    async def send_direct_gpu_v1(self, ticket, reads, remote_ref, slab_bytes):
        return await self.transfer.run(
            self.transfer.send_direct, ticket, reads, remote_ref, slab_bytes
        )

    def release_chunk(self, ticket):
        self.transfer.release_direct(ticket)
        room = int(ticket.split(":", 1)[0])
        state = self.rooms.get(room)
        if state:
            state["released"].add(ticket)

    async def release_remote_direct_gpu_v1(self, rank, ticket):
        ref = await xo.actor_ref(
            address=self._world_addresses[rank], uid=f"{self.default_uid()}-{rank}"
        )
        await ref.release_chunk(ticket)

    def done(self, room):
        self.transfer.poll_direct()
        state = self.rooms[room]
        return state["sent"] == state["total"] and len(state["released"]) == len(
            state["chunks"]
        )

    def get_stats(self):
        return dict(
            self.transfer.metrics,
            active_rooms=len(self.rooms),
            active_transfers=len(self.transfer.direct_requests),
        )

    async def _receive(self, room, destinations, aux_index):
        deadline = time.monotonic() + 120
        source = None
        while source is None:
            source = await self.directory.source(room)
            if time.monotonic() > deadline:
                raise TimeoutError("SGLang Xavier producer bootstrap timed out")
            if source is None:
                await asyncio.sleep(0.005)
        rank = source["rank"]
        self._world_addresses[rank] = source["address"]
        sender = await xo.actor_ref(
            address=source["address"], uid=f"{self.default_uid()}-{rank}"
        )
        offset, index, nbytes = 0, 0, 0
        while True:
            chunk = await sender.chunk(room, index)
            if chunk is None:
                if time.monotonic() > deadline:
                    raise TimeoutError("SGLang Xavier producer KV transfer timed out")
                await asyncio.sleep(0.001)
                continue
            pages = chunk["pages"]
            targets = destinations[offset : offset + len(pages)]
            if len(targets) != len(pages):
                raise ValueError(
                    "SGLang Xavier source and destination page counts differ"
                )
            mapping = dict(zip(pages, targets))
            if len(mapping) != len(pages):
                raise ValueError("Duplicate SGLang Xavier source pages")
            invalid = await self.transfer.run(
                self.transfer.load_direct,
                [{rank: {name: mapping for name in self.caches}}],
                [chunk["ticket"]],
            )
            if invalid:
                raise RuntimeError("SGLang Xavier GPU handoff is unavailable")
            offset += len(pages)
            nbytes += len(pages) * sum(
                cache[0].numel() for cache in self.caches.values()
            )
            index += 1
            if chunk["final"]:
                if offset != len(destinations) or len(chunk["aux"]) != len(self.aux):
                    raise ValueError(
                        "SGLang Xavier KV or auxiliary metadata geometry differs"
                    )
                for buf, payload in zip(self.aux, chunk["aux"]):
                    if len(payload) != buf.shape[1]:
                        raise ValueError(
                            "SGLang Xavier auxiliary metadata size differs"
                        )
                # KV writes have drained. Copy first-token metadata and publish
                # the room marker last, before native decode can commit the slots.
                for buf, payload in zip(self.aux, chunk["aux"]):
                    buf[aux_index].copy_(
                        torch.from_numpy(np.frombuffer(payload, dtype=np.uint8).copy())
                    )
                await self.directory.complete(room, nbytes)
                return

    async def receive(self, room, destinations, aux_index):
        task = self.tasks[room] = asyncio.create_task(
            self._receive(room, destinations, aux_index)
        )
        try:
            await finish_before_cancel(task)
        finally:
            self.tasks.pop(room, None)

    async def abort(self, room):
        task = self.tasks.get(room)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        # Do not return engine slots while a peer's gather is still reading them.
        async with self.transfer.send_lock:
            state = self.rooms.get(room)
            if state:
                state["aborted"] = True
                for chunk in state["chunks"]:
                    self.transfer.release_direct(chunk["ticket"])

    def clear(self, room):
        self.rooms.pop(room, None)


class XavierKVManager(BaseKVManager):
    def __init__(self, args, disaggregation_mode, server_args, is_mla_backend=False):
        config = json.loads(os.environ[GPU_CONFIG_ENV])
        contract = KVCacheContract.from_dict(config["contract"])
        if is_mla_backend:
            raise ValueError("SGLang Xavier requires ordinary full-attention GPU KV")
        self.kv_args = args
        self.req_to_decode_prefix_len = {}
        self.transfer_infos = {}
        # All supported deployments have DP=1. Actor rendezvous replaces an
        # extra HTTP bootstrap service; no fabricated KV result is used.
        self.prefill_info_table = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self.config = config
        self._closed = False
        self.pool = self.actor = None
        try:
            self.call(self._start(args, contract))
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    async def _start(self, args, contract):
        from importlib.metadata import version

        from ...xavier.transport import gpu_pool_options

        options = gpu_pool_options(self.config["host"], os.environ)
        torch.cuda.set_device(args.gpu_id)
        namespace = fingerprint_metadata(
            dict(
                contract=contract.to_dict(),
                engine="sglang",
                version=version("sglang"),
                layout="sglang-pd-layer-k-v-v1",
                item_bytes=args.kv_item_lens,
                aux_bytes=args.aux_item_lens,
            )
        )
        directory = await xo.actor_ref(
            address=self.config["address"], uid=self.config["uid"]
        )
        await directory.configure(namespace)
        self.pool = await xo.create_actor_pool(options["external_address"], n_process=0)
        await self.pool.start()
        self.actor = await xo.create_actor(
            XavierGPUActor,
            args,
            contract,
            directory,
            namespace,
            self.config["rank"],
            address=self.pool.external_address,
            uid=f"{XavierGPUActor.default_uid()}-{self.config['rank']}",
        )
        await directory.register_peer(self.config["rank"], self.actor.address)

    def call(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(
            timeout=180
        )

    def submit(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def try_ensure_parallel_info(self, address):
        self.prefill_info_table[address] = SimpleNamespace(dp_size=1)
        return True

    def register_to_bootstrap(self):
        pass

    def close(self):
        if self._closed:
            return
        self._closed = True

        async def stop():
            try:
                if self.actor is not None:
                    await xo.destroy_actor(self.actor)
            finally:
                if self.pool is not None:
                    await self.pool.stop()

        try:
            self.call(stop())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            if not self._thread.is_alive():
                self._loop.close()
            atexit.unregister(self.close)


class XavierKVSender(BaseKVSender):
    def __init__(
        self,
        mgr,
        bootstrap_addr,
        bootstrap_room,
        dest_tp_ranks,
        pp_rank,
        req_has_disagg_prefill_dp_rank=False,
    ):
        self.kv_mgr, self.room = mgr, bootstrap_room
        self.inited = False
        self.aborted = False
        mgr.call(mgr.actor.open(self.room))

    def init(self, num_kv_indices, aux_index=None):
        self.kv_mgr.call(self.kv_mgr.actor.init(self.room, num_kv_indices, aux_index))
        self.inited = True

    def send(self, kv_indices, state_indices=None, num_kv_tokens=None):
        if state_indices:
            raise ValueError(
                "SGLang Xavier does not transfer auxiliary attention state"
            )
        # The native scheduler has completed the forward pass before send().
        # Fence the producer stream before another thread gathers its slots.
        torch.cuda.synchronize(self.kv_mgr.kv_args.gpu_id)
        self.kv_mgr.call(self.kv_mgr.actor.add_chunk(self.room, kv_indices.tolist()))

    def poll(self):
        if self.aborted:
            return KVPoll.Failed
        if self.inited and self.kv_mgr.call(self.kv_mgr.actor.done(self.room)):
            return KVPoll.Success
        return KVPoll.WaitingForInput if not self.inited else KVPoll.Transferring

    def get_transfer_metric(self):
        return KVTransferMetric()

    def failure_exception(self):
        raise RuntimeError("SGLang Xavier producer GPU transfer failed")

    def abort(self):
        self.kv_mgr.call(self.kv_mgr.actor.abort(self.room))
        self.aborted = True

    def clear(self):
        self.kv_mgr.call(self.kv_mgr.actor.clear(self.room))


class XavierKVReceiver(BaseKVReceiver):
    def __init__(self, mgr, bootstrap_addr, bootstrap_room=None):
        self.kv_mgr, self.room = mgr, bootstrap_room
        self.require_staging = False
        self.abort_notified = False
        self.ready = False
        self.future = None
        self.aborted = False

    def init(self, prefill_dp_rank):
        self.ready = True

    def send_metadata(
        self,
        kv_indices,
        aux_index=None,
        state_indices=None,
        decode_prefix_len=None,
        destination=KVTransferDestination.DEVICE,
    ):
        if (
            state_indices
            or decode_prefix_len
            or destination != KVTransferDestination.DEVICE
        ):
            raise ValueError("SGLang Xavier requires a complete GPU destination")
        self.future = self.kv_mgr.submit(
            self.kv_mgr.actor.receive(self.room, kv_indices.tolist(), aux_index)
        )

    def poll(self):
        if self.aborted:
            return KVPoll.Failed
        if not self.ready:
            return KVPoll.Bootstrapping
        if self.future is None:
            return KVPoll.WaitingForInput
        if not self.future.done():
            return KVPoll.Transferring
        return KVPoll.Failed if self.future.exception() else KVPoll.Success

    def failure_exception(self):
        if self.future is not None and self.future.done():
            self.future.result()
        raise RuntimeError("SGLang Xavier decode GPU transfer failed")

    def abort(self):
        self.kv_mgr.call(self.kv_mgr.actor.abort(self.room))
        self.aborted = True

    def clear(self):
        self.kv_mgr.call(self.kv_mgr.actor.clear(self.room))

    def ensure_abort_notified(self, *, force_arm=False):
        self.abort()
