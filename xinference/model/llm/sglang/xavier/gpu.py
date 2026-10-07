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
from typing import Any

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
from ..gc_lifecycle import InitializationGCFreeze
from .settings import GPU_CONFIG_ENV, transfer_timeout


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

    def __init__(
        self,
        args,
        contract,
        directory,
        namespace,
        rank,
        ipc_descriptors: dict[str, tuple] | None = None,
    ):
        super().__init__()
        self.args, self.contract = args, contract
        self.ipc_descriptors = ipc_descriptors
        self.directory, self.namespace, self.rank = directory, namespace, rank
        self._world_addresses: dict[int, str] = {}
        self.rooms: dict[int, dict[str, Any]] = {}
        self.tasks: dict[int, asyncio.Task] = {}
        self._gc_freeze = InitializationGCFreeze()

    async def __post_create__(self):
        if self.ipc_descriptors is None:
            self.caches, self.aux = _buffers(self.args, self.contract)
        else:
            from torch.multiprocessing.reductions import rebuild_cuda_tensor

            torch.cuda.set_device(self.args.gpu_id)
            self.caches = {
                name: rebuild_cuda_tensor(*desc)
                for name, desc in self.ipc_descriptors.items()
            }
            # Raw CPU addresses are meaningful only in the engine process.
            # The manager commits metadata there after these GPU writes drain.
            self.aux = None
        self._snapshot_store = SimpleNamespace(
            capacity=len(next(iter(self.caches.values())))
        )
        self.transfer = DirectGPUTransfer(self, self.caches, 0, slab_bytes=64 * 1024**2)
        # A 64-token SGLang page can exceed the default 256 KiB view. Avoid
        # padding small prompts and final chunks to the full slab.
        self.transfer.add_slab_views(
            (1024**2, 4 * 1024**2, 8 * 1024**2, 16 * 1024**2, 32 * 1024**2)
        )
        self.chunk_capacity = max(
            1,
            self.transfer.slab_bytes
            // sum(cache[0].numel() for cache in self.caches.values()),
        )
        if self.ipc_descriptors is not None:
            # Only the independent importer process owns this lifetime. A full
            # scan of the imported Torch graph otherwise stalls every queued
            # transfer, even when the KV payload itself takes a few milliseconds.
            self._gc_freeze.start()

    async def __pre_destroy__(self):
        try:
            for room in set(self.tasks) | set(self.rooms):
                await self.abort(room)
            await self.transfer.close()
        finally:
            self._gc_freeze.close()

    async def open(self, room):
        if room in self.rooms:
            raise ValueError("Duplicate SGLang Xavier GPU room")
        self.rooms[room] = dict(
            chunks=[],
            pending=[],
            released=set(),
            total=None,
            sent=0,
            completed=asyncio.Event(),
            changed=asyncio.Event(),
            deadline=time.monotonic() + transfer_timeout(),
        )
        try:
            await self.directory.publish_source(
                room, dict(address=self.address, rank=self.rank)
            )
        except BaseException:
            self.rooms.pop(room, None)
            raise

    def init(self, room, count, aux_index):
        state = self.rooms[room]
        state.update(total=count, aux_index=aux_index)

    async def add_chunk(self, room, pages, aux_payload: list[bytes] | None = None):
        state = self.rooms[room]
        state["deadline"] = time.monotonic() + transfer_timeout()
        state["sent"] += len(pages)
        final = state["sent"] == state["total"]
        if state["sent"] > state["total"]:
            raise ValueError("SGLang Xavier sent too many KV pages")
        state["pending"].extend(pages)
        # SGLang submits cached prefix pages separately from the final prefill
        # page. Coalesce small chunks, keeping large prefill transfers pipelined.
        if not final and len(state["pending"]) < self.chunk_capacity:
            return
        pages, state["pending"] = state["pending"], []
        ticket = f"{room}:{len(state['chunks'])}"
        aux = None
        if final:
            aux = (
                aux_payload
                if self.aux is None
                else [bytes(buf[state["aux_index"]].numpy()) for buf in self.aux]
            )
            if aux is None:
                raise ValueError("Missing SGLang Xavier first-token metadata")
        self.transfer.register_direct(
            ticket, ticket, pages, lease_timeout=transfer_timeout()
        )
        # Scheduler-owned source slots remain pinned until every chunk is read.
        state["chunks"].append(dict(ticket=ticket, pages=pages, final=final, aux=aux))
        state["changed"].set()

    def chunk(self, room, index):
        state = self.rooms.get(room)
        if state is None or state.get("aborted"):
            raise RuntimeError("SGLang Xavier producer was cancelled")
        return state["chunks"][index] if index < len(state["chunks"]) else None

    async def wait_chunk(self, room, index):
        state = self.rooms[room]
        state["changed"].clear()
        chunk = self.chunk(room, index)
        while chunk is None:
            await asyncio.wait_for(state["changed"].wait(), timeout=transfer_timeout())
            state["changed"].clear()
            chunk = self.chunk(room, index)
        return chunk

    async def send_direct_gpu_v1(self, ticket, reads, remote_ref, slab_bytes):
        return await self.transfer.run(
            self.transfer.send_direct, ticket, reads, remote_ref, slab_bytes
        )

    def release_chunk(self, ticket):
        self.transfer.release_direct(ticket)
        room = int(ticket.split(":", 1)[0])
        state = self.rooms.get(room)
        if state:
            state["deadline"] = time.monotonic() + transfer_timeout()
            state["released"].add(ticket)
            if state["sent"] == state["total"] and len(state["released"]) == len(
                state["chunks"]
            ):
                state["completed"].set()

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

    async def wait_done(self, room):
        state = self.rooms[room]
        while not state["completed"].is_set():
            deadline = state["deadline"]
            try:
                await asyncio.wait_for(
                    state["completed"].wait(),
                    timeout=max(0, deadline - time.monotonic()),
                )
            except asyncio.TimeoutError:
                if state["deadline"] > time.monotonic():
                    continue  # Source publication or a completed pull made progress.
                await self.abort(room)
                raise TimeoutError("SGLang Xavier producer KV transfer timed out")
        if state.get("aborted"):
            raise RuntimeError("SGLang Xavier producer was cancelled")
        return self.done(room)

    def get_stats(self):
        return dict(
            self.transfer.metrics,
            transfer_pid=os.getpid(),
            active_rooms=len(self.rooms),
            active_transfers=len(self.transfer.direct_requests),
        )

    async def _receive(self, room, destinations, aux_index):
        source = await self.directory.wait_source(room)
        rank = source["rank"]
        self._world_addresses[rank] = source["address"]
        sender = await xo.actor_ref(
            address=source["address"], uid=f"{self.default_uid()}-{rank}"
        )
        offset, index, nbytes = 0, 0, 0
        while True:
            chunk = await sender.wait_chunk(room, index)
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
                if offset != len(destinations) or len(chunk["aux"]) != len(
                    self.args.aux_item_lens
                ):
                    raise ValueError(
                        "SGLang Xavier KV or auxiliary metadata geometry differs"
                    )
                for size, payload in zip(self.args.aux_item_lens, chunk["aux"]):
                    if len(payload) != size:
                        raise ValueError(
                            "SGLang Xavier auxiliary metadata size differs"
                        )
                if self.aux is None:
                    return nbytes, chunk["aux"]
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
            return await finish_before_cancel(task)
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
                state["completed"].set()
                state["changed"].set()

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
        self.directory = None
        self._ipc_caches: dict[str, torch.Tensor] = {}
        self.aux: list[torch.Tensor] = []
        self._receive_tasks: dict[int, asyncio.Task] = {}
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
        actor_class = XavierGPUActor
        if self.config.get("heterogeneous"):
            from ...xavier.backends.torch.pd import CrossEngineGPUActor

            actor_class = CrossEngineGPUActor
            namespace = contract.fingerprint
        directory = await xo.actor_ref(
            address=self.config["address"], uid=self.config["uid"]
        )
        if self.config.get("heterogeneous"):
            await directory.configure(namespace, contract.to_dict())
        else:
            await directory.configure(namespace)
        from torch.multiprocessing.reductions import reduce_tensor
        from xoscar.backends.allocate_strategy import ProcessIndex

        # Preserve the exporting wrappers until the importing process stops.
        # Export the engine allocations once, as the vLLM connector does.
        self._ipc_caches, self.aux = _buffers(args, contract)
        if self.config.get("heterogeneous"):
            self._ipc_caches = {
                name: cache.view(torch.float16).reshape(
                    len(cache),
                    contract.block_size,
                    contract.num_kv_heads,
                    contract.head_dim,
                )
                for name, cache in self._ipc_caches.items()
            }
        descriptors = {
            name: reduce_tensor(cache)[1] for name, cache in self._ipc_caches.items()
        }
        torch.cuda.synchronize(args.gpu_id)
        self.directory = directory
        # Device-wide NIXL fences and Python GPU-copy dispatch must not run in
        # the scheduler's CUDA context or compete with its forward-pass thread.
        self.pool = await xo.create_actor_pool(
            options["external_address"],
            n_process=1,
            subprocess_start_method="spawn",
            # Exported IPC counters and request-owned slots cannot be replayed
            # by actor-pool recovery. A failed process requires a model relaunch.
            auto_recover=False,
        )
        await self.pool.start()
        self.actor = await xo.create_actor(
            actor_class,
            SimpleNamespace(gpu_id=args.gpu_id, aux_item_lens=args.aux_item_lens),
            contract,
            directory,
            namespace,
            self.config["rank"],
            ipc_descriptors=descriptors,
            allocate_strategy=ProcessIndex(1),
            address=self.pool.external_address,
            uid=f"{actor_class.default_uid()}-{self.config['rank']}",
        )
        await directory.register_peer(self.config["rank"], self.actor.address)

    async def receive(self, room, destinations, aux_index):
        task = self._receive_tasks[room] = asyncio.create_task(
            self._receive(room, destinations, aux_index)
        )
        try:
            return await finish_before_cancel(task)
        finally:
            self._receive_tasks.pop(room, None)

    async def _receive(self, room, destinations, aux_index):
        nbytes, payloads = await self.actor.receive(room, destinations, aux_index)
        # The transfer actor validates all payload lengths before returning.
        # Publish completion only after the engine's actual CPU buffers update.
        for buf, payload in zip(self.aux, payloads):
            buf[aux_index].copy_(
                torch.from_numpy(np.frombuffer(payload, dtype=np.uint8).copy())
            )
        await self.directory.complete(room, nbytes)

    async def abort(self, room):
        await self.actor.abort(room)
        # Also drain the local metadata commit. The GPU actor can finish just
        # before abort(), while its reply has yet to reach this event loop.
        task = self._receive_tasks.get(room)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

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
                for room in list(self._receive_tasks):
                    await self.abort(room)
                if self.actor is not None:
                    if self.directory is not None:
                        await self.directory.unregister_peer(
                            self.config["rank"], self.actor.address
                        )
                    await xo.destroy_actor(self.actor)
            finally:
                if self.pool is not None:
                    await self.pool.stop()
            self._ipc_caches.clear()
            self.aux.clear()

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
        self.future = None
        self.total, self.sent, self.aux_index = 0, 0, None
        self._operation = None
        self._error = None
        self._enqueue("open", self.room)

    def _enqueue(self, method, *args):
        previous = self._operation

        async def run():
            if previous is not None:
                await asyncio.wrap_future(previous)
            return await getattr(self.kv_mgr.actor, method)(*args)

        coroutine = run()
        try:
            self._operation = self.kv_mgr.submit(coroutine)
        except Exception as error:
            coroutine.close()
            self._error = error
        return self._operation

    def init(self, num_kv_indices, aux_index=None):
        if self.aborted:
            return
        self.total, self.sent, self.aux_index = num_kv_indices, 0, aux_index
        initialized = self._enqueue("init", self.room, num_kv_indices, aux_index)

        async def wait_done():
            if initialized is not None:
                await asyncio.wrap_future(initialized)
            return await self.kv_mgr.actor.wait_done(self.room)

        coroutine = wait_done()
        try:
            self.future = self.kv_mgr.submit(coroutine)
        except Exception as error:
            coroutine.close()
            self._error = error
        self.inited = True

    def send(self, kv_indices, state_indices=None, num_kv_tokens=None):
        if self.aborted:
            return
        if state_indices:
            raise ValueError(
                "SGLang Xavier does not transfer auxiliary attention state"
            )
        # The native scheduler has completed the forward pass before send().
        # Fence the producer stream before the transfer process reads its slots.
        torch.cuda.synchronize(self.kv_mgr.kv_args.gpu_id)
        pages = kv_indices.tolist()
        self.sent += len(pages)
        aux = (
            [bytes(buf[self.aux_index].numpy()) for buf in self.kv_mgr.aux]
            if self.sent == self.total
            else None
        )
        self._enqueue("add_chunk", self.room, pages, aux)

    def poll(self):
        if self.aborted:
            return KVPoll.Failed
        operation = getattr(self, "_operation", None)
        if getattr(self, "_error", None) is not None or (
            operation is not None
            and operation.done()
            and (operation.cancelled() or operation.exception())
        ):
            return KVPoll.Failed
        if not self.inited:
            return KVPoll.WaitingForInput
        if not self.future.done():
            return KVPoll.Transferring
        return KVPoll.Failed if self.future.exception() else KVPoll.Success

    def get_transfer_metric(self):
        return KVTransferMetric()

    def failure_exception(self):
        if self._error is not None:
            raise self._error
        if self._operation is not None and self._operation.done():
            self._operation.result()
        if self.future is not None and self.future.done():
            self.future.result()
        raise RuntimeError("SGLang Xavier producer GPU transfer failed")

    def abort(self):
        # SGLang reclaims source pages as soon as abort() returns. Mark failure
        # locally, but retain the drain fence before allowing that reclamation.
        self.aborted = True

        async def drain_and_abort():
            if self._operation is not None:
                await asyncio.gather(
                    asyncio.wrap_future(self._operation), return_exceptions=True
                )
            await self.kv_mgr.abort(self.room)

        self.kv_mgr.call(drain_and_abort())
        if self.future is not None:
            self.future.cancel()

    def clear(self):
        async def drain_and_clear():
            if self._operation is not None:
                await asyncio.gather(
                    asyncio.wrap_future(self._operation), return_exceptions=True
                )
            await self.kv_mgr.actor.clear(self.room)

        self.kv_mgr.submit(drain_and_clear())


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
            self.kv_mgr.receive(self.room, kv_indices.tolist(), aux_index)
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
        self.aborted = True
        # Decode can immediately free destinations after abort/failed poll.
        # Its GPU writes and local metadata commit must drain before returning.
        self.kv_mgr.call(self.kv_mgr.abort(self.room))

    def clear(self):
        self.kv_mgr.submit(self.kv_mgr.actor.clear(self.room))

    def ensure_abort_notified(self, *, force_arm=False):
        self.abort()
