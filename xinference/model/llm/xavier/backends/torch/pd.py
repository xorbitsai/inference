# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Canonical per-layer K/V pages, using Xavier's persistent GPU NIXL slabs."""

import asyncio
import os
import re
import struct
import time
from types import SimpleNamespace
from typing import Any

import torch
import xoscar as xo

from ....sglang.gc_lifecycle import InitializationGCFreeze
from .direct_handoff import DirectGPUTransfer
from .gpu_transfer import finish_before_cancel
from .snapshot import block_major_view


def sglang_first_token_payload(sizes, room, token, prompt_tokens):
    # SGLang >=0.5.21 MetadataBuffers: output_ids, cached_tokens, logprobs,
    # top-logprobs, speculative buffers, hidden states, bootstrap room. Logprobs,
    # speculation, sampling masks and checksums are rejected for this protocol.
    if len(sizes) != 10 or sizes[0] != 64 or sizes[1] != 64 or sizes[-1] != 64:
        raise ValueError("Unsupported SGLang first-token metadata layout")
    payload = [bytearray(size) for size in sizes]
    struct.pack_into("<i", payload[0], 0, token)
    struct.pack_into("<i", payload[1], 0, prompt_tokens)
    struct.pack_into("<Q", payload[-1], 0, room)
    return [bytes(item) for item in payload]


def canonical_vllm_views(caches, num_blocks, contract):
    layers = {}
    for name, tensor in caches.items():
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if match is None or not isinstance(tensor, torch.Tensor):
            raise ValueError(
                "Cross-engine Xavier requires global full-attention layer indices"
            )
        index = int(match.group(1))
        cache = block_major_view(tensor, num_blocks)
        # vLLM 0.28 FlashAttention packs K/V in the content dimension.
        # Preserve the engine allocation and its physical strides.
        if tuple(cache.shape[1:]) == (
            contract.num_kv_heads,
            contract.block_size,
            2 * contract.head_dim,
        ):
            cache = cache.unflatten(-1, (2, contract.head_dim)).permute(0, 3, 2, 1, 4)
        if (
            tuple(cache.shape[1:])
            != (2, contract.block_size, contract.num_kv_heads, contract.head_dim)
            or cache.dtype != torch.float16
            or index in layers
        ):
            raise ValueError(
                f"Unsupported vLLM cross-engine KV layout: layer={name}, shape={tuple(cache.shape)}, stride={cache.stride()}, dtype={cache.dtype}, expected={contract.to_dict()}"
            )
        layers[index] = cache
    if set(layers) != set(range(contract.num_layers)):
        raise ValueError("Incomplete cross-engine vLLM KV layer set")
    return {
        str(index + kind * contract.num_layers): layers[index][:, kind]
        for kind in range(2)
        for index in range(contract.num_layers)
    }


class CrossEngineGPUActor(xo.StatelessActor):
    @classmethod
    def default_uid(cls):
        return "xavier-cross-engine-transfer"

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
            raise ValueError("Cross-engine PD requires CUDA IPC descriptors")
        from torch.multiprocessing.reductions import rebuild_cuda_tensor

        torch.cuda.set_device(self.args.gpu_id)
        self.caches = {
            name: rebuild_cuda_tensor(*desc)
            for name, desc in self.ipc_descriptors.items()
        }
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
            // sum(
                cache[0].numel() * cache.element_size()
                for cache in self.caches.values()
            ),
        )
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
        )
        try:
            await self.directory.publish_source(
                room, dict(address=self.address, rank=self.rank)
            )
        except BaseException:
            self.rooms.pop(room, None)
            raise

    async def register_prefill(
        self, room, request_id, pages, first_token, prompt_tokens
    ):
        await self.open(room)
        self.init(room, len(pages), 0)
        state = self.rooms[room]
        ticket = f"{room}:0"
        self.transfer.register_direct(ticket, request_id, pages)
        state["sent"] = len(pages)
        state["chunks"].append(
            dict(
                ticket=ticket,
                pages=pages,
                final=True,
                aux=dict(first_token=first_token, prompt_tokens=prompt_tokens),
            )
        )

    async def poll_direct_gpu_v1(self):
        finished = self.transfer.poll_direct()
        for room, state in list(self.rooms.items()):
            if state["completed"].is_set():
                await self.directory.release(room, "prefill")
                self.rooms.pop(room, None)
        return finished

    def init(self, room, count, aux_index):
        state = self.rooms[room]
        if state.get("aborted"):
            raise RuntimeError("Cross-engine producer was cancelled")
        state.update(total=count, aux_index=aux_index)

    async def add_chunk(self, room, pages, aux_payload: list[bytes] | None = None):
        state = self.rooms[room]
        if state.get("aborted"):
            raise RuntimeError("Cross-engine producer was cancelled")
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
            aux = aux_payload
            if aux is None:
                raise ValueError("Missing SGLang Xavier first-token metadata")
        self.transfer.register_direct(ticket, ticket, pages)
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
        await state["completed"].wait()
        if state.get("aborted"):
            raise RuntimeError("SGLang Xavier producer was cancelled")
        return self.done(room)

    def get_stats(self):
        return dict(
            self.transfer.metrics,
            transfer_pid=os.getpid(),
            active_rooms=len(self.rooms),
            active_transfers=len(self.transfer.direct_requests),
            active_receives=len(self.tasks),
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
                cache[0].numel() * cache.element_size()
                for cache in self.caches.values()
            )
            index += 1
            if chunk["final"]:
                if not self.args.aux_item_lens:
                    if offset != len(destinations):
                        raise ValueError("Cross-engine KV page counts differ")
                    stats = await self.directory.request_info(room)
                    await self.directory.complete(
                        room, nbytes, max(stats["prompt_tokens"] - 1, 0)
                    )
                    await self.directory.release(room, "decode")
                    return set()
                if isinstance(chunk["aux"], dict):
                    chunk["aux"] = sglang_first_token_payload(
                        self.args.aux_item_lens,
                        room,
                        chunk["aux"]["first_token"],
                        chunk["aux"]["prompt_tokens"],
                    )
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
                # Raw CPU metadata addresses belong to the engine process.
                # Its manager publishes the first-token/room marker after these
                # GPU writes drain, before native decode can commit the slots.
                return nbytes, chunk["aux"]

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
                # Native bootstrap aborts need not call sender.clear(). Waiters
                # already hold the state and will observe its aborted marker.
                self.rooms.pop(room, None)

    def clear(self, room):
        self.rooms.pop(room, None)
