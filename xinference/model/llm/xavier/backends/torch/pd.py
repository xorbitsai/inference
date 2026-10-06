# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Canonical per-layer K/V pages, using Xavier's persistent GPU NIXL slabs."""

import asyncio
import logging
import math
import os
import re
import struct
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import xoscar as xo

from ....sglang.gc_lifecycle import InitializationGCFreeze
from ....sglang.xavier.settings import transfer_timeout
from .direct_handoff import DirectGPUTransfer
from .gpu_transfer import finish_before_cancel
from .snapshot import block_major_view

logger = logging.getLogger(__name__)


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

    async def register_prefill(
        self, room, request_id, pages, first_token, prompt_tokens
    ):
        await self.open(room)
        self.init(room, len(pages), 0)
        state = self.rooms[room]
        ticket = f"{room}:0"
        self.transfer.register_direct(
            ticket, request_id, pages, lease_timeout=transfer_timeout()
        )
        state["sent"] = len(pages)
        state["chunks"].append(
            dict(
                ticket=ticket,
                pages=pages,
                final=True,
                aux=dict(first_token=first_token, prompt_tokens=prompt_tokens),
            )
        )
        state["changed"].set()

    async def poll_direct_gpu_v1(self):
        finished = self.transfer.poll_direct()
        for room, state in list(self.rooms.items()):
            expired = any(
                chunk["ticket"] not in self.transfer.direct_requests
                and chunk["ticket"] not in state["released"]
                for chunk in state["chunks"]
            )
            if expired:
                await self.abort(room)
                finished.update(self.transfer.poll_direct())
            if expired or state["completed"].is_set():
                self.rooms.pop(room, None)
                try:
                    await self.directory.release(room, "prefill")
                except Exception:
                    logger.warning(
                        "Cross-engine room release failed: %s", room, exc_info=True
                    )
        return finished

    def _producer_state(self, room):
        state = self.rooms.get(room)
        if state is None:
            raise RuntimeError("Cross-engine producer was cancelled")
        return state

    def init(self, room, count, aux_index):
        state = self._producer_state(room)
        state.update(total=count, aux_index=aux_index)

    async def add_chunk(self, room, pages, aux_payload: list[bytes] | None = None):
        state = self._producer_state(room)
        state["sent"] += len(pages)
        state["deadline"] = time.monotonic() + transfer_timeout()
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
        self.transfer.register_direct(
            ticket, ticket, pages, lease_timeout=transfer_timeout()
        )
        # Scheduler-owned source slots remain pinned until every chunk is read.
        state["chunks"].append(dict(ticket=ticket, pages=pages, final=final, aux=aux))
        state["changed"].set()

    def chunk(self, room, index):
        state = self._producer_state(room)
        return state["chunks"][index] if index < len(state["chunks"]) else None

    async def wait_chunk(self, room, index):
        state = self._producer_state(room)
        state["changed"].clear()
        chunk = self.chunk(room, index)
        while chunk is None:
            deadline = state["deadline"]
            try:
                await asyncio.wait_for(
                    state["changed"].wait(), timeout=max(0, deadline - time.monotonic())
                )
            except asyncio.TimeoutError:
                if state["deadline"] > time.monotonic():
                    continue
                raise TimeoutError("Cross-engine producer KV transfer timed out")
            state["changed"].clear()
            chunk = self.chunk(room, index)
        return chunk

    async def send_direct_gpu_v1(self, ticket, reads, remote_ref, slab_bytes):
        result = await self.transfer.run(
            self.transfer.send_direct, ticket, reads, remote_ref, slab_bytes
        )
        state = self.transfer.direct_requests.get(ticket)
        if state is not None:
            state.deadline = time.monotonic() + transfer_timeout()
        room = self.rooms.get(int(ticket.split(":", 1)[0]))
        if room is not None:
            room["deadline"] = time.monotonic() + transfer_timeout()
        return result

    def _export_host_pages(self, pages, offset, prefix_tokens):
        """Copy owned canonical pages, clearing unused engine allocation bytes."""
        device = next(iter(self.caches.values())).device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        try:
            return self._copy_host_pages(pages, offset, prefix_tokens, device)
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    def _copy_host_pages(self, pages, offset, prefix_tokens, device):
        c = self.contract
        indices = torch.tensor(pages, dtype=torch.long, device=device)
        data = torch.stack(
            [
                torch.stack(
                    [
                        self.caches[str(layer + kind * c.num_layers)].index_select(
                            0, indices
                        )
                        for kind in (0, 1)
                    ],
                    dim=1,
                )
                for layer in range(c.num_layers)
            ],
            dim=1,
        ).cpu()
        for index in range(len(pages)):
            valid = max(
                0, min(c.block_size, prefix_tokens - (offset + index) * c.block_size)
            )
            data[index, :, :, valid:] = 0
        return [page.numpy().astype("<f2", copy=False).tobytes() for page in data]

    async def export_host_pages(self, room, index, start=0):
        """Bounded CUDA-to-CPU RPC for a Metal consumer; source slots stay pinned."""
        async with self.transfer.send_lock:
            chunk = self.chunk(room, index)
            if (
                chunk is None
                or type(start) is not int
                or not 0 <= start < len(chunk["pages"])
            ):
                raise ValueError("Invalid cross-engine host page cursor")
            state = self.transfer._live_direct(chunk["ticket"])
            if state is None:
                raise RuntimeError("Cross-engine source pages expired")
            c = self.contract
            if c.layer_nbytes * c.num_layers > 64 * 1024**2:
                raise ValueError("Cross-engine host page exceeds 64 MiB")
            capacity = max(1, min(16, 64 * 1024**2 // (c.layer_nbytes * c.num_layers)))
            pages = chunk["pages"][start : start + capacity]
            if any(
                not set(pages) <= allowed for allowed in state.layer_blocks.values()
            ):
                raise ValueError("Unowned cross-engine source pages")
            state.reading = True
            state.claimed = True
            try:
                info = await self.directory.request_info(room)
                offset = (
                    sum(
                        len(item["pages"])
                        for item in self.rooms[room]["chunks"][:index]
                    )
                    + start
                )
                # Cancellation must drain the CUDA read before abort can free
                # engine slots. The returned bytes then own their CPU storage.
                payload = await finish_before_cancel(
                    asyncio.create_task(
                        asyncio.to_thread(
                            self._export_host_pages,
                            pages,
                            offset,
                            info["prompt_tokens"] - 1,
                        )
                    )
                )
                self.transfer.metrics["host_bytes"] = self.transfer.metrics.get(
                    "host_bytes", 0
                ) + sum(map(len, payload))
                return dict(pages=payload, next=start + len(pages))
            finally:
                state.reading = False
                state.deadline = time.monotonic() + 600
                if state.released:
                    self.transfer.release_direct(chunk["ticket"])

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
        state = self._producer_state(room)
        return state["sent"] == state["total"] and len(state["released"]) == len(
            state["chunks"]
        )

    async def wait_done(self, room):
        state = self._producer_state(room)
        while not state["completed"].is_set():
            deadline = state["deadline"]
            try:
                await asyncio.wait_for(
                    state["completed"].wait(),
                    timeout=max(0, deadline - time.monotonic()),
                )
            except asyncio.TimeoutError:
                if state["deadline"] > time.monotonic():
                    continue
                await self.abort(room)
                raise TimeoutError("Cross-engine producer KV transfer timed out")
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
        info = await self.directory.request_info(room)
        prompt_tokens = info["prompt_tokens"]
        source_pages = math.ceil(prompt_tokens / self.contract.block_size)
        receive_pages = (
            source_pages
            if self.args.aux_item_lens
            else math.ceil((prompt_tokens - 1) / self.contract.block_size)
        )
        if len(destinations) < receive_pages or len(destinations) > source_pages:
            raise ValueError("Cross-engine KV destination page counts differ")
        destinations = destinations[:receive_pages]
        source = await asyncio.wait_for(
            self.directory.wait_source(room), timeout=transfer_timeout()
        )
        if source.get("transport") == "host":
            return await self._receive_host(room, source, destinations)
        rank = source["rank"]
        self._world_addresses[rank] = source["address"]
        sender = await xo.actor_ref(
            address=source["address"], uid=f"{self.default_uid()}-{rank}"
        )
        offset, index, nbytes = 0, 0, 0
        while True:
            chunk = await asyncio.wait_for(
                sender.wait_chunk(room, index), timeout=transfer_timeout()
            )
            pages = chunk["pages"]
            targets = destinations[offset : offset + len(pages)]
            if offset + len(pages) > source_pages:
                raise ValueError(
                    "SGLang Xavier source and destination page counts differ"
                )
            mapping = dict(zip(pages[: len(targets)], targets))
            if len(mapping) != len(targets) or len(set(pages)) != len(pages):
                raise ValueError("Duplicate SGLang Xavier source pages")
            if mapping:
                invalid = await asyncio.wait_for(
                    self.transfer.run(
                        self.transfer.load_direct,
                        [{rank: {name: mapping for name in self.caches}}],
                        [chunk["ticket"]],
                    ),
                    timeout=transfer_timeout(),
                )
            else:
                # A 64k+1 prompt has one source-only final page: vLLM
                # recomputes its final token after the external load.
                await asyncio.wait_for(
                    sender.release_chunk(chunk["ticket"]), timeout=transfer_timeout()
                )
                invalid = set()
            if invalid:
                raise RuntimeError("SGLang Xavier GPU handoff is unavailable")
            offset += len(pages)
            nbytes += len(targets) * sum(
                cache[0].numel() * cache.element_size()
                for cache in self.caches.values()
            )
            index += 1
            if chunk["final"]:
                if offset != source_pages:
                    raise ValueError("Cross-engine KV source page counts differ")
                if not self.args.aux_item_lens:
                    await self.directory.complete(
                        room, nbytes, max(prompt_tokens - 1, 0)
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

    def _import_host_pages(self, pages, destinations):
        device = next(iter(self.caches.values())).device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        try:
            c = self.contract
            data = torch.from_numpy(
                np.stack(
                    [
                        np.frombuffer(page, dtype="<f2").reshape(
                            c.num_layers, 2, c.block_size, c.num_kv_heads, c.head_dim
                        )
                        for page in pages
                    ]
                )
            ).to(device)
            indices = torch.tensor(destinations, device=device, dtype=torch.long)
            for layer in range(c.num_layers):
                for kind in (0, 1):
                    self.caches[str(layer + kind * c.num_layers)].index_copy_(
                        0, indices, data[:, layer, kind]
                    )
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    async def _receive_host(self, room, source, destinations):
        from ....sglang.xavier.settings import transfer_timeout

        sender = await xo.actor_ref(address=source["address"], uid=source["uid"])
        info = await self.directory.request_info(room)
        prompt_tokens = info["prompt_tokens"]
        c = self.contract
        expected = (prompt_tokens + c.block_size - 1) // c.block_size
        imported = prompt_tokens if self.args.aux_item_lens else prompt_tokens - 1
        if len(destinations) != (imported + c.block_size - 1) // c.block_size:
            raise ValueError("Cross-engine host destination page count differs")
        page_bytes = c.num_layers * c.layer_nbytes
        start, nbytes, first_token = 0, 0, None
        deadline = time.monotonic() + transfer_timeout()
        try:
            while start < expected:
                result = await asyncio.wait_for(
                    sender.read(room, start),
                    timeout=max(0, deadline - time.monotonic()),
                )
                pages = result["pages"]
                if (
                    not pages
                    or len(pages) > 16
                    or len(pages) * page_bytes > 64 * 1024**2
                    or result["next"] != start + len(pages)
                    or result["next"] > expected
                    or result["total"] != expected
                    or result["prompt_tokens"] != prompt_tokens
                    or type(result["first_token"]) is not int
                    or result["first_token"] < 0
                    or (
                        first_token is not None and first_token != result["first_token"]
                    )
                    or any(type(p) is not bytes or len(p) != page_bytes for p in pages)
                ):
                    raise ValueError("Incomplete cross-engine host prefix")
                first_token = result["first_token"]
                targets = destinations[start : result["next"]]
                if targets:
                    task = asyncio.create_task(
                        asyncio.to_thread(
                            self._import_host_pages, pages[: len(targets)], targets
                        )
                    )
                    # Slot reuse after abort must wait for CUDA writes to drain,
                    # including errors and repeated cancellation during H2D.
                    await finish_before_cancel(task)
                nbytes += len(pages) * page_bytes
                start = result["next"]
            await sender.release(room)
            self.transfer.metrics["host_imported_bytes"] = (
                self.transfer.metrics.get("host_imported_bytes", 0) + nbytes
            )
            if not self.args.aux_item_lens:
                await self.directory.complete(room, 0, imported, nbytes)
                await self.directory.release(room, "decode")
                return set()
            return nbytes, sglang_first_token_payload(
                self.args.aux_item_lens, room, first_token, prompt_tokens
            )
        except BaseException:
            await sender.abort(room)
            raise

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
                # Native bootstrap aborts need not call sender.clear(). Waiters
                # already hold the state and will observe its aborted marker.
                self.rooms.pop(room, None)

    def clear(self, room):
        self.rooms.pop(room, None)
