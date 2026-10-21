# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""TRT HND pages exposed as canonical, bit-preserving BF16 GPU views."""

import time
from types import SimpleNamespace

import torch

from ..xavier.backends.torch.pd import CrossEngineGPUActor
from .gpu import close_import, import_pool


def canonical_trt_views(cache, contract):
    expected = (
        contract.num_layers,
        2,
        contract.block_size * contract.num_kv_heads * contract.head_dim,
    )
    if (
        cache.ndim != 4
        or tuple(cache.shape[1:]) != expected
        or not cache.is_contiguous()
        or cache.dtype != torch.bfloat16
        or contract.logical_dtype != "bfloat16"
    ):
        raise ValueError("TRT-LLM KV differs from the canonical BF16 contract")
    # TRT 1.2.1 uses HND inside each layer's K/V block. These strided views
    # retain the original C++ allocation; gather/scatter stays on the GPU.
    pages = cache.view(
        len(cache),
        contract.num_layers,
        2,
        contract.num_kv_heads,
        contract.block_size,
        contract.head_dim,
    )
    return {
        str(layer + kind * contract.num_layers): pages[:, layer, kind].transpose(1, 2)
        for kind in (0, 1)
        for layer in range(contract.num_layers)
    }


class TRTCrossEngineGPUActor(CrossEngineGPUActor):
    def __init__(self, descriptor, device, contract, directory, rank):
        super().__init__(
            SimpleNamespace(gpu_id=device, aux_item_lens=[]),
            contract,
            directory,
            contract.fingerprint,
            rank,
        )
        self.descriptor, self.device = descriptor, device
        self._ipc_pointer = None
        self.aborted: dict[int, float] = {}

    async def __post_create__(self):
        self._ipc_pointer, cache = import_pool(self.descriptor, self.device)
        try:
            self.caches = canonical_trt_views(cache, self.contract)
            self._initialize_transfer()
        except BaseException:
            self._close_import()
            raise

    def _close_import(self):
        if self._ipc_pointer is not None:
            close_import(self._ipc_pointer)
            self._ipc_pointer = None

    async def __pre_destroy__(self):
        await super().__pre_destroy__()
        self._close_import()

    def _prune_aborted(self):
        now = time.monotonic()
        self.aborted = {
            room: deadline for room, deadline in self.aborted.items() if deadline > now
        }

    async def abort(self, room):
        self._prune_aborted()
        self.aborted[room] = time.monotonic() + 300
        await super().abort(room)

    async def poll_direct_gpu_v1(self):
        self._prune_aborted()
        return await super().poll_direct_gpu_v1()

    async def publish(self, room, request_id, blocks):
        self._prune_aborted()
        if room in self.aborted:
            raise RuntimeError("TRT cross-engine producer was cancelled")
        info = await self.directory.request_info(room)
        if room in self.aborted:
            raise RuntimeError("TRT cross-engine producer was cancelled")
        count = (
            info["prompt_tokens"] + self.contract.block_size - 1
        ) // self.contract.block_size
        if len(blocks) != count or len(set(blocks)) != count:
            raise ValueError("Invalid TRT cross-engine source KV slots")
        # Pin before TRT's request_finished callback returns slots. The Python
        # output supplies the first token later, before the chunk becomes ready.
        await self.register_prefill(
            room, request_id, blocks, None, info["prompt_tokens"]
        )

    def chunk(self, room, index):
        chunk = super().chunk(room, index)
        if chunk is not None and chunk["aux"]["first_token"] is None:
            return None
        return chunk

    def finalize(self, room, first_token):
        state = self._producer_state(room)
        if type(first_token) is not int or first_token < 0:
            raise ValueError("Invalid TRT cross-engine first token")
        state["chunks"][0]["aux"]["first_token"] = first_token
        state["changed"].set()

    async def load(self, room, destinations):
        return await self.receive(room, destinations, 0)
