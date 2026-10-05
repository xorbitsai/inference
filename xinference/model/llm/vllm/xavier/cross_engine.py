# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""vLLM V1 adapter for canonical Xavier GPU pages shared with SGLang."""

import asyncio
import logging
import math
import os
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
import xoscar as xo
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

from ...xavier.actor_loop import release_actor_loop
from ...xavier.backends.torch.pd import CrossEngineGPUActor, canonical_vllm_views
from ...xavier.contract import KVCacheContract
from ...xavier.pd_contract import build_pd_contract, prompt_digest
from ...xavier.transport import gpu_pool_options
from .v1_connector import XavierConnector

logger = logging.getLogger(__name__)


def configure_cross_engine(model_path, model_config, cache_config):
    if model_config is None:
        raise ValueError("Missing cross-engine vLLM model configuration")
    if any(
        model_config.get(name)
        for name in (
            "hf_overrides",
            "rope_scaling",
            "rope_theta",
            "lora_modules",
            "speculative_config",
            "kv_transfer_config",
            "tokenizer",
            "tokenizer_revision",
            "model_loader_extra_config",
        )
    ) or model_config.get("load_format", "auto") not in ("auto", "safetensors"):
        raise ValueError(
            "Cross-engine Xavier requires local weights without semantic overrides"
        )
    if model_config.get("dtype", "auto") not in (
        None,
        "auto",
        "half",
        "float16",
    ) or model_config.get("kv_cache_dtype", "auto") not in ("auto", "float16"):
        raise ValueError("Cross-engine Xavier PD requires FP16 KV")
    if model_config.get("quantization") not in (None, "none"):
        raise ValueError("Cross-engine Xavier PD requires unquantized weights")
    if model_config.get("block_size", 64) != 64:
        raise ValueError("Cross-engine Xavier PD requires 64-token pages")
    contract = build_pd_contract(model_path, model_config.get("max_model_len"))
    cache_config["contract"] = contract.to_dict()
    model_config.update(dtype="float16", block_size=64)
    # Prefix reuse within P remains native. D owns complete per-request pages.
    if cache_config["role"] == "decode":
        model_config["enable_prefix_caching"] = False


@dataclass
class CrossEngineMetadata(KVConnectorMetadata):
    direct_sends: set[str] = field(default_factory=set)
    direct_store: bool = False
    loads: list[tuple[str, int, list[int]]] = field(default_factory=list)


class CrossEngineConnector(XavierConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self.contract = KVCacheContract.from_dict(self._xavier_config["contract"])
        if (
            self._block_size != self.contract.block_size
            or vllm_config.model_config.dtype != torch.float16
        ):
            raise ValueError("Cross-engine vLLM KV differs from the FP16 contract")
        self._history_enabled = False
        self._pool = None
        self._directory = None
        self._cross_requests = {}
        self._cross_allocated = {}
        self._prepared = set()

    async def directory(self):
        if self._directory is None:
            self._directory = await xo.actor_ref(
                address=self._xavier_config["address"], uid=self._xavier_config["uid"]
            )
        return self._directory

    async def _get_transfer_ref(self):
        if self._transfer_ref is None:
            directory = await self.directory()
            peers = (await directory.get_stats())["peers"]
            self._transfer_ref = await xo.actor_ref(
                address=peers[self._rank],
                uid=f"{CrossEngineGPUActor.default_uid()}-{self._rank}",
            )
        return self._transfer_ref

    async def _ensure_gpu_cache_mapping(self):
        from torch.multiprocessing.reductions import reduce_tensor
        from xoscar.backends.allocate_strategy import ProcessIndex

        async with self._gpu_mapping_lock:
            if not self._gpu_cache_mapped:
                caches = canonical_vllm_views(
                    self._registered_kv_caches, self._num_cache_blocks, self.contract
                )
                directory = await self.directory()
                await directory.configure(self.contract.fingerprint)
                options = gpu_pool_options(self._xavier_config["host"], os.environ)
                self._pool = await xo.create_actor_pool(
                    options["external_address"],
                    n_process=1,
                    subprocess_start_method="spawn",
                    auto_recover=False,
                )
                await self._pool.start()
                descriptors = {
                    name: reduce_tensor(cache)[1] for name, cache in caches.items()
                }
                torch.cuda.synchronize()
                self._transfer_ref = await xo.create_actor(
                    CrossEngineGPUActor,
                    SimpleNamespace(
                        gpu_id=torch.cuda.current_device(), aux_item_lens=[]
                    ),
                    self.contract,
                    directory,
                    self.contract.fingerprint,
                    self._rank,
                    ipc_descriptors=descriptors,
                    allocate_strategy=ProcessIndex(1),
                    address=self._pool.external_address,
                    uid=f"{CrossEngineGPUActor.default_uid()}-{self._rank}",
                )
                await directory.register_peer(self._rank, self._transfer_ref.address)
                self._gpu_cache_mapped = True
        return self._transfer_ref

    def register_kv_caches(self, kv_caches):
        super().register_kv_caches(kv_caches)
        self._call(self._ensure_gpu_cache_mapping())

    async def _prepare(self, request):
        params = request.kv_transfer_params or {}
        handoff = params.get("sglang_xavier")
        if not isinstance(handoff, dict) or handoff.get("mode") != "gpu":
            raise ValueError("Missing cross-engine Xavier GPU room")
        room = handoff["room"]
        if request.request_id not in self._prepared:
            directory = await self.directory()
            await directory.prepare(
                room,
                self.contract.fingerprint,
                prompt_digest(request.prompt_token_ids),
                self._xavier_config["role"],
                rank=self._xavier_config["rank"],
                prompt_tokens=len(request.prompt_token_ids),
            )
            self._prepared.add(request.request_id)
        return room

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if not self._is_consumer:
            return 0, False
        params = request.kv_transfer_params or {}
        if params.get("do_remote_prefill") is False:
            return 0, False
        if len(request.prompt_token_ids) < 2:
            raise ValueError(
                "Cross-engine vLLM decode requires at least two prompt tokens"
            )
        room = self._call(self._prepare(request))
        tokens = max(len(request.prompt_token_ids) - 1 - num_computed_tokens, 0)
        if tokens:
            self._cross_requests[request.request_id] = room
        return tokens, bool(tokens)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if request.request_id not in self._cross_requests or not num_external_tokens:
            return
        ids = blocks.get_block_ids()
        self._cross_allocated[request.request_id] = (
            self._cross_requests[request.request_id],
            list(ids[0]),
        )
        request.kv_transfer_params["do_remote_prefill"] = False

    def build_connector_meta(self, scheduler_output):
        meta = CrossEngineMetadata(
            self._direct_sends, bool(scheduler_output.num_scheduled_tokens)
        )
        self._direct_sends = set()
        for request_id, (room, destinations) in self._cross_allocated.items():
            meta.loads.append((request_id, room, destinations))
        self._cross_allocated.clear()
        return meta

    def start_load_kv(self, forward_context, **kwargs):
        meta = self._get_connector_metadata()
        self._direct_sends.update(meta.direct_sends)
        for request_id, room, destinations in meta.loads:

            async def submit(
                room=room, destinations=destinations, request_id=request_id
            ):
                actor = await self._ensure_gpu_cache_mapping()
                task = asyncio.create_task(
                    self._receive_pages(actor, room, destinations)
                )
                self._gpu_load_jobs[task] = [request_id]

            self._call(submit())

    async def _receive_pages(self, actor, room, destinations):
        try:
            return await actor.receive(room, destinations, 0)
        except Exception:
            # The transfer actor fences writes before reporting an error. Return
            # load errors through vLLM's per-request callback, including after an
            # abort removes the room. Raising from get_finished kills EngineCore.
            # kv_load_failure_policy=fail prevents fallback to local prefill.
            logger.warning(
                "Cross-engine KV load failed for room %s", room, exc_info=True
            )
            return set(destinations)

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        if self._is_producer and self._get_connector_metadata().direct_store:
            torch.cuda.synchronize()

    def request_finished(self, request, block_ids):
        self._cross_requests.pop(request.request_id, None)
        if self._is_producer and (request.kv_transfer_params or {}).get(
            "do_remote_decode"
        ):
            from vllm.v1.request import RequestStatus

            if request.status == RequestStatus.FINISHED_ABORTED:
                return False, None
            room = self._call(self._prepare(request))
            count = math.ceil(len(request.prompt_token_ids) / self._block_size)
            pages = block_ids[:count]
            if len(pages) != count or not request.output_token_ids:
                raise RuntimeError("Incomplete cross-engine vLLM prefill")

            async def register():
                actor = await self._get_transfer_ref()
                await actor.register_prefill(
                    room,
                    request.request_id,
                    pages,
                    request.output_token_ids[0],
                    len(request.prompt_token_ids),
                )

            self._call(register())
            self._direct_sends.add(request.request_id)
            self._prepared.discard(request.request_id)
            return True, dict(
                do_remote_prefill=True,
                sglang_xavier=request.kv_transfer_params["sglang_xavier"],
            )
        self._prepared.discard(request.request_id)
        return False, None

    def request_finished_all_groups(self, request, block_ids):
        return self.request_finished(request, list(block_ids[0]))

    def shutdown(self):
        if self._loop is None:
            return

        async def stop():
            if self._gpu_load_jobs:
                await asyncio.gather(*self._gpu_load_jobs, return_exceptions=True)
            if self._pool is not None:
                await xo.destroy_actor(self._transfer_ref)
                await self._pool.stop()

        try:
            self._call(stop())
        finally:
            release_actor_loop(self._loop)
            self._loop = None
