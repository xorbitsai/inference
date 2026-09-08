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
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import xoscar as xo
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.core.sched.output import SchedulerOutput

from .block_tracker import VLLMBlockTracker
from .snapshot import block_major_view
from .transfer import XAVIER_BF16_TRANSPORT_DTYPE, TransferActor
from .utils import hash_block_tokens

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)


@dataclass
class XavierStoreRequest:
    request_id: str
    block_ids: List[int]
    block_hashes: List[int]
    block_ids_by_group: List[List[int]]


@dataclass
class XavierLoadRequest:
    request_id: str
    transfers: Dict[int, Dict[int, int]]
    lease: str = ""
    local_transfers_by_group: Dict[int, Dict[int, Dict[int, int]]] = field(
        default_factory=dict
    )


@dataclass
class XavierConnectorMetadata(KVConnectorMetadata):
    store_requests: List[XavierStoreRequest] = field(default_factory=list)
    load_requests: List[XavierLoadRequest] = field(default_factory=list)


class XavierConnector(KVConnectorBase_V1, SupportsHMA):
    # same as vllm.core.block.prefix_caching_block.PrefixCachingBlock._none_hash
    _none_hash = -1

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._xavier_config = dict(
            self._kv_transfer_config.get_from_extra_config("xavier_config", {}) or {}
        )
        self._block_size = vllm_config.cache_config.block_size
        has_recurrent_cache = any(
            hasattr(group.kv_cache_spec, "mamba_cache_mode")
            for group in kv_cache_config.kv_cache_groups
        )
        if has_recurrent_cache:
            raise ValueError(
                "Xavier PD does not yet support hybrid/recurrent attention caches "
                "(for example Qwen3.5). Launch this model without PD/Xavier, "
                "or use a full-attention model such as Qwen3 for PD."
            )
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("Xavier V1 currently requires TP=1 and PP=1")
        if (
            vllm_config.lora_config is not None
            or vllm_config.model_config.is_multimodal_model
        ):
            raise ValueError(
                "Xavier V1 currently supports text-only models without LoRA"
            )
        self._rank = int(self._xavier_config.get("rank", 0))
        self._is_producer = self._kv_transfer_config.is_kv_producer
        self._is_consumer = self._kv_transfer_config.is_kv_consumer
        self._requests_need_load: Dict[str, XavierLoadRequest] = {}
        self._pending_store_requests: Dict[str, XavierStoreRequest] = {}
        self._leased_requests: Dict[str, XavierLoadRequest] = {}
        self._num_cache_blocks = kv_cache_config.num_blocks
        self._request_staged_layers: Dict[str, set[str]] = {}
        self._registered_kv_caches: Dict[str, torch.Tensor | Sequence[torch.Tensor]] = (
            {}
        )
        self._chunked_prefill: Dict[str, Tuple[List[List[int]], List[int]]] = {}
        self._layer_group_ids = self._build_layer_group_index()
        self._tracker_ref: Optional[xo.ActorRefType["VLLMBlockTracker"]] = None
        self._transfer_ref: Optional[xo.ActorRefType["TransferActor"]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def shutdown(self):
        if self._loop is None:
            return
        if self._loop.is_closed():
            return
        self._loop.close()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        if not self._is_consumer:
            return

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, XavierConnectorMetadata)
        if not metadata.load_requests:
            return

        try:
            if self._registered_kv_caches:
                for layer_name, kv_layer in self._registered_kv_caches.items():
                    for request in metadata.load_requests:
                        self._load_layer_blocks(layer_name, kv_layer, request)
            else:
                layers = getattr(forward_context, "no_compile_layers", {}) or {}
                for layer_name, layer in layers.items():
                    kv_layer = getattr(layer, "kv_cache", None)
                    if kv_layer is not None:
                        for request in metadata.load_requests:
                            self._load_layer_blocks(layer_name, kv_layer, request)
        finally:
            for request in metadata.load_requests:
                if request.lease:
                    self._call(self._release_load_request(request))

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def register_kv_caches(
        self, kv_caches: Dict[str, torch.Tensor | Sequence[torch.Tensor]]
    ):
        self._registered_kv_caches = dict(kv_caches)
        cache_groups = {
            layer_name: self._get_layer_group_id(layer_name)
            for layer_name in self._registered_kv_caches
        }
        logger.debug(
            "Xavier V1 registered KV caches: rank=%s, layers=%s, groups=%s",
            self._rank,
            list(self._registered_kv_caches),
            cache_groups,
        )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        if not self._is_producer:
            return

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, XavierConnectorMetadata)
        if not metadata.store_requests:
            return

        for request in metadata.store_requests:
            if not request.block_ids:
                continue
            self._stage_kv_layer_for_request(request, layer_name, kv_layer)
            self._pending_store_requests[request.request_id] = request

    def wait_for_save(self):
        if not self._is_producer or not self._pending_store_requests:
            return

        pending = list(self._pending_store_requests.values())
        self._pending_store_requests.clear()
        try:
            self._stage_missing_registered_layers(pending)
            self._call(self._register_blocks(pending))
        finally:
            for request in pending:
                self._request_staged_layers.pop(request.request_id, None)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if (
            getattr(request, "lora_request", None)
            or getattr(request, "mm_features", None)
            or getattr(request, "prompt_embeds", None) is not None
            or getattr(request, "cache_salt", None)
        ):
            raise ValueError(
                "Xavier V1 does not support LoRA, multimodal, embedding or salted prompts"
            )
        self._requests_need_load.pop(request.request_id, None)
        previous = self._leased_requests.pop(request.request_id, None)
        if previous is not None:
            self._call(self._release_load_request(previous))
        if not self._is_consumer:
            return 0, False

        token_ids = list(request.prompt_token_ids or [])
        external_token_count = max(len(token_ids) - 1, 0)
        if external_token_count <= num_computed_tokens:
            return 0, False

        hashes = self._build_xavier_hashes(token_ids[:external_token_count])
        if not hashes:
            return 0, False

        local_hit_blocks = num_computed_tokens // self._block_size
        query_hashes = hashes[local_hit_blocks:]
        if not query_hashes:
            return 0, False

        remote = self._call(self._query_remote_blocks(request.request_id, query_hashes))
        transfers, matched_blocks = self._build_contiguous_transfers(
            query_hashes, remote
        )
        if matched_blocks == 0:
            return 0, False

        matched_tokens = min(
            matched_blocks * self._block_size,
            external_token_count - num_computed_tokens,
        )
        load = XavierLoadRequest(
            request_id=request.request_id,
            transfers=transfers,
            lease=f"{self._rank}:{uuid.uuid4().hex}",
        )
        if not self._call(self._reserve_load_request(load)):
            return 0, False
        self._requests_need_load[request.request_id] = load
        self._leased_requests[request.request_id] = load
        logger.debug(
            "Xavier V1 external cache hit: request=%s, blocks=%s, tokens=%s",
            request.request_id,
            matched_blocks,
            matched_tokens,
        )
        return matched_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        if not self._is_consumer or num_external_tokens <= 0:
            return

        load_request = self._requests_need_load.get(request.request_id)
        if load_request is None:
            return

        block_ids_by_group = self._normalize_block_groups(blocks.get_block_ids())
        local_transfers_by_group: Dict[int, Dict[int, Dict[int, int]]] = {}
        for group_id, group_block_ids in enumerate(block_ids_by_group):
            updated: Dict[int, Dict[int, int]] = {}
            for from_rank, remote_to_placeholder in load_request.transfers.items():
                # Placeholder indices are absolute positions in the prompt.
                # Allocation includes locally cached prefix blocks: remote
                # suffix blocks must not overwrite that prefix. Indexing by
                # position also preserves order across multiple source ranks.
                updated[from_rank] = {
                    remote_block_id: group_block_ids[placeholder]
                    for remote_block_id, placeholder in remote_to_placeholder.items()
                }
            local_transfers_by_group[group_id] = updated

        self._requests_need_load[request.request_id] = XavierLoadRequest(
            request_id=request.request_id,
            transfers=load_request.transfers,
            lease=load_request.lease,
            local_transfers_by_group=local_transfers_by_group,
        )
        logger.debug(
            "Xavier V1 allocated local blocks: request=%s, transfers=%s",
            request.request_id,
            local_transfers_by_group,
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = XavierConnectorMetadata()
        if self._is_producer:
            self._build_store_meta(scheduler_output, meta)
        if self._is_consumer:
            self._build_load_meta(scheduler_output, meta)
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: List[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._chunked_prefill.pop(request.request_id, None)
        self._requests_need_load.pop(request.request_id, None)
        load = self._leased_requests.pop(request.request_id, None)
        if load is not None:
            self._call(self._release_load_request(load))
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: Tuple[List[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, block_ids[0] if block_ids else [])

    def take_events(self) -> Iterable:
        return ()

    def _call(self, coro):
        # vLLM creates the KV connector inside EngineCore after CUDA/JIT
        # initialization. Starting a helper thread here can trip glibc static
        # TLS allocation in CUDA-heavy environments, so run actor calls on a
        # connector-local loop in the EngineCore thread.
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    async def _get_tracker_ref(self) -> xo.ActorRefType["VLLMBlockTracker"]:
        if self._tracker_ref is None:
            self._tracker_ref = await xo.actor_ref(
                address=self._xavier_config.get("block_tracker_address"),
                uid=self._xavier_config.get("block_tracker_uid"),
            )
        return self._tracker_ref

    async def _get_transfer_ref(self) -> xo.ActorRefType["TransferActor"]:
        if self._transfer_ref is None:
            self._transfer_ref = await xo.actor_ref(
                address=self._xavier_config.get("rank_address"),
                uid=f"{TransferActor.default_uid()}-{self._rank}",
            )
            await self._transfer_ref.configure_snapshots_v1(self._num_cache_blocks)
        return self._transfer_ref

    async def _reserve_load_request(self, request):
        transfer = await self._get_transfer_ref()
        return await transfer.reserve_remote_blocks_v1(request.lease, request.transfers)

    async def _release_load_request(self, request):
        transfer = await self._get_transfer_ref()
        await transfer.release_remote_blocks_v1(request.lease, request.transfers)

    async def _query_remote_blocks(
        self,
        request_id: str,
        query_hashes: List[Tuple[int, int]],
    ) -> Dict[int, set[Tuple[int, int, int]]]:
        tracker_ref = await self._get_tracker_ref()
        remote = await tracker_ref.query_blocks(
            0,
            query_hashes,
            exclude_rank=self._rank,
        )
        logger.debug("Xavier V1 query result for request %s: %s", request_id, remote)
        return remote

    async def _stage_layer_blocks(
        self,
        request_id: str,
        layer_name: str,
        block_ids: List[int],
        blocks: torch.Tensor,
    ):
        transfer_ref = await self._get_transfer_ref()
        await transfer_ref.stage_layer_blocks_v1(
            request_id,
            layer_name,
            block_ids,
            blocks,
        )

    async def _register_blocks(self, requests: List[XavierStoreRequest]):
        tracker_ref = await self._get_tracker_ref()
        transfer_ref = await self._get_transfer_ref()
        expected_layers = {
            name
            for layer, cache in self._registered_kv_caches.items()
            for name, _ in self._iter_kv_tensors(layer, cache)
        }
        for request in requests:
            layers = expected_layers or self._request_staged_layers.get(
                request.request_id, set()
            )
            available, evicted = await transfer_ref.publish_blocks_v1(
                request.block_hashes, layers
            )
            if evicted:
                await tracker_ref.unregister_blocks(0, self._rank, evicted)
            # Transport addresses identify immutable content, not recyclable GPU slots.
            await tracker_ref.register_blocks(
                0, [(key, key) for key in available], self._rank
            )
            logger.debug(
                "Xavier V1 registered blocks: request=%s, rank=%s, blocks=%s",
                request.request_id,
                self._rank,
                available,
            )

    async def _read_layer_blocks(
        self,
        layer_name: str,
        from_rank: int,
        src_to_dst: Dict[int, int],
        recv_shape: Tuple[int, ...],
        recv_dtype: torch.dtype,
    ):
        transfer_ref = await self._get_transfer_ref()
        # Bound each actor reply; packed hybrid cache blocks can be several
        # MiB each. Consume replies incrementally rather than sending a large
        # tensor through the engine's synchronous actor bridge.
        blocks = []
        for src, dst in src_to_dst.items():
            blocks.append(
                await transfer_ref.read_layer_blocks_v1(
                    from_rank, layer_name, {src: dst}, (1, *recv_shape[1:]), recv_dtype
                )
            )
        return torch.cat(blocks, dim=0)

    def _load_layer_blocks(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        request: XavierLoadRequest,
    ) -> None:
        for from_rank, src_to_dst in request.transfers.items():
            if not src_to_dst:
                continue
            for kv_layer_name, kv_tensor in self._iter_kv_tensors(layer_name, kv_layer):
                src_to_dst = self._get_local_transfer_map(
                    request, kv_layer_name, from_rank
                )
                if not src_to_dst:
                    continue
                kv_tensor = block_major_view(kv_tensor, self._num_cache_blocks)
                local_block_ids = list(src_to_dst.values())
                # Must match the producer-side staging dtype in
                # TransferActor.stage_layer_blocks_v1: bf16 is transported as
                # float32 (not float16) to avoid overflow that corrupts KV.
                transfer_dtype = (
                    XAVIER_BF16_TRANSPORT_DTYPE
                    if kv_tensor.dtype is torch.bfloat16
                    else kv_tensor.dtype
                )
                recv_shape = (len(local_block_ids), *tuple(kv_tensor.shape[1:]))
                blocks = self._call(
                    self._read_layer_blocks(
                        kv_layer_name,
                        from_rank,
                        src_to_dst,
                        recv_shape,
                        transfer_dtype,
                    )
                )
                if blocks.dtype != kv_tensor.dtype:
                    blocks = blocks.to(dtype=kv_tensor.dtype)
                kv_tensor[torch.tensor(local_block_ids, device=kv_tensor.device)] = (
                    blocks.to(device=kv_tensor.device, non_blocking=True)
                )
                logger.debug(
                    "Load Xavier V1 blocks: request=%s rank=%s from_rank=%s layer=%s blocks=%s",
                    request.request_id,
                    self._rank,
                    from_rank,
                    kv_layer_name,
                    len(local_block_ids),
                )

    def _stage_kv_layer_for_request(
        self,
        request: XavierStoreRequest,
        layer_name: str,
        kv_layer: torch.Tensor | Sequence[torch.Tensor],
    ) -> None:
        staged_layers = self._request_staged_layers.setdefault(
            request.request_id, set()
        )
        for kv_layer_name, kv_tensor in self._iter_kv_tensors(layer_name, kv_layer):
            if kv_layer_name in staged_layers:
                continue
            kv_tensor = block_major_view(kv_tensor, self._num_cache_blocks)
            source_block_ids = self._get_source_block_ids(request, kv_layer_name)
            num_blocks = min(len(request.block_ids), len(source_block_ids))
            if num_blocks <= 0:
                continue
            transport_block_ids = request.block_hashes[:num_blocks]
            source_block_ids = source_block_ids[:num_blocks]
            block_ids_tensor = torch.tensor(
                source_block_ids, device=kv_tensor.device, dtype=torch.long
            )
            blocks = (
                kv_tensor.index_select(0, block_ids_tensor).detach().cpu().contiguous()
            )
            self._call(
                self._stage_layer_blocks(
                    request.request_id,
                    kv_layer_name,
                    transport_block_ids,
                    blocks,
                )
            )
            staged_layers.add(kv_layer_name)

    def _stage_missing_registered_layers(
        self, requests: List[XavierStoreRequest]
    ) -> None:
        if not self._registered_kv_caches:
            return

        for request in requests:
            if not request.block_ids:
                continue
            for layer_name, kv_layer in self._registered_kv_caches.items():
                self._stage_kv_layer_for_request(request, layer_name, kv_layer)

    @staticmethod
    def _iter_kv_tensors(
        layer_name: str, kv_layer: torch.Tensor | Sequence[torch.Tensor]
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        if isinstance(kv_layer, torch.Tensor):
            yield layer_name, kv_layer
            return

        if not isinstance(kv_layer, (list, tuple)):
            raise TypeError(
                f"Unsupported Xavier V1 KV cache type for {layer_name}: "
                f"{type(kv_layer)!r}"
            )

        tensors = list(kv_layer)
        if not tensors:
            return

        use_base_name = len(tensors) == 1
        for idx, tensor in enumerate(tensors):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Unsupported Xavier V1 KV cache item type for {layer_name}"
                    f"[{idx}]: {type(tensor)!r}"
                )
            tensor_layer_name = layer_name if use_base_name else f"{layer_name}#{idx}"
            yield tensor_layer_name, tensor

    def _build_store_meta(
        self, scheduler_output: SchedulerOutput, meta: XavierConnectorMetadata
    ) -> None:
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = list(new_req.prompt_token_ids or [])
            block_ids_by_group = self._normalize_block_groups(new_req.block_ids)
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[new_req.req_id]
            num_tokens = num_scheduled_tokens + new_req.num_computed_tokens
            if num_tokens < len(token_ids):
                self._chunked_prefill[new_req.req_id] = (
                    block_ids_by_group,
                    token_ids,
                )
                continue
            self._add_store_request(meta, new_req.req_id, token_ids, block_ids_by_group)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            if req_id not in self._chunked_prefill:
                continue
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_tokens = num_scheduled_tokens + num_computed_tokens
            new_block_ids = cached_reqs.new_block_ids[i]
            existing_block_ids_by_group, prompt_token_ids = self._chunked_prefill[
                req_id
            ]
            if req_id in getattr(cached_reqs, "resumed_req_ids", set()):
                block_ids_by_group = self._normalize_block_groups(new_block_ids)
            else:
                block_ids_by_group = self._merge_block_groups(
                    existing_block_ids_by_group,
                    self._normalize_block_groups(new_block_ids),
                )
            if num_tokens < len(prompt_token_ids):
                self._chunked_prefill[req_id] = (
                    block_ids_by_group,
                    prompt_token_ids,
                )
                continue
            self._add_store_request(meta, req_id, prompt_token_ids, block_ids_by_group)
            self._chunked_prefill.pop(req_id, None)

    def _build_load_meta(
        self, scheduler_output: SchedulerOutput, meta: XavierConnectorMetadata
    ) -> None:
        for new_req in scheduler_output.scheduled_new_reqs:
            load_request = self._requests_need_load.pop(new_req.req_id, None)
            if load_request is not None:
                meta.load_requests.append(load_request)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for req_id in cached_reqs.req_ids:
            load_request = self._requests_need_load.pop(req_id, None)
            if load_request is not None:
                meta.load_requests.append(load_request)

    def _add_store_request(
        self,
        meta: XavierConnectorMetadata,
        request_id: str,
        token_ids: List[int],
        block_ids_by_group: List[List[int]],
    ) -> None:
        external_token_count = max(len(token_ids) - 1, 0)
        hashes = self._build_xavier_hashes(token_ids[:external_token_count])
        if not hashes:
            return
        if not block_ids_by_group:
            return
        group_lengths = [len(block_ids) for block_ids in block_ids_by_group]
        num_blocks = min(len(hashes), *group_lengths)
        if num_blocks <= 0:
            return
        block_ids = block_ids_by_group[0]
        request = XavierStoreRequest(
            request_id=request_id,
            block_ids=list(block_ids[:num_blocks]),
            block_hashes=[content_hash for content_hash, _ in hashes[:num_blocks]],
            block_ids_by_group=[
                list(group_block_ids[:num_blocks])
                for group_block_ids in block_ids_by_group
            ],
        )
        meta.store_requests.append(request)

    def _build_layer_group_index(self) -> Dict[str, int]:
        groups: Dict[int, List[str]] = {}
        layer_group_ids: Dict[str, int] = {}
        for group_id, group in enumerate(self._kv_cache_config.kv_cache_groups):
            layer_names = list(getattr(group, "layer_names", []) or [])
            groups[group_id] = layer_names
            for layer_name in layer_names:
                layer_group_ids[layer_name] = group_id
        logger.debug(
            "Xavier V1 KV cache groups: rank=%s, groups=%s",
            self._rank,
            groups,
        )
        return layer_group_ids

    @staticmethod
    def _base_layer_name(layer_name: str) -> str:
        return layer_name.split("#", 1)[0]

    def _get_layer_group_id(self, layer_name: str) -> int:
        return self._layer_group_ids.get(self._base_layer_name(layer_name), 0)

    @staticmethod
    def _normalize_block_groups(block_ids: Any) -> List[List[int]]:
        if block_ids is None:
            return []
        if isinstance(block_ids, tuple):
            return [list(group) for group in block_ids]
        if isinstance(block_ids, list):
            if not block_ids:
                return []
            if all(isinstance(block_id, int) for block_id in block_ids):
                return [list(block_ids)]
            return [list(group) for group in block_ids]
        return [list(block_ids)]

    @staticmethod
    def _merge_block_groups(
        existing: List[List[int]], new: List[List[int]]
    ) -> List[List[int]]:
        num_groups = max(len(existing), len(new))
        merged: List[List[int]] = []
        for group_id in range(num_groups):
            existing_group = existing[group_id] if group_id < len(existing) else []
            new_group = new[group_id] if group_id < len(new) else []
            merged.append(list(existing_group) + list(new_group))
        return merged

    def _get_source_block_ids(
        self, request: XavierStoreRequest, layer_name: str
    ) -> List[int]:
        group_id = self._get_layer_group_id(layer_name)
        if group_id >= len(request.block_ids_by_group):
            logger.warning(
                "Xavier V1 layer %s maps to missing cache group %s; falling "
                "back to group 0",
                layer_name,
                group_id,
            )
            group_id = 0
        return list(request.block_ids_by_group[group_id])

    def _get_local_transfer_map(
        self,
        request: XavierLoadRequest,
        layer_name: str,
        from_rank: int,
    ) -> Dict[int, int]:
        group_id = self._get_layer_group_id(layer_name)
        group_transfers = request.local_transfers_by_group.get(group_id)
        if group_transfers is None:
            logger.warning(
                "Xavier V1 layer %s maps to missing local cache group %s; "
                "falling back to group 0",
                layer_name,
                group_id,
            )
            group_transfers = request.local_transfers_by_group.get(0, {})
        return dict(group_transfers.get(from_rank, {}))

    def _build_xavier_hashes(self, token_ids: List[int]) -> List[Tuple[int, int]]:
        hashes: List[Tuple[int, int]] = []
        prev_hash: Optional[int] = None
        for block_idx, start in enumerate(range(0, len(token_ids), self._block_size)):
            chunk = token_ids[start : start + self._block_size]
            if not chunk:
                continue
            content_hash = hash_block_tokens(
                block_idx == 0,
                self._none_hash if block_idx == 0 else prev_hash,
                chunk,
                self._none_hash,
                self._none_hash,
            )
            hashes.append((content_hash, block_idx))
            prev_hash = content_hash
        return hashes

    @staticmethod
    def _build_contiguous_transfers(
        query_hashes: List[Tuple[int, int]],
        remote: Dict[int, set[Tuple[int, int, int]]],
    ) -> Tuple[Dict[int, Dict[int, int]], int]:
        by_placeholder: Dict[int, Tuple[int, int]] = {}
        for from_rank, remote_details in remote.items():
            for _, remote_block_id, placeholder_block_id in remote_details:
                by_placeholder[placeholder_block_id] = (from_rank, remote_block_id)

        transfers: Dict[int, Dict[int, int]] = {}
        matched_blocks = 0
        for _, placeholder_block_id in query_hashes:
            detail = by_placeholder.get(placeholder_block_id)
            if detail is None:
                break
            from_rank, remote_block_id = detail
            transfers.setdefault(from_rank, {})[remote_block_id] = placeholder_block_id
            matched_blocks += 1
        return transfers, matched_blocks
