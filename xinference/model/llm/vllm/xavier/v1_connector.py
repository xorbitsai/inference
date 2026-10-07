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
import math
import sys
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import xoscar as xo
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.core.sched.output import SchedulerOutput

from ...xavier.actor_loop import acquire_actor_loop, release_actor_loop
from ...xavier.backends.torch.snapshot import block_major_view
from ...xavier.block_tracker import VLLMBlockTracker
from ...xavier.profiling import profile_stage
from ...xavier.utils import hash_block_tokens
from .transfer import XAVIER_BF16_TRANSPORT_DTYPE, TransferActor
from .transport import uses_direct_handoff

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

    from ...xavier.backends.torch.gpu_export import GPUExportArena
    from ...xavier.backends.torch.local_read import LocalReadBuffer
    from ...xavier.backends.torch.packed_gather import PackedGather
    from ...xavier.local_directory import LocalBlockDirectory

logger = logging.getLogger(__name__)

_MAX_READ_BYTES = 1024 * 1024
_MAX_READ_BLOCKS = 64
_MAX_EXPORT_BYTES = 32 * 1024 * 1024
_MAX_PENDING_EXPORT_BYTES = 8 * _MAX_EXPORT_BYTES
_EXPORT_REFRESH_SECONDS = 1.0
_MAX_HASH_CACHE_TOKENS = 128 * 1024
_MAX_HASH_CACHE_ENTRIES = 128


@dataclass
class _PackedExportEntries:
    keys: List[int]
    packed: torch.Tensor
    layers: List[Tuple[str, Tuple[int, ...], torch.dtype, int, int]]
    slot: Optional[int] = None

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> Tuple[str, List[int], torch.Tensor]:
        name, shape, dtype, offset, nbytes = self.layers[index]
        return (
            name,
            self.keys,
            self.packed[:, offset : offset + nbytes]
            .view(dtype)
            .view(len(self.keys), *shape),
        )

    def __iter__(self) -> Iterator[Tuple[str, List[int], torch.Tensor]]:
        for index in range(len(self)):
            yield self[index]


_CPUExportBatch = Tuple[
    Union[List[Tuple[str, List[int], torch.Tensor]], _PackedExportEntries],
    List[torch.Tensor],
    List[torch.cuda.Event],
]


@dataclass
class XavierKVSchema(KVConnectorHandshakeMetadata):
    """Missing or mismatched schemas cause a cache miss and local recomputation."""

    block_size: int
    layers: Dict[str, Tuple[Tuple[int, ...], torch.dtype]]
    cache_dtype: str


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
    remote_blocks_by_group: List[List[int]] = field(default_factory=list)
    local_hit_tokens: int = 0


@dataclass
class XavierConnectorMetadata(KVConnectorMetadata):
    direct_sends: set[str] = field(default_factory=set)
    direct_store: bool = False
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
        self._cache_dtype = vllm_config.cache_config.cache_dtype
        self._recurrent_groups = {
            i
            for i, group in enumerate(kv_cache_config.kv_cache_groups)
            if hasattr(group.kv_cache_spec, "mamba_cache_mode")
        }
        self._has_recurrent_cache = bool(self._recurrent_groups)
        if self._has_recurrent_cache:
            if getattr(
                vllm_config.cache_config, "enable_prefix_caching", False
            ) or getattr(
                getattr(vllm_config, "scheduler_config", None),
                "async_scheduling",
                False,
            ):
                raise ValueError(
                    "Xavier recurrent handoff requires enable_prefix_caching=False "
                    "and async_scheduling=False"
                )
            if len(self._recurrent_groups) == len(kv_cache_config.kv_cache_groups):
                raise ValueError(
                    "Xavier recurrent handoff requires a full-attention group"
                )
            if getattr(vllm_config, "speculative_config", None) is not None:
                raise ValueError(
                    "Xavier recurrent handoff does not support speculative decoding"
                )
            for i in self._recurrent_groups:
                if (
                    kv_cache_config.kv_cache_groups[i].kv_cache_spec.mamba_cache_mode
                    != "none"
                ):
                    raise ValueError(
                        "Xavier recurrent handoff requires mamba_cache_mode=none"
                    )
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("Xavier V1 currently requires TP=1 and PP=1")
        if vllm_config.lora_config is not None or (
            vllm_config.model_config.is_multimodal_model
            and not getattr(
                getattr(vllm_config.model_config, "multimodal_config", None),
                "language_model_only",
                False,
            )
        ):
            raise ValueError(
                "Xavier V1 requires text-only models or language_model_only=True, without LoRA"
            )
        self._direct_handoff = uses_direct_handoff(self._xavier_config)
        self._gpu_budget = self._xavier_config.get("gpu_cache_bytes")
        if (
            self._direct_handoff
            and len(kv_cache_config.kv_cache_groups) != 1
            and not self._has_recurrent_cache
        ):
            raise ValueError(
                "Direct handoff requires P/D roles, GPU transport and one KV group"
            )
        if self._has_recurrent_cache and not self._direct_handoff:
            raise ValueError(
                "Xavier recurrent caches require prefill/decode GPU handoff"
            )
        self._gpu_cache_mapped = False
        self._gpu_mapping_lock = asyncio.Lock()
        self._kv_schema: Optional[XavierKVSchema] = None
        self._rank = int(self._xavier_config.get("rank", 0))
        self._is_producer = self._kv_transfer_config.is_kv_producer
        self._is_consumer = self._kv_transfer_config.is_kv_consumer
        self._history_enabled = (
            self._direct_handoff
            and bool(self._gpu_budget)
            and not self._has_recurrent_cache
        )
        if self._has_recurrent_cache and self._gpu_budget:
            logger.info(
                "Xavier recurrent handoff uses engine GPU states; historical prefix "
                "reuse is unavailable, so no history cache budget is allocated."
            )
        if self._history_enabled and self._is_producer:
            # P may restore its own independent history before computing a suffix.
            self._is_consumer = True
        self._requests_need_load: Dict[str, XavierLoadRequest] = {}
        self._leased_requests: Dict[str, XavierLoadRequest] = {}
        self._gpu_load_jobs: Dict[asyncio.Task, List[str]] = {}
        self._direct_sends: set[str] = set()
        self._invalid_block_ids: set[int] = set()
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
        self._exported_keys: set[int] = set()
        self._export_refresh_at = 0.0
        self._cpu_export_streams: Dict[torch.device, torch.cuda.Stream] = {}
        self._ipc_export_jobs: Dict[str, Tuple[set[int], int, List[torch.Tensor]]] = {}
        self._ipc_export_poll_at = 0.0
        self._ipc_export_poll: Optional[asyncio.Task] = None
        self._ipc_export_enqueues: Dict[str, asyncio.Task] = {}
        self._ipc_export_budget = _MAX_PENDING_EXPORT_BYTES
        self._packed_gather: Optional["PackedGather"] = None
        self._gpu_export_arena: Optional["GPUExportArena"] = None
        self._gpu_export_tickets: Dict[str, List[int]] = {}
        self._gpu_export_registered = False
        self._hash_cache: OrderedDict[
            Tuple[int, int, Tuple[int, ...]], Tuple[Tuple[int, int], ...]
        ] = OrderedDict()
        self._hash_cache_tokens = 0
        self._local_directory: Optional["LocalBlockDirectory"] = None
        self._directory_retry_at = 0.0
        self._local_read: Optional["LocalReadBuffer"] = None
        self._local_read_checked = False

    def shutdown(self):
        local_read = getattr(self, "_local_read", None)
        if local_read is not None:
            local_read.close()
            self._local_read = None
        directory = getattr(self, "_local_directory", None)
        if directory is not None:
            directory.close()
            self._local_directory = None
        if self._loop is None:
            arena = getattr(self, "_gpu_export_arena", None)
            if arena is not None:
                arena.close()
                self._gpu_export_arena = None
            return
        try:
            if getattr(self, "_ipc_export_jobs", None):
                try:
                    self._poll_ipc_exports(wait=True)
                except Exception:
                    logger.warning(
                        "Xavier IPC export failed during shutdown", exc_info=True
                    )
            arena = getattr(self, "_gpu_export_arena", None)
            if arena is not None:
                try:
                    if self._transfer_ref is not None:
                        self._call(
                            self._transfer_ref.close_snapshot_export_source_v1(
                                arena.source
                            )
                        )
                finally:
                    arena.close()
                    self._gpu_export_arena = None
            if getattr(self, "_gpu_load_jobs", None):
                results = self._call(
                    asyncio.gather(*self._gpu_load_jobs, return_exceptions=True)
                )
                for result in results:
                    if isinstance(result, BaseException):
                        logger.warning(
                            "Xavier GPU load failed during shutdown",
                            exc_info=(type(result), result, result.__traceback__),
                        )
                self._gpu_load_jobs.clear()
            if getattr(self, "_gpu_cache_mapped", False):
                try:
                    self._call(self._transfer_ref.close_gpu_caches_v1())
                except Exception:
                    logger.warning(
                        "Failed to close Xavier GPU caches during shutdown; continuing",
                        exc_info=True,
                    )
        finally:
            try:
                release_actor_loop(self._loop)
            finally:
                self._loop = None

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, XavierConnectorMetadata)
        self._direct_sends.update(metadata.direct_sends)
        if not self._is_consumer or not metadata.load_requests:
            return

        if getattr(self, "_gpu_budget", None) is not None:
            self._call(self._submit_gpu_requests(metadata.load_requests))
            return

        load_failed = False
        try:
            if self._registered_kv_caches:
                for request in metadata.load_requests:
                    self._load_request_blocks(request)
            else:
                layers = getattr(forward_context, "no_compile_layers", {}) or {}
                for layer_name, layer in layers.items():
                    kv_layer = getattr(layer, "kv_cache", None)
                    if kv_layer is not None:
                        for request in metadata.load_requests:
                            self._load_layer_blocks(layer_name, kv_layer, request)
        except BaseException:
            load_failed = True
            raise
        finally:
            release_error = None
            for request in metadata.load_requests:
                if request.lease:
                    try:
                        self._call(self._release_load_request(request))
                    except Exception as error:
                        release_error = release_error or error
                        logger.warning(
                            "Failed to release Xavier snapshot lease", exc_info=True
                        )
            if release_error is not None and not load_failed:
                raise release_error

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        sent = set()
        self._poll_ipc_exports()
        if self._direct_handoff and self._is_producer and self._direct_sends:
            assert self._transfer_ref is not None
            sent = self._call(self._transfer_ref.poll_direct_gpu_v1())
            self._direct_sends.difference_update(sent)
        if not self._gpu_load_jobs:
            return sent, set()
        # Pump responses on the shared actor loop without waiting for any RPC.
        # CUDA writes and lease cleanup progress in the independent actor process.
        self._call(asyncio.sleep(0))
        received = set()
        for task in list(self._gpu_load_jobs):
            if task.done():
                self._invalid_block_ids.update(task.result() or ())
                request_ids = self._gpu_load_jobs.pop(task)
                received.update(request_ids)
                for request_id in request_ids:
                    logger.debug(
                        "Finished Xavier async KV load: request=%s", request_id
                    )
        # Include aborted requests: vLLM retains their destination blocks until
        # finished_recving arrives. Never cancel writes or report them early.
        # If the last request aborts while a load is pending, an idle EngineCore
        # may not poll again until new work arrives. Destination reclamation then
        # waits for that step; actor-side writes and lease release still progress.
        return sent, received

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid = self._invalid_block_ids
        self._invalid_block_ids = set()
        return invalid

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def register_kv_caches(
        self, kv_caches: Dict[str, torch.Tensor | Sequence[torch.Tensor]]
    ):
        self._registered_kv_caches = dict(kv_caches)
        if (
            sys.platform == "linux"
            and self._gpu_budget is None
            and not self._direct_handoff
        ):
            devices = {
                tensor.device
                for layer, cache in kv_caches.items()
                for _, tensor in self._iter_kv_tensors(layer, cache)
                if tensor.is_cuda
            }
            if devices:
                # Two GPU copies can briefly coexist during packing. Leave
                # headroom for graph capture, activations and allocator segments.
                free = min(torch.cuda.mem_get_info(device)[0] for device in devices)
                self._ipc_export_budget = min(
                    _MAX_PENDING_EXPORT_BYTES, max(2 * _MAX_EXPORT_BYTES, free // 8)
                )
                try:
                    from ...xavier.backends.torch.packed_gather import PackedGather

                    caches = {
                        name: block_major_view(tensor, self._num_cache_blocks)
                        for layer, cache in kv_caches.items()
                        for name, tensor in self._iter_kv_tensors(layer, cache)
                    }
                    self._packed_gather = PackedGather.try_create(
                        caches,
                        {name: self._get_layer_group_id(name) for name in caches},
                    )
                    if self._packed_gather is not None:
                        self._packed_gather.warmup()
                        if self._packed_gather.row_bytes <= _MAX_EXPORT_BYTES:
                            from ...xavier.backends.torch.gpu_export import (
                                GPUExportArena,
                            )

                            self._gpu_export_arena = GPUExportArena(
                                _MAX_EXPORT_BYTES,
                                self._ipc_export_budget // _MAX_EXPORT_BYTES,
                                self._packed_gather.device,
                            )
                except Exception:
                    self._packed_gather = None
                    logger.warning(
                        "Fused Xavier export unavailable; using per-layer gathers",
                        exc_info=True,
                    )
        schema = {
            name: (
                tuple(self._cache_block_view(layer, tensor).shape[1:]),
                tensor.dtype,
            )
            for layer, cache in self._registered_kv_caches.items()
            for name, tensor in self._iter_kv_tensors(layer, cache)
        }

        self._kv_schema = XavierKVSchema(self._block_size, schema, self._cache_dtype)
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

    def get_handshake_metadata(self):
        return self._kv_schema

    def set_xfer_handshake_metadata(self, metadata):
        if len(metadata) != 1:
            raise ValueError("Xavier KV schema requires one worker (TP=1)")
        self._kv_schema = next(iter(metadata.values()))

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        # This callback runs inside attention and CUDA graph capture/replay.
        # Export registered caches in wait_for_save, outside the model forward,
        # using the current metadata rather than capture-time request IDs.
        return

    def wait_for_save(self):
        if self._direct_handoff and self._is_producer:
            metadata = self._get_connector_metadata()
            if metadata.direct_store:
                self._call(self._ensure_gpu_cache_mapping())
                torch.cuda.synchronize()
            return
        if not self._is_producer:
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, XavierConnectorMetadata)
        pending = metadata.store_requests
        if not pending:
            return
        try:
            self._poll_ipc_exports()
            keys = {key for request in pending for key in request.block_hashes}
            pending_keys = {
                key for keys, _, _ in self._ipc_export_jobs.values() for key in keys
            }
            if keys.issubset(pending_keys):
                return
            # Only this worker writes its actor's snapshots. Publication reports
            # every eviction; successful keys need no per-request control RPC.
            # Periodic authoritative refresh also repairs discovery after recovery.
            if time.monotonic() < self._export_refresh_at and keys.issubset(
                self._exported_keys
            ):
                return
            if time.monotonic() < self._export_refresh_at and (
                keys - self._exported_keys
            ).issubset(pending_keys):
                return
            # New content cannot require authoritative recovery of a mirrored
            # hit. Avoid a readiness RPC on every cold prefill; existing keys
            # still use the periodic refresh above and below.
            ready = None
            if self._export_refresh_at:
                if time.monotonic() >= self._export_refresh_at and (
                    keys & self._exported_keys
                ):
                    # Cold traffic may reuse a small published prefix. Repair
                    # its authoritative publication once per refresh interval,
                    # rather than leaving the deadline expired and querying on
                    # every subsequent cold request.
                    self._call(self._register_blocks(pending))
                if time.monotonic() < self._export_refresh_at:
                    ready = (keys & self._exported_keys) | pending_keys
                elif not (keys & self._exported_keys):
                    ready = pending_keys.copy()
            if ready is not None and keys.issubset(ready):
                return
            missing = (
                self._missing_store_requests(pending, ready)
                if ready is not None
                else self._call(self._filter_store_requests(pending))
            )
            if missing and self._can_queue_ipc_export(missing):
                nbytes = self._cpu_export_size(missing)
                count = sum(len(request.block_hashes) for request in missing)
                per_batch = max(1, _MAX_EXPORT_BYTES // (nbytes // count))
                slots = (count + per_batch - 1) // per_batch
                if (
                    nbytes + sum(entry[1] for entry in self._ipc_export_jobs.values())
                    > self._ipc_export_budget
                ) or (
                    self._gpu_export_arena is not None
                    and slots > len(self._gpu_export_arena.available)
                ):
                    self._poll_ipc_exports(wait=True)
                    missing = self._call(self._filter_store_requests(pending))
                    nbytes = self._cpu_export_size(missing)
                # Gather every source before returning to vLLM. Later engine
                # steps may preempt requests or reuse their original GPU slots.
                batches = list(self._cpu_export_batches(missing, gpu_only=True))
                self._enqueue_ipc_export(batches, nbytes)
                return
            if self._ipc_export_jobs:
                self._poll_ipc_exports(wait=True)
                missing = self._call(self._filter_store_requests(pending))
            if missing:
                self._stage_missing_registered_layers(missing)
            # Re-publish all requested keys, including reused snapshots. This
            # refreshes discovery and retries a failed tracker registration
            # without copying already published KV payloads again.
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
        if self._direct_handoff:
            params = getattr(request, "kv_transfer_params", None) or {}
            handoff = params.get("xavier_direct")
            if (
                self._has_recurrent_cache
                and self._is_producer
                and params.get("do_remote_decode")
            ):
                if (
                    not params.get("_xavier_truncated")
                    and request.num_prompt_tokens > 1
                ):
                    request.prompt_token_ids.pop()
                    request._all_token_ids.pop()
                    request.num_prompt_tokens -= 1
                    request.max_tokens = 1
                    params["_xavier_truncated"] = True
                return 0, False
            if self._is_producer and self._history_enabled:
                previous = self._requests_need_load.get(request.request_id)
                if previous is not None:
                    if previous.local_transfers_by_group:
                        raise RuntimeError("History load already allocated")
                    self._requests_need_load.pop(request.request_id)
                    self._call(self._release_history_request(previous.lease))
                tokens = max(len(request.prompt_token_ids) - 1, 0)
                # A local prefix hit can leave only a partial final block.
                # Restoring that tiny tail costs more actor/layer work than
                # computing it. Keep the local-cache fast path free of RPCs.
                if tokens - num_computed_tokens < self._block_size:
                    return 0, False
                start = num_computed_tokens // self._block_size
                hashes = self._build_xavier_hashes(request.prompt_token_ids[:tokens])[
                    start:
                ]
                lease = "history:" + uuid.uuid4().hex

                async def reserve():
                    ref = await self._get_transfer_ref()
                    return await ref.reserve_direct_history_v1(
                        lease, [key for key, _ in hashes]
                    )

                matched = self._call(reserve())
                if not matched:
                    return 0, False
                self._requests_need_load[request.request_id] = XavierLoadRequest(
                    request.request_id,
                    {self._rank: {key: start + i for i, key in enumerate(matched)}},
                    lease=lease,
                )
                return (
                    min(len(matched) * self._block_size, tokens - num_computed_tokens),
                    True,
                )
            if not self._is_consumer:
                return 0, False
            if handoff is None:
                raise ValueError("Missing direct Xavier handoff metadata")
            if params.get("do_remote_prefill") is False:
                # A preempted decode recomputes its prompt AND generated tokens.
                # Its single-use producer ticket was consumed at allocation.
                return 0, False
            tokens = handoff["tokens"]
            rank, ticket = handoff["rank"], handoff["ticket"]
            if tokens <= num_computed_tokens:
                if ticket:

                    async def release():
                        ref = await self._get_transfer_ref()
                        await ref.release_remote_direct_gpu_v1(rank, ticket)

                    self._call(release())
                params["do_remote_prefill"] = False
                return 0, False

            remote_groups = handoff.get("blocks_by_group", [])
            if self._has_recurrent_cache or remote_groups:
                expected_blocks = (tokens + self._block_size - 1) // self._block_size
                valid_groups = (
                    isinstance(remote_groups, (list, tuple))
                    and len(remote_groups) == len(self._kv_cache_config.kv_cache_groups)
                    and all(
                        isinstance(group, (list, tuple))
                        and len(group)
                        == (1 if i in self._recurrent_groups else expected_blocks)
                        and all(type(block) is int and block >= 0 for block in group)
                        for i, group in enumerate(remote_groups)
                    )
                )
                if not valid_groups:
                    logger.warning(
                        "Incompatible Xavier cache groups; recomputing request %s",
                        request.request_id,
                    )
                    params["do_remote_prefill"] = False
                    self._requests_need_load.pop(request.request_id, None)
                    return 0, False

            async def claim():
                ref = await self._get_transfer_ref()
                return await ref.claim_remote_direct_gpu_v1(rank, ticket)

            if not self._call(claim()):
                params["do_remote_prefill"] = False
                self._requests_need_load.pop(request.request_id, None)
                return 0, False
            start = num_computed_tokens // self._block_size
            blocks = handoff["blocks"]
            self._requests_need_load[request.request_id] = XavierLoadRequest(
                request.request_id,
                {rank: {block: i for i, block in enumerate(blocks) if i >= start}},
                lease=ticket,
                remote_blocks_by_group=remote_groups,
                local_hit_tokens=num_computed_tokens,
            )
            return tokens - num_computed_tokens, True
        self._requests_need_load.pop(request.request_id, None)
        previous = self._leased_requests.pop(request.request_id, None)
        if previous is not None:
            self._call(self._release_load_request(previous))
        if not self._is_consumer:
            return 0, False

        token_ids = list(request.prompt_token_ids or [])
        external_token_count = max(len(token_ids) - 1, 0)
        if external_token_count - num_computed_tokens < self._block_size:
            return 0, False

        local_hit_blocks = num_computed_tokens // self._block_size
        probe_tokens = min(
            external_token_count, (local_hit_blocks + 1) * self._block_size
        )
        hashes = self._build_xavier_hashes(token_ids[:probe_tokens])
        if not hashes:
            return 0, False

        query_hashes = hashes[local_hit_blocks:]
        if not query_hashes:
            return 0, False

        if self._local_directory is not None:
            present = self._local_directory.contains(query_hashes[0][0])
            if present is False:
                return 0, False
            if not self._local_directory.valid:
                self._local_directory.close()
                self._local_directory = None
                self._directory_retry_at = 0.0
        remote = self._call(self._query_remote_blocks(request.request_id, query_hashes))
        # A remote prefix must contain the first uncached block. Cold requests
        # need neither hashes nor a directory RPC payload for the whole prompt.
        # Positive probes still use authoritative queries and snapshot leases.
        if not self._build_contiguous_transfers(query_hashes, remote)[1]:
            return 0, False
        if probe_tokens < external_token_count:
            hashes = self._build_xavier_hashes(token_ids[:external_token_count])
            query_hashes = hashes[local_hit_blocks:]
            remainder = self._call(
                self._query_remote_blocks(request.request_id, query_hashes[1:])
            )
            for rank, blocks in remainder.items():
                remote.setdefault(rank, set()).update(blocks)
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
        return matched_tokens, self._gpu_budget is not None

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        if not self._is_consumer or num_external_tokens <= 0:
            return

        load_request = self._requests_need_load.get(request.request_id)
        if load_request is None:
            return

        block_ids_by_group = self._normalize_block_groups(blocks.get_block_ids())
        # Sliding-window groups use shared null blocks for skipped positions.
        # They are placeholders, not writable destinations for remote KV.
        null_positions = {
            (group_id, index)
            for group_id, group in enumerate(getattr(blocks, "blocks", ()))
            for index, block in enumerate(group)
            if block.is_null
        }
        local_transfers_by_group: Dict[int, Dict[int, Dict[int, int]]] = {}
        for group_id, group_block_ids in enumerate(block_ids_by_group):
            updated: Dict[int, Dict[int, int]] = {}
            for from_rank, remote_to_placeholder in load_request.transfers.items():
                if load_request.remote_blocks_by_group:
                    remote = load_request.remote_blocks_by_group[group_id]
                    if group_id in self._recurrent_groups:
                        if len(remote) != 1 or len(group_block_ids) != 1:
                            raise ValueError(
                                "Expected one recurrent state block per request"
                            )
                        updated[from_rank] = {remote[0]: group_block_ids[0]}
                    else:
                        start = load_request.local_hit_tokens // self._block_size
                        updated[from_rank] = {
                            block: group_block_ids[i]
                            for i, block in enumerate(remote)
                            if i >= start and (group_id, i) not in null_positions
                        }
                    continue
                # Placeholder indices are absolute positions in the prompt.
                # Allocation includes locally cached prefix blocks: remote
                # suffix blocks must not overwrite that prefix. Indexing by
                # position also preserves order across multiple source ranks.
                updated[from_rank] = {
                    remote_block_id: group_block_ids[placeholder]
                    for remote_block_id, placeholder in remote_to_placeholder.items()
                    if (group_id, placeholder) not in null_positions
                }
            local_transfers_by_group[group_id] = updated

        self._requests_need_load[request.request_id] = XavierLoadRequest(
            request_id=request.request_id,
            transfers=load_request.transfers,
            lease=load_request.lease,
            local_transfers_by_group=local_transfers_by_group,
        )
        if self._direct_handoff and not load_request.lease.startswith("history:"):
            request.kv_transfer_params["do_remote_prefill"] = False
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
        if self._direct_handoff and self._is_producer:
            meta.direct_sends = self._direct_sends
            self._direct_sends = set()
            meta.direct_store = bool(scheduler_output.num_scheduled_tokens)
        if self._is_producer and not self._direct_handoff:
            self._build_store_meta(scheduler_output, meta)
        if self._is_consumer:
            self._build_load_meta(scheduler_output, meta)
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: List[int],
        block_ids_by_group: Optional[Tuple[List[int], ...]] = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        if self._direct_handoff:
            pending = self._requests_need_load.get(request.request_id)
            if pending is not None and pending.lease.startswith("history:"):
                if not pending.local_transfers_by_group:
                    self._requests_need_load.pop(request.request_id)
                    self._call(self._release_history_request(pending.lease))
                else:
                    # Allocated async loads must still drain after an abort.
                    return False, None
            params = getattr(request, "kv_transfer_params", None) or {}
            if self._is_producer and params.get("do_remote_decode"):
                from vllm.v1.request import RequestStatus

                if request.status == RequestStatus.FINISHED_ABORTED:
                    return False, None
                token_count = max(len(request.prompt_token_ids) - 1, 0)
                if self._has_recurrent_cache:
                    if not params.get("_xavier_truncated"):
                        return False, {
                            "do_remote_prefill": True,
                            "xavier_direct": {
                                "rank": self._rank,
                                "ticket": "",
                                "blocks": [],
                                "tokens": 0,
                            },
                        }
                    token_count = len(request.prompt_token_ids)
                count = min(
                    len(block_ids),
                    (token_count + self._block_size - 1) // self._block_size,
                )
                blocks = block_ids[:count]
                grouped = None
                registered: List[int] | Dict[str, List[int]] = blocks
                if block_ids_by_group is not None:
                    grouped = [
                        list(ids) if i in self._recurrent_groups else list(ids[:count])
                        for i, ids in enumerate(block_ids_by_group)
                    ]
                    registered = {
                        name: grouped[group_id]
                        for name, group_id in self._layer_group_ids.items()
                    }
                ticket = uuid.uuid4().hex if blocks else ""
                if blocks:

                    async def register():
                        ref = await self._get_transfer_ref()
                        history_args = ()
                        if self._history_enabled:
                            hashes = self._build_xavier_hashes(
                                request.prompt_token_ids[:token_count]
                            )
                            history_args = ([key for key, _ in hashes[:count]],)
                        await ref.register_direct_gpu_v1(
                            ticket, request.request_id, registered, *history_args
                        )

                    self._call(register())
                    self._direct_sends.add(request.request_id)
                    logger.debug(
                        "Register Xavier direct handoff: request=%s blocks=%s",
                        request.request_id,
                        len(blocks),
                    )
                return bool(blocks), {
                    "do_remote_prefill": True,
                    "xavier_direct": {
                        "rank": self._rank,
                        "address": self._xavier_config.get("rank_address"),
                        "ticket": ticket,
                        "blocks": blocks,
                        **({"blocks_by_group": grouped} if grouped is not None else {}),
                        "tokens": min(token_count, count * self._block_size),
                    },
                }
            pending = self._requests_need_load.get(request.request_id)
            if pending is not None and pending.local_transfers_by_group:
                # The worker must drain all writes before freeing destinations.
                return False, None
            self._requests_need_load.pop(request.request_id, None)
            handoff = params.get("xavier_direct")
            if handoff and params.get("do_remote_prefill") is not False:

                async def release():
                    ref = await self._get_transfer_ref()
                    await ref.release_remote_direct_gpu_v1(
                        handoff["rank"], handoff["ticket"]
                    )

                self._call(release())
                params["do_remote_prefill"] = False
            return False, None
        self._chunked_prefill.pop(request.request_id, None)
        pending = self._requests_need_load.get(request.request_id)
        if (
            self._gpu_budget is not None
            and pending is not None
            and pending.local_transfers_by_group
        ):
            # Allocation already put this request in WAITING_FOR_REMOTE_KVS.
            # Submit even if aborted before metadata is emitted; completion is
            # needed to release the scheduler's retained destination blocks.
            return False, None
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
        if self._has_recurrent_cache and block_ids:
            full_group = next(
                i for i in range(len(block_ids)) if i not in self._recurrent_groups
            )
            return self.request_finished(request, block_ids[full_group], block_ids)
        return self.request_finished(request, block_ids[0] if block_ids else [])

    def take_events(self) -> Iterable:
        return ()

    def _call(self, coro):
        # vLLM creates the KV connector inside EngineCore after CUDA/JIT
        # initialization. Starting a helper thread here can trip glibc static
        # TLS allocation in CUDA-heavy environments, so run actor calls on a
        # shared loop in the EngineCore thread. Scheduler and worker connectors
        # must not switch loops and invalidate xoscar's connection cache.
        if self._loop is None:
            self._loop = acquire_actor_loop()
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
            ref = await xo.actor_ref(
                address=self._xavier_config.get("rank_address"),
                uid=f"{TransferActor.default_uid()}-{self._rank}",
            )
            await ref.configure_snapshots_v1(self._num_cache_blocks)
            if self._kv_schema is not None:
                await ref.configure_kv_schema_v1(
                    self._kv_schema.block_size,
                    self._kv_schema.layers,
                    self._kv_schema.cache_dtype,
                )
            self._transfer_ref = ref
        return self._transfer_ref

    async def _release_history_request(self, lease):
        transfer = await self._get_transfer_ref()
        await transfer.release_direct_history_v1(lease)

    async def _reserve_load_request(self, request):
        transfer = await self._get_transfer_ref()
        return await transfer.reserve_remote_blocks_v1(request.lease, request.transfers)

    async def _release_load_request(self, request):
        transfer = await self._get_transfer_ref()
        if self._direct_handoff:
            if request.lease.startswith("history:"):
                await transfer.release_direct_history_v1(request.lease)
            else:
                await transfer.release_remote_direct_gpu_v1(
                    next(iter(request.transfers)), request.lease
                )
        else:
            await transfer.release_remote_blocks_v1(request.lease, request.transfers)

    async def _query_remote_blocks(
        self,
        request_id: str,
        query_hashes: List[Tuple[int, int]],
    ) -> Dict[int, set[Tuple[int, int, int]]]:
        tracker_ref = await self._get_tracker_ref()
        if (
            sys.platform == "linux"
            and self._gpu_budget is None
            and self._local_directory is None
            and time.monotonic() >= self._directory_retry_at
        ):
            from ...xavier.local_directory import LocalBlockDirectory

            self._directory_retry_at = time.monotonic() + 1.0
            self._local_directory = LocalBlockDirectory.attach(
                await tracker_ref.get_snapshot_directory(0)
            )
            if (
                self._local_directory is not None
                and self._local_directory.contains(query_hashes[0][0]) is False
            ):
                return {}
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

    async def _filter_store_requests(
        self, requests: List[XavierStoreRequest], ready: Optional[set[int]] = None
    ) -> List[XavierStoreRequest]:
        if ready is None:
            transfer_ref = await self._get_transfer_ref()
            ready = set(
                await transfer_ref.ready_blocks_for_export_v1(
                    [key for request in requests for key in request.block_hashes]
                )
            )
        ready.update(
            key for keys, _, _ in self._ipc_export_jobs.values() for key in keys
        )
        return self._missing_store_requests(requests, ready)

    @staticmethod
    def _missing_store_requests(
        requests: List[XavierStoreRequest], ready: set[int]
    ) -> List[XavierStoreRequest]:
        missing = []
        for request in requests:
            indices = []
            for i, key in enumerate(request.block_hashes):
                if key not in ready:
                    indices.append(i)
                    ready.add(key)
            if indices:
                missing.append(
                    XavierStoreRequest(
                        request.request_id,
                        [request.block_ids[i] for i in indices],
                        [request.block_hashes[i] for i in indices],
                        [
                            [group[i] for i in indices]
                            for group in request.block_ids_by_group
                        ],
                    )
                )
        return missing

    async def _register_blocks(self, requests: List[XavierStoreRequest]):
        if not requests:
            return
        tracker_ref = await self._get_tracker_ref()
        transfer_ref = await self._get_transfer_ref()
        expected_layers = {
            name
            for layer, cache in self._registered_kv_caches.items()
            for name, _ in self._iter_kv_tensors(layer, cache)
        }
        if expected_layers:
            # Registered caches share one schema. Avoid constructing layer
            # groups and copying the actor's already unique result on hot hits.
            keys = list(
                dict.fromkeys(
                    key for request in requests for key in request.block_hashes
                )
            )
            available, evicted = await transfer_ref.publish_blocks_v1(
                keys, expected_layers
            )
        else:
            groups: Dict[frozenset[str], Dict[int, None]] = {}
            for request in requests:
                layers = frozenset(
                    self._request_staged_layers.get(request.request_id, set())
                )
                groups.setdefault(layers, {}).update(
                    dict.fromkeys(request.block_hashes)
                )
            available_keys: Dict[int, None] = {}
            evicted_keys: Dict[int, None] = {}
            for layers, group_keys in groups.items():
                ready, removed = await transfer_ref.publish_blocks_v1(
                    list(group_keys), set(layers)
                )
                available_keys.update(dict.fromkeys(ready))
                evicted_keys.update(dict.fromkeys(removed))
            available, evicted = list(available_keys), list(evicted_keys)
        requested_keys = {key for request in requests for key in request.block_hashes}
        self._exported_keys.difference_update(requested_keys - set(available))
        if evicted:
            self._exported_keys.difference_update(evicted)
            await tracker_ref.unregister_blocks(0, self._rank, evicted)
        # Retry discovery even when ready snapshots required no payload export.
        # Transport addresses identify immutable content, not recyclable GPU slots.
        await tracker_ref.register_snapshot_blocks(0, available, self._rank)
        self._exported_keys.update(available)
        self._export_refresh_at = time.monotonic() + _EXPORT_REFRESH_SECONDS
        logger.debug(
            "Xavier V1 registered blocks: requests=%s, rank=%s, blocks=%s",
            len(requests),
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
        # Amortize actor/control RPCs for small attention blocks, while keeping
        # large cache replies bounded. A single oversized block stays indivisible.
        block_bytes = math.prod(recv_shape[1:]) * recv_dtype.itemsize
        batch_size = max(1, min(_MAX_READ_BLOCKS, _MAX_READ_BYTES // block_bytes))
        items = list(src_to_dst.items())
        blocks = []
        for offset in range(0, len(items), batch_size):
            batch = dict(items[offset : offset + batch_size])
            blocks.append(
                await transfer_ref.read_layer_blocks_v1(
                    from_rank,
                    layer_name,
                    batch,
                    (len(batch), *recv_shape[1:]),
                    recv_dtype,
                )
            )
        return torch.cat(blocks, dim=0)

    async def _read_request_payload(self, transfer, rank, reads, *, pin_memory):
        local = self._local_read
        if local is not None and not local.valid():
            local.close()
            self._local_read = None
            self._local_read_checked = False
        if sys.platform == "linux" and not self._local_read_checked:
            from ...xavier.backends.torch.local_read import LocalReadBuffer

            self._local_read = LocalReadBuffer.attach(
                await transfer.get_local_read_metadata_v1()
            )
            self._local_read_checked = True
        local = self._local_read
        if local is not None:
            lease = await transfer.read_request_blocks_local_v1(
                rank, reads, local.token
            )
            if lease is not None:
                try:
                    return local.copy(
                        lease, sum(read.nbytes for read in reads), pin_memory=pin_memory
                    )
                finally:
                    await transfer.release_local_read_v1(lease)
        return await transfer.read_request_blocks_v1(rank, reads)

    def _load_request_blocks(self, request):
        from ...xavier.backends.torch.request_transfer import (
            LayerRead,
            batch_reads,
            unpack_reads,
        )

        caches = {
            name: block_major_view(tensor, self._num_cache_blocks)
            for layer, cache in self._registered_kv_caches.items()
            for name, tensor in self._iter_kv_tensors(layer, cache)
        }

        async def load():
            transfer = await self._get_transfer_ref()
            indices_by_destination = {}
            for rank in request.transfers:
                reads = []
                for layer, tensor in caches.items():
                    mapping = self._get_local_transfer_map(request, layer, rank)
                    if mapping:
                        dtype = (
                            XAVIER_BF16_TRANSPORT_DTYPE
                            if tensor.dtype == torch.bfloat16
                            else tensor.dtype
                        )
                        reads.append(
                            LayerRead(
                                layer,
                                list(mapping),
                                list(mapping.values()),
                                tuple(tensor.shape[1:]),
                                dtype,
                                tensor.dtype,
                            )
                        )
                for batch in batch_reads(reads):
                    devices = {caches[read.layer].device for read in batch}
                    cuda = len(devices) == 1 and next(iter(devices)).type == "cuda"
                    with profile_stage(
                        "load_rpc",
                        request_id=request.request_id,
                        rank=self._rank,
                        nbytes=sum(read.nbytes for read in batch),
                        blocks=sum(len(read.keys) for read in batch),
                    ):
                        payload = await self._read_request_payload(
                            transfer, rank, batch, pin_memory=cuda
                        )
                    if cuda:
                        device = next(iter(devices))
                        with profile_stage(
                            "load_h2d",
                            device=device,
                            request_id=request.request_id,
                            rank=self._rank,
                            nbytes=payload.numel(),
                        ):
                            # Upload one packed payload instead of synchronizing
                            # a pageable copy and index upload for every layer.
                            payload = payload.pin_memory().to(device, non_blocking=True)
                    for read, blocks in unpack_reads(payload, batch):
                        cache = caches[read.layer]
                        with profile_stage(
                            "load_h2d",
                            device=cache.device,
                            request_id=request.request_id,
                            layer=read.layer,
                            rank=self._rank,
                            nbytes=read.nbytes,
                            blocks=len(read.keys),
                        ):
                            if (
                                cache.dtype == torch.bfloat16
                                and blocks.dtype == XAVIER_BF16_TRANSPORT_DTYPE
                            ):
                                blocks = blocks.view(torch.bfloat16)
                            elif blocks.dtype != cache.dtype:
                                raise RuntimeError(
                                    f"Unexpected Xavier KV dtype {blocks.dtype} for cache {cache.dtype}"
                                )
                            destination = (cache.device, tuple(read.destinations))
                            if destination not in indices_by_destination:
                                if cache.is_cuda:
                                    indices_by_destination[destination] = torch.tensor(
                                        read.destinations,
                                        dtype=torch.long,
                                        pin_memory=True,
                                    ).to(cache.device, non_blocking=True)
                                else:
                                    indices_by_destination[destination] = torch.tensor(
                                        read.destinations, device=cache.device
                                    )
                            cache[indices_by_destination[destination]] = blocks.to(
                                cache.device, non_blocking=True
                            )

        self._call(load())

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
                # Pass the logical dtype so the producer can validate its snapshot.
                recv_shape = (len(local_block_ids), *tuple(kv_tensor.shape[1:]))
                with profile_stage(
                    "load_rpc",
                    request_id=request.request_id,
                    layer=kv_layer_name,
                    blocks=len(local_block_ids),
                    rank=self._rank,
                ):
                    blocks = self._call(
                        self._read_layer_blocks(
                            kv_layer_name,
                            from_rank,
                            src_to_dst,
                            recv_shape,
                            kv_tensor.dtype,
                        )
                    )
                with profile_stage(
                    "load_h2d",
                    device=kv_tensor.device,
                    request_id=request.request_id,
                    layer=kv_layer_name,
                    nbytes=blocks.numel() * blocks.element_size(),
                    rank=self._rank,
                ):
                    if (
                        kv_tensor.dtype == torch.bfloat16
                        and blocks.dtype == XAVIER_BF16_TRANSPORT_DTYPE
                    ):
                        blocks = blocks.view(torch.bfloat16)
                    elif blocks.dtype != kv_tensor.dtype:
                        raise RuntimeError(
                            f"Unexpected Xavier KV dtype {blocks.dtype} for cache {kv_tensor.dtype}"
                        )
                    kv_tensor[
                        torch.tensor(local_block_ids, device=kv_tensor.device)
                    ] = blocks.to(device=kv_tensor.device, non_blocking=True)
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
            with profile_stage(
                "store_d2h",
                device=kv_tensor.device,
                request_id=request.request_id,
                layer=kv_layer_name,
                blocks=num_blocks,
                rank=self._rank,
            ):
                block_ids_tensor = torch.tensor(
                    source_block_ids, device=kv_tensor.device, dtype=torch.long
                )
                blocks = (
                    kv_tensor.index_select(0, block_ids_tensor)
                    .detach()
                    .cpu()
                    .contiguous()
                )
            with profile_stage(
                "store_rpc",
                request_id=request.request_id,
                layer=kv_layer_name,
                nbytes=blocks.numel() * blocks.element_size(),
                rank=self._rank,
            ):
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
        if getattr(self, "_gpu_budget", None) is not None:
            self._call(self._stage_gpu_requests(requests))
            return
        if not self._registered_kv_caches:
            return

        self._call(self._stage_cpu_requests(requests))

    def _cpu_export_batches(
        self, requests: List[XavierStoreRequest], *, gpu_only: bool = False
    ) -> Iterator[_CPUExportBatch]:
        caches = {
            name: block_major_view(tensor, self._num_cache_blocks)
            for layer, cache in self._registered_kv_caches.items()
            for name, tensor in self._iter_kv_tensors(layer, cache)
        }
        block_bytes = sum(
            (
                ((math.prod(tensor.shape[1:]) * tensor.element_size() + 7) // 8 * 8)
                if gpu_only
                else math.prod(tensor.shape[1:]) * tensor.element_size()
            )
            for tensor in caches.values()
        )
        if not block_bytes:
            return
        # Keep every layer of a block in the same RPC. One oversized block is
        # indivisible, just as on the read path.
        batch_size = max(1, _MAX_EXPORT_BYTES // block_bytes)
        keys: List[int] = []
        sources: Dict[str, List[int]] = {name: [] for name in caches}
        for request in requests:
            layer_ids = {
                name: self._get_source_block_ids(request, name) for name in caches
            }
            count = min(
                len(request.block_hashes),
                len(request.block_ids),
                *(len(ids) for ids in layer_ids.values()),
            )
            for i in range(count):
                keys.append(request.block_hashes[i])
                for name, ids in layer_ids.items():
                    sources[name].append(ids[i])
                if len(keys) == batch_size:
                    if gpu_only:
                        yield self._prepare_cpu_export(
                            caches, keys, sources, gpu_only=True
                        )
                    else:
                        yield self._prepare_cpu_export(caches, keys, sources)
                    keys = []
                    sources = {name: [] for name in caches}
        if keys:
            if gpu_only:
                yield self._prepare_cpu_export(caches, keys, sources, gpu_only=True)
            else:
                yield self._prepare_cpu_export(caches, keys, sources)

    def _prepare_cpu_export(
        self,
        caches: Dict[str, torch.Tensor],
        keys: List[int],
        sources: Dict[str, List[int]],
        *,
        gpu_only: bool = False,
    ) -> _CPUExportBatch:
        if gpu_only and self._packed_gather is not None:
            slot, output = None, None
            arena = self._gpu_export_arena
            if (
                arena is not None
                and len(keys) * self._packed_gather.row_bytes <= arena.slot_bytes
            ):
                slot, output = arena.allocate(
                    (len(keys), self._packed_gather.row_bytes)
                )
            with profile_stage(
                "store_gather",
                device=self._packed_gather.device,
                blocks=len(keys),
                rank=self._rank,
            ):
                try:
                    if output is None:
                        packed, owners = self._packed_gather(sources)
                    else:
                        packed, owners = self._packed_gather(sources, output)
                        assert arena is not None and slot is not None
                        arena.record(slot, tuple(packed.shape))
                except BaseException:
                    if slot is not None:
                        torch.cuda.current_stream(
                            self._packed_gather.device
                        ).synchronize()
                        assert arena is not None
                        arena.release([slot])
                    raise
            return (
                _PackedExportEntries(keys, packed, self._packed_gather.layers, slot),
                owners,
                [],
            )
        entries = []
        owners = []
        events = {}
        indices_by_source = {}
        try:
            for name, tensor in caches.items():
                with profile_stage(
                    "store_d2h",
                    device=tensor.device,
                    layer=name,
                    blocks=len(keys),
                    rank=self._rank,
                ):
                    source = (tensor.device, tuple(sources[name]))
                    if source not in indices_by_source:
                        if tensor.is_cuda:
                            host_indices = torch.tensor(
                                sources[name], dtype=torch.long, pin_memory=True
                            )
                            indices_by_source[source] = host_indices.to(
                                tensor.device, non_blocking=True
                            )
                            owners.append(host_indices)
                        else:
                            indices_by_source[source] = torch.tensor(
                                sources[name], device=tensor.device, dtype=torch.long
                            )
                    indices = indices_by_source[source]
                    gathered = tensor.index_select(0, indices).detach()
                    if gpu_only:
                        entries.append((name, keys, gathered))
                        continue
                    if tensor.is_cuda:
                        if tensor.device not in events:
                            events[tensor.device] = torch.cuda.Event()
                        event = events[tensor.device]
                        blocks = torch.empty_like(
                            gathered, device="cpu", pin_memory=True
                        )
                        # Retain both sides until the event has completed, including
                        # when allocation, staging or an actor RPC fails.
                        owners.append(gathered)
                        entries.append((name, keys, blocks))
                        stream = self._cpu_export_streams.get(tensor.device)
                        if stream is None:
                            stream = torch.cuda.Stream(device=tensor.device)
                            self._cpu_export_streams[tensor.device] = stream
                        # Source gathers precede future writes on the model
                        # stream. Only immutable gathered buffers cross to the
                        # copy stream, so subsequent forwards need not wait for
                        # PCIe D2H traffic.
                        stream.wait_stream(torch.cuda.current_stream(tensor.device))
                        with torch.cuda.stream(stream):
                            blocks.copy_(gathered, non_blocking=True)
                            event.record(stream)
                    else:
                        entries.append((name, keys, gathered.contiguous()))
        except BaseException:
            for event in events.values():
                event.synchronize()
            raise
        return entries, owners, list(events.values())

    def _cpu_export_size(self, requests: List[XavierStoreRequest]) -> int:
        return sum(len(request.block_hashes) for request in requests) * sum(
            (
                math.prod(block_major_view(tensor, self._num_cache_blocks).shape[1:])
                * tensor.element_size()
                + 7
            )
            // 8
            * 8
            for layer, cache in self._registered_kv_caches.items()
            for _, tensor in self._iter_kv_tensors(layer, cache)
        )

    def _can_queue_ipc_export(self, requests: List[XavierStoreRequest]) -> bool:
        return (
            sys.platform == "linux"
            and self._gpu_budget is None
            and bool(self._registered_kv_caches)
            and all(
                tensor.is_cuda
                for layer, cache in self._registered_kv_caches.items()
                for _, tensor in self._iter_kv_tensors(layer, cache)
            )
            and self._cpu_export_size(requests) <= self._ipc_export_budget
        )

    def _prepare_ipc_export(
        self, batches: List[_CPUExportBatch], nbytes: int
    ) -> Tuple[str, list, List[int]]:
        from torch.multiprocessing.reductions import reduce_tensor

        ticket = uuid.uuid4().hex
        descriptors = []
        owners = []
        slots = []
        for entries, indices, _ in batches:
            if not entries:
                continue
            if isinstance(entries, _PackedExportEntries):
                packed, layers = entries.packed, entries.layers
            else:
                count = len(entries[0][1])
                pieces, layers, offset = [], [], 0
                for name, keys, value in entries:
                    if offset % 8:
                        padding = 8 - offset % 8
                        pieces.append(
                            torch.zeros(
                                (count, padding), dtype=torch.uint8, device=value.device
                            )
                        )
                        offset += padding
                    part = value.view(torch.uint8).reshape(count, -1)
                    layers.append(
                        (
                            name,
                            tuple(value.shape[1:]),
                            value.dtype,
                            offset,
                            part.shape[1],
                        )
                    )
                    pieces.append(part)
                    offset += part.shape[1]
                if offset % 8:
                    pieces.append(
                        torch.zeros(
                            (count, 8 - offset % 8),
                            dtype=torch.uint8,
                            device=pieces[0].device,
                        )
                    )
                packed = torch.cat(pieces, dim=1)
            if isinstance(entries, _PackedExportEntries) and entries.slot is not None:
                assert self._gpu_export_arena is not None
                slots.append(entries.slot)
                descriptor = {
                    "gpu_export": self._gpu_export_arena.source,
                    "slot": entries.slot,
                    "shape": tuple(packed.shape),
                }
            else:
                descriptor = reduce_tensor(packed)[1]
            keys = (
                entries.keys
                if isinstance(entries, _PackedExportEntries)
                else entries[0][1]
            )
            descriptors.append((keys, descriptor, layers))
            owners.append(packed)
            owners.extend(indices)
        all_keys = {key for keys, _, _ in descriptors for key in keys}
        # Keep ownership even if an enqueue acknowledgement is lost; the actor
        # may already have started copying these IPC buffers.
        self._ipc_export_jobs[ticket] = (all_keys, nbytes, owners)
        if slots:
            self._gpu_export_tickets[ticket] = slots

        return ticket, descriptors, slots

    async def _send_ipc_export(
        self, ticket: str, descriptors: list, slots: List[int]
    ) -> None:
        transfer = await self._get_transfer_ref()
        # Serialize registration and recovery. Each mapping owns one set of CUDA
        # IPC references for the lifetime of the actor, shared by all exports.
        async with self._gpu_mapping_lock:

            async def register():
                assert self._gpu_export_arena is not None
                await transfer.register_snapshot_export_source_v1(
                    self._gpu_export_arena.source, self._gpu_export_arena.metadata()
                )
                self._gpu_export_registered = True

            async def enqueue():
                return await transfer.enqueue_snapshot_export_v1(
                    ticket,
                    descriptors,
                    self._xavier_config.get("block_tracker_address"),
                    self._xavier_config.get("block_tracker_uid"),
                )

            try:
                if slots and not self._gpu_export_registered:
                    await register()
                if await enqueue() is False:
                    self._exported_keys.clear()
                    self._export_refresh_at = 0
                    await register()
                    if await enqueue() is False:
                        raise RuntimeError("Xavier export source registration lost")
            except BaseException:
                self._gpu_export_registered = False
                raise

    async def _queue_ipc_export(
        self, batches: List[_CPUExportBatch], nbytes: int
    ) -> None:
        await self._send_ipc_export(*self._prepare_ipc_export(batches, nbytes))

    def _enqueue_ipc_export(self, batches: List[_CPUExportBatch], nbytes: int) -> None:
        # Only the reusable arena has a source-close handshake that drains late
        # submissions before freeing GPU storage. Legacy per-batch CUDA IPC
        # exports retain their synchronous acknowledgement and cleanup behavior.
        if self._gpu_export_arena is None or not all(
            isinstance(entries, _PackedExportEntries) and entries.slot is not None
            for entries, _, _ in batches
        ):
            self._call(self._queue_ipc_export(batches, nbytes))
            return
        ticket, descriptors, slots = self._prepare_ipc_export(batches, nbytes)
        if self._loop is None:
            self._loop = acquire_actor_loop()
        # Own slots and index tensors before scheduling the RPC. Advance the
        # existing EngineCore loop to submit it without waiting for an ACK.
        # Subsequent polling progresses the task; the actor copies and publishes
        # independently once it receives the immutable arena descriptor.
        self._ipc_export_enqueues[ticket] = self._loop.create_task(
            self._send_ipc_export(ticket, descriptors, slots)
        )
        self._call(asyncio.sleep(0))

    def _poll_ipc_exports(self, *, wait: bool = False) -> None:
        if self._ipc_export_enqueues or self._ipc_export_poll is not None:
            self._call(asyncio.sleep(0))
        submission_error = None
        for ticket, submitted in list(self._ipc_export_enqueues.items()):
            if wait or submitted.done():
                del self._ipc_export_enqueues[ticket]
                # On a lost acknowledgement keep the GPU owners until the
                # source-close handshake has drained any accepted copy.
                try:
                    if wait and not submitted.done():
                        self._call(asyncio.shield(submitted))
                    submitted.result()
                except Exception as exc:
                    submission_error = submission_error or exc
        if submission_error is not None:
            # Drain every submission before shutdown can invalidate the source.
            # A late registration must never reopen a freed arena mapping.
            raise submission_error
        future = self._ipc_export_poll
        if not self._ipc_export_jobs or (
            future is None
            and (not wait and time.monotonic() < self._ipc_export_poll_at)
        ):
            return

        tickets = [
            ticket
            for ticket in self._ipc_export_jobs
            if ticket not in self._ipc_export_enqueues
        ]
        if not tickets and future is None:
            return

        async def poll():
            transfer = await self._get_transfer_ref()
            return await transfer.poll_snapshot_exports_v1(tickets, wait=wait)

        if future is not None:
            if not wait and not future.done():
                return
            try:
                if wait and not future.done():
                    self._call(asyncio.shield(future))
                results = future.result()
            finally:
                self._ipc_export_poll = None
        elif not wait and self._loop is not None:
            self._ipc_export_poll = self._loop.create_task(poll())
            self._call(asyncio.sleep(0))
            return
        else:
            results = self._call(poll())
        error = None
        for ticket, (available, evicted) in results.items():
            slots = self._gpu_export_tickets.get(ticket)
            if slots is not None:
                assert self._gpu_export_arena is not None
                self._gpu_export_arena.release(slots)
                del self._gpu_export_tickets[ticket]
            self._ipc_export_jobs.pop(ticket)
            if available is None:
                self._exported_keys.clear()
                self._export_refresh_at = 0
                error = error or RuntimeError(
                    f"Xavier snapshot export failed: {evicted}"
                )
                continue
            self._exported_keys.difference_update(evicted)
            self._exported_keys.update(available)
            if not self._export_refresh_at:
                self._export_refresh_at = time.monotonic() + _EXPORT_REFRESH_SECONDS
        self._ipc_export_poll_at = time.monotonic() + 0.01
        if error is not None:
            raise error
        if wait and future is not None and self._ipc_export_jobs:
            self._poll_ipc_exports(wait=True)

    async def _stage_cpu_requests(self, requests: List[XavierStoreRequest]) -> None:
        if not requests or not self._registered_kv_caches:
            return
        transfer = await self._get_transfer_ref()
        pending: Optional[_CPUExportBatch] = None

        async def stage(batch: _CPUExportBatch) -> None:
            entries, _, events = batch
            for event in events:
                event.synchronize()
            with profile_stage(
                "store_rpc",
                rank=self._rank,
                nbytes=sum(t.numel() * t.element_size() for _, _, t in entries),
            ):
                # xoscar's Torch serializer exposes a NumPy memoryview. Socket
                # backpressure requires len(buffer) to count bytes, so send
                # flat uint8 views rather than multidimensional typed buffers.
                await transfer.stage_layer_batches_v1(
                    [
                        (name, keys, t.view(torch.uint8).reshape(-1))
                        for name, keys, t in entries
                    ],
                    [(tuple(t.shape), t.dtype) for _, _, t in entries],
                )

        try:
            for current in self._cpu_export_batches(requests):
                previous, pending = pending, current
                if previous is not None:
                    # The next D2H copy overlaps the previous batch's actor RPC.
                    # At most two bounded batches own GPU and pinned CPU buffers.
                    await stage(previous)
                    previous = None
            if pending is not None:
                await stage(pending)
        finally:
            if pending is not None:
                for event in pending[2]:
                    event.synchronize()

    async def _ensure_gpu_cache_mapping(self):
        from torch.multiprocessing.reductions import reduce_tensor

        transfer = await self._get_transfer_ref()
        async with self._gpu_mapping_lock:
            if not self._gpu_cache_mapped:
                descriptors = {}
                for name, cache in self._registered_kv_caches.items():
                    for layer, tensor in self._iter_kv_tensors(name, cache):
                        tensor = self._cache_block_view(name, tensor)
                        if not tensor.is_cuda:
                            raise ValueError(
                                "Xavier GPU transfer requires CUDA KV caches"
                            )
                        _, descriptors[layer] = reduce_tensor(tensor)
                if not descriptors:
                    raise ValueError(
                        "Xavier GPU transfer requires registered KV caches"
                    )
                torch.cuda.synchronize()
                await transfer.map_gpu_caches_v1(
                    descriptors,
                    0 if self._has_recurrent_cache else self._gpu_budget,
                    direct_handoff=self._direct_handoff,
                )
                self._gpu_cache_mapped = True
        return transfer

    async def _stage_gpu_requests(self, requests):
        if not requests:
            return
        transfer = await self._ensure_gpu_cache_mapping()
        entries = []
        for request in requests:
            layers = {}
            for name, cache in self._registered_kv_caches.items():
                for layer, _ in self._iter_kv_tensors(name, cache):
                    ids = self._get_source_block_ids(request, layer)
                    count = min(len(request.block_ids), len(ids))
                    if count:
                        layers[layer] = (request.block_hashes[:count], ids[:count])
            entries.append(layers)
        torch.cuda.synchronize()
        await transfer.stage_gpu_requests_v1(entries)

    async def _submit_gpu_requests(self, requests):
        try:
            transfer = await self._ensure_gpu_cache_mapping()
            entries = []
            for request in requests:
                ranks = {}
                for rank in request.transfers:
                    layers = {}
                    for name, cache in self._registered_kv_caches.items():
                        for layer, _ in self._iter_kv_tensors(name, cache):
                            mapping = self._get_local_transfer_map(request, layer, rank)
                            if mapping:
                                layers[layer] = mapping
                    if layers:
                        ranks[rank] = layers
                entries.append(ranks)
            torch.cuda.synchronize()

            async def load():
                if self._direct_handoff:
                    historical = [
                        i
                        for i, r in enumerate(requests)
                        if r.lease.startswith("history:")
                    ]
                    direct = [
                        i
                        for i, r in enumerate(requests)
                        if not r.lease.startswith("history:")
                    ]
                    if historical:
                        await transfer.load_direct_history_v1(
                            [entries[i] for i in historical],
                            [requests[i].lease for i in historical],
                        )
                    if direct:
                        return await transfer.load_direct_gpu_v1(
                            [entries[i] for i in direct],
                            [requests[i].lease for i in direct],
                        )
                else:
                    await transfer.load_gpu_requests_v1(
                        entries, [(r.lease, r.transfers) for r in requests]
                    )

            task = asyncio.create_task(load())
        except BaseException:
            # Worker metadata already owns these leases. Until task creation
            # succeeds, the TransferActor has no operation that can release them.
            results = await asyncio.gather(
                *(self._release_load_request(r) for r in requests if r.lease),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    logger.warning(
                        "Failed to release Xavier lease after submission failure",
                        exc_info=(type(result), result, result.__traceback__),
                    )
            raise
        self._gpu_load_jobs[task] = [r.request_id for r in requests]
        # Dispatch before returning to forward execution, without waiting for
        # transfer completion or introducing another polling RPC per engine step.
        await asyncio.sleep(0)

    def _cache_block_view(self, layer_name: str, tensor: torch.Tensor) -> torch.Tensor:
        if not self._has_recurrent_cache:
            return block_major_view(tensor, self._num_cache_blocks)
        if self._get_layer_group_id(layer_name) in self._recurrent_groups:
            return block_major_view(tensor, self._num_cache_blocks)
        # HMA attention kernels may split a logical block into multiple physical
        # blocks. Keep that extra dimension inside the transfer unit, as a view.
        view = block_major_view(tensor, self._num_cache_blocks, allow_multiple=True)
        physical = view.shape[0]
        return view.unflatten(
            0, (self._num_cache_blocks, physical // self._num_cache_blocks)
        )

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
        if self._gpu_budget is not None:
            # Async loads are waiting for KV, not scheduled for a forward pass.
            # Transfer lease ownership only after destination allocation.
            for req_id, load in list(self._requests_need_load.items()):
                if load.local_transfers_by_group:
                    meta.load_requests.append(load)
                    del self._requests_need_load[req_id]
                    self._leased_requests.pop(req_id, None)
            return
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
        cacheable = len(token_ids) <= _MAX_HASH_CACHE_TOKENS
        cache_key = (
            (self._block_size, self._none_hash, tuple(token_ids)) if cacheable else None
        )
        if cache_key is not None and cache_key in self._hash_cache:
            self._hash_cache.move_to_end(cache_key)
            return list(self._hash_cache[cache_key])
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
        if cache_key is not None:
            self._hash_cache[cache_key] = tuple(hashes)
            self._hash_cache_tokens += len(token_ids)
            while (
                self._hash_cache_tokens > _MAX_HASH_CACHE_TOKENS
                or len(self._hash_cache) > _MAX_HASH_CACHE_ENTRIES
            ):
                key, _ = self._hash_cache.popitem(last=False)
                self._hash_cache_tokens -= len(key[2])
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
