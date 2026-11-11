# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Share hybrid prefixes from vLLM's resident GPU blocks."""

import asyncio
import logging
import uuid
from collections import OrderedDict
from dataclasses import dataclass

from vllm import __version__ as VLLM_VERSION
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.model_executor.layers.mamba.mamba_utils import get_conv_state_layout
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.request import RequestStatus

from ...xavier.cache_lifecycle import CACHE_SESSION_ARG
from ...xavier.contract import fingerprint_metadata
from .cache_lifecycle import order_free_session_blocks
from .v1_connector import XavierConnector, XavierConnectorMetadata, XavierLoadRequest

logger = logging.getLogger(__name__)
_MAX_OFFERS = 16
_LOOKUP_TIMEOUT = 0.5


@dataclass
class _PublishedPrefix:
    key: tuple[str, bytes]
    block_ids: list[int]
    session_id: str | None = None
    partial: bool = False


class XavierHybridTieredConnector(MultiConnector):
    """Try the GPU connector before triggering CPU lookup or promotion."""

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self._requests_to_connector.pop(request.request_id, None)
        for index, connector in enumerate(self._connectors):
            tokens, asynchronous = connector.get_num_new_matched_tokens(
                request, num_computed_tokens
            )
            if tokens is None:
                return None, False
            if tokens > 0:
                self._requests_to_connector[request.request_id] = index
                return tokens, asynchronous
        return 0, False


class XavierHybridGPUConnector(XavierConnector):
    _shards_snapshots = False

    @property
    def requires_kv_delivery(self) -> bool:
        return False

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(
            vllm_config, role, kv_cache_config, shared_recurrent_prefix=True
        )
        self._offers: OrderedDict[str, _PublishedPrefix] = OrderedDict()
        self._offers_by_block: dict[int, set[str]] = {}
        self._attempted: set[str] = set()
        self._mappings: dict[str, XavierLoadRequest] = {}
        self._session_admissions: set[str] = set()
        self._session_phases: dict[str, str] = {}

        if role == KVConnectorRole.SCHEDULER:
            groups = kv_cache_config.kv_cache_groups
            if (
                not self._has_recurrent_cache
                or len(self._recurrent_groups) == len(groups)
                or any(
                    not isinstance(group.kv_cache_spec, (FullAttentionSpec, MambaSpec))
                    or group.kv_cache_spec.block_size != self._block_size
                    for group in groups
                )
            ):
                raise ValueError(
                    "Xavier GPU hybrid prefix sharing requires aligned full-attention "
                    "and recurrent KV cache groups"
                )
            self._identity = fingerprint_metadata(
                {
                    "vllm": VLLM_VERSION,
                    "model": vllm_config.model_config.compute_hash(),
                    "cache": vllm_config.cache_config.compute_hash(),
                    "layout": vllm_config.cache_config.kv_cache_layout,
                    "conv_layout": get_conv_state_layout(),
                    "groups": [(g.layer_names, repr(g.kv_cache_spec)) for g in groups],
                }
            )

    def _prefix_key(self, request, tokens):
        hash_size = self._kv_cache_manager.block_pool.hash_block_size
        return self._identity, request.block_hashes[tokens // hash_size - 1]

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        first_admission = request.request_id not in self._session_admissions
        self._session_admissions.add(request.request_id)
        sessions = (
            list(
                {
                    offer.session_id
                    for offer in self._offers.values()
                    if offer.session_id
                }
            )
            if first_admission
            else []
        )
        self._requests_need_load.pop(request.request_id, None)
        keys = []
        lookup_options = {}
        if (
            request.request_id not in self._attempted
            and not request.skip_reading_prefix_cache
        ):
            hash_size = self._kv_cache_manager.block_pool.hash_block_size
            end = request.num_tokens - 1
            if hash_size < self._block_size:
                first = num_computed_tokens // hash_size
                hashes = request.block_hashes[first : end // hash_size]
                if hashes:
                    lookup_options["fine_prefix"] = (
                        self._identity,
                        hash_size,
                        first,
                        hashes,
                    )
            else:
                keys = [
                    self._prefix_key(request, tokens)
                    for tokens in range(
                        end - end % self._block_size,
                        num_computed_tokens,
                        -self._block_size,
                    )
                ]
        if not keys and not lookup_options and not sessions:
            return 0, False
        if sessions:
            lookup_options["session_ids"] = sessions

        async def lookup():
            tracker = await self._get_tracker_ref()
            return await tracker.query_gpu_prefix(self._rank, keys, **lookup_options)

        try:
            offer = self._call(asyncio.wait_for(lookup(), _LOOKUP_TIMEOUT))
        except Exception:
            logger.debug("Xavier GPU prefix lookup unavailable", exc_info=True)
            return 0, False
        if sessions:
            offer, phases = offer
            self._order_session_prefixes(phases)
        if offer is None:
            return 0, False
        rank, ticket, tokens, groups = offer
        if not num_computed_tokens < tokens < request.num_tokens:
            return 0, False
        self._requests_need_load[request.request_id] = XavierLoadRequest(
            request.request_id,
            {rank: {}},
            lease=ticket,
            remote_blocks_by_group=groups,
            local_hit_tokens=num_computed_tokens,
        )
        return tokens - num_computed_tokens, True

    def get_external_cache_hit_sources(self, request, num_external_tokens):
        return {CacheHitSource.P2P: num_external_tokens}

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        load = self._requests_need_load.pop(request.request_id, None)
        if load is None or num_external_tokens <= 0:
            return
        self._attempted.add(request.request_id)
        rank = next(iter(load.transfers))
        start = load.local_hit_tokens // self._block_size
        for group_id, (source, dest) in enumerate(
            zip(load.remote_blocks_by_group, blocks.blocks)
        ):
            positions = (
                (len(source) - 1,)
                if group_id in self._recurrent_groups
                else range(start, len(source))
            )
            mapping = {
                source[i]: dest[i].block_id
                for i in positions
                if i < len(dest) and not dest[i].is_null
            }
            if len(mapping) != len(positions):
                # Let vLLM release its remote-load allocation and retry.
                load.local_transfers_by_group.clear()
                break
            load.local_transfers_by_group[group_id] = {rank: mapping}
        self._mappings[request.request_id] = load

    def build_connector_meta(self, scheduler_output):
        metadata = XavierConnectorMetadata(load_requests=list(self._mappings.values()))
        self._mappings.clear()
        return metadata

    def request_finished_all_groups(self, request, block_ids):
        self._session_admissions.discard(request.request_id)
        self._requests_need_load.pop(request.request_id, None)
        self._attempted.discard(request.request_id)
        if (
            request.status
            in (
                RequestStatus.FINISHED_STOPPED,
                RequestStatus.FINISHED_LENGTH_CAPPED,
            )
            and not request.skip_reading_prefix_cache
        ):
            self._publish(request, block_ids)
        return False, None

    def _publish(self, request, finished_blocks=()):
        manager = self._kv_cache_manager
        end = min(request.num_prompt_tokens - 1, request.num_computed_tokens)
        if end <= 0:
            return
        groups, tokens, _ = manager.coordinator.find_longest_cache_hit(
            request.block_hashes, end
        )
        if tokens % self._block_size and self._can_borrow_partial(
            groups, finished_blocks
        ):
            self._publish_checkpoint(request, groups, tokens, partial=True)
        # Earlier blocks may also have only a partial checkpoint remaining.
        # Find a whole checkpoint for requests that diverge before the tail.
        while tokens % self._block_size:
            groups, tokens, _ = manager.coordinator.find_longest_cache_hit(
                request.block_hashes, tokens - tokens % self._block_size
            )
        if tokens:
            self._publish_checkpoint(request, groups, tokens)

    @staticmethod
    def _can_borrow_partial(groups, finished_blocks):
        # A running request may move its prefix-cache hash and continue writing
        # to the original slot without allocating it again. Only borrow tails
        # that are already free or owned solely by the request that just ended.
        return all(
            group
            and (
                group[-1].ref_cnt == 0
                or (
                    group[-1].ref_cnt == 1
                    and index < len(finished_blocks)
                    and group[-1].block_id in finished_blocks[index]
                )
            )
            for index, group in enumerate(groups)
        )

    def _publish_checkpoint(self, request, groups, tokens, *, partial=False):
        pool = self._kv_cache_manager.block_pool
        # Recurrent groups have null placeholders before the final state. The
        # final physical block can contain fewer than block_size valid tokens.
        count = (tokens + self._block_size - 1) // self._block_size
        for group_id, group in enumerate(groups):
            if len(group) != count:
                return
            indices = [index for index, block in enumerate(group) if not block.is_null]
            expected = (
                [count - 1]
                if group_id in self._recurrent_groups
                else list(range(count))
            )
            if indices != expected:
                return
        group_ids = [[block.block_id for block in group] for group in groups]
        owned = [block for group in groups for block in group if not block.is_null]
        key = self._prefix_key(request, tokens)
        ticket = uuid.uuid4().hex
        layer_blocks = {
            name: [block.block_id for block in groups[group_id] if not block.is_null]
            for name, group_id in self._layer_group_ids.items()
        }
        extra_args = (
            getattr(getattr(request, "sampling_params", None), "extra_args", None) or {}
        )
        session = extra_args.get(CACHE_SESSION_ARG)
        session_id = session["id"] if session is not None else None
        if session_id is not None:
            for previous, offer in tuple(self._offers.items()):
                if offer.session_id == session_id and offer.partial == partial:
                    self._revoke(previous)
        same_kind = [
            ticket for ticket, offer in self._offers.items() if offer.partial == partial
        ]
        if len(same_kind) >= _MAX_OFFERS:
            victim = next(
                (
                    ticket
                    for ticket in same_kind
                    if self._session_phases.get(self._offers[ticket].session_id)
                    != "waiting"
                ),
                same_kind[0],
            )
            self._revoke(victim)

        offer = _PublishedPrefix(
            key,
            list(dict.fromkeys(block.block_id for block in owned)),
            session_id,
            partial,
        )
        self._offers[ticket] = offer
        for block_id in offer.block_ids:
            self._offers_by_block.setdefault(block_id, set()).add(ticket)
        # Put borrowed blocks back on the normal free queue. The pool calls
        # our fence before the first slot can be allocated for different data.
        pool.touch(owned)
        pool.unpin_blocks(owned, self._on_reuse)

        async def publish():
            transfer = await self._get_transfer_ref()
            await transfer.register_shared_prefix_v1(ticket, layer_blocks)
            tracker = await self._get_tracker_ref()
            await tracker.publish_gpu_prefix(self._rank, key, ticket, tokens, group_ids)

        try:
            self._call(asyncio.wait_for(publish(), _LOOKUP_TIMEOUT))
        except Exception:
            # Keep the fence even if publication timed out after the RPC ran.
            logger.debug("Xavier GPU prefix publication unavailable", exc_info=True)

    def _on_reuse(self, block):
        for ticket in tuple(self._offers_by_block.get(block.block_id, ())):
            self._revoke(ticket)

    def _order_session_prefixes(self, phases):
        self._session_phases = phases
        try:
            waiting = order_free_session_blocks(
                self._kv_cache_manager.block_pool, self._offers.values(), phases
            )
        except Exception:
            logger.debug("Xavier cache lifecycle ordering unavailable", exc_info=True)
            return
        if waiting:
            logger.debug(
                "Xavier cache lifecycle ordering: waiting_blocks=%d",
                waiting,
            )

    def _revoke(self, ticket):
        offer = self._offers[ticket]

        async def fence():
            transfer = await self._get_transfer_ref()
            await transfer.revoke_shared_prefix_v1(ticket)

        # If the fence fails, allocation must fail before the source can be
        # overwritten. A directory miss alone cannot prove readers drained.
        self._call(fence())
        del self._offers[ticket]
        for block_id in offer.block_ids:
            tickets = self._offers_by_block[block_id]
            tickets.remove(ticket)
            if not tickets:
                del self._offers_by_block[block_id]

        async def unpublish():
            tracker = await self._get_tracker_ref()
            await tracker.remove_gpu_prefix(self._rank, offer.key, ticket)

        try:
            self._call(asyncio.wait_for(unpublish(), _LOOKUP_TIMEOUT))
        except Exception:
            logger.debug("Xavier GPU prefix directory unavailable", exc_info=True)

    def wait_for_save(self):
        if not self._gpu_cache_mapped:
            self._call(self._ensure_gpu_cache_mapping())

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        if metadata.load_requests:
            self._call(self._start_loads(metadata.load_requests))

    async def _start_loads(self, loads):
        for load in loads:
            self._gpu_load_jobs[asyncio.create_task(self._load_prefix(load))] = [
                load.request_id
            ]

    async def _load_prefix(self, load):
        try:
            if len(load.local_transfers_by_group) != len(load.remote_blocks_by_group):
                return False
            transfer = await self._ensure_gpu_cache_mapping()
            rank = next(iter(load.transfers))
            layers = {
                name: load.local_transfers_by_group[self._get_layer_group_id(name)][
                    rank
                ]
                for layer, cache in self._registered_kv_caches.items()
                for name, _ in self._iter_kv_tensors(layer, cache)
            }
            loaded = await transfer.load_shared_prefix_v1({rank: layers}, load.lease)
            if loaded:
                logger.debug(
                    "Xavier GPU prefix imported: request=%s source_rank=%s destination_rank=%s",
                    load.request_id,
                    rank,
                    self._rank,
                )
            return loaded
        except Exception:
            logger.warning("Xavier GPU prefix load failed", exc_info=True)
            return False

    def get_transfer_results(self, finished_req_ids):
        results = KVConnectorTransferResults()
        if self._gpu_load_jobs:
            self._call(asyncio.sleep(0))
        for job in tuple(self._gpu_load_jobs):
            if job.done():
                request_ids = self._gpu_load_jobs.pop(job)
                results.finished_recving.update(request_ids)
                if not job.result():
                    results.failed_recving.update(request_ids)
        return results
