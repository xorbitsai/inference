import asyncio
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple, no_type_check

import xoscar as xo
from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.block.interfaces import Block
from vllm.core.interfaces import BlockSpaceManager
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.sequence import (
    SequenceData,
    SequenceGroup,
    SequenceGroupMetadata,
    SequenceGroupMetadataDelta,
    SequenceStage,
    SequenceStatus,
)

from .block_manager import XavierBlockManager


class XavierScheduler(Scheduler):
    @staticmethod
    def _get_block_space_manager_class(version: str):
        print(f"Here!!! MyBlockManager")
        return XavierBlockManager

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
        xavier_config: Optional[Dict] = None,
        virtual_engine: Optional[int] = 0,
    ) -> None:
        BlockSpaceManager.get_block_space_manager_class = (
            self._get_block_space_manager_class
        )
        super().__init__(
            scheduler_config,
            cache_config,
            lora_config,
            pipeline_parallel_size,
            output_proc_callback,
        )
        xavier_config["virtual_engine"] = virtual_engine  # type: ignore
        self.block_manager.xavier_config = xavier_config
        self._xavier_config = xavier_config
        self._virtual_engine = virtual_engine
        self._block_tracker_ref = None
        self._transfer_ref = None
        self._transferring: Deque[SequenceGroup] = deque()
        self._transfer_status: Dict[SequenceGroup, Set[int]] = {}

    async def _get_block_tracker_ref(self):
        from .block_tracker import VLLMBlockTracker

        if self._block_tracker_ref is None:
            block_tracker_address = self._xavier_config.get("block_tracker_address")
            self._block_tracker_ref = await xo.actor_ref(
                address=block_tracker_address, uid=VLLMBlockTracker.default_uid()
            )
        return self._block_tracker_ref

    async def _get_transfer_ref(self):
        from .transfer import TransferActor

        if self._transfer_ref is None:
            transfer_address = self._xavier_config.get("rank_address")
            rank = self._xavier_config.get("rank")
            self._transfer_ref = await xo.actor_ref(
                address=transfer_address, uid=f"{TransferActor.default_uid()}-{rank}"
            )
        return self._transfer_ref

    async def _get_transfer_details(
        self,
        virtual_engine: int,
        block_tables: Dict[int, List[int]],
        seq_group: SequenceGroup,
    ) -> Tuple[Set[int], Dict[str, Set[Tuple[int, int, int]]]]:
        print(f"======Scheduled blocks: {block_tables}")
        details: Set[Tuple[int, int]] = set()
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            block_ids = block_tables[seq.seq_id]
            for _id in block_ids:
                block: Block = self.block_manager.get_block_by_block_id(seq.seq_id, _id)
                detail = (block.content_hash, _id)
                if (
                    (block.content_hash is not None)
                    and (
                        not self.block_manager.get_block_status_by_block_id(
                            "transferred", block.block_id
                        )
                    )
                    and (
                        not self.block_manager.get_block_status_by_block_id(
                            "executed", block.block_id
                        )
                    )
                ):
                    details.add(detail)
        tracker_ref = await self._get_block_tracker_ref()
        remote = await tracker_ref.query_blocks(virtual_engine, list(details))
        local: Set[int] = set()
        for _, remote_details in remote.items():
            for _, _, local_block_id in remote_details:
                local.add(local_block_id)
        return local, remote

    async def _do_transfer_inner(
        self, virtual_engine: int, remote: Dict[str, Set[Tuple[int, int, int]]]
    ):
        transfer_ref = await self._get_transfer_ref()
        for addr, hash_and_block_id in remote.items():
            src_to_dst: Dict[int, int] = {x[1]: x[2] for x in hash_and_block_id}
            await transfer_ref.recv(virtual_engine, addr, src_to_dst)

    async def _do_transfer(
        self,
        virtual_engine: int,
        local: Set[int],
        remote: Dict[str, Set[Tuple[int, int, int]]],
        seq_group: SequenceGroup,
        is_prefill: bool,
    ):
        await self._do_transfer_inner(virtual_engine, remote)
        self._transfer_status[seq_group] = local
        for _id in local:
            self.block_manager.set_block_status_by_block_id("transferred", _id, True)
        if is_prefill:
            self.waiting.appendleft(seq_group)
        else:
            self.running.appendleft(seq_group)
        self._transferring.remove(seq_group)
        print(f"=========Scheduler: Receive done")

    @no_type_check
    async def schedule(
        self,
    ) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, bool]:
        virtual_engine = self._virtual_engine

        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_start_time = time.perf_counter()

        scheduler_outputs: SchedulerOutputs = self._schedule()
        now = time.time()

        if not self.cache_config.enable_prefix_caching:
            common_computed_block_nums = []

        allow_async_output_proc: bool = self.use_async_output_proc

        # TODO
        scheduled_seq_groups = []
        has_transferring = False

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for i, scheduled_seq_group in enumerate(scheduler_outputs.scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)

            seq_group_metadata = self._seq_group_metadata_cache[
                self.cache_id
            ].get_object()
            seq_group_metadata.seq_data.clear()
            seq_group_metadata.block_tables.clear()

            # seq_id -> SequenceData
            seq_data: Dict[int, SequenceData] = {}
            # seq_id -> physical block numbers
            block_tables: Dict[int, List[int]] = {}

            if seq_group.is_encoder_decoder():
                # Encoder associated with SequenceGroup
                encoder_seq = seq_group.get_encoder_seq()
                assert encoder_seq is not None
                encoder_seq_data = encoder_seq.data
                # Block table for cross-attention
                # Also managed at SequenceGroup level
                cross_block_table = self.block_manager.get_cross_block_table(seq_group)
            else:
                encoder_seq_data = None
                cross_block_table = None

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                self.block_manager.access_all_blocks_in_seq(seq, now)

            local, remote = await self._get_transfer_details(
                virtual_engine, block_tables, seq_group
            )
            if remote:
                running_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
                is_prefill = token_chunk_size != 1
                for seq in running_seqs:
                    seq.status = (
                        SequenceStatus.WAITING if is_prefill else SequenceStatus.RUNNING
                    )
                    seq.transferred = True
                    seq.data._stage = (
                        SequenceStage.PREFILL if is_prefill else SequenceStage.DECODE
                    )
                self._transfer_status[seq_group] = set()
                asyncio.create_task(
                    self._do_transfer(
                        virtual_engine, local, remote, seq_group, is_prefill
                    )
                )
                self._transferring.append(seq_group)
                has_transferring = True
                continue
            else:
                scheduled_seq_groups.append(seq_group)

            if self.cache_config.enable_prefix_caching:
                common_computed_block_nums = (
                    self.block_manager.get_common_computed_block_ids(
                        seq_group.get_seqs(status=SequenceStatus.RUNNING)
                    )
                )
                if seq_group in self._transfer_status:
                    transferred_blocks = self._transfer_status[seq_group]
                    if transferred_blocks:
                        common_computed_block_nums.extend(transferred_blocks)
                        common_computed_block_nums = list(
                            sorted(common_computed_block_nums)
                        )
                        del self._transfer_status[seq_group]

            do_sample = True
            is_prompt = seq_group.is_prefill()
            # We should send the metadata to workers when the first prefill
            # is sent. Subsequent requests could be chunked prefill or decode.
            is_first_prefill = False
            if is_prompt:
                seqs = seq_group.get_seqs()
                # Prefill has only 1 sequence.
                assert len(seqs) == 1
                num_computed_tokens = seqs[0].data.get_num_computed_tokens()
                is_first_prefill = num_computed_tokens == 0
                # In the next iteration, all prompt tokens are not computed.
                # It means the prefill is chunked, and we don't need sampling.
                # NOTE: We use get_len instead of get_prompt_len because when
                # a sequence is preempted, prefill includes previous generated
                # output tokens.
                if token_chunk_size + num_computed_tokens < seqs[0].data.get_len():
                    do_sample = False

            # It assumes the scheduled_seq_groups is ordered by
            # prefill < decoding.
            if is_first_prefill or not self.scheduler_config.send_delta_data:
                seq_group_metadata = SequenceGroupMetadata(
                    request_id=seq_group.request_id,
                    is_prompt=is_prompt,
                    seq_data=seq_data,
                    sampling_params=seq_group.sampling_params,
                    block_tables=block_tables,
                    do_sample=do_sample,
                    pooling_params=seq_group.pooling_params,
                    token_chunk_size=token_chunk_size,
                    lora_request=seq_group.lora_request,
                    computed_block_nums=common_computed_block_nums,
                    encoder_seq_data=encoder_seq_data,
                    cross_block_table=cross_block_table,
                    state=seq_group.state,
                    token_type_ids=seq_group.token_type_ids,
                    # `multi_modal_data` will only be present for the 1st comm
                    # between engine and worker.
                    # the subsequent comms can still use delta, but
                    # `multi_modal_data` will be None.
                    multi_modal_data=seq_group.multi_modal_data
                    if scheduler_outputs.num_prefill_groups > 0
                    else None,
                    multi_modal_placeholders=seq_group.multi_modal_placeholders
                    if scheduler_outputs.num_prefill_groups > 0
                    else None,
                    mm_processor_kwargs=seq_group.mm_processor_kwargs,
                    prompt_adapter_request=seq_group.prompt_adapter_request,
                )
            else:
                # When SPMD mode is enabled, we only send delta data except for
                # the first request to reduce serialization cost.
                seq_data_delta = {}
                for id, data in seq_data.items():
                    seq_data_delta[id] = data.get_delta_and_reset()
                seq_group_metadata = SequenceGroupMetadataDelta(
                    seq_data_delta,
                    seq_group.request_id,
                    block_tables,
                    is_prompt,
                    do_sample=do_sample,
                    token_chunk_size=token_chunk_size,
                    computed_block_nums=common_computed_block_nums,
                )
            seq_group_metadata_list.append(seq_group_metadata)

            if allow_async_output_proc:
                allow_async_output_proc = self._allow_async_output_proc(seq_group)

        if has_transferring:
            scheduler_outputs.scheduled_seq_groups = scheduled_seq_groups
            for seq_group in self.running.copy():
                if seq_group in self._transfer_status:
                    self.running.remove(seq_group)
                    print(f"=======Remove: {seq_group}")

        # Now that the batch has been created, we can assume all blocks in the
        # batch will have been computed before the next scheduling invocation.
        # This is because the engine assumes that a failure in model execution
        # will crash the vLLM instance / will not retry.
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group, scheduled_seq_group.token_chunk_size
            )

        self._seq_group_metadata_cache[self.next_cache_id].reset()

        scheduler_time = time.perf_counter() - scheduler_start_time
        # Add this to scheduler time to all the sequences that are currently
        # running. This will help estimate if the scheduler is a significant
        # component in the e2e latency.
        for seq_group in self.running:
            if seq_group is not None and seq_group.metrics is not None:
                if seq_group.metrics.scheduler_time is not None:
                    seq_group.metrics.scheduler_time += scheduler_time
                else:
                    seq_group.metrics.scheduler_time = scheduler_time

        # Move to next cache (if exists)
        self.cache_id = self.next_cache_id

        # Return results
        return (seq_group_metadata_list, scheduler_outputs, allow_async_output_proc)

    def has_unfinished_seqs(self) -> bool:
        res = super().has_unfinished_seqs()
        return res or len(self._transferring) != 0

    def get_num_unfinished_seq_groups(self) -> int:
        res = super().get_num_unfinished_seq_groups()
        return res + len(self._transferring)
