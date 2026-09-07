import asyncio
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from vllm.core.block.interfaces import Block
from vllm.core.scheduler import ScheduledSequenceGroup, Scheduler, SchedulerOutputs
from vllm.sequence import (
    ExecuteModelRequest,
    SequenceGroup,
    SequenceStage,
    SequenceStatus,
)
from vllm.utils import Device, is_pin_memory_available
from vllm.worker.cache_engine import CacheEngine

from .executor import XavierExecutor
from .scheduler import XavierScheduler
from .scheduler_hook import EngineHook
from .utils import hash_block_tokens
from .xavier_remote_kvcache_manager import XavierRemoteKVCacheManager

logger = logging.getLogger(__name__)


class XavierEngineHook(EngineHook):
    # same as vllm.core.block.prefix_caching_block.PrefixCachingBlock._none_hash
    _none_hash: int = -1

    def __init__(self):
        super().__init__()
        # Used to store the information collected during the hook.
        self._executor_context: Dict[str, Any] = {}  # type: ignore
        self._scheduler_context: Dict[str, Any] = {}  # type: ignore
        self._swap_stream = torch.cuda.Stream()
        self._num_attn_layers = 0
        self._scheduler: Optional[List[Scheduler]] = None  # type: ignore

    def post_scheduler_init(self, scheduler: XavierScheduler):
        scheduler._block_tracker_ref = None
        scheduler._transfer_ref = None
        scheduler._transferring = deque()
        scheduler._transfer_status = {}

        self._scheduler_context = {
            "scheduled_seq_groups": [],
            "has_transferring": False,
        }

    async def _get_scheduler_block_tracker_ref(
        self, scheduler: XavierScheduler
    ) -> XavierRemoteKVCacheManager:
        if scheduler._block_tracker_ref is None:
            scheduler._block_tracker_ref = XavierRemoteKVCacheManager()
            await scheduler._block_tracker_ref.setup(scheduler._xavier_config or {})
        return scheduler._block_tracker_ref

    async def _get_scheduler_transfer_ref(
        self, scheduler: XavierScheduler
    ) -> XavierRemoteKVCacheManager:
        if scheduler._transfer_ref is None:
            scheduler._transfer_ref = XavierRemoteKVCacheManager()
            await scheduler._transfer_ref.setup(scheduler._xavier_config or {})
        return scheduler._transfer_ref

    async def _get_transfer_details(
        self,
        scheduler: XavierScheduler,
        virtual_engine: int,
        block_tables: Dict[int, List[int]],
        seq_group: SequenceGroup,
    ) -> Tuple[Set[int], Dict[int, Set[Tuple[int, int, int]]]]:
        # If the `seq_group` has the `force_calculation` attribute set to `True`,
        # it indicates that there were issues during the transmission process.
        # In this case, force the computation and exclude it from the Xavier process.
        if getattr(seq_group, "force_calculation", False):
            return set(), dict()
        """
        Retrieve information from other replicas to check if any blocks have already been computed,
        for the purpose of data transfer.
        """
        details: Set[Tuple[int, int]] = set()
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            block_ids = block_tables[seq.seq_id]
            tmp_content_hash = None
            for _id in block_ids:
                block: Block = scheduler.block_manager.get_block_by_block_id(
                    seq.seq_id, _id
                )
                if block._prev_block is None:
                    content_hash = hash_block_tokens(
                        True,
                        self._none_hash,
                        block.token_ids,
                        block._extra_hash,
                        self._none_hash,
                    )

                else:
                    prev_content_hash: Optional[int] = tmp_content_hash
                    content_hash = hash_block_tokens(
                        False,
                        prev_content_hash,
                        block.token_ids,
                        block._extra_hash,
                        self._none_hash,
                    )
                tmp_content_hash = content_hash
                detail = (content_hash, _id)
                """
                1. `block.content_hash is not None` means that the block has been filled with tokens.
                Unless it is evicted from the cache, the computation result of this block is constant.
                2. Check the `transferred` status of the block.
                If it is `True`, it means the block has already been transferred locally
                and does not need to be transferred again.
                3. Check the `executed` status of the block.
                If it is `True`, it means the block has already been computed locally
                and does not need to be transferred.
                """
                if (
                    (content_hash is not None)
                    and (
                        not scheduler.block_manager.get_block_status_by_block_id(
                            "transferred", block.block_id
                        )
                    )
                    and (
                        not scheduler.block_manager.get_block_status_by_block_id(
                            "executed", block.block_id
                        )
                    )
                ):
                    details.add(detail)

        logger.debug(f"Xaiver scheduler details: {details}")
        if details:
            engine_metadata = {
                "virtual_engine": virtual_engine,
                "rank": (
                    scheduler._xavier_config.get("rank")
                    if scheduler._xavier_config
                    else -1
                ),
            }
            cache_metadatas: List[Dict[str, Union[str, int]]] = [
                {"content_hash": content_hash, "block_id": block_id}
                for content_hash, block_id in details
            ]
            tracker_ref = await self._get_scheduler_block_tracker_ref(scheduler)
            remote = await tracker_ref.query_blocks(engine_metadata, cache_metadatas)
            # Not all queried blocks have corresponding results in other replicas.
            # Therefore, it is necessary to record which local block data was actually transferred.
            local: Set[int] = set()
            for _, remote_details in remote.items():
                for _, _, local_block_id in remote_details:
                    local.add(local_block_id)
            if local:
                logger.debug(
                    f"Data in local blocks: {local} will be transmitted from the remote."
                )
            return local, remote
        else:
            return set(), dict()

    @staticmethod
    def _get_swap_block_ids(src_to_dst: Dict[int, int], is_sender: bool) -> List[int]:
        return list(sorted([r if is_sender else l for r, l in src_to_dst.items()]))

    def _incr_count_for_block_id(self, virtual_engine: int, block_ids: List[int]):
        """
        The reference count of the `block_id` involved in the transfer is incremented by 1
        to ensure it is not reclaimed.
        """
        scheduler = self._scheduler[virtual_engine]  # type: ignore
        gpu_allocator = scheduler.block_manager.block_allocator._allocators[Device.GPU]

        for _id in block_ids:
            gpu_allocator._refcounter.incr(_id)

    def _decr_count_for_block_id(self, virtual_engine: int, block_ids: List[int]):
        """
        After the transfer, the reference count is decremented by 1.
        """
        scheduler = self._scheduler[virtual_engine]  # type: ignore
        gpu_allocator = scheduler.block_manager.block_allocator._allocators[Device.GPU]

        for _id in block_ids:
            gpu_allocator._refcounter.decr(_id)

    def _swap_in_from_buffer(
        self, cache_engine: CacheEngine, cpu_buf: torch.Tensor, block_ids: List[int]
    ) -> None:
        src_to_dst = torch.tensor(
            [(idx, block_num) for idx, block_num in enumerate(block_ids)],
            device="cpu",
            dtype=torch.int64,
        ).view(-1, 2)
        with torch.cuda.stream(self._swap_stream):
            for i in range(self._num_attn_layers):
                cache_engine.attn_backend.swap_blocks(
                    cpu_buf[i], cache_engine.gpu_cache[i], src_to_dst
                )
        torch.cuda.Stream.synchronize(self._swap_stream)

    async def _swap_to_cache_engine(
        self,
        transfer_ref: XavierRemoteKVCacheManager,
        virtual_engine: int,
        from_rank: int,
        local: Set[int],
        remote_block_metadata: Dict[int, int],
    ):
        block_ids = list(local)
        self._incr_count_for_block_id(virtual_engine, block_ids)
        cache_engine = self._cache_engine[virtual_engine]

        engine_metadata = {
            "virtual_engine": virtual_engine,
        }
        cache_metadata: List[Dict[str, Union[str, int, Dict[int, int]]]] = [
            {
                "from_rank": from_rank,
                "remote_block_metadata": remote_block_metadata,
            }
        ]

        cpu_buf_index = None
        try:
            recvbuf, recv_block_ids, cpu_buf_index = await transfer_ref.read_blocks(
                engine_metadata, cache_metadata
            )
            self._swap_in_from_buffer(cache_engine, recvbuf, recv_block_ids)
        finally:
            self._decr_count_for_block_id(virtual_engine, block_ids)
            buffer_metadata = {
                "cpu_buf_index_dict": cpu_buf_index,
            }
            if cpu_buf_index is not None:
                await transfer_ref.free_blocks(buffer_metadata)

    async def _do_transfer_inner(
        self,
        scheduler: XavierScheduler,
        virtual_engine: int,
        local: Set[int],
        remote: Dict[int, Set[Tuple[int, int, int]]],
    ):
        transfer_ref = await self._get_scheduler_transfer_ref(scheduler)
        # In xaiver remote_block_metadata is hash_and_block_id
        for from_rank, remote_block_metadata in remote.items():
            await self._swap_to_cache_engine(
                transfer_ref, virtual_engine, from_rank, local, remote_block_metadata  # type: ignore
            )

    async def _do_transfer(
        self,
        scheduler: XavierScheduler,
        virtual_engine: int,
        local: Set[int],
        remote: Dict[int, Set[Tuple[int, int, int]]],
        seq_group: SequenceGroup,
    ):
        try:
            await self._do_transfer_inner(scheduler, virtual_engine, local, remote)
        except Exception as e:
            """
            The exception here is most likely due to the sender triggering recovery during the transmission process.
            In this case, fallback to performing computation during the prefill stage.
            """
            logger.error(f"Transfer failed: {e}")
            # Force this `seq_group` to perform computation.
            seq_group.force_calculation = True
            scheduler._transfer_status.pop(seq_group, None)
            scheduler.waiting.appendleft(seq_group)
            scheduler._transferring.remove(seq_group)

            # Unpin prefill instance kvcache
            unpin_handle = scheduler._unpin_handlers.get(seq_group.request_id, None)
            if unpin_handle is not None:
                await unpin_handle.free_prefill_model_cache(seq_group.request_id)
            scheduler.remove__unpin_handler(seq_group.request_id)

        else:
            # After the transfer is completed, update the corresponding metadata.
            scheduler._transfer_status[seq_group] = local
            for _id in local:
                logger.info(
                    f"scheduler.block_manager type: {type(scheduler.block_manager)}"
                )
                scheduler.block_manager.set_block_status_by_block_id(
                    "transferred", _id, True
                )
            # After the transfer, place the `seq_group` back into the `waiting` queue to
            # wait for the next scheduling execution.
            scheduler.waiting.appendleft(seq_group)
            scheduler._transferring.remove(seq_group)

            # Unpin prefill instance kvcache
            unpin_handle = scheduler._unpin_handlers.get(seq_group.request_id, None)
            if unpin_handle is not None:
                await unpin_handle.free_prefill_model_cache(seq_group.request_id)
            scheduler.remove__unpin_handler(seq_group.request_id)

    async def pre_scheduler_prefill(
        self,
        scheduler: XavierScheduler,
        scheduled_seq_group: ScheduledSequenceGroup,
        block_tables: Dict[int, List[int]],
    ) -> bool:
        """Xinference Change!!!
        Additional data structures required by Xavier. Clean current context here.
        """
        self._scheduler_context["scheduled_seq_groups"] = []
        self._scheduler_context["has_transferring"] = False

        """
        After completing the scheduling, the blocks have been allocated.
        Therefore, it is possible to check whether some blocks have already been computed on other replicas based on this information,
        and subsequently initiate the transfer.
        According to the internal code comments in vllm,
        whether `token_chunk_size` is 1 can indicate whether the `seq_group` is in the decode or prefill stage.
        It is noted that data transmission is only applied during the prefill stage.
        In the decode stage, it only applies to the last token of the block, which can negatively impact throughput.
        """
        virtual_engine = (
            scheduler._virtual_engine if scheduler._virtual_engine is not None else 0
        )
        seq_group = scheduled_seq_group.seq_group
        token_chunk_size = scheduled_seq_group.token_chunk_size
        is_prefill: bool = token_chunk_size != 1
        # must query remote in decode
        if is_prefill:
            local, remote = await self._get_transfer_details(
                scheduler, virtual_engine, block_tables, seq_group
            )
            if remote:
                running_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
                for seq in running_seqs:
                    seq.status = SequenceStatus.WAITING
                    # Additional attribute `transferred` to mark that this `seq_group` involves a transfer process.
                    # During the next scheduling, block allocation will no longer be required
                    # since it has already been completed.
                    seq.transferred = True
                    seq.data._stage = SequenceStage.PREFILL
                scheduler._transfer_status[seq_group] = set()
                # Use `create_task` to avoid blocking subsequent scheduling.
                asyncio.create_task(
                    self._do_transfer(
                        scheduler, virtual_engine, local, remote, seq_group
                    )
                )
                # The `seq_group` that is currently being transferred enters a new queue.
                scheduler._transferring.append(seq_group)
                self._scheduler_context["has_transferring"] = True
                return True
            else:
                import vllm.core.scheduler

                self._scheduler_context["scheduled_seq_groups"].append(
                    vllm.core.scheduler.ScheduledSequenceGroup(
                        seq_group, token_chunk_size
                    )
                )
                return False
        return False

        if scheduler.cache_config.enable_prefix_caching:
            common_computed_block_nums = (
                scheduler.block_manager.get_common_computed_block_ids(
                    seq_group.get_seqs(status=SequenceStatus.RUNNING)
                )
            )
            """Xinference Change!!!
            This is very important and is the core of Xavier.
            `computed_block_nums` is the key attribute that determines which blocks do not need to be computed,
            as decided by the `model_runner`.
            Therefore, after the transfer is completed, this attribute needs to be updated.
            """
            if seq_group in scheduler._transfer_status:
                transferred_blocks = scheduler._transfer_status[seq_group]
                if transferred_blocks:
                    common_computed_block_nums.extend(transferred_blocks)
                    common_computed_block_nums = list(
                        sorted(common_computed_block_nums)
                    )
                    del scheduler._transfer_status[seq_group]

    async def post_scheduler_prefill(
        self,
        scheduler: XavierScheduler,
        scheduler_outputs: SchedulerOutputs,
        scheduled_seq_groups: List[SequenceGroup],
    ):
        """Xinference Change!!!
        If the `seq_group` in this scheduling triggers a transfer,
        it needs to be removed from the running queue (as it is already in the transferring queue).
        It should remain in the transferring queue until the transfer is complete,
        and then it can be placed back into the appropriate queue for scheduling.
        """
        has_transferring = self._scheduler_context.get("has_transferring", False)  # type: ignore
        scheduled_seq_groups = self._scheduler_context.get("scheduled_seq_groups")  # type: ignore

        if has_transferring and scheduled_seq_groups is not None:
            scheduler_outputs.scheduled_seq_groups = scheduled_seq_groups
            for seq_group in scheduler.running.copy():
                if seq_group in scheduler._transfer_status:
                    scheduler.running.remove(seq_group)

    async def pre_scheduler_decode(
        self,
        scheduler: XavierScheduler,
        scheduled_seq_group: ScheduledSequenceGroup,
        block_tables: Dict[int, List[int]],
    ):
        """
        Before the _schedule() is ready, we need to do something before decode stage.
        """
        pass

    async def post_scheduler_decode(
        self,
        scheduler: XavierScheduler,
        scheduler_outputs: SchedulerOutputs,
        scheduled_seq_groups: List[SequenceGroup],
    ):
        """
        Before the _schedule() is ready, we need to do something after decode stage.
        """
        pass

    async def _get_executor_block_tracker_ref(
        self, executor: XavierExecutor
    ) -> XavierRemoteKVCacheManager:
        if executor._block_tracker_ref is None:
            executor._block_tracker_ref = XavierRemoteKVCacheManager()
            await executor._block_tracker_ref.setup(executor.vllm_config.xavier_config)
        return executor._block_tracker_ref

    async def _get_executor_transfer_ref(
        self, executor: XavierExecutor
    ) -> XavierRemoteKVCacheManager:
        if executor._transfer_ref is None:
            executor._transfer_ref = XavierRemoteKVCacheManager()
            await executor._transfer_ref.setup(executor.vllm_config.xavier_config)
        return executor._transfer_ref

    async def post_execute_init(self, executor: XavierExecutor):
        """
        In vllm, the `cache_engine` is the entity that truly manages the KV cache tensors.
        Retrieve the necessary transmission information from the `cache_engine`.
        """
        transfer_ref = await self._get_executor_transfer_ref(executor)
        ref_cache_engine: CacheEngine = executor.driver_worker.cache_engine[0]
        buffer_dtype = ref_cache_engine.dtype
        buffer_device = "cpu"
        buffer_pin_memory = (
            is_pin_memory_available() if buffer_device != "cpu" else False
        )
        num_attn_layers = ref_cache_engine.num_attention_layers
        kv_cache_shape = ref_cache_engine.gpu_cache[0].shape
        assert kv_cache_shape[0] == 2
        buffer_num = 2
        transfer_block_num = executor.vllm_config.xavier_config.get(
            "transfer_block_num"
        )
        buffer_shape = (
            transfer_block_num,
            num_attn_layers,
            kv_cache_shape[0],
            *kv_cache_shape[2:],
        )

        self._num_attn_layers = num_attn_layers
        self._cache_engine = executor.driver_worker.cache_engine
        self._scheduler = executor.scheduler

        transfer_metadata = {
            "cache_engine": executor.driver_worker.cache_engine,
            "scheduler": executor.scheduler,
            "num_buffer": buffer_num,
            "buffer_shape": buffer_shape,
            "buffer_dtype": buffer_dtype,
            "buffer_device": buffer_device,
            "pin_memory": buffer_pin_memory,
        }

        await transfer_ref.setup(
            xavier_config=executor.vllm_config.xavier_config,
            transfer_metadata=transfer_metadata,
        )

    def _get_rank(self, executor: XavierExecutor) -> int:
        return executor.vllm_config.xavier_config.get("rank")

    async def pre_execute(
        self, executor: XavierExecutor, execute_model_req: ExecuteModelRequest
    ):
        """
        Collect information about the blocks involved in the execution before the vllm `ModelRunner` executes.
        This information will be used by the tracker after execution to register the locally computed blocks.
        """
        virtual_engine = execute_model_req.virtual_engine
        assert executor.scheduler is not None
        scheduler = executor.scheduler[virtual_engine]  # type: ignore
        executed_blocks_details: Set[Tuple[int, int]] = set()
        for meta in execute_model_req.seq_group_metadata_list:
            block_tables = meta.block_tables
            tmp_content_hash = None
            for seq_id, block_ids in block_tables.items():
                for _id in block_ids:
                    b = scheduler.block_manager.get_block_by_block_id(seq_id, _id)
                    # The `executed` attribute is used to prevent duplicate registration of the block.
                    executed = scheduler.block_manager.get_block_status_by_block_id(
                        "executed", _id
                    )
                    # The `transferred` attribute indicates the block was received from another rank,
                    # so it should not be registered again by this rank.
                    transferred = scheduler.block_manager.get_block_status_by_block_id(
                        "transferred", _id
                    )
                    if b._prev_block is None:
                        content_hash = hash_block_tokens(
                            True,
                            self._none_hash,
                            b.token_ids,
                            b._extra_hash,
                            self._none_hash,
                        )

                    else:
                        prev_content_hash: Optional[int] = tmp_content_hash
                        content_hash = hash_block_tokens(
                            False,
                            prev_content_hash,
                            b.token_ids,
                            b._extra_hash,
                            self._none_hash,
                        )
                    tmp_content_hash = content_hash
                    detail = (content_hash, b.block_id)

                    if (
                        (content_hash is not None)
                        and (not executed)
                        and (not transferred)
                    ):
                        executed_blocks_details.add(detail)

        # Add to hook context
        executed_blocks_details: List[Dict[str, Union[str, int]]] = [  # type: ignore
            {"content_hash": content_hash, "block_id": block_id}
            for content_hash, block_id in executed_blocks_details
        ]
        self._executor_context["executed_blocks_details"] = executed_blocks_details

    async def post_execute(
        self, executor: XavierExecutor, execute_model_req: ExecuteModelRequest
    ):
        executed_blocks_details = self._executor_context.get(
            "executed_blocks_details", None
        )
        rank = self._get_rank(executor)
        block_tracker_ref = await self._get_executor_block_tracker_ref(executor)
        virtual_engine = execute_model_req.virtual_engine
        assert executor.scheduler is not None
        scheduler = executor.scheduler[virtual_engine]

        if executed_blocks_details:
            """
            Why not collect and register the information after execution?
            Because after execution, the model's execution callback hook will release the block_id,
            causing the block manager to lose access to the correct information.
            """
            engine_metadata = {
                "virtual_engine": virtual_engine,
                "rank": rank,
            }
            cache_metadatas = executed_blocks_details

            await block_tracker_ref.register_blocks(
                engine_metadata,
                cache_metadatas,
            )

            for executed_block in executed_blocks_details:
                _id = executed_block["block_id"]
                logger.debug(f"Register block {_id} to rank {rank}")
                scheduler.block_manager.set_block_status_by_block_id(
                    "executed", _id, True
                )

        # Clear the context
        self._executor_context.pop("executed_blocks_details", None)
