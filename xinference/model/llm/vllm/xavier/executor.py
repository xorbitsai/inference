from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union

import xoscar as xo
from vllm.executor.gpu_executor import GPUExecutorAsync
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, PoolerOutput

if TYPE_CHECKING:
    from .scheduler import XavierScheduler


class XavierExecutor(GPUExecutorAsync):
    scheduler: Optional[List["XavierScheduler"]] = None

    def _init_executor(self) -> None:
        self.vllm_config.parallel_config.worker_cls = (
            "xinference.model.llm.vllm.xavier.worker.XavierWorker"
        )
        super()._init_executor()
        self._transfer_ref = None
        self._block_tracker_ref = None

    async def init_transfer(self):
        transfer_ref = await self._get_transfer_ref()
        await transfer_ref.setup(self.driver_worker.cache_engine, self.scheduler)

    async def _get_block_tracker_ref(self):
        from .block_tracker import VLLMBlockTracker

        if self._block_tracker_ref is None:
            block_tracker_address = self.vllm_config.xavier_config.get(
                "block_tracker_address"
            )
            self._block_tracker_ref = await xo.actor_ref(
                address=block_tracker_address, uid=VLLMBlockTracker.default_uid()
            )
        return self._block_tracker_ref

    async def _get_transfer_ref(self):
        from .transfer import TransferActor

        if self._transfer_ref is None:
            transfer_address = self.vllm_config.xavier_config.get("rank_address")
            rank = self.vllm_config.xavier_config.get("rank")
            self._transfer_ref = await xo.actor_ref(
                address=transfer_address, uid=f"{TransferActor.default_uid()}-{rank}"
            )
        return self._transfer_ref

    def get_rank_address(self) -> str:
        return self.vllm_config.xavier_config.get("rank_address")

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[Union[SamplerOutput, PoolerOutput]]:
        virtual_engine = execute_model_req.virtual_engine
        block_tracker_ref = await self._get_block_tracker_ref()
        scheduler = self.scheduler[virtual_engine]  # type: ignore
        rank_address = self.get_rank_address()
        block_infos: Set[Tuple[int, int]] = set()
        for meta in execute_model_req.seq_group_metadata_list:
            block_tables = meta.block_tables
            for seq_id, block_ids in block_tables.items():
                for _id in block_ids:
                    b = scheduler.block_manager.get_block_by_block_id(seq_id, _id)
                    if b.content_hash is not None:
                        block_infos.add((b.content_hash, b.block_id))

        res = await super().execute_model_async(execute_model_req)

        await block_tracker_ref.set_blocks(
            virtual_engine, list(block_infos), rank_address
        )
        scheduler.update_blocks_local(block_infos)
        return res
