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
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union

import xoscar as xo
from vllm.executor.mp_distributed_executor import MultiprocessingDistributedExecutor
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, PoolerOutput
from vllm.utils import is_pin_memory_available
from vllm.worker.cache_engine import CacheEngine

from .utils import hash_block_tokens

if TYPE_CHECKING:
    from .scheduler import XavierScheduler


class XavierExecutor(MultiprocessingDistributedExecutor):
    scheduler: Optional[List["XavierScheduler"]] = None
    # same as vllm.core.block.prefix_caching_block.PrefixCachingBlock._none_hash
    _none_hash: int = -1

    def _init_executor(self) -> None:
        super()._init_executor()
        self._transfer_ref = None
        self._block_tracker_ref = None

    async def init_transfer(self):
        """
        In vllm, the `cache_engine` is the entity that truly manages the KV cache tensors.
        Retrieve the necessary transmission information from the `cache_engine`.
        """
        transfer_ref = await self._get_transfer_ref()
        ref_cache_engine: CacheEngine = self.driver_worker.cache_engine[0]  # type: ignore
        buffer_dtype = ref_cache_engine.dtype
        buffer_device = "cpu"
        buffer_pin_memory = is_pin_memory_available()
        num_attn_layers = ref_cache_engine.num_attention_layers
        kv_cache_shape = ref_cache_engine.gpu_cache[0].shape
        assert kv_cache_shape[0] == 2
        buffer_num = 2
        transfer_block_num = self.vllm_config.xavier_config.get("transfer_block_num")
        buffer_shape = (
            transfer_block_num,
            num_attn_layers,
            kv_cache_shape[0],
            *kv_cache_shape[2:],
        )
        await transfer_ref.setup(
            self.driver_worker.cache_engine,
            self.scheduler,
            num_buffer=buffer_num,
            buffer_shape=buffer_shape,
            buffer_dtype=buffer_dtype,
            buffer_device=buffer_device,
            pin_memory=buffer_pin_memory,
        )

    async def _get_block_tracker_ref(self):
        if self._block_tracker_ref is None:
            block_tracker_address = self.vllm_config.xavier_config.get(
                "block_tracker_address"
            )
            block_tracker_uid = self.vllm_config.xavier_config.get("block_tracker_uid")
            self._block_tracker_ref = await xo.actor_ref(
                address=block_tracker_address, uid=block_tracker_uid
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

    def get_rank(self) -> int:
        return self.vllm_config.xavier_config.get("rank")

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[Union[SamplerOutput, PoolerOutput]]:
        """
        Collect information about the blocks involved in the execution before the vllm `ModelRunner` executes.
        This information will be used by the tracker after execution to register the locally computed blocks.
        """
        virtual_engine = execute_model_req.virtual_engine
        block_tracker_ref = await self._get_block_tracker_ref()
        scheduler = self.scheduler[virtual_engine]  # type: ignore
        rank = self.get_rank()
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

                    if (content_hash is not None) and (not executed):
                        executed_blocks_details.add(detail)

        res = await super().execute_model_async(execute_model_req)

        if executed_blocks_details:
            """
            Why not collect and register the information after execution?
            Because after execution, the model's execution callback hook will release the block_id,
            causing the block manager to lose access to the correct information.
            """
            await block_tracker_ref.register_blocks(
                virtual_engine, list(executed_blocks_details), rank
            )

            for _, _id in executed_blocks_details:
                scheduler.block_manager.set_block_status_by_block_id(
                    "executed", _id, True
                )

        return res
