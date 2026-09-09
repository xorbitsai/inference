# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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
from logging import getLogger
from typing import TYPE_CHECKING, List, Optional, Union

from vllm.executor.mp_distributed_executor import MultiprocessingDistributedExecutor
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, PoolerOutput

from .xavier_remote_kvcache_manager import XavierRemoteKVCacheManager

if TYPE_CHECKING:
    from .scheduler import XavierScheduler


logger = getLogger(__name__)


class XavierExecutor(MultiprocessingDistributedExecutor):
    scheduler: Optional[List["XavierScheduler"]] = None
    # same as vllm.core.block.prefix_caching_block.PrefixCachingBlock._none_hash
    _none_hash: int = -1

    def _init_executor(self) -> None:
        super()._init_executor()
        self._transfer_ref: Optional[XavierRemoteKVCacheManager] = None
        self._block_tracker_ref: Optional[XavierRemoteKVCacheManager] = None

    async def init_transfer(self):
        from .xavier_scheduler_hook import XavierEngineHook

        self._engine_hook: XavierEngineHook = XavierEngineHook()  # type: ignore
        await self._engine_hook.post_execute_init(self)

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[Union[SamplerOutput, PoolerOutput]]:
        """
        Before execute model, we need to do some pre-processing.
        """
        await self._engine_hook.pre_execute(self, execute_model_req)

        res = await super().execute_model_async(execute_model_req)

        """
        After execute model, we need to do some post-processing.
        """
        await self._engine_hook.post_execute(self, execute_model_req)

        return res
