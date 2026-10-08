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
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, no_type_check

import xoscar as xo

from ...xavier.collective_manager import CollectiveManager as CollectiveManager
from ...xavier.collective_manager import with_lock as with_lock

if TYPE_CHECKING:
    from .transfer import Rank0TransferActor


logger = logging.getLogger(__name__)


class Rank0ModelActor(xo.StatelessActor):
    @classmethod
    def default_uid(cls):
        return "rank0-model-actor"

    def __init__(self, xavier_config: Dict[str, Any]):
        super().__init__()
        self._rank = 0
        self._xavier_config = xavier_config
        self._transfer_ref: Optional[xo.ActorRefType["Rank0TransferActor"]] = None

    async def __pre_destroy__(self):
        if self._transfer_ref is not None:
            try:
                await xo.destroy_actor(self._transfer_ref)
                del self._transfer_ref
            except Exception as e:
                logger.debug(
                    f"Destroy transfer actor failed, rank: {self._rank}, address: {self.address}, error: {e}"
                )

    @no_type_check
    async def start_transfer_for_vllm(self, rank_addresses: List[str]):
        from .transfer import Rank0TransferActor

        self._transfer_ref = await xo.create_actor(
            Rank0TransferActor,
            address=self.address,
            uid=f"{Rank0TransferActor.default_uid()}-{self._rank}",
            rank=self._rank,
            world_size=self._xavier_config.get("world_size"),  # type: ignore
            rank_address=self._xavier_config.get("rank_address"),  # type: ignore
            store_address=self._xavier_config.get("store_address"),  # type: ignore
            store_port=self._xavier_config.get("store_port"),  # type: ignore
            world_addresses=rank_addresses,
        )
        logger.debug(
            f"Init transfer actor: {self._transfer_ref.address}, rank: {self._rank} done for vllm."  # type: ignore
        )
