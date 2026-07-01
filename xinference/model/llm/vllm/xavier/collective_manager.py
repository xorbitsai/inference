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
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, no_type_check

import xoscar as xo

from .block_tracker import VLLMBlockTracker

if TYPE_CHECKING:
    from .transfer import Rank0TransferActor, TransferActor


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


def with_lock(method):
    async def wrapper(self, *args, **kwargs):
        async with self._lock:
            return await method(self, *args, **kwargs)

    return wrapper


class CollectiveManager(xo.StatelessActor):
    @classmethod
    def default_uid(cls):
        return f"xavier-collective-manager"

    def __init__(self, model_uid: str):
        super().__init__()
        self._model_uid = model_uid
        self._tracker_ref: Optional[xo.ActorRefType["VLLMBlockTracker"]] = None
        self._rank_to_ref: Dict[int, xo.ActorRefType["TransferActor"]] = {}
        self._lock = asyncio.Lock()

    async def __post_create__(self):
        self._tracker_ref = await xo.actor_ref(
            address=self.address,
            uid=f"{VLLMBlockTracker.default_uid()}-{self._model_uid}",
        )

    async def unregister_rank(self, rank: int):
        self._rank_to_ref.pop(rank, None)
        await self._tracker_ref.unregister_rank(rank)  # type: ignore
        logger.debug(f"Unregister rank: {rank}")

    async def register_rank(self, rank: int, address: str, update: bool = False):
        from .transfer import TransferActor

        rank_ref = await xo.actor_ref(
            address=address, uid=f"{TransferActor.default_uid()}-{rank}"
        )
        self._rank_to_ref[rank] = rank_ref
        logger.debug(f"Register rank: {rank}, address: {address}")
        if update:
            await self._update_world()
            await self._tracker_ref.register_rank(rank)  # type: ignore

    @with_lock
    async def _update_world(self):
        """
        Locking is used to prevent chaos when multiple replicas trigger recovery simultaneously.
        """
        from .....core.utils import gen_random_string

        prefix = gen_random_string(6)
        tasks = []
        rank_to_ref = self._rank_to_ref.copy()
        world_addresses = [ref.address for _, ref in sorted(rank_to_ref.items())]
        for rank, ref in rank_to_ref.items():
            tasks.append(ref.connect_full_mesh(prefix, world_addresses))
        try:
            logger.debug(
                f"Rebuild collective communication with world_addresses: {world_addresses}, prefix: {prefix}"
            )
            await asyncio.gather(*tasks)
            logger.debug(
                f"Rebuild collective communication with world_addresses: {world_addresses}, prefix: {prefix} done."
            )
        except Exception as e:
            """
            The exception here is most likely due to another replica triggering recovery during the recovery process,
            causing `connect_full_mesh` to time out.
            Simply log the exception and
            let the subsequent update process handle the reconstruction of the collective communication world.
            """
            logger.error(
                f"Rebuild collective communication with world_addresses: {world_addresses} failed. "
                f"Exception: {e}"
            )
            # Print the complete error stack
            traceback.print_exception(type(e), e, e.__traceback__)
