# Copyright 2022-2025 XProbe Inc.
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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Union

import xoscar as xo

from .block_tracker import VLLMBlockTracker

if TYPE_CHECKING:
    from .transfer import TransferActor


logger = logging.getLogger(__name__)


def with_lock(method):
    async def wrapper(self, *args, **kwargs):
        async with self._lock:
            return await method(self, *args, **kwargs)

    return wrapper


@dataclass
class NullRef:
    address: str


class CollectiveManager(xo.StatelessActor):
    @classmethod
    def default_uid(cls):
        return f"xavier-collective-manager"

    def __init__(self, model_uid: str, world_size: int, store_port: int):
        super().__init__()
        self._model_uid = model_uid
        self._rank = 0
        self._world_size = world_size
        self._store_port = store_port
        self._device = None
        self._tcp_store = None
        self._context = None
        self._tracker_ref: Optional[xo.ActorRefType["VLLMBlockTracker"]] = None
        self._rank_to_ref: Dict[
            int, Union[xo.ActorRefType["TransferActor"], NullRef]
        ] = {}
        self._lock = asyncio.Lock()

    async def __post_create__(self):
        self._rank_to_ref[self._rank] = NullRef(address=self.address)
        self._tracker_ref = await xo.actor_ref(
            address=self.address,
            uid=f"{VLLMBlockTracker.default_uid()}-{self._model_uid}",
        )

        from xoscar.collective import xoscar_pygloo as xp

        self._context = xp.rendezvous.Context(self._rank, self._world_size)
        ip = self.address.split(":")[0]
        attr = xp.transport.tcp.attr(ip)
        self._device = xp.transport.tcp.CreateDevice(attr)

    def _connect_full_mesh_inner(self, prefix: Optional[str] = None):
        from xoscar.collective import xoscar_pygloo as xp

        assert self._device is not None
        assert self._context is not None
        if self._tcp_store is None:
            ip = self.address.split(":")[0]
            opt = xp.rendezvous.TCPStoreOptions()
            opt.port = self._store_port
            opt.numWorkers = self._world_size
            opt.isServer = True
            self._tcp_store = xp.rendezvous.TCPStore(ip, opt)

        prefix_store = xp.rendezvous.PrefixStore(
            prefix or str(self._world_size), self._tcp_store
        )
        self._context.connectFullMesh(prefix_store, self._device)
        logger.debug(f"Rank {self._rank} arrives successfully")

    async def connect_full_mesh(self, prefix: Optional[str] = None):
        await asyncio.to_thread(self._connect_full_mesh_inner, prefix)

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
        from .....core.utils import gen_random_string

        prefix = gen_random_string(6)
        tasks = []
        rank_to_ref = self._rank_to_ref.copy()
        world_addresses = [ref.address for _, ref in sorted(rank_to_ref.items())]
        for rank, ref in rank_to_ref.items():
            if rank != self._rank:
                tasks.append(ref.connect_full_mesh(prefix, world_addresses))
        try:
            logger.debug(
                f"Rebuild collective communication with world_addresses: {world_addresses}, prefix: {prefix}"
            )
            await asyncio.gather(self.connect_full_mesh(prefix), *tasks)
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
