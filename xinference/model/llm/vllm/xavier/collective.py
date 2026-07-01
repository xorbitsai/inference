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
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class CollectiveRank:
    def __init__(
        self,
        rank: int,
        world_size: int,
        rank_address: str,
        store_address: str,
        store_port: int,
        world_addresses: List[str],
    ):
        self._rank = rank
        self._world_size = world_size
        self._rank_address = rank_address
        self._world_addresses = world_addresses
        self._store_address = store_address
        self._store_port = store_port
        self._device = None
        self._tcp_store = None
        self._context = None

    def init_rank(self):
        from xoscar.collective import xoscar_pygloo as xp

        self._context = xp.rendezvous.Context(self._rank, self._world_size)

        attr = xp.transport.tcp.attr(self._rank_address.split(":")[0])
        self._device = xp.transport.tcp.CreateDevice(attr)

        opt = xp.rendezvous.TCPStoreOptions()
        opt.port = self._store_port
        opt.numWorkers = self._world_size
        opt.isServer = self._rank == 0
        opt.waitWorkers = False

        self._tcp_store = xp.rendezvous.TCPStore(self._store_address, opt)
        if self._world_addresses:
            self.connect_full_mesh()

    def connect_full_mesh(
        self, prefix: Optional[str] = None, world_addresses: Optional[List[str]] = None
    ):
        from xoscar.collective import xoscar_pygloo as xp

        assert self._device is not None
        assert self._tcp_store is not None
        assert self._context is not None
        if world_addresses is not None:
            self._world_addresses = world_addresses
        prefix_store = xp.rendezvous.PrefixStore(
            prefix or str(self._world_size), self._tcp_store
        )
        self._context.connectFullMesh(prefix_store, self._device)
        logger.debug(
            f"Rank {self._rank} arrives successfully, world addresses: {self._world_addresses}"
        )
