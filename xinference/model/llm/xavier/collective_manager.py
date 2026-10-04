# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import logging
import traceback
from typing import Any, Dict, Optional

import xoscar as xo

from .block_tracker import BlockTracker
from .constants import DEFAULT_TRANSFER_ACTOR_UID

logger = logging.getLogger(__name__)


def with_lock(method):
    async def wrapper(self, *args, **kwargs):
        async with self._lock:
            return await method(self, *args, **kwargs)

    return wrapper


class CollectiveManager(xo.StatelessActor):
    @classmethod
    def default_uid(cls):
        return f"xavier-collective-manager"

    def __init__(
        self, model_uid: str, transfer_actor_uid: str = DEFAULT_TRANSFER_ACTOR_UID
    ):
        super().__init__()
        self._model_uid = model_uid
        self._transfer_actor_uid = transfer_actor_uid
        self._tracker_ref: Optional[xo.ActorRefType["BlockTracker"]] = None
        self._rank_to_ref: Dict[int, xo.ActorRefType[Any]] = {}
        self._lock = asyncio.Lock()

    async def __post_create__(self):
        self._tracker_ref = await xo.actor_ref(
            address=self.address,
            uid=f"{BlockTracker.default_uid()}-{self._model_uid}",
        )

    async def unregister_rank(self, rank: int):
        self._rank_to_ref.pop(rank, None)
        await self._tracker_ref.unregister_rank(rank)  # type: ignore
        await asyncio.gather(
            *(
                ref.release_consumer_leases_v1(rank)
                for other, ref in self._rank_to_ref.items()
                if other != 0
            ),
            return_exceptions=True,
        )
        logger.debug(f"Unregister rank: {rank}")

    async def register_rank(self, rank: int, address: str, update: bool = False):
        rank_ref = await xo.actor_ref(
            address=address, uid=f"{self._transfer_actor_uid}-{rank}"
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
        from ....core.utils import gen_random_string

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
