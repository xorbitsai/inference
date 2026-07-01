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
from typing import Any, Dict, Optional

import xoscar as xo
from vllm.core.block.interfaces import BlockId
from vllm.core.block.prefix_caching_block import (
    BlockTracker,
    PrefixCachingBlockAllocator,
)

from .....isolation import Isolation

logger = logging.getLogger(__name__)


class XavierInnerBlockTracker(BlockTracker):
    """Used to track the status of a block inside the prefix caching allocator"""

    """
    Here, two fixed attributes, `transferred` and `executed`,
    have been added to the `BlockTracker` class to mark the status of the corresponding `block_id`.
    We cannot directly set attributes on the `Block` object
    because the `Block` objects are dynamically allocated with each scheduling.
    The `Block` objects executed in two different scheduling steps may have the same `id`, `hash`, etc.,
    but the instance objects may differ.
    The BlockTracker object inside vllm is one-to-one with the block_id.
    """
    __slots__ = ("active", "last_accessed", "computed", "transferred", "executed")

    def __init__(self):
        super().__init__()
        self.transferred = False
        self.executed = False


class XavierPrefixCachingBlockAllocator(PrefixCachingBlockAllocator):
    def __init__(self, *args, run_isolation: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        for _id in self._block_tracker.keys():
            self._block_tracker[_id] = XavierInnerBlockTracker()

        self._xavier_config: Optional[Dict[str, Any]] = None
        self._block_tracker_ref = None
        if run_isolation:
            self._isolation = Isolation(
                asyncio.new_event_loop(), threaded=True, daemon=True
            )
            self._isolation.start()
        else:
            self._isolation = None  # type: ignore

    def __del__(self):
        if self._isolation is not None:
            self._isolation.stop()

    @property
    def xavier_config(self):
        return self._xavier_config

    @xavier_config.setter
    def xavier_config(self, v: Dict[str, Any]):
        self._xavier_config = v

    async def _get_block_tracker_ref(self):
        if self._block_tracker_ref is None:
            block_tracker_address = self.xavier_config.get("block_tracker_address")
            block_tracker_uid = self.xavier_config.get("block_tracker_uid")
            self._block_tracker_ref = await xo.actor_ref(
                address=block_tracker_address, uid=block_tracker_uid
            )
        return self._block_tracker_ref

    async def unregister_block(self, block_id: int):
        assert self._xavier_config is not None
        tracker_ref = await self._get_block_tracker_ref()
        await tracker_ref.unregister_block(
            self.xavier_config.get("virtual_engine"),
            self.xavier_config.get("rank"),
            block_id,
        )

    def _maybe_allocate_evicted_block_id(self) -> Optional[BlockId]:
        """
        This is the only entry point where the `block_id` is evicted from the cache.
        Therefore, when the `block_id` is evicted, the tracker actor needs to unregister the block information.
        At the same time, make sure to reset the attributes corresponding to that `block_id`.
        """
        evicted_block_id = super()._maybe_allocate_evicted_block_id()
        logger.debug(f"block_id: {evicted_block_id} will be evicted from the cache.")
        if evicted_block_id is not None and self._isolation is not None:
            tracker = self._block_tracker[evicted_block_id]
            assert isinstance(tracker, XavierInnerBlockTracker)
            tracker.transferred = False
            tracker.executed = False
            self._isolation.call(self.unregister_block(evicted_block_id))
            logger.debug(f"block_id: {evicted_block_id} will be used again.")
        return evicted_block_id
