import asyncio
from typing import Any, Dict, Optional

import xoscar as xo
from vllm.core.block.interfaces import BlockId
from vllm.core.block.prefix_caching_block import (
    BlockTracker,
    PrefixCachingBlockAllocator,
)

from .....isolation import Isolation


class XavierInnerBlockTracker(BlockTracker):
    """Used to track the status of a block inside the prefix caching allocator"""

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
        from .block_tracker import VLLMBlockTracker

        if self._block_tracker_ref is None:
            block_tracker_address = self.xavier_config.get("block_tracker_address")
            self._block_tracker_ref = await xo.actor_ref(
                address=block_tracker_address, uid=VLLMBlockTracker.default_uid()
            )
        return self._block_tracker_ref

    async def unregister_block(self, block_id: int):
        assert self._xavier_config is not None
        tracker_ref = await self._get_block_tracker_ref()
        await tracker_ref.unregister_block(
            self.xavier_config.get("virtual_engine"),
            self.xavier_config.get("rank_address"),
            block_id,
        )

    def _maybe_allocate_evicted_block_id(self) -> Optional[BlockId]:
        evicted_block_id = super()._maybe_allocate_evicted_block_id()
        if evicted_block_id is not None and self._isolation is not None:
            tracker = self._block_tracker[evicted_block_id]
            assert isinstance(tracker, XavierInnerBlockTracker)
            tracker.transferred = False
            tracker.executed = False
            self._isolation.call(self.unregister_block(evicted_block_id))
        return evicted_block_id
