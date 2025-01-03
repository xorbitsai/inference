from typing import Optional

from vllm.core.block.interfaces import BlockId
from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator


class XavierPrefixCachingBlockAllocator(PrefixCachingBlockAllocator):
    def _maybe_allocate_evicted_block_id(self) -> Optional[BlockId]:
        evicted_block_id = super()._maybe_allocate_evicted_block_id()
        if evicted_block_id is not None:
            pass
        return evicted_block_id
