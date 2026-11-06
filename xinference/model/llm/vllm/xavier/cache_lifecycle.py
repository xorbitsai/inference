# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

from typing import Any, Callable, Deque, Dict, Iterable


class PDCacheLifecycleMixin:
    """Only prefill defers finished-block cleanup until decode releases it."""

    running: Deque[Any]
    waiting: Deque[Any]
    swapped: Deque[Any]
    _transferring: Deque[Any]
    _free_finished_seq_group: Callable[[Any], None]

    def free_finished_seq_groups(self):
        if self._role != "prefill":
            super().free_finished_seq_groups()

    def free_seq_cache(self, request_id: str):
        for queue in (self.running, self._transferring, self.waiting, self.swapped):
            for seq_group in list(queue):
                if seq_group.request_id == request_id and seq_group.is_finished():
                    self._free_finished_seq_group(seq_group)
                    queue.remove(seq_group)


def order_free_session_blocks(
    pool: Any, offers: Iterable[Any], phases: Dict[str, str]
) -> int:
    """Move waiting chains to the tail, ordered so their heads survive longest.

    All blocks remain free. Active requests and in-progress transfers retain
    their existing ownership and reuse fences. Active, finished and expired
    hints leave blocks in their ordinary engine eviction order.
    """
    positions: Dict[int, int] = {}
    for offer in offers:
        if phases.get(offer.session_id) != "waiting":
            continue
        for position, block_id in enumerate(offer.block_ids):
            positions[block_id] = max(positions.get(block_id, -1), position)
    queue = pool.free_block_queue
    moved = 0
    for block_id in sorted(positions, key=positions.__getitem__, reverse=True):
        block = pool.blocks[block_id]
        if (
            block.ref_cnt
            or block.is_null
            or block.prev_free_block is None
            or block.next_free_block is None
        ):
            continue
        queue.remove(block)
        queue.append_n([block])
        moved += 1
    return moved
