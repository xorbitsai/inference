# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

from typing import Any, Callable, Deque


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
