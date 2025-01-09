import logging
from typing import Any, Dict, Optional

from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.interfaces import Block
from vllm.core.block_manager import SelfAttnBlockSpaceManager
from vllm.sequence import SequenceGroup, SequenceStatus
from vllm.utils import Device

from .allocator import XavierCpuGpuBlockAllocator

logger = logging.getLogger(__name__)


class XavierBlockManager(SelfAttnBlockSpaceManager):
    def __init__(self, *args, **kwargs):
        # Monkey patch
        CpuGpuBlockAllocator.create = XavierCpuGpuBlockAllocator.create
        super().__init__(*args, **kwargs)
        self._xavier_config: Optional[Dict[str, Any]] = None
        logger.debug("Init xavier block manager done.")

    @property
    def xavier_config(self):
        return self._xavier_config

    @xavier_config.setter
    def xavier_config(self, value: Dict[str, Any]):
        self._xavier_config = value
        self.block_allocator.xavier_config = value

    def get_block_by_block_id(self, seq_id: int, block_id: int) -> Block:
        table = self.block_tables[seq_id]
        for b in table.blocks:
            if b.block_id == block_id:
                return b

    def get_block_status_by_block_id(self, status_name: str, block_id: int) -> bool:
        tracker = self.block_allocator._allocators[Device.GPU]._block_tracker[block_id]
        return getattr(tracker, status_name)

    def set_block_status_by_block_id(
        self, status_name: str, block_id: int, status: bool
    ) -> None:
        tracker = self.block_allocator._allocators[Device.GPU]._block_tracker[block_id]
        assert getattr(tracker, status_name, None) is not None
        setattr(tracker, status_name, status)

    def allocate(self, seq_group: SequenceGroup) -> None:
        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        if all([getattr(s, "transferred", False) for s in waiting_seqs]):
            return
        super().allocate(seq_group)
