from typing import Dict
from weakref import WeakValueDictionary

from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.interfaces import Block
from vllm.core.block_manager import SelfAttnBlockSpaceManager
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus

from .allocator import XavierCpuGpuBlockAllocator


class XavierBlockManager(SelfAttnBlockSpaceManager):
    def __init__(self, *args, **kwargs):
        # Monkey patch
        CpuGpuBlockAllocator.create = XavierCpuGpuBlockAllocator.create
        super().__init__(*args, **kwargs)
        self.index: Dict[int, WeakValueDictionary[int, Block]] = {}
        print(f"==========Here block manager: {type(self.block_allocator)}")

    def get_block_by_block_id(self, seq_id: int, block_id: int) -> Block:
        if seq_id not in self.index:
            self.index[seq_id] = WeakValueDictionary()
        if block_id in self.index[seq_id]:
            return self.index[seq_id][block_id]
        table = self.block_tables[seq_id]
        for b in table.blocks:
            if b.block_id == block_id:
                self.index[seq_id][block_id] = b
                return b

    def allocate(self, seq_group: SequenceGroup) -> None:
        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        if all([getattr(s, "transferred", False) for s in waiting_seqs]):
            return
        super().allocate(seq_group)

    def free(self, seq: Sequence) -> None:
        super().free(seq)
        seq_id = seq.seq_id
        if seq_id not in self.index:
            return
        del self.index[seq_id]
