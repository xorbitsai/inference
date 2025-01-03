from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.interfaces import DeviceAwareBlockAllocator
from vllm.platforms import current_platform

from .block import XavierPrefixCachingBlockAllocator


class XavierCpuGpuBlockAllocator(CpuGpuBlockAllocator):
    @staticmethod
    def create(
        allocator_type: str,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        block_size: int,
    ) -> DeviceAwareBlockAllocator:
        # For HPU, block id 0 is used only for padding
        reserved_blocks = 1 if current_platform.is_hpu() else 0
        block_ids = list(range(reserved_blocks, num_gpu_blocks + num_cpu_blocks))
        num_gpu_blocks -= reserved_blocks
        gpu_block_ids = block_ids[:num_gpu_blocks]
        cpu_block_ids = block_ids[num_gpu_blocks:]

        gpu_allocator = XavierPrefixCachingBlockAllocator(
            num_blocks=num_gpu_blocks,
            block_size=block_size,
            block_ids=gpu_block_ids,
        )

        cpu_allocator = XavierPrefixCachingBlockAllocator(
            num_blocks=num_cpu_blocks,
            block_size=block_size,
            block_ids=cpu_block_ids,
        )

        return XavierCpuGpuBlockAllocator(
            cpu_block_allocator=cpu_allocator,
            gpu_block_allocator=gpu_allocator,
        )
