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
from typing import Any, Dict, Optional

from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.interfaces import DeviceAwareBlockAllocator
from vllm.platforms import current_platform
from vllm.utils import Device

from .block import XavierPrefixCachingBlockAllocator


class XavierCpuGpuBlockAllocator(CpuGpuBlockAllocator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._xavier_config: Optional[Dict[str, Any]] = None  # type: ignore

    @property
    def xavier_config(self):
        return self._xavier_config

    @xavier_config.setter
    def xavier_config(self, v: Dict[str, Any]):
        self._xavier_config = v
        self._allocators[Device.GPU].xavier_config = v

    @staticmethod
    def create(
        allocator_type: str,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        block_size: int,
    ) -> DeviceAwareBlockAllocator:
        """Xinference Change!!!
        1. The code is copied here because the `allocator` needs to be instantiated as a subclass.
        2. Why not re-instantiate it externally?
        Re-instantiating the `allocator` is costly because it requires initializing many tensors.
        """

        # For HPU, block id 0 is used only for padding
        reserved_blocks = 1 if current_platform.is_hpu() else 0
        block_ids = list(range(reserved_blocks, num_gpu_blocks + num_cpu_blocks))
        num_gpu_blocks -= reserved_blocks
        gpu_block_ids = block_ids[:num_gpu_blocks]
        cpu_block_ids = block_ids[num_gpu_blocks:]

        gpu_allocator = XavierPrefixCachingBlockAllocator(
            run_isolation=True,
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
