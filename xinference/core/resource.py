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

from dataclasses import dataclass
from typing import Dict, Union

import psutil

from ..device_utils import get_nvidia_gpu_info


@dataclass
class ResourceStatus:
    usage: float
    total: float
    memory_used: float
    memory_available: float
    memory_total: float


@dataclass
class GPUStatus:
    name: str
    mem_total: float
    mem_free: float
    mem_used: float
    mem_usage: float
    gpu_util: float


def gather_node_info() -> Dict[str, Union[ResourceStatus, GPUStatus]]:
    node_resource = dict()
    mem_info = psutil.virtual_memory()
    node_resource["cpu"] = ResourceStatus(
        usage=psutil.cpu_percent() / 100.0,
        total=psutil.cpu_count(),
        memory_used=mem_info.used,
        memory_available=mem_info.available,
        memory_total=mem_info.total,
    )
    for gpu_idx, gpu_info in get_nvidia_gpu_info().items():
        mem_total = gpu_info["total"]
        mem_usage = (gpu_info["used"] / mem_total) if mem_total else 0.0
        node_resource[gpu_idx] = GPUStatus(  # type: ignore
            name=gpu_info["name"],
            mem_total=mem_total,
            mem_used=gpu_info["used"],
            mem_free=gpu_info["free"],
            mem_usage=mem_usage,
            gpu_util=gpu_info["util"],
        )

    return node_resource  # type: ignore
