# Copyright 2022-2023 XProbe Inc.
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
from typing import Dict

from xorbits._mars import resource


@dataclass
class ResourceStatus:
    available: float
    total: float
    memory_available: float
    memory_total: float


def gather_node_info() -> Dict[str, ResourceStatus]:
    node_resource = dict()
    mem_info = resource.virtual_memory()
    node_resource["cpu"] = ResourceStatus(
        available=resource.cpu_percent() / 100.0,
        total=resource.cpu_count(),
        memory_available=mem_info.available,
        memory_total=mem_info.total,
    )
    for idx, gpu_card_stat in enumerate(resource.cuda_card_stats()):
        node_resource[f"gpu-{idx}"] = ResourceStatus(
            available=gpu_card_stat.gpu_usage / 100.0,
            total=1,
            memory_available=gpu_card_stat.fb_mem_info.available,
            memory_total=gpu_card_stat.fb_mem_info.total,
        )

    return node_resource
