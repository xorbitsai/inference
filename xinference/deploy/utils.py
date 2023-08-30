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

import os
from typing import TYPE_CHECKING, Any

import xoscar as xo

if TYPE_CHECKING:
    from xoscar.backends.pool import MainActorPoolType


async def create_worker_actor_pool(
    address: str, logging_conf: Any = None
) -> "MainActorPoolType":
    from xorbits._mars.resource import cuda_count

    subprocess_start_method = "forkserver" if os.name != "nt" else "spawn"

    cuda_device_indices = []
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        cuda_device_indices.extend([int(i) for i in cuda_visible_devices.split(",")])
    else:
        cuda_device_indices = list(range(cuda_count()))

    if cuda_device_indices:
        envs = []
        labels = ["main"]
        for i in cuda_device_indices:
            envs.append({"CUDA_VISIBLE_DEVICES": str(i)})
            labels.append(f"gpu-{i}")

        n_process = len(cuda_device_indices)
        pool = await xo.create_actor_pool(
            address=address,
            n_process=n_process,
            labels=labels,
            envs=envs,
            subprocess_start_method=subprocess_start_method,
            logging_conf=logging_conf,
        )
        return pool
    else:
        from xorbits._mars.resource import cpu_count

        # create a process for every 4 CPUs.
        cpu_indices = [i for i in range(cpu_count()) if i % 4 == 0]

        labels = ["main"]
        for i in cpu_indices:
            labels.append(f"cpu-{i}")

        n_process = len(cpu_indices)
        pool = await xo.create_actor_pool(
            address=address,
            n_process=n_process,
            labels=labels,
            subprocess_start_method=subprocess_start_method,
            logging_conf=logging_conf,
        )
        return pool
