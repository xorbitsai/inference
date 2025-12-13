# Copyright 2022-2025 XProbe Inc.
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

import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import xoscar as xo

from .resource import GPUStatus

if TYPE_CHECKING:
    from .supervisor import WorkerStatus


class LaunchStrategy(ABC):
    """
    Base class for model replica launch strategies.
    """

    @abstractmethod
    def select_worker(
        self, worker_candidates: List[Dict]
    ) -> Tuple[xo.ActorRefType, Optional[List[int]]]:
        """
        Pick a worker and the gpu_idx list for the next replica.

        Args:
            worker_candidates: List of dicts that contain at least `ref` (worker ref)
            and `count` (current load).

        Returns:
            worker_ref, gpu_idx list (None lets worker decide).
        """


class IdleFirstLaunchStrategy(LaunchStrategy):
    """
    Prefer the GPU with the most available memory across the cluster.

    The strategy reads the latest worker heartbeat stored in supervisor,
    books a fixed amount of memory for every assignment to avoid stacking
    all replicas onto a single GPU, and falls back to least-load worker
    if no heartbeat is available.
    """

    def __init__(
        self,
        worker_status: Dict[str, "WorkerStatus"],
        reserve_bytes: int = 1 * 1024 * 1024 * 1024,
    ):
        self._worker_status = worker_status
        self._reserve_bytes = reserve_bytes
        self._booked: Dict[str, Dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._fallback_load: Dict[str, int] = defaultdict(int)

    def _parse_gpu_idx(self, gpu_key: str) -> Optional[int]:
        if gpu_key == "cpu":
            return None
        if gpu_key.startswith("gpu-"):
            try:
                return int(gpu_key.split("-")[1])
            except Exception:
                return None
        try:
            return int(gpu_key)
        except Exception:
            return None

    def _select_idle_gpu(
        self, worker_candidates: List[Dict]
    ) -> Optional[Tuple[xo.ActorRefType, int]]:
        best = None
        for candidate in worker_candidates:
            ref = candidate["ref"]
            status = self._worker_status.get(ref.address)
            if not status:
                continue
            for gpu_key, gpu_status in status.status.items():  # type: ignore
                if not isinstance(gpu_status, GPUStatus):
                    continue
                gpu_idx = self._parse_gpu_idx(str(gpu_key))
                if gpu_idx is None:
                    continue
                booked = self._booked[ref.address][gpu_idx]
                usable_mem = gpu_status.mem_free - booked
                # Skip GPUs that are already overbooked
                if usable_mem <= 0:
                    continue
                if (
                    best is None
                    or usable_mem > best["usable_mem"]
                    or (
                        usable_mem == best["usable_mem"]
                        and gpu_status.mem_total > best["mem_total"]
                    )
                ):
                    best = {
                        "ref": ref,
                        "gpu_idx": gpu_idx,
                        "usable_mem": usable_mem,
                        "mem_total": gpu_status.mem_total,
                    }
        if best is None:
            return None
        self._booked[best["ref"].address][best["gpu_idx"]] += self._reserve_bytes
        return best["ref"], best["gpu_idx"]

    def select_worker(
        self, worker_candidates: List[Dict]
    ) -> Tuple[xo.ActorRefType, Optional[List[int]]]:
        # No heartbeat: treat GPUs as unlimited, pick random for the first replica,
        # then prefer the least booked worker.
        if not self._worker_status:
            for candidate in worker_candidates:
                self._fallback_load.setdefault(candidate["ref"].address, 0)
            min_load = min(
                self._fallback_load[candidate["ref"].address]
                for candidate in worker_candidates
            )
            least_loaded = [
                candidate
                for candidate in worker_candidates
                if self._fallback_load[candidate["ref"].address] == min_load
            ]
            chosen = random.choice(least_loaded)
            self._fallback_load[chosen["ref"].address] += 1
            return chosen["ref"], None

        idle_gpu = self._select_idle_gpu(worker_candidates)
        if idle_gpu is not None:
            worker_ref, gpu_idx = idle_gpu
            return worker_ref, [gpu_idx]

        # Fallback: use least-loaded worker
        worker_candidates.sort(key=lambda x: (x["count"], x["ref"].address))
        chosen = worker_candidates[0]["ref"]
        return chosen, None
