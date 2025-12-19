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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import xoscar as xo

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
            `count` (current load), and optionally `alloc` (GPU allocation snapshot).

        Returns:
            worker_ref, gpu_idx list (None lets worker decide).
        """


class IdleFirstLaunchStrategy(LaunchStrategy):
    """
    Count-based scheduler: choose the (worker, gpu) slot with the fewest active models.
    Ignores memory; falls back to least-loaded worker if no GPU allocation snapshot is available.
    """

    def __init__(self, worker_status: Dict[str, "WorkerStatus"]):
        self._worker_status = worker_status
        self._fallback_load: Dict[str, int] = {}

    def _select_least_loaded_gpu(
        self, worker_candidates: List[Dict]
    ) -> Optional[Tuple[xo.ActorRefType, List[int]]]:
        best = None
        for candidate in worker_candidates:
            ref = candidate["ref"]
            alloc = candidate.get("alloc")
            if not alloc or "models" not in alloc:
                continue
            models = alloc.get("models", {})
            embeddings = alloc.get("embeddings", {})
            user_specified = alloc.get("user_specified", {})
            for dev_key, models_on_dev in models.items():
                try:
                    dev = int(dev_key)
                except Exception:
                    continue
                load = len(models_on_dev)
                load += len(embeddings.get(dev, []))
                load += len(user_specified.get(dev, []))
                if best is None or load < best["load"] or (
                    load == best["load"] and ref.address < best["ref"].address
                ):
                    best = {"ref": ref, "gpu_idx": [dev], "load": load}
        return (best["ref"], best["gpu_idx"]) if best else None

    def select_worker(
        self, worker_candidates: List[Dict]
    ) -> Tuple[xo.ActorRefType, Optional[List[int]]]:
        random.shuffle(worker_candidates)

        # Use allocation snapshot to pick the least-loaded GPU slot.
        gpu_choice = self._select_least_loaded_gpu(worker_candidates)
        if gpu_choice is not None:
            return gpu_choice

        # Fallback: least-loaded worker by count, stable by address.
        for candidate in worker_candidates:
            self._fallback_load.setdefault(candidate["ref"].address, 0)
        worker_candidates.sort(key=lambda x: (x["count"], x["ref"].address))
        min_count = worker_candidates[0]["count"]
        least_loaded = [c for c in worker_candidates if c["count"] == min_count]
        chosen = random.choice(least_loaded)
        self._fallback_load[chosen["ref"].address] += 1
        return chosen["ref"], None
