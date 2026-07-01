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
        self, worker_candidates: List[Dict], n_gpu: Optional[int] = None
    ) -> Tuple[xo.ActorRefType, Optional[List[int]]]:
        """
        Pick a worker and the gpu_idx list for the next replica.

        Args:
            worker_candidates: List of dicts that contain at least `ref` (worker ref)
            `count` (current load), and optionally `alloc` (GPU allocation snapshot).
            n_gpu: Optional requested GPU count for this replica.

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
        self, worker_candidates: List[Dict], n_gpu: int
    ) -> Optional[Tuple[xo.ActorRefType, List[int]]]:
        best = None
        for candidate in worker_candidates:
            ref = candidate["ref"]
            alloc = candidate.get("alloc")
            if not alloc:
                continue
            total = alloc.get("total", [])
            models = alloc.get("models", {})
            user_specified = alloc.get("user_specified", {})
            if total:
                dev_iter = total
            else:
                dev_iter = []
                keys = set(models.keys())
                keys.update(user_specified.keys())
                for dev_key in keys:
                    try:
                        dev_iter.append(int(dev_key))
                    except Exception:
                        continue
            loads = []
            for dev in dev_iter:
                models_on_dev = models.get(dev, [])
                load = len(models_on_dev)
                load += len(user_specified.get(dev, []))
                loads.append((load, dev))
            if len(loads) < n_gpu:
                continue
            loads.sort(key=lambda x: (x[0], x[1]))
            selected = loads[:n_gpu]
            score = tuple([sum(load for load, _ in selected), ref.address])
            if best is None or score < best["score"]:
                best = {
                    "ref": ref,
                    "gpu_idx": [dev for _, dev in selected],
                    "score": score,
                }
        return (best["ref"], best["gpu_idx"]) if best else None

    def _reserve_slot(
        self, worker_candidates: List[Dict], ref: xo.ActorRefType, dev: int
    ):
        """
        Update local snapshot so subsequent replicas see the reserved slot.
        """
        for candidate in worker_candidates:
            if candidate["ref"] != ref:
                continue
            alloc = candidate.setdefault("alloc", {})
            models = alloc.setdefault("models", {})
            models.setdefault(dev, [])
            models[dev].append("__reserved__")
            break

    def select_worker(
        self, worker_candidates: List[Dict], n_gpu: Optional[int] = None
    ) -> Tuple[xo.ActorRefType, Optional[List[int]]]:
        random.shuffle(worker_candidates)

        # Use allocation snapshot to pick the least-loaded GPU slot.
        requested_gpu = n_gpu if isinstance(n_gpu, int) and n_gpu > 0 else 1
        gpu_choice = self._select_least_loaded_gpu(worker_candidates, requested_gpu)
        if gpu_choice is not None:
            ref, gpu_idx = gpu_choice
            # Update local snapshot to reflect the reservation.
            for dev in gpu_idx:
                self._reserve_slot(worker_candidates, ref, dev)
            return ref, gpu_idx

        # Fallback: least-loaded worker by count, stable by address.
        for candidate in worker_candidates:
            self._fallback_load.setdefault(candidate["ref"].address, 0)
        worker_candidates.sort(key=lambda x: (x["count"], x["ref"].address))
        min_count = worker_candidates[0]["count"]
        least_loaded = [c for c in worker_candidates if c["count"] == min_count]
        chosen = random.choice(least_loaded)
        self._fallback_load[chosen["ref"].address] += 1
        return chosen["ref"], None
