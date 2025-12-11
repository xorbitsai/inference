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

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Set, Tuple, Union

from ..device_utils import initialize_gpu_memory_info, update_gpu_memory_info
from .utils import parse_replica_model_uid

logger = logging.getLogger(__name__)


@dataclass
class LaunchModelSpec:
    """Specification for model launch"""

    model_uid: str
    n_gpu: int
    model_name: Optional[str] = None
    model_size: Optional[Union[int, str]] = None
    model_format: Optional[str] = None
    quantization: Optional[str] = None


class IdleFirstLaunchStrategy:
    """
    Prefer the GPU running Xinference, otherwise keep allocating onto the emptiest
    remaining GPU.
    """

    def __init__(
        self,
        total_gpu_devices: List[int],
        allowed_devices: Optional[Set[int]] = None,
        gpu_memory_info: Optional[Dict[int, Dict[str, Union[int, float]]]] = None,
        model_spread_used_gpus: Optional[Dict[str, Set[int]]] = None,
        active_model_counts: Optional[Dict[str, int]] = None,
    ):
        self._allowed_devices = allowed_devices
        self._total_gpu_devices = self._filter_allowed(total_gpu_devices)
        self._gpu_memory_info = gpu_memory_info or initialize_gpu_memory_info(
            self._total_gpu_devices, logger=logger
        )
        # Track which GPUs have been used in the first round for each model
        self._model_spread_used_gpus: Dict[str, Set[int]] = (
            model_spread_used_gpus if model_spread_used_gpus is not None else {}
        )
        # Track active replicas per base model to clean spread history
        self._active_model_counts: Dict[str, int] = (
            active_model_counts if active_model_counts is not None else {}
        )

    def _filter_allowed(self, total_gpu_devices: List[int]) -> List[int]:
        if self._allowed_devices is None:
            return total_gpu_devices
        return [dev for dev in total_gpu_devices if dev in self._allowed_devices]

    def _select_emptiest_gpu(
        self,
        candidates: List[int],
        pending_gpu_counts: Dict[int, int],
        allocated_gpus: Mapping[int, Set[str]],
    ) -> Optional[int]:
        if not candidates:
            return None

        scored: List[Tuple[int, Union[int, float]]] = []
        for dev in candidates:
            update_gpu_memory_info(self._gpu_memory_info, dev, logger=logger)
            available = self._gpu_memory_info.get(dev, {}).get("available", 0)
            # Penalize GPUs already planned/allocated to avoid stacking too early
            penalty = pending_gpu_counts.get(dev, 0) + len(
                allocated_gpus.get(dev, set())
            )
            scored.append((dev, available - penalty))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[0][0] if scored else None

    def allocate(
        self,
        spec: LaunchModelSpec,
        total_gpu_devices: List[int],
        user_specified_allocated_devices: Set[int],
        allocated_gpus: Mapping[int, Set[str]],
    ) -> List[int]:
        available_total = self._filter_allowed(total_gpu_devices)
        if not available_total:
            raise RuntimeError("No available slot found for the model")

        model_uid = spec.model_uid
        try:
            base_model_uid, _ = parse_replica_model_uid(model_uid)
        except Exception:
            base_model_uid = model_uid
        used_in_spread = self._model_spread_used_gpus.setdefault(base_model_uid, set())
        n_gpu = spec.n_gpu

        pending_gpu_counts: Dict[int, int] = {}
        selected: List[int] = []

        while len(selected) < n_gpu:
            # If some GPUs haven't received a replica for this model yet, try them first
            if len(used_in_spread) < len(available_total):
                candidate_pool = [
                    dev
                    for dev in available_total
                    if dev not in user_specified_allocated_devices
                    and dev not in used_in_spread
                ]
                if not candidate_pool:
                    candidate_pool = [
                        dev
                        for dev in available_total
                        if dev not in user_specified_allocated_devices
                    ]
            else:
                candidate_pool = [
                    dev
                    for dev in available_total
                    if dev not in user_specified_allocated_devices
                ]
            emptiest_gpu = self._select_emptiest_gpu(
                candidate_pool, pending_gpu_counts, allocated_gpus
            )
            if emptiest_gpu is None:
                raise RuntimeError("No available slot found for the model")

            selected.append(emptiest_gpu)
            pending_gpu_counts[emptiest_gpu] = (
                pending_gpu_counts.get(emptiest_gpu, 0) + 1
            )
            used_in_spread.add(emptiest_gpu)

        # Persist spread history for this base model
        self._model_spread_used_gpus[base_model_uid] = used_in_spread
        self._active_model_counts[base_model_uid] = (
            self._active_model_counts.get(base_model_uid, 0) + 1
        )
        return selected

    def release(self, model_uid: str, devices: List[int]) -> None:
        try:
            base_model_uid, _ = parse_replica_model_uid(model_uid)
        except Exception:
            base_model_uid = model_uid
        count = self._active_model_counts.get(base_model_uid, 0)
        if count <= 1:
            self._active_model_counts.pop(base_model_uid, None)
            self._model_spread_used_gpus.pop(base_model_uid, None)
        else:
            self._active_model_counts[base_model_uid] = count - 1


def create_launch_strategy(
    strategy_name: str,
    total_gpu_devices: List[int],
    allowed_devices: Optional[Set[int]] = None,
    gpu_memory_info: Optional[Dict[int, Dict[str, Union[int, float]]]] = None,
    model_spread_used_gpus: Optional[Dict[str, Set[int]]] = None,
    active_model_counts: Optional[Dict[str, int]] = None,
) -> IdleFirstLaunchStrategy:
    normalized = strategy_name.lower()
    if normalized != "idle_first":
        logger.warning(
            f"Unknown launch strategy '{strategy_name}', falling back to idle_first"
        )
    return IdleFirstLaunchStrategy(
        total_gpu_devices,
        allowed_devices=allowed_devices,
        gpu_memory_info=gpu_memory_info,
        model_spread_used_gpus=model_spread_used_gpus,
        active_model_counts=active_model_counts,
    )
