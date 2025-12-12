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

from ..device_utils import update_gpu_memory_info
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


class LaunchStrategy:
    """
    Base class for launch strategies.
    Concrete implementations should override allocate/release/is_idle.
    """

    def allocate(
        self,
        spec: LaunchModelSpec,
        total_gpu_devices: List[int],
        user_specified_allocated_devices: Set[int],
        allocated_gpus: Mapping[int, Set[str]],
    ) -> List[int]:
        raise NotImplementedError

    def release(self, model_uid: str, devices: List[int]) -> None:
        raise NotImplementedError

    def is_idle(self) -> bool:
        raise NotImplementedError


class IdleFirstLaunchStrategy(LaunchStrategy):
    """
    Prefer the GPU running Xinference, otherwise keep allocating onto the emptiest
    remaining GPU.
    """

    _DEFAULT_BOOKED_MB = 1024  # logical reservation per replica

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
        if gpu_memory_info is None:
            raise ValueError("gpu_memory_info must be provided for launch strategy")
        self._gpu_memory_info = gpu_memory_info
        # Track which GPUs have been used in the first round for each model
        self._model_spread_used_gpus: Dict[str, Set[int]] = (
            model_spread_used_gpus if model_spread_used_gpus is not None else {}
        )
        # Track active replicas per base model to clean spread history
        self._active_model_counts: Dict[str, int] = (
            active_model_counts if active_model_counts is not None else {}
        )
        # Logical reservations (MB) per GPU for this strategy's base model
        self._reserved_memory_mb: Dict[int, float] = {}

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
            # Deduct logical reservations to avoid stacking replicas too quickly
            available -= self._reserved_memory_mb.get(dev, 0)
            # Penalize GPUs already planned/allocated to avoid stacking too early
            penalty = pending_gpu_counts.get(dev, 0) + len(
                allocated_gpus.get(dev, set())
            )
            scored.append((dev, available - penalty))

        scored.sort(key=lambda item: (-item[1], item[0]))
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
            # Prefer truly idle GPUs first: those without existing allocations
            unoccupied_gpus = [
                dev
                for dev in available_total
                if dev not in user_specified_allocated_devices
                and not allocated_gpus.get(dev)
            ]
            spreading_phase = bool(unoccupied_gpus) and len(used_in_spread) < len(
                unoccupied_gpus
            )
            if spreading_phase:
                # First round: try to place replicas on distinct, unoccupied GPUs
                candidate_pool = [
                    dev for dev in unoccupied_gpus if dev not in used_in_spread
                ]
                if not candidate_pool:
                    candidate_pool = [dev for dev in unoccupied_gpus]
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
        # Reserve logical memory for selected GPUs
        for dev in selected:
            self._reserved_memory_mb[dev] = (
                self._reserved_memory_mb.get(dev, 0.0) + self._DEFAULT_BOOKED_MB
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
            for dev in devices:
                if dev in self._reserved_memory_mb:
                    self._reserved_memory_mb[dev] -= self._DEFAULT_BOOKED_MB
                    if self._reserved_memory_mb[dev] <= 0:
                        self._reserved_memory_mb.pop(dev, None)
        else:
            self._active_model_counts[base_model_uid] = count - 1
            for dev in devices:
                if dev in self._reserved_memory_mb:
                    self._reserved_memory_mb[dev] -= self._DEFAULT_BOOKED_MB
                    if self._reserved_memory_mb[dev] <= 0:
                        self._reserved_memory_mb.pop(dev, None)

    def is_idle(self) -> bool:
        """Return True when no active models are tracked by this strategy."""
        return not self._active_model_counts


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
