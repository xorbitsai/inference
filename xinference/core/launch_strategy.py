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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union

from ..device_utils import initialize_gpu_memory_info, update_gpu_memory_info

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


class LaunchStrategy(ABC):
    """Abstract base class for GPU allocation strategies"""

    @abstractmethod
    def allocate(
        self,
        spec: LaunchModelSpec,
        total_gpu_devices: List[int],
        user_specified_allocated_devices: Set[int],
        allocated_gpus: Dict[int, str],
    ) -> List[int]:
        """
        Allocate GPUs for model launch

        Args:
            spec: Model launch specification
            total_gpu_devices: List of all available GPU indices
            user_specified_allocated_devices: Set of user-specified allocated devices
            allocated_gpus: Dictionary mapping GPU index to model UID

        Returns:
            List of allocated GPU indices
        """
        pass

    @abstractmethod
    def release(self, model_uid: str, devices: List[int]) -> None:
        """
        Release GPUs allocated for a model

        Args:
            model_uid: Model identifier
            devices: List of GPU indices to release
        """
        pass


class MemoryAwareLaunchStrategy(LaunchStrategy):
    """
    Launch strategy that prefers the GPU running xinference first and otherwise
    chooses the most idle GPU from the remaining pool.
    """

    def __init__(
        self,
        total_gpu_devices: List[int],
        allowed_devices: Optional[Set[int]] = None,
        gpu_memory_info: Optional[Dict[int, Dict[str, Union[int, float]]]] = None,
    ):
        self._total_gpu_devices = total_gpu_devices
        self._allowed_devices = allowed_devices
        self._gpu_memory_info = gpu_memory_info or {}
        self._model_memory_usage: Dict[str, int] = {}
        self._current_gpu = self._determine_current_gpu()

        if not self._gpu_memory_info and self._total_gpu_devices:
            self._initialize_gpu_memory_tracking()

    def _determine_current_gpu(self) -> Optional[int]:
        """Pick the GPU xinference is running on (first visible GPU)."""
        if not self._total_gpu_devices:
            return None

        ordered_allowed = [
            dev for dev in self._total_gpu_devices if self._is_gpu_allowed(dev)
        ]
        if ordered_allowed:
            return ordered_allowed[0]
        return self._total_gpu_devices[0]

    def _is_gpu_allowed(self, gpu_idx: int) -> bool:
        return self._allowed_devices is None or gpu_idx in self._allowed_devices

    def _initialize_gpu_memory_tracking(self):
        """Initialize GPU memory tracking for allowed GPUs"""
        target_gpus = [
            gpu_idx
            for gpu_idx in self._total_gpu_devices
            if self._is_gpu_allowed(gpu_idx)
        ]
        if not target_gpus:
            return
        try:
            self._gpu_memory_info = initialize_gpu_memory_info(
                target_gpus, logger=logger
            )
        except Exception as e:
            logger.warning(f"Failed to initialize GPU memory tracking: {e}")
            for gpu_idx in target_gpus:
                self._gpu_memory_info[gpu_idx] = {
                    "total": 0,
                    "used": 0,
                    "available": 0,
                }

    def _estimate_model_memory_usage(
        self,
        model_name: str,
        model_size: Union[int, str],
        model_format: Optional[str],
        quantization: Optional[str],
    ) -> int:
        """Estimate memory usage for a model based on its characteristics"""
        if isinstance(model_size, str):
            if "B" in model_size:
                size_gb = float(model_size.replace("B", ""))
            else:
                size_gb = float(model_size)
        else:
            size_gb = float(model_size)

        base_memory_mb = int(size_gb * 1024 * 1.5)

        if quantization:
            if "4bit" in quantization.lower() or "4-bit" in quantization.lower():
                base_memory_mb = base_memory_mb // 3
            elif "8bit" in quantization.lower() or "8-bit" in quantization.lower():
                base_memory_mb = base_memory_mb // 2

        if model_format and "mlx" in model_format.lower():
            base_memory_mb = int(base_memory_mb * 0.8)

        return max(base_memory_mb, 1024)

    def _filter_available_devices(
        self, total_gpu_devices: List[int], user_specified_allocated_devices: Set[int]
    ) -> List[int]:
        """Respect allowed devices and drop user-specified occupied GPUs."""
        return [
            dev
            for dev in total_gpu_devices
            if self._is_gpu_allowed(dev) and dev not in user_specified_allocated_devices
        ]

    def _init_memory_snapshot(
        self, candidate_gpus: List[int]
    ) -> Dict[int, Dict[str, Union[int, float]]]:
        memory_snapshot: Dict[int, Dict[str, Union[int, float]]] = {}
        for gpu_idx in candidate_gpus:
            update_gpu_memory_info(self._gpu_memory_info, gpu_idx, logger=logger)
            memory_snapshot[gpu_idx] = dict(self._gpu_memory_info[gpu_idx])
        return memory_snapshot

    def _can_fit_model_on_gpu(
        self,
        gpu_idx: int,
        estimated_memory_mb: int,
        memory_snapshot: Dict[int, Dict[str, Union[int, float]]],
    ) -> bool:
        if estimated_memory_mb <= 0:
            return True
        available_memory = memory_snapshot.get(gpu_idx, {}).get("available", 0)
        return estimated_memory_mb <= available_memory

    def _consume_allocation(
        self,
        gpu_idx: int,
        estimated_memory_mb: int,
        memory_snapshot: Dict[int, Dict[str, Union[int, float]]],
    ) -> None:
        if estimated_memory_mb <= 0:
            return
        if gpu_idx not in memory_snapshot:
            memory_snapshot[gpu_idx] = {
                "total": 0,
                "used": 0,
                "available": 0,
            }
        memory_snapshot[gpu_idx]["used"] += estimated_memory_mb
        memory_snapshot[gpu_idx]["available"] = max(
            memory_snapshot[gpu_idx]["available"] - estimated_memory_mb, 0
        )

    def _select_most_idle_gpu(
        self,
        candidate_gpus: List[int],
        in_use_gpus: Set[int],
        memory_snapshot: Dict[int, Dict[str, Union[int, float]]],
        estimated_memory_mb: int,
    ) -> Optional[int]:
        if not candidate_gpus:
            return None

        available_candidates = [g for g in candidate_gpus if g not in in_use_gpus]
        if not available_candidates:
            available_candidates = candidate_gpus

        sorted_candidates = sorted(
            available_candidates,
            key=lambda g: (
                memory_snapshot.get(g, {}).get("available", 0),
                -g,
            ),
            reverse=True,
        )

        for gpu in sorted_candidates:
            if self._can_fit_model_on_gpu(gpu, estimated_memory_mb, memory_snapshot):
                return gpu

        return sorted_candidates[0] if sorted_candidates else None

    def allocate(
        self,
        spec: LaunchModelSpec,
        total_gpu_devices: List[int],
        user_specified_allocated_devices: Set[int],
        allocated_gpus: Dict[int, str],
    ) -> List[int]:
        """
        Allocate GPUs using the current-GPU-first strategy.
        """
        model_uid = spec.model_uid
        n_gpu = spec.n_gpu

        candidate_gpus = self._filter_available_devices(
            total_gpu_devices, user_specified_allocated_devices
        )
        if not candidate_gpus:
            raise RuntimeError("No available slot found for the model")

        memory_snapshot = self._init_memory_snapshot(candidate_gpus)

        estimated_memory_mb = 0
        if spec.model_name and spec.model_size:
            estimated_memory_mb = self._estimate_model_memory_usage(
                spec.model_name, spec.model_size, spec.model_format, spec.quantization
            )
            self._model_memory_usage[model_uid] = estimated_memory_mb
        selected: List[int] = []
        in_use_gpus: Set[int] = set(user_specified_allocated_devices)
        in_use_gpus.update(allocated_gpus.keys())

        preferred_gpu = (
            self._current_gpu if self._current_gpu in candidate_gpus else None
        )
        if preferred_gpu is not None and preferred_gpu not in in_use_gpus:
            if self._can_fit_model_on_gpu(
                preferred_gpu, estimated_memory_mb, memory_snapshot
            ):
                selected.append(preferred_gpu)
                self._consume_allocation(
                    preferred_gpu, estimated_memory_mb, memory_snapshot
                )
                in_use_gpus.add(preferred_gpu)

        while len(selected) < n_gpu:
            gpu_idx = self._select_most_idle_gpu(
                candidate_gpus, in_use_gpus, memory_snapshot, estimated_memory_mb
            )
            if gpu_idx is None:
                raise RuntimeError("No available slot found for the model")

            selected.append(gpu_idx)
            self._consume_allocation(gpu_idx, estimated_memory_mb, memory_snapshot)
            in_use_gpus.add(gpu_idx)

        if estimated_memory_mb > 0:
            for gpu_idx in selected:
                if gpu_idx in self._gpu_memory_info:
                    self._gpu_memory_info[gpu_idx]["used"] = memory_snapshot[gpu_idx][
                        "used"
                    ]
                    self._gpu_memory_info[gpu_idx]["available"] = memory_snapshot[
                        gpu_idx
                    ]["available"]

        return selected

    def release(self, model_uid: str, devices: List[int]) -> None:
        """Release allocated GPUs and roll back memory accounting"""
        if model_uid not in self._model_memory_usage:
            return

        memory_used = self._model_memory_usage[model_uid]
        for gpu_idx in devices:
            if gpu_idx in self._gpu_memory_info:
                self._gpu_memory_info[gpu_idx]["used"] = max(
                    self._gpu_memory_info[gpu_idx]["used"] - memory_used, 0
                )
                self._gpu_memory_info[gpu_idx]["available"] = min(
                    self._gpu_memory_info[gpu_idx]["available"] + memory_used,
                    self._gpu_memory_info[gpu_idx].get("total", float("inf")),
                )

        del self._model_memory_usage[model_uid]


def create_launch_strategy(
    strategy_name: str,
    total_gpu_devices: List[int],
    allowed_devices: Optional[Set[int]] = None,
    gpu_memory_info: Optional[Dict[int, Dict[str, Union[int, float]]]] = None,
) -> LaunchStrategy:
    if strategy_name.lower() != "memory_aware":
        logger.warning(
            f"Unknown launch strategy '{strategy_name}', falling back to memory_aware"
        )
    return MemoryAwareLaunchStrategy(
        total_gpu_devices,
        allowed_devices=allowed_devices,
        gpu_memory_info=gpu_memory_info,
    )
