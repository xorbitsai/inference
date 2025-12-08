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
from typing import Dict, List, Optional, Set, Tuple, Union

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
    """Memory-aware GPU allocation strategy supporting single-GPU multi-replica"""

    def __init__(
        self,
        total_gpu_devices: List[int],
        gpu_memory_info: Optional[Dict[int, Dict[str, Union[int, float]]]] = None,
    ):
        self._total_gpu_devices = total_gpu_devices
        self._gpu_memory_info = gpu_memory_info or {}
        self._model_memory_usage: Dict[str, int] = {}

        # Initialize memory tracking for all GPUs if not provided
        if not self._gpu_memory_info:
            self._initialize_gpu_memory_tracking()

    def _initialize_gpu_memory_tracking(self):
        """Initialize GPU memory tracking for all available GPUs"""
        try:
            from ..device_utils import initialize_gpu_memory_info

            self._gpu_memory_info = initialize_gpu_memory_info(
                self._total_gpu_devices, logger=logger
            )
        except Exception as e:
            logger.warning(f"Failed to initialize GPU memory tracking: {e}")
            # Fallback to basic tracking without actual memory info
            for gpu_idx in self._total_gpu_devices:
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
        # Basic estimation logic - this can be enhanced with more sophisticated calculations
        if isinstance(model_size, str):
            # Convert string size like "7B" to integer
            if "B" in model_size:
                size_gb = float(model_size.replace("B", ""))
            else:
                size_gb = float(model_size)
        else:
            size_gb = float(model_size)

        # Base memory estimation (rough calculation)
        base_memory_mb = int(size_gb * 1024 * 1.5)  # 1.5GB per billion parameters

        # Adjust based on quantization
        if quantization:
            if "4bit" in quantization.lower() or "4-bit" in quantization.lower():
                base_memory_mb = base_memory_mb // 3
            elif "8bit" in quantization.lower() or "8-bit" in quantization.lower():
                base_memory_mb = base_memory_mb // 2

        # Adjust based on model format
        if model_format:
            if "mlx" in model_format.lower():
                base_memory_mb = int(
                    base_memory_mb * 0.8
                )  # MLX is generally more memory efficient

        return max(base_memory_mb, 1024)  # Minimum 1GB

    def _can_fit_model_on_gpu(self, gpu_idx: int, estimated_memory_mb: int) -> bool:
        """Check if a model can fit on a specific GPU"""
        # Update memory info for the GPU
        update_gpu_memory_info(self._gpu_memory_info, gpu_idx, logger=logger)

        available_memory = self._gpu_memory_info[gpu_idx]["available"]
        can_fit = estimated_memory_mb <= available_memory

        if can_fit:
            logger.info(
                f"Model can fit on GPU {gpu_idx}: needs {estimated_memory_mb}MB, has {available_memory}MB available"
            )
        else:
            logger.warning(
                f"Model cannot fit on GPU {gpu_idx}: needs {estimated_memory_mb}MB, has {available_memory}MB available"
            )

        return can_fit

    def _get_gpu_with_most_available_memory(self) -> int:
        """Find the GPU with the most available memory"""
        max_available_gpu = -1
        max_available_memory: Union[int, float] = -1

        for gpu_idx in self._total_gpu_devices:
            update_gpu_memory_info(self._gpu_memory_info, gpu_idx, logger=logger)
            available_memory = self._gpu_memory_info[gpu_idx]["available"]

            if available_memory > max_available_memory:
                max_available_memory = available_memory
                max_available_gpu = gpu_idx

        if max_available_gpu == -1:
            raise RuntimeError("No suitable GPU found")

        return max_available_gpu

    def allocate(
        self,
        spec: LaunchModelSpec,
        total_gpu_devices: List[int],
        user_specified_allocated_devices: Set[int],
        allocated_gpus: Dict[int, str],
    ) -> List[int]:
        """
        Allocate GPUs using memory-aware strategy

        Strategy:
        1. Prefer completely free GPUs
        2. If not enough, use GPUs with most available memory
        3. Support single-GPU multi-replica deployment
        """
        model_uid = spec.model_uid
        n_gpu = spec.n_gpu

        # Estimate model memory usage if model info is provided
        estimated_memory_mb = 0
        if spec.model_name and spec.model_size:
            estimated_memory_mb = self._estimate_model_memory_usage(
                spec.model_name, spec.model_size, spec.model_format, spec.quantization
            )
            self._model_memory_usage[model_uid] = estimated_memory_mb

        # Check for completely available GPUs first
        completely_available_gpus = [
            dev
            for dev in total_gpu_devices
            if dev not in allocated_gpus and dev not in user_specified_allocated_devices
        ]

        # If all visible GPUs are already occupied (by allocated or user-specified),
        # keep legacy behavior and fail fast instead of oversubscribing.
        if len(completely_available_gpus) < n_gpu:
            raise RuntimeError("No available slot found for the model")

        if estimated_memory_mb > 0:
            suitable_with_mem: List[Tuple[int, Union[int, float]]] = []

            # Include completely available GPUs first
            for gpu_idx in completely_available_gpus:
                if self._can_fit_model_on_gpu(gpu_idx, estimated_memory_mb):
                    available = self._gpu_memory_info[gpu_idx]["available"]
                    suitable_with_mem.append((gpu_idx, available))

            # Check already allocated GPUs for possible reuse
            for dev in total_gpu_devices:
                if dev in allocated_gpus:
                    if self._can_fit_model_on_gpu(dev, estimated_memory_mb):
                        available = self._gpu_memory_info[dev]["available"]
                        suitable_with_mem.append((dev, available))

            if suitable_with_mem:
                suitable_with_mem.sort(key=lambda x: x[1], reverse=True)
                selected = [dev for dev, _ in suitable_with_mem[:n_gpu]]
            else:
                # Not enough GPUs with sufficient memory, pick the best GPU and reuse it
                best_gpu = self._get_gpu_with_most_available_memory()
                selected = [best_gpu] if n_gpu == 1 else [best_gpu] * n_gpu
        else:
            # No memory estimation available, use basic strategy
            if len(completely_available_gpus) >= n_gpu:
                selected = completely_available_gpus[:n_gpu]
            else:
                # For single GPU deployment without memory estimation, allow sharing
                if n_gpu == 1 and total_gpu_devices:
                    # Use the first available GPU or the one with most available memory
                    if completely_available_gpus:
                        selected = [completely_available_gpus[0]]
                    else:
                        # No completely available GPU, find one with most available memory
                        best_gpu = self._get_gpu_with_most_available_memory()
                        selected = [best_gpu]
                else:
                    # Use GPUs with most available memory
                    remaining_needed = n_gpu - len(completely_available_gpus)
                    candidate_gpus = [
                        dev
                        for dev in total_gpu_devices
                        if dev not in completely_available_gpus
                        and dev not in allocated_gpus
                    ]

                    gpu_memory_list = []
                    for dev in candidate_gpus:
                        update_gpu_memory_info(
                            self._gpu_memory_info, dev, logger=logger
                        )
                        available_memory = self._gpu_memory_info[dev]["available"]
                        gpu_memory_list.append((dev, available_memory))

                    # Sort by available memory (descending)
                    gpu_memory_list.sort(key=lambda x: x[1], reverse=True)

                    selected = completely_available_gpus.copy()
                    for dev, available_memory in gpu_memory_list[:remaining_needed]:
                        selected.append(dev)

        if len(selected) < n_gpu:
            if not selected:
                best_gpu = self._get_gpu_with_most_available_memory()
                selected.append(best_gpu)
            fill_gpu = selected[0]
            while len(selected) < n_gpu:
                selected.append(fill_gpu)

        # Update memory usage accounting
        for gpu_idx in selected:
            if gpu_idx in self._gpu_memory_info and estimated_memory_mb > 0:
                self._gpu_memory_info[gpu_idx]["used"] += estimated_memory_mb
                self._gpu_memory_info[gpu_idx]["available"] -= estimated_memory_mb

        return selected

    def release(self, model_uid: str, devices: List[int]) -> None:
        """Release allocated GPUs and roll back memory accounting"""
        # Roll back memory usage accounting
        if model_uid in self._model_memory_usage:
            memory_used = self._model_memory_usage[model_uid]
            for gpu_idx in devices:
                if gpu_idx in self._gpu_memory_info:
                    # Roll back memory usage
                    self._gpu_memory_info[gpu_idx]["used"] -= memory_used
                    self._gpu_memory_info[gpu_idx]["available"] += memory_used

            # Remove model from memory tracking
            del self._model_memory_usage[model_uid]


class PackingFirstLaunchStrategy(LaunchStrategy):
    """
    Prefer filling one GPU before moving to the next highest-available GPU.
    Allows GPU reuse when requested replicas exceed available distinct GPUs.
    """

    def __init__(
        self,
        total_gpu_devices: List[int],
        gpu_memory_info: Optional[Dict[int, Dict[str, Union[int, float]]]] = None,
    ):
        self._total_gpu_devices = total_gpu_devices
        self._gpu_memory_info = gpu_memory_info or initialize_gpu_memory_info(
            total_gpu_devices, logger=logger
        )

    def allocate(
        self,
        spec: LaunchModelSpec,
        total_gpu_devices: List[int],
        user_specified_allocated_devices: Set[int],
        allocated_gpus: Dict[int, str],
    ) -> List[int]:
        candidates = [
            dev
            for dev in total_gpu_devices
            if dev not in allocated_gpus and dev not in user_specified_allocated_devices
        ]
        if not candidates:
            raise RuntimeError("No available slot found for the model")

        for dev in candidates:
            update_gpu_memory_info(self._gpu_memory_info, dev, logger=logger)
        candidates.sort(
            key=lambda d: self._gpu_memory_info.get(d, {}).get("available", 0),
            reverse=True,
        )

        selected: List[int] = []
        idx = 0
        while len(selected) < spec.n_gpu:
            chosen = candidates[min(idx, len(candidates) - 1)]
            selected.append(chosen)
            if idx < len(candidates) - 1:
                idx += 1

        return selected

    def release(self, model_uid: str, devices: List[int]) -> None:
        # No internal accounting maintained here
        return


class SpreadFirstLaunchStrategy(LaunchStrategy):
    """
    Prefer spreading replicas across distinct GPUs before reusing any GPU.
    Falls back to reuse when replicas exceed distinct GPUs.
    """

    def __init__(
        self,
        total_gpu_devices: List[int],
        gpu_memory_info: Optional[Dict[int, Dict[str, Union[int, float]]]] = None,
    ):
        self._total_gpu_devices = total_gpu_devices
        self._gpu_memory_info = gpu_memory_info or initialize_gpu_memory_info(
            total_gpu_devices, logger=logger
        )

    def allocate(
        self,
        spec: LaunchModelSpec,
        total_gpu_devices: List[int],
        user_specified_allocated_devices: Set[int],
        allocated_gpus: Dict[int, str],
    ) -> List[int]:
        candidates = [
            dev
            for dev in total_gpu_devices
            if dev not in allocated_gpus and dev not in user_specified_allocated_devices
        ]
        if not candidates:
            raise RuntimeError("No available slot found for the model")

        for dev in candidates:
            update_gpu_memory_info(self._gpu_memory_info, dev, logger=logger)
        candidates.sort(
            key=lambda d: self._gpu_memory_info.get(d, {}).get("available", 0),
            reverse=True,
        )

        selected: List[int] = []
        idx = 0
        while len(selected) < spec.n_gpu:
            chosen = candidates[idx % len(candidates)]
            selected.append(chosen)
            idx += 1

        return selected

    def release(self, model_uid: str, devices: List[int]) -> None:
        return


class QuotaAwareLaunchStrategy(LaunchStrategy):
    """
    Restrict allocation to an allowed set of GPUs, then spread-first within that set.
    """

    def __init__(
        self,
        total_gpu_devices: List[int],
        allowed_devices: Optional[Set[int]] = None,
        gpu_memory_info: Optional[Dict[int, Dict[str, Union[int, float]]]] = None,
    ):
        self._total_gpu_devices = total_gpu_devices
        self._allowed_devices = allowed_devices
        self._gpu_memory_info = gpu_memory_info or initialize_gpu_memory_info(
            total_gpu_devices, logger=logger
        )

    def allocate(
        self,
        spec: LaunchModelSpec,
        total_gpu_devices: List[int],
        user_specified_allocated_devices: Set[int],
        allocated_gpus: Dict[int, str],
    ) -> List[int]:
        device_pool = (
            [dev for dev in total_gpu_devices if dev in self._allowed_devices]
            if self._allowed_devices is not None
            else total_gpu_devices
        )
        candidates = [
            dev
            for dev in device_pool
            if dev not in allocated_gpus and dev not in user_specified_allocated_devices
        ]
        if not candidates:
            raise RuntimeError("No available slot found for the model")

        for dev in candidates:
            update_gpu_memory_info(self._gpu_memory_info, dev, logger=logger)
        candidates.sort(
            key=lambda d: self._gpu_memory_info.get(d, {}).get("available", 0),
            reverse=True,
        )

        selected: List[int] = []
        idx = 0
        while len(selected) < spec.n_gpu:
            chosen = candidates[idx % len(candidates)]
            selected.append(chosen)
            idx += 1

        return selected

    def release(self, model_uid: str, devices: List[int]) -> None:
        return


def create_launch_strategy(
    strategy_name: str,
    total_gpu_devices: List[int],
    allowed_devices: Optional[Set[int]] = None,
    gpu_memory_info: Optional[Dict[int, Dict[str, Union[int, float]]]] = None,
) -> LaunchStrategy:
    strategy_name = strategy_name.lower()
    if strategy_name == "memory_aware":
        return MemoryAwareLaunchStrategy(total_gpu_devices, gpu_memory_info)
    if strategy_name == "packing_first":
        return PackingFirstLaunchStrategy(total_gpu_devices, gpu_memory_info)
    if strategy_name == "spread_first":
        return SpreadFirstLaunchStrategy(total_gpu_devices, gpu_memory_info)
    if strategy_name == "quota_aware":
        return QuotaAwareLaunchStrategy(
            total_gpu_devices, allowed_devices, gpu_memory_info
        )
    logger.warning(
        f"Unknown launch strategy '{strategy_name}', falling back to memory_aware"
    )
    return MemoryAwareLaunchStrategy(total_gpu_devices, gpu_memory_info)
