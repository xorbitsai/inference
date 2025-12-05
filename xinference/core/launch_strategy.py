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
from ..model.llm.memory import estimate_llm_gpu_memory

logger = logging.getLogger(__name__)


@dataclass
class LaunchModelSpec:
    """Specification for model launch"""

    model_uid: str
    n_gpu: int
    context_length: Optional[int] = None
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

    _DEFAULT_MIN_MEMORY_MB = 1024
    _DEFAULT_CONTEXT_LENGTH = 2048

    _DEFAULT_MIN_MEMORY_MB = 1024

    def __init__(
        self,
        total_gpu_devices: List[int],
        allowed_devices: Optional[Set[int]] = None,
        gpu_memory_info: Optional[Dict[int, Dict[str, Union[int, float]]]] = None,
    ):
        self._total_gpu_devices = sorted(total_gpu_devices)
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

    def _normalize_quantization(self, quantization: Optional[str]) -> Optional[str]:
        if quantization is None:
            return None
        quant_str = quantization.strip().lower()
        if quant_str in {"", "none", "null"}:
            return None
        return quantization

    def _estimate_model_memory_usage(
        self,
        model_name: str,
        model_size: Union[int, str],
        model_format: Optional[str],
        quantization: Optional[str],
        context_length: Optional[int],
    ) -> int:
        """Estimate memory usage using cal-model-mem algorithm."""
        ctx_len = context_length or self._DEFAULT_CONTEXT_LENGTH
        fmt = model_format or "pytorch"
        quant_norm = self._normalize_quantization(quantization)
        try:
            mem_info = estimate_llm_gpu_memory(
                model_size_in_billions=model_size,
                quantization=quant_norm,
                context_length=ctx_len,
                model_format=fmt,
                model_name=model_name,
            )
            if mem_info is not None:
                return max(int(mem_info.total), self._DEFAULT_MIN_MEMORY_MB)
        except Exception as e:
            logger.warning(f"Failed to estimate memory for {model_name}: {e}")

        return self._DEFAULT_MIN_MEMORY_MB

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

        return None

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
                spec.model_name,
                spec.model_size,
                spec.model_format,
                spec.quantization,
                spec.context_length,
            )
            self._model_memory_usage[model_uid] = estimated_memory_mb
        else:
            estimated_memory_mb = self._DEFAULT_MIN_MEMORY_MB
            self._model_memory_usage[model_uid] = estimated_memory_mb
        selected: List[int] = []
        # user-specified slots不能被自动策略占用，其余已分配的GPU允许复用，靠可用显存判断是否还能放得下
        in_use_gpus: Set[int] = set(user_specified_allocated_devices)

        target_gpu = self._current_gpu if self._current_gpu in candidate_gpus else None

        while len(selected) < n_gpu:
            if target_gpu is not None and self._can_fit_model_on_gpu(
                target_gpu, estimated_memory_mb, memory_snapshot
            ):
                selected.append(target_gpu)
                self._consume_allocation(
                    target_gpu, estimated_memory_mb, memory_snapshot
                )
                continue

            next_gpu = self._select_most_idle_gpu(
                candidate_gpus, in_use_gpus, memory_snapshot, estimated_memory_mb
            )
            if next_gpu is None:
                raise RuntimeError("No available slot found for the model")

            target_gpu = next_gpu
            in_use_gpus.add(next_gpu)

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
