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
from typing import Dict, List, Mapping, Optional, Set, Tuple, Union

import torch

from ..device_utils import initialize_gpu_memory_info, update_gpu_memory_info
from ..model.llm.llm_family import DEFAULT_CONTEXT_LENGTH
from ..model.llm.memory import estimate_llm_gpu_memory

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
        allocated_gpus: Mapping[int, Set[str]],
    ) -> List[int]:
        """Allocate GPUs for model launch"""
        pass

    @abstractmethod
    def release(self, model_uid: str, devices: List[int]) -> None:
        """Release GPUs allocated for a model"""
        pass


class LocalFirstLaunchStrategy(LaunchStrategy):
    """
    Prefer the GPU running Xinference, otherwise keep allocating onto the emptiest
    remaining GPU.
    """

    def __init__(
        self,
        total_gpu_devices: List[int],
        allowed_devices: Optional[Set[int]] = None,
        gpu_memory_info: Optional[Dict[int, Dict[str, Union[int, float]]]] = None,
    ):
        self._allowed_devices = allowed_devices
        self._total_gpu_devices = self._filter_allowed(total_gpu_devices)
        self._gpu_memory_info = gpu_memory_info or initialize_gpu_memory_info(
            self._total_gpu_devices, logger=logger
        )
        self._model_memory_usage: Dict[str, Tuple[int, Dict[int, int]]] = {}
        self._preferred_gpu = self._detect_preferred_gpu()

    def _filter_allowed(self, total_gpu_devices: List[int]) -> List[int]:
        if self._allowed_devices is None:
            return total_gpu_devices
        return [dev for dev in total_gpu_devices if dev in self._allowed_devices]

    def _detect_preferred_gpu(self) -> Optional[int]:
        try:
            if torch.cuda.is_available():
                gpu_idx = torch.cuda.current_device()
                if gpu_idx in self._total_gpu_devices:
                    return gpu_idx
        except Exception:
            pass
        return self._total_gpu_devices[0] if self._total_gpu_devices else None

    def _estimate_model_memory_usage(
        self,
        model_name: Optional[str],
        model_size: Optional[Union[int, str]],
        model_format: Optional[str],
        quantization: Optional[str],
    ) -> int:
        """Estimate memory usage using the documented cal-model-mem algorithm."""
        if model_size is None:
            return 1024

        def _normalize_size(size: Union[int, str]) -> str:
            if isinstance(size, str):
                normalized = size.strip().lower().rstrip("b")
                return normalized if normalized else "0"
            return str(size)

        size_in_billions = _normalize_size(model_size)
        model_format = model_format or "pytorch"

        try:
            mem_info = estimate_llm_gpu_memory(
                model_size_in_billions=size_in_billions,
                quantization=quantization,
                context_length=DEFAULT_CONTEXT_LENGTH,
                model_format=model_format,
                model_name=model_name,
                kv_cache_dtype=16,
            )
            if mem_info is None and model_name:
                mem_info = estimate_llm_gpu_memory(
                    model_size_in_billions=size_in_billions,
                    quantization=quantization,
                    context_length=DEFAULT_CONTEXT_LENGTH,
                    model_format=model_format,
                    model_name=None,
                    kv_cache_dtype=16,
                )
            if mem_info is not None:
                return max(int(mem_info.total), 1024)
        except Exception:
            logger.debug("Failed to estimate memory via cal-model-mem", exc_info=True)

        # If estimation fails, keep minimal guard to avoid zero/negative allocation.
        return 1024

    def _has_capacity(
        self,
        gpu_idx: int,
        estimated_memory_mb: int,
        pending_gpu_counts: Dict[int, int],
        allocated_gpus: Mapping[int, Set[str]],
    ) -> bool:
        if estimated_memory_mb <= 0:
            return True

        update_gpu_memory_info(self._gpu_memory_info, gpu_idx, logger=logger)
        gpu_info = self._gpu_memory_info.get(gpu_idx, {})
        available = gpu_info.get("available", 0)
        total = gpu_info.get("total", 0)
        # If we cannot get valid memory info, assume capacity is available to avoid false negatives.
        if total == 0 and available == 0:
            return True
        planned_usage = (
            pending_gpu_counts.get(gpu_idx, 0) + len(allocated_gpus.get(gpu_idx, set()))
        ) * estimated_memory_mb
        return available - planned_usage >= estimated_memory_mb

    def _is_available(
        self,
        gpu_idx: int,
        user_specified_allocated_devices: Set[int],
        allocated_gpus: Mapping[int, Set[str]],
        estimated_memory_mb: int,
        pending_gpu_counts: Dict[int, int],
    ) -> bool:
        if gpu_idx in user_specified_allocated_devices:
            return False
        return self._has_capacity(
            gpu_idx, estimated_memory_mb, pending_gpu_counts, allocated_gpus
        )

    def _select_emptiest_gpu(
        self,
        candidates: List[int],
        estimated_memory_mb: int,
        pending_gpu_counts: Dict[int, int],
        allocated_gpus: Mapping[int, Set[str]],
    ) -> Optional[int]:
        if not candidates:
            return None

        scored: List[Tuple[int, Union[int, float]]] = []
        for dev in candidates:
            if not self._has_capacity(
                dev, estimated_memory_mb, pending_gpu_counts, allocated_gpus
            ):
                continue
            update_gpu_memory_info(self._gpu_memory_info, dev, logger=logger)
            available = self._gpu_memory_info.get(dev, {}).get("available", 0)
            available -= (
                pending_gpu_counts.get(dev, 0) + len(allocated_gpus.get(dev, set()))
            ) * estimated_memory_mb
            scored.append((dev, available))

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
        n_gpu = spec.n_gpu
        estimated_memory_mb = self._estimate_model_memory_usage(
            spec.model_name, spec.model_size, spec.model_format, spec.quantization
        )
        logger.info(
            "Launch estimate for %s: %s MB (name=%s, size=%s, format=%s, quant=%s)",
            spec.model_uid,
            estimated_memory_mb,
            spec.model_name,
            spec.model_size,
            spec.model_format,
            spec.quantization,
        )

        pending_gpu_counts: Dict[int, int] = {}
        selected: List[int] = []

        preferred_gpu = (
            self._preferred_gpu
            if self._preferred_gpu in available_total
            else (available_total[0] if available_total else None)
        )

        if preferred_gpu is not None and self._is_available(
            preferred_gpu,
            user_specified_allocated_devices,
            allocated_gpus,
            estimated_memory_mb,
            pending_gpu_counts,
        ):
            while len(selected) < n_gpu and self._is_available(
                preferred_gpu,
                user_specified_allocated_devices,
                allocated_gpus,
                estimated_memory_mb,
                pending_gpu_counts,
            ):
                selected.append(preferred_gpu)
                pending_gpu_counts[preferred_gpu] = (
                    pending_gpu_counts.get(preferred_gpu, 0) + 1
                )

        if len(selected) < n_gpu:
            candidate_pool = [
                dev
                for dev in available_total
                if dev != preferred_gpu and dev not in user_specified_allocated_devices
            ]
            emptiest_gpu = self._select_emptiest_gpu(
                candidate_pool, estimated_memory_mb, pending_gpu_counts, allocated_gpus
            )
            if emptiest_gpu is None:
                raise RuntimeError("No available slot found for the model")

            while len(selected) < n_gpu and self._is_available(
                emptiest_gpu,
                user_specified_allocated_devices,
                allocated_gpus,
                estimated_memory_mb,
                pending_gpu_counts,
            ):
                selected.append(emptiest_gpu)
                pending_gpu_counts[emptiest_gpu] = (
                    pending_gpu_counts.get(emptiest_gpu, 0) + 1
                )

        if len(selected) < n_gpu:
            raise RuntimeError("No available slot found for the model")

        if estimated_memory_mb > 0:
            for gpu_idx, count in pending_gpu_counts.items():
                if gpu_idx in self._gpu_memory_info:
                    self._gpu_memory_info[gpu_idx]["used"] += (
                        estimated_memory_mb * count
                    )
                    self._gpu_memory_info[gpu_idx]["available"] -= (
                        estimated_memory_mb * count
                    )

        self._model_memory_usage[model_uid] = (estimated_memory_mb, pending_gpu_counts)
        return selected

    def release(self, model_uid: str, devices: List[int]) -> None:
        record = self._model_memory_usage.pop(model_uid, None)
        if not record:
            return
        estimated_memory_mb, gpu_counts = record
        if estimated_memory_mb <= 0:
            return

        for gpu_idx, count in gpu_counts.items():
            if gpu_idx in self._gpu_memory_info:
                self._gpu_memory_info[gpu_idx]["used"] -= estimated_memory_mb * count
                self._gpu_memory_info[gpu_idx]["available"] += (
                    estimated_memory_mb * count
                )


def create_launch_strategy(
    strategy_name: str,
    total_gpu_devices: List[int],
    allowed_devices: Optional[Set[int]] = None,
    gpu_memory_info: Optional[Dict[int, Dict[str, Union[int, float]]]] = None,
) -> LaunchStrategy:
    normalized = strategy_name.lower()
    supported = {
        "local_first",
        "memory_aware",
        "packing_first",
        "spread_first",
        "quota_aware",
    }
    if normalized not in supported:
        logger.warning(
            f"Unknown launch strategy '{strategy_name}', falling back to local_first"
        )
    return LocalFirstLaunchStrategy(
        total_gpu_devices,
        allowed_devices=allowed_devices,
        gpu_memory_info=gpu_memory_info,
    )
