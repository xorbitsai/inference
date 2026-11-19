from __future__ import annotations

import os
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from multiprocessing import Queue
from typing import Iterator

import torch
from loguru import logger

from ..utils.logging import categorized_log as logc

INFERENCE_STAGE_NAMES = (
    "FM",
    "latent_encoder",
    "patch_encoder",
    "LLM",
    "latent_decoder",
    "speaker_encoder",
    "vocoder",
)

_TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_INFERENCE_STAGE_NAME_MAP = {name.lower(): name for name in INFERENCE_STAGE_NAMES}
_CURRENT_INFERENCE_PROFILER: ContextVar[InferenceProfiler | None] = ContextVar(
    "current_inference_profiler",
    default=None,
)


def normalize_inference_stage_name(name: str) -> str:
    canonical = _INFERENCE_STAGE_NAME_MAP.get(name.strip().lower())
    if canonical is None:
        raise ValueError(
            f"Unsupported inference stage '{name}'. "
            f"Expected one of: {', '.join(INFERENCE_STAGE_NAMES)}."
        )
    return canonical


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_ENV_VALUES


@dataclass(slots=True)
class InferenceStageStat:
    seconds: float = 0.0
    count: int = 0


@dataclass(frozen=True, slots=True)
class ProfileEvent:
    stage: str
    seconds: float
    count: int
    pid: int


class DataProfiler:
    def __init__(self, queue: Queue | None = None):
        self._queue = queue
        self._pid = os.getpid()

    @property
    def enabled(self) -> bool:
        return self._queue is not None

    @contextmanager
    def measure(self, stage: str, *, count: int = 1) -> Iterator[None]:
        if self._queue is None:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self._queue.put(
                ProfileEvent(
                    stage=stage,
                    seconds=time.perf_counter() - start,
                    count=int(count),
                    pid=self._pid,
                )
            )

    def child(self) -> DataProfiler:
        return DataProfiler(self._queue)


def ensure_data_profiler(profiler: DataProfiler | None) -> DataProfiler:
    return DataProfiler() if profiler is None else profiler


class InferenceProfiler:
    def __init__(
        self,
        device: torch.device,
        *,
        log_calls: bool | None = None,
        request_id: str | None = None,
    ):
        self._device = device
        self._log_calls = (
            _env_flag("DOTS_TTS_PROFILE_EACH_CALL")
            if log_calls is None
            else bool(log_calls)
        )
        self._request_id = request_id
        self._stats = {stage: InferenceStageStat() for stage in INFERENCE_STAGE_NAMES}
        self._call_index = 0
        self._stage_call_indices = {stage: 0 for stage in INFERENCE_STAGE_NAMES}

    def _sync(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def _log_call(
        self,
        *,
        stage: str,
        seconds: float,
        count: int,
        phase: str | None,
        step: int | str | None,
    ) -> None:
        if not self._log_calls:
            return
        self._call_index += 1
        self._stage_call_indices[stage] += 1
        context = ""
        if phase is not None:
            context += f" phase={phase}"
        if step is not None:
            context += f" step={step}"
        logger.info(
            logc(
                "profile",
                "Inference profiling call: request_id={} stage={} call_index={} "
                "stage_call_index={} seconds={:.4f} count={}{}",
            ),
            self._request_id if self._request_id is not None else "-",
            stage,
            self._call_index,
            self._stage_call_indices[stage],
            seconds,
            int(count),
            context,
        )

    @contextmanager
    def measure(
        self,
        stage: str,
        *,
        count: int = 1,
        phase: str | None = None,
        step: int | str | None = None,
    ) -> Iterator[None]:
        stage = normalize_inference_stage_name(stage)
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            stat = self._stats[stage]
            elapsed = time.perf_counter() - start
            stat.seconds += elapsed
            stat.count += int(count)
            self._log_call(
                stage=stage,
                seconds=elapsed,
                count=int(count),
                phase=phase,
                step=step,
            )

    def summary(
        self,
        *,
        duration_seconds: float | None = None,
    ) -> dict[str, dict[str, float | int]]:
        summary: dict[str, dict[str, float | int]] = {}
        for stage in INFERENCE_STAGE_NAMES:
            stat = self._stats[stage]
            payload: dict[str, float | int] = {
                "seconds": stat.seconds,
                "count": stat.count,
            }
            if duration_seconds is not None:
                payload["rtf"] = (
                    stat.seconds / duration_seconds
                    if duration_seconds > 0
                    else float("inf")
                )
            summary[stage] = payload
        return summary


@contextmanager
def inference_profiling(
    *,
    enabled: bool,
    device: torch.device,
    log_calls: bool | None = None,
    request_id: str | None = None,
) -> Iterator[InferenceProfiler | None]:
    profiler = (
        InferenceProfiler(device, log_calls=log_calls, request_id=request_id)
        if enabled
        else None
    )
    with activate_inference_profiler(profiler):
        yield profiler


@contextmanager
def activate_inference_profiler(
    profiler: InferenceProfiler | None,
) -> Iterator[InferenceProfiler | None]:
    if profiler is None:
        yield None
        return
    token: Token[InferenceProfiler | None] = _CURRENT_INFERENCE_PROFILER.set(profiler)
    try:
        yield profiler
    finally:
        _CURRENT_INFERENCE_PROFILER.reset(token)


@contextmanager
def measure_inference(
    stage: str,
    *,
    count: int = 1,
    phase: str | None = None,
    step: int | str | None = None,
) -> Iterator[None]:
    profiler = _CURRENT_INFERENCE_PROFILER.get()
    if profiler is None:
        yield
        return
    with profiler.measure(stage, count=count, phase=phase, step=step):
        yield


def log_inference_profile(
    *,
    request_id: str,
    profiling: dict[str, dict[str, float | int]],
    duration_seconds: float,
) -> None:
    active_stages = [
        stage for stage in INFERENCE_STAGE_NAMES if int(profiling[stage]["count"]) > 0
    ]
    if not active_stages:
        logger.info(
            logc(
                "profile",
                "Inference profiling summary: request_id={} no_profiled_stages duration_seconds={:.3f}",
            ),
            request_id,
            duration_seconds,
        )
        return
    for stage in active_stages:
        stats = profiling[stage]
        logger.info(
            logc(
                "profile",
                "Inference profiling: request_id={} stage={} seconds={:.4f} count={} rtf={:.4f}",
            ),
            request_id,
            stage,
            float(stats["seconds"]),
            int(stats["count"]),
            float(stats["rtf"]),
        )


__all__ = [
    "DataProfiler",
    "ProfileEvent",
    "INFERENCE_STAGE_NAMES",
    "activate_inference_profiler",
    "ensure_data_profiler",
    "InferenceProfiler",
    "InferenceStageStat",
    "inference_profiling",
    "log_inference_profile",
    "measure_inference",
    "normalize_inference_stage_name",
]
