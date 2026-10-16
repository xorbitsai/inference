from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from time import perf_counter

from tqdm import tqdm


class TrainProgress:
    def __init__(
        self,
        *,
        max_steps: int,
        start_step: int,
        rank: int,
        world_size: int,
        enabled: bool,
        show_all_ranks: bool,
        description: str,
        smooth_window: int = 20,
    ) -> None:
        self._show_progress = bool(enabled) and (world_size == 1 or show_all_ranks or rank == 0)
        bar_desc = description
        if world_size > 1:
            bar_desc = f"{description} [rank {rank + 1}/{world_size}]"
        position = rank if show_all_ranks else 0
        self._pbar = tqdm(
            total=max_steps,
            initial=start_step,
            desc=bar_desc,
            unit="step",
            dynamic_ncols=True,
            disable=not self._show_progress,
            position=position,
            leave=False,
        )
        self._smooth_window = max(1, int(smooth_window))
        self._metric_history: dict[str, deque[float]] = {}
        self._last_log_step = int(start_step)
        self._last_log_time = perf_counter()

    def update(self, step: int) -> None:
        delta = int(step) - int(self._pbar.n)
        if delta > 0:
            self._pbar.update(delta)

    def log(
        self,
        *,
        step: int,
        epoch: int,
        epoch_step: int | None = None,
        epoch_total: int | None = None,
        metrics: Mapping[str, float],
        global_batch_size: int | None = None,
    ) -> None:
        now = perf_counter()
        steps_delta = max(0, int(step) - self._last_log_step)
        dt = max(now - self._last_log_time, 1e-6)

        if self._show_progress:
            postfix: dict[str, str] = {"epoch": str(epoch)}
            if epoch_step is not None and epoch_total is not None and epoch_total > 0:
                epoch_pct = 100.0 * float(epoch_step) / float(epoch_total)
                postfix["epoch_step"] = f"{epoch_step}/{epoch_total}"
                postfix["epoch%"] = f"{epoch_pct:.1f}%"
            if steps_delta > 0:
                iter_per_sec = steps_delta / dt
                postfix["it/s"] = f"{iter_per_sec:.2f}"
                if global_batch_size is not None and global_batch_size > 0:
                    postfix["samples/s"] = f"{iter_per_sec * global_batch_size:.1f}"
            for key, value in metrics.items():
                history = self._metric_history.setdefault(key, deque(maxlen=self._smooth_window))
                history.append(float(value))
                smoothed = sum(history) / float(len(history))
                postfix[key] = self._format_metric(key, smoothed)
            self._pbar.set_postfix(postfix, refresh=False)

        self._last_log_step = int(step)
        self._last_log_time = now

    def write(self, message: str) -> None:
        if self._show_progress:
            self._pbar.write(message)
            return
        print(message)

    def close(self) -> None:
        self._pbar.close()

    @staticmethod
    def _format_metric(name: str, value: float) -> str:
        if name.lower() == "lr":
            return f"{value:.2e}"
        abs_value = abs(value)
        if abs_value >= 1000.0 or (0.0 < abs_value < 1e-3):
            return f"{value:.2e}"
        return f"{value:.4f}"
