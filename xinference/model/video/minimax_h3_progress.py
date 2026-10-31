# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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

from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Optional

PROGRESS_PATH_KEY = "_xinference_progress_path"
WORKER_EXTENSION = "xinference.model.video.minimax_h3_progress.MiniMaxH3WorkerExtension"
_active_progress: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "xinference_h3_progress", default=None
)


def _write_progress(path: Optional[str], value: float) -> None:
    if path:
        # Workers share the request directory with the model actor. Atomic
        # replacement lets the actor poll without reading a partial number.
        target = Path(path)
        temporary = target.with_suffix(".tmp")
        try:
            temporary.write_text(str(value))
            temporary.replace(target)
        except OSError:
            # Cancellation may remove the directory while a worker drains.
            pass


def _sampling_path(sampling: Any) -> Optional[str]:
    return (getattr(sampling, "extra_args", None) or {}).get(PROGRESS_PATH_KEY)


class _ProgressBar:
    def __init__(self, bar: Any, state: Dict[str, Any], index: int):
        self._bar, self._state, self._index = bar, state, index

    def __enter__(self):
        self._bar.__enter__()
        return self

    def __exit__(self, *args):
        return self._bar.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self._bar, name)

    def update(self, count=1):
        result = self._bar.update(count)
        if self._bar.total:
            fraction = min(
                (self._index + self._bar.n / self._bar.total) / self._state["n"], 1.0
            )
            _write_progress(self._state["path"], 0.1 + 0.7 * fraction)
        return result


def _install_progress(pipeline: Any) -> None:
    if getattr(pipeline, "_xinference_progress_installed", False):
        return
    forward, progress_bar = pipeline.forward, pipeline.progress_bar

    def tracked_forward(request, *args, **kwargs):
        sampling = request.sampling_params
        path = _sampling_path(sampling)
        state = {"path": path, "n": sampling.num_outputs_per_prompt, "index": 0}
        token = _active_progress.set(state if path else None)
        _write_progress(path, 0.05)
        try:
            result = forward(request, *args, **kwargs)
            _write_progress(path, 0.9)
            return result
        finally:
            _active_progress.reset(token)

    def tracked_bar(*args, **kwargs):
        bar = progress_bar(*args, **kwargs)
        state = _active_progress.get()
        if state is None:
            return bar
        index = state["index"]
        state["index"] += 1
        _write_progress(state["path"], 0.1 + 0.7 * min(index / state["n"], 1.0))
        return _ProgressBar(bar, state, index)

    pipeline.forward, pipeline.progress_bar = tracked_forward, tracked_bar
    # Step execution does not call forward() or create a tqdm bar. Each state
    # carries its own sampling parameters even in a mixed request batch.
    if hasattr(pipeline, "step_scheduler"):
        scheduler = pipeline.step_scheduler

        def tracked_scheduler(state, *args, **kwargs):
            result = scheduler(state, *args, **kwargs)
            fraction = min(state.step_index / state.sampling.num_inference_steps, 1.0)
            _write_progress(_sampling_path(state.sampling), 0.1 + 0.7 * fraction)
            return result

        pipeline.step_scheduler = tracked_scheduler
    if hasattr(pipeline, "post_decode"):
        decode = pipeline.post_decode

        def tracked_decode(state, *args, **kwargs):
            result = decode(state, *args, **kwargs)
            _write_progress(_sampling_path(state.sampling), 0.9)
            return result

        pipeline.post_decode = tracked_decode
    pipeline._xinference_progress_installed = True


class MiniMaxH3WorkerExtension:
    model_runner: Any

    def xinference_enable_h3_progress(self) -> None:
        import torch.distributed as dist

        # Every GPU executes the pipeline, but only one rank publishes progress.
        if not dist.is_initialized() or dist.get_rank() == 0:
            _install_progress(self.model_runner.pipeline)
