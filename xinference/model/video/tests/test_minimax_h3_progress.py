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

# CI: linux

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from ..minimax_h3_progress import (
    PROGRESS_PATH_KEY,
    MiniMaxH3WorkerExtension,
    _install_progress,
    _write_progress,
)


class _Bar:
    def __init__(self, total):
        self.n, self.total = 0, total

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, count=1):
        self.n += count


@pytest.mark.parametrize("n", [1, 2])
def test_full_forward_reports_steps_and_reserves_decode_progress(tmp_path, n):
    path = tmp_path / "progress.txt"
    values = []
    pipeline = SimpleNamespace(progress_bar=lambda **kw: _Bar(**kw))

    def forward(request):
        assert float(path.read_text()) == 0.05
        for _ in range(n):
            with pipeline.progress_bar(total=4) as bar:
                for _ in range(4):
                    bar.update()
                    values.append(float(path.read_text()))
        return "result"

    pipeline.forward = forward
    _install_progress(pipeline)
    installed = pipeline.forward
    _install_progress(pipeline)
    assert pipeline.forward is installed
    sampling = SimpleNamespace(
        extra_args={PROGRESS_PATH_KEY: str(path)}, num_outputs_per_prompt=n
    )
    assert pipeline.forward(SimpleNamespace(sampling_params=sampling)) == "result"
    assert values == pytest.approx(
        [0.1 + 0.7 * step / (4 * n) for step in range(1, 4 * n + 1)]
    )
    assert float(path.read_text()) == 0.9


def test_concurrent_forwards_do_not_mix_request_progress(tmp_path):
    barrier = Barrier(2, timeout=5)
    pipeline = SimpleNamespace(progress_bar=lambda **kw: _Bar(**kw))

    def forward(request):
        with pipeline.progress_bar(total=4) as bar:
            for _ in range(4):
                barrier.wait()
                bar.update()
        return request.sampling_params.extra_args[PROGRESS_PATH_KEY]

    pipeline.forward = forward
    _install_progress(pipeline)
    requests = [
        SimpleNamespace(
            sampling_params=SimpleNamespace(
                extra_args={PROGRESS_PATH_KEY: str(tmp_path / f"{index}.txt")},
                num_outputs_per_prompt=1,
            )
        )
        for index in range(2)
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(pipeline.forward, requests))
    assert len(set(paths)) == 2
    assert all(float(Path(path).read_text()) == 0.9 for path in paths)


def test_step_execution_reports_each_request_and_decode_separately(tmp_path):
    def schedule(state):
        state.step_index += 1

    pipeline = SimpleNamespace(
        forward=lambda req: None,
        progress_bar=lambda **kw: _Bar(**kw),
        step_scheduler=schedule,
        post_decode=lambda state: "decoded",
    )
    _install_progress(pipeline)
    path = tmp_path / "step.txt"
    state = SimpleNamespace(
        step_index=0,
        sampling=SimpleNamespace(
            extra_args={PROGRESS_PATH_KEY: str(path)},
            num_inference_steps=4,
        ),
    )
    for step in range(1, 5):
        pipeline.step_scheduler(state)
        assert float(path.read_text()) == pytest.approx(0.1 + 0.7 * step / 4)
    assert pipeline.post_decode(state) == "decoded"
    assert float(path.read_text()) == 0.9


def test_missing_request_directory_does_not_break_worker(tmp_path):
    _write_progress(str(tmp_path / "removed" / "progress.txt"), 0.5)
    assert not list(tmp_path.iterdir())


def test_untracked_forward_uses_the_original_progress_bar():
    bar = nullcontext()
    pipeline = SimpleNamespace(progress_bar=lambda **kw: bar)

    def forward(request):
        assert pipeline.progress_bar(total=4) is bar

    pipeline.forward = forward
    _install_progress(pipeline)
    pipeline.forward(
        SimpleNamespace(
            sampling_params=SimpleNamespace(
                extra_args={},
                num_outputs_per_prompt=1,
            )
        )
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_worker_extension_only_installs_on_rank_zero(monkeypatch, rank):
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: rank)
    worker = MiniMaxH3WorkerExtension()
    pipeline = SimpleNamespace(
        forward=lambda req: None, progress_bar=lambda **kw: _Bar(**kw)
    )
    worker.model_runner = SimpleNamespace(pipeline=pipeline)
    worker.xinference_enable_h3_progress()
    assert getattr(pipeline, "_xinference_progress_installed", False) is (rank == 0)
