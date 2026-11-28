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

from functools import partial
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ... import constants
from ...model.image.cache_manager import ImageCacheManager
from ...model.image.core import match_diffusion
from ...model.image.sensenova_u1 import get_sensenova_checkpoint_files
from .. import worker as worker_module
from ..supervisor import SupervisorActor
from ..worker import WorkerActor


@pytest.mark.asyncio
async def test_remove_cache_reaches_every_worker_after_a_failure():
    failing = SimpleNamespace(confirm_and_remove_model=AsyncMock(return_value=False))
    healthy = SimpleNamespace(confirm_and_remove_model=AsyncMock(return_value=True))
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._worker_address_to_worker = {"w1:1": failing, "w2:2": healthy}

    assert not await SupervisorActor.confirm_and_remove_model(supervisor, "demo")

    for worker in (failing, healthy):
        worker.confirm_and_remove_model.assert_awaited_once_with(model_version="demo")


@pytest.mark.asyncio
@pytest.mark.parametrize("hub", ["huggingface", "modelscope"])
@pytest.mark.parametrize("order", [("Q4_0", "Q8_0"), ("Q8_0", "Q4_0")])
@pytest.mark.parametrize("quantization_to_remove", ["Q4_0", "Q8_0"])
async def test_remove_sensenova_gguf_cache_reclaims_only_selected_weights(
    tmp_path, monkeypatch, hub, order, quantization_to_remove
):
    cache_root = str(tmp_path / "cache")
    monkeypatch.setattr(constants, "XINFERENCE_CACHE_DIR", cache_root)
    monkeypatch.setattr(constants, "XINFERENCE_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(worker_module, "XINFERENCE_CACHE_DIR", cache_root)
    monkeypatch.setattr("xinference.model.utils.IS_NEW_HUGGINGFACE_HUB", True)

    if hub == "huggingface":
        hub_root = tmp_path / "huggingface"
        monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub_root))
        repo = hub_root / "models--sensenova--SenseNova-U1.6-8B-MoT-GGUF"
        snapshot = repo / "snapshots" / "revision"
        blobs = repo / "blobs"
        download_module = "huggingface_hub"
        pattern_key = "allow_patterns"
    else:
        hub_root = tmp_path / "modelscope"
        monkeypatch.setenv("MODELSCOPE_CACHE", str(hub_root))
        snapshot = hub_root / "models" / "SenseNova" / "SenseNova-U1.6-8B-MoT-GGUF"
        blobs = snapshot
        download_module = "modelscope.hub.snapshot_download"
        pattern_key = "allow_file_pattern"

    def download(_model_id, **kwargs):
        snapshot.mkdir(parents=True, exist_ok=True)
        blobs.mkdir(parents=True, exist_ok=True)
        for name in kwargs[pattern_key]:
            file = snapshot / name
            blob = blobs / name
            if not blob.exists():
                blob.write_text(name, encoding="utf-8")
                if file != blob:
                    file.symlink_to(blob)
        return str(snapshot)

    monkeypatch.setattr(import_module(download_module), "snapshot_download", download)

    model_name = "SenseNova-U1.6-8B-MoT-GGUF"
    filenames = {}
    managers = {}
    for quantization in order:
        spec = match_diffusion(model_name, hub, quantization=quantization)
        filenames[quantization] = get_sensenova_checkpoint_files(spec)
        manager = ImageCacheManager(spec)
        manager.cache()
        assert manager.get_cache_status()
        managers[quantization] = manager

    removed = managers[quantization_to_remove]
    removed_weight = blobs / filenames[quantization_to_remove][0]
    tracker = SimpleNamespace(
        list_deletable_models=AsyncMock(return_value=removed.get_cache_dir()),
        confirm_and_remove_model=AsyncMock(),
    )
    worker = SimpleNamespace(_cache_tracker_ref=tracker, address="127.0.0.1:0")
    worker.list_deletable_models = partial(WorkerActor.list_deletable_models, worker)

    for quantization, manager in managers.items():
        assert {p.name for p in Path(manager.get_cache_dir()).iterdir()} == set(
            filenames[quantization]
        )
    assert removed_weight.is_file()
    assert await WorkerActor.confirm_and_remove_model(
        worker, f"{model_name}--{quantization_to_remove}"
    )
    assert not removed.get_cache_status()
    assert not removed_weight.exists()

    remaining = next(q for q in order if q != quantization_to_remove)
    remaining_manager = managers[remaining]
    assert remaining_manager.get_cache_status()
    for name in filenames[remaining]:
        assert (Path(remaining_manager.get_cache_dir()) / name).read_text(
            encoding="utf-8"
        ) == name
