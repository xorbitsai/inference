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

import sys
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import xoscar as xo

from .... import constants
from ....core import worker as worker_module
from ....core.cache_metadata import get_model_version_infos
from ....core.cache_tracker import CacheTrackerActor
from ... import video
from ...cache_manager import CacheManager
from .. import core, engine_family
from ..ltx25_checkpoint import LTX25_BASE_FILES


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_cache", [False, True], ids=["factory", "legacy"])
async def test_modelscope_phai_m1_inventory_after_restart(
    tmp_path, monkeypatch, legacy_cache
):
    cache_dir = tmp_path / "cache"
    model_dir = tmp_path / "models"
    hub_dir = tmp_path / "modelscope"
    monkeypatch.setattr(constants, "XINFERENCE_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(constants, "XINFERENCE_MODEL_DIR", str(model_dir))
    monkeypatch.setattr(video, "XINFERENCE_MODEL_DIR", str(model_dir))
    monkeypatch.setattr(worker_module, "XINFERENCE_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("MODELSCOPE_CACHE", str(hub_dir))
    monkeypatch.setattr(CacheManager, "is_initialized", False)

    # Isolate the registries that the normal installation and factory publish.
    families, descriptions, engines = {}, {}, {}
    monkeypatch.setattr(video, "BUILTIN_VIDEO_MODELS", families)
    monkeypatch.setattr(video, "VIDEO_MODEL_DESCRIPTIONS", descriptions)
    monkeypatch.setattr(core, "VIDEO_MODEL_DESCRIPTIONS", descriptions)
    monkeypatch.setattr(video, "VIDEO_ENGINES", engines)
    monkeypatch.setattr(engine_family, "VIDEO_ENGINES", engines)
    video._install()

    model_name = "Phai-M1-dev"
    sibling_name = "Phai-M1-distilled"
    filename = "Phai-M1-22b-dev-transformer-bf16.safetensors"
    relative_checkpoint = f"diffusion_models/{filename}"
    checkpoint = hub_dir / "models" / "PhaiModelLab" / "Phai-M1" / relative_checkpoint
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    snapshot = hub_dir / "models" / "Lightricks" / "LTX-2.5-Diffusers"
    for name in LTX25_BASE_FILES:
        artifact = snapshot / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.touch()

    download_checkpoint = Mock(return_value=str(checkpoint))
    download_base = Mock(return_value=str(snapshot))
    monkeypatch.setitem(
        sys.modules,
        "modelscope.hub.file_download",
        SimpleNamespace(model_file_download=download_checkpoint),
    )
    monkeypatch.setitem(
        sys.modules,
        "modelscope.hub.snapshot_download",
        SimpleNamespace(snapshot_download=download_base),
    )
    for name in ("hf_hub_download", "snapshot_download"):
        monkeypatch.setattr(
            f"huggingface_hub.{name}",
            lambda *args, **kwargs: pytest.fail("Unexpected Hugging Face download"),
        )

    variant_cache = cache_dir / "v2" / model_name
    if legacy_cache:
        # Older versions kept the base beside the variant, keyed by hub.
        variant_cache.mkdir()
        (variant_cache / filename).symlink_to(checkpoint)
        base_cache = cache_dir / "v2" / "Phai-M1-LTX-2.5-modelscope"
        for name in LTX25_BASE_FILES:
            link = base_cache / name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(snapshot / name)
    else:
        core.create_video_model_instance(
            "phai-m1-dev",
            model_name,
            model_engine="diffusers",
            download_hub="modelscope",
            enable_virtual_env=False,
        )
        download_checkpoint.assert_called_once_with(
            "PhaiModelLab/Phai-M1", relative_checkpoint, revision="master"
        )
        download_base.assert_called_once()

    video._install()
    inventory = get_model_version_infos()
    assert len(inventory[model_name]) == len(inventory[sibling_name]) == 1
    if legacy_cache:
        download_checkpoint.assert_not_called()
        download_base.assert_not_called()

    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        tracker = await xo.create_actor(
            CacheTrackerActor, address=pool.external_address
        )
        worker = SimpleNamespace(address="127.0.0.1:12345", _cache_tracker_ref=tracker)
        await tracker.record_model_version(inventory, worker.address)
        assert await tracker.list_cached_models(worker.address, model_name) == [
            {
                "cache_status": True,
                "model_file_location": str(variant_cache),
                "model_name": model_name,
                "model_type": "video",
                "model_version": model_name,
            }
        ]
        assert await tracker.list_cached_models(worker.address, sibling_name) == []
        deletable = await worker_module.WorkerActor.list_deletable_models(
            worker, model_name
        )
        assert str(variant_cache) in deletable
        assert (
            await worker_module.WorkerActor.list_deletable_models(worker, sibling_name)
            == []
        )
        worker.list_deletable_models = MethodType(
            worker_module.WorkerActor.list_deletable_models, worker
        )
        assert await worker_module.WorkerActor.confirm_and_remove_model(
            worker, model_name
        )
        assert not variant_cache.exists()
        assert not checkpoint.exists()
        assert await tracker.list_cached_models(worker.address, model_name) == []
