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
from contextlib import asynccontextmanager
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
from ..cache_manager import VideoCacheManager
from ..ltx25_checkpoint import LTX25_BASE_FILES

MODEL_NAME = "Phai-M1-dev"
SIBLING_NAME = "Phai-M1-distilled"
CHECKPOINT_NAME = "Phai-M1-22b-dev-transformer-bf16.safetensors"


@pytest.fixture
def hub_cache(request, tmp_path, monkeypatch):
    hub = getattr(request, "param", "modelscope")
    cache_dir = tmp_path / "cache"
    model_dir = tmp_path / "models"
    hub_dir = tmp_path / hub
    monkeypatch.setattr(constants, "XINFERENCE_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(constants, "XINFERENCE_MODEL_DIR", str(model_dir))
    monkeypatch.setattr(video, "XINFERENCE_MODEL_DIR", str(model_dir))
    monkeypatch.setattr(worker_module, "XINFERENCE_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(CacheManager, "is_initialized", False)

    # Isolate the registries that the normal installation and factory publish.
    families, descriptions, engines = {}, {}, {}
    monkeypatch.setattr(video, "BUILTIN_VIDEO_MODELS", families)
    monkeypatch.setattr(video, "VIDEO_MODEL_DESCRIPTIONS", descriptions)
    monkeypatch.setattr(core, "VIDEO_MODEL_DESCRIPTIONS", descriptions)
    monkeypatch.setattr(video, "VIDEO_ENGINES", engines)
    monkeypatch.setattr(engine_family, "VIDEO_ENGINES", engines)
    video._install()

    if hub == "modelscope":
        monkeypatch.setenv("MODELSCOPE_CACHE", str(hub_dir))
        relative_checkpoint = f"diffusion_models/{CHECKPOINT_NAME}"
        checkpoint = (
            hub_dir / "models" / "PhaiModelLab" / "Phai-M1" / relative_checkpoint
        )
        snapshot = hub_dir / "models" / "Lightricks" / "LTX-2___5-Diffusers"
    else:
        monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub_dir))
        relative_checkpoint = CHECKPOINT_NAME
        checkpoint = (
            hub_dir
            / "models--PhaiModelLab--Phai-M1"
            / "snapshots"
            / "dev"
            / CHECKPOINT_NAME
        )
        snapshot = (
            hub_dir / "models--Lightricks--LTX-2.5-Diffusers" / "snapshots" / "base"
        )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    for name in LTX25_BASE_FILES:
        artifact = snapshot / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.touch()

    download_checkpoint = Mock(return_value=str(checkpoint))
    download_base = Mock(return_value=str(snapshot))

    def unexpected_download(*args, **kwargs):
        pytest.fail("Unexpected download from an unselected Hub")

    monkeypatch.setitem(
        sys.modules,
        "modelscope.hub.file_download",
        SimpleNamespace(
            model_file_download=(
                download_checkpoint if hub == "modelscope" else unexpected_download
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "modelscope.hub.snapshot_download",
        SimpleNamespace(
            snapshot_download=(
                download_base if hub == "modelscope" else unexpected_download
            )
        ),
    )
    for name, download in (
        ("hf_hub_download", download_checkpoint),
        ("snapshot_download", download_base),
    ):
        monkeypatch.setattr(
            f"huggingface_hub.{name}",
            download if hub == "huggingface" else unexpected_download,
        )
    return SimpleNamespace(
        hub=hub,
        cache_root=cache_dir / "v2",
        variant_cache=cache_dir / "v2" / MODEL_NAME,
        checkpoint=checkpoint,
        relative_checkpoint=relative_checkpoint,
        snapshot=snapshot,
        download_checkpoint=download_checkpoint,
        download_base=download_base,
    )


@asynccontextmanager
async def restarted_cache_tracker():
    video._install()
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        tracker = await xo.create_actor(
            CacheTrackerActor, address=pool.external_address
        )
        worker = SimpleNamespace(address="127.0.0.1:12345", _cache_tracker_ref=tracker)
        worker.list_deletable_models = MethodType(
            worker_module.WorkerActor.list_deletable_models, worker
        )
        await tracker.record_model_version(get_model_version_infos(), worker.address)
        yield tracker, worker


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_cache", [False, True], ids=["factory", "legacy"])
async def test_modelscope_phai_m1_inventory_after_restart(hub_cache, legacy_cache):
    variant_cache = hub_cache.variant_cache
    if legacy_cache:
        # Older versions kept the base beside the variant, keyed by hub.
        variant_cache.mkdir()
        (variant_cache / CHECKPOINT_NAME).symlink_to(hub_cache.checkpoint)
        base_cache = hub_cache.cache_root / "Phai-M1-LTX-2.5-modelscope"
        for name in LTX25_BASE_FILES:
            link = base_cache / name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(hub_cache.snapshot / name)
    else:
        core.create_video_model_instance(
            "phai-m1-dev",
            MODEL_NAME,
            model_engine="diffusers",
            download_hub="modelscope",
            enable_virtual_env=False,
        )
        hub_cache.download_checkpoint.assert_called_once_with(
            "PhaiModelLab/Phai-M1", hub_cache.relative_checkpoint, revision="master"
        )
        hub_cache.download_base.assert_called_once()

    async with restarted_cache_tracker() as (tracker, worker):
        assert await tracker.get_model_version_count(MODEL_NAME) == 1
        assert await tracker.get_model_version_count(SIBLING_NAME) == 1
        if legacy_cache:
            hub_cache.download_checkpoint.assert_not_called()
            hub_cache.download_base.assert_not_called()
        assert await tracker.list_cached_models(worker.address, MODEL_NAME) == [
            {
                "cache_status": True,
                "model_file_location": str(variant_cache),
                "model_name": MODEL_NAME,
                "model_type": "video",
                "model_version": MODEL_NAME,
            }
        ]
        assert await tracker.list_cached_models(worker.address, SIBLING_NAME) == []
        assert str(variant_cache) in await worker.list_deletable_models(MODEL_NAME)
        assert await worker.list_deletable_models(SIBLING_NAME) == []
        assert await worker_module.WorkerActor.confirm_and_remove_model(
            worker, MODEL_NAME
        )
        assert not variant_cache.exists()
        assert not hub_cache.checkpoint.exists()
        assert await tracker.list_cached_models(worker.address, MODEL_NAME) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("hub_cache", ["huggingface", "modelscope"], indirect=True)
@pytest.mark.parametrize("override", ["model_path", "base_model_path"])
async def test_mixed_phai_m1_cache_is_deletable_after_restart(
    hub_cache, tmp_path, override
):
    variant_cache = hub_cache.variant_cache
    local_checkpoint = override == "model_path"
    if local_checkpoint:
        source = tmp_path / CHECKPOINT_NAME
        local_path = source
    else:
        local_path = tmp_path / "local-base"
        local_path.mkdir()
        source = local_path / "model_index.json"
    source.write_bytes(b"user supplied")

    model = core.create_video_model_instance(
        "mixed-phai-m1-dev",
        MODEL_NAME,
        model_engine="diffusers",
        download_hub=hub_cache.hub,
        enable_virtual_env=False,
        **{override: str(local_path)},
    )
    assert not VideoCacheManager(model.model_spec).get_cache_status()

    if local_checkpoint:
        hub_cache.download_checkpoint.assert_not_called()
        hub_cache.download_base.assert_called_once()
        shared = hub_cache.snapshot / "scheduler/scheduler_config.json"
        other_link = hub_cache.cache_root / "other-model" / shared.name
        other_link.parent.mkdir()
        other_link.symlink_to(shared)
        managed = {hub_cache.snapshot / name for name in LTX25_BASE_FILES} - {shared}
    else:
        hub_cache.download_checkpoint.assert_called_once()
        hub_cache.download_base.assert_not_called()
        managed = {hub_cache.checkpoint}

    async with restarted_cache_tracker() as (tracker, worker):
        cached = await tracker.list_cached_models(worker.address, MODEL_NAME)
        assert [item["model_file_location"] for item in cached] == [str(variant_cache)]
        assert set(await worker.list_deletable_models(MODEL_NAME)) == {
            str(variant_cache),
            *(str(path) for path in managed),
        }
        assert await worker_module.WorkerActor.confirm_and_remove_model(
            worker, MODEL_NAME
        )
        assert await tracker.list_cached_models(worker.address, MODEL_NAME) == []

    assert not variant_cache.exists()
    assert all(not path.exists() for path in managed)
    assert source.read_bytes() == b"user supplied"
    if local_checkpoint:
        assert other_link.resolve(strict=True) == shared
