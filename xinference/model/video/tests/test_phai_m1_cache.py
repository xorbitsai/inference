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

import asyncio
import inspect
import sys
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from .... import constants
from ....core import worker as worker_module
from ....core.worker import WorkerActor
from ...cache_manager import CacheManager
from .. import BUILTIN_VIDEO_MODELS, load_model_family_from_json, ltx2, ltx25_checkpoint
from ..cache_manager import VideoCacheManager
from ..core import create_video_model_instance
from ..engine_family import VIDEO_ENGINES

HUBS = ("huggingface", "modelscope")
REVISIONS = {"huggingface": "main", "modelscope": "master"}
VARIANTS = ("dev", "distilled")


def _checkpoint_name(variant: str) -> str:
    return f"Phai-M1-22b-{variant}-transformer-bf16.safetensors"


def _remote_checkpoint(hub: str, variant: str) -> str:
    name = _checkpoint_name(variant)
    return name if hub == "huggingface" else f"diffusion_models/{name}"


def _write_fixture(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(path.name.encode())
    return path


@pytest.fixture(scope="module")
def catalog():
    families = {}
    load_model_family_from_json("models", families)
    return {
        (spec.model_hub, variant): spec
        for variant in VARIANTS
        for spec in families[f"Phai-M1-{variant}"]
    }


@pytest.fixture
def registered_catalog(catalog, monkeypatch):
    for variant in VARIANTS:
        specs = [catalog[hub, variant] for hub in HUBS]
        model_name = specs[0].model_name
        monkeypatch.setitem(BUILTIN_VIDEO_MODELS, model_name, specs)
        monkeypatch.setitem(
            VIDEO_ENGINES,
            model_name,
            {
                "diffusers": [
                    {
                        "model_name": model_name,
                        "model_format": "diffusers",
                        "quantization": "none",
                        "video_class": ltx2.LTX2VideoModel,
                    }
                ]
            },
        )
    return catalog


@pytest.fixture
def hub_cache(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    hf_root = tmp_path / "hf"
    ms_root = tmp_path / "ms"
    monkeypatch.setattr(constants, "XINFERENCE_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(constants, "XINFERENCE_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(worker_module, "XINFERENCE_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hf_root))
    monkeypatch.setenv("MODELSCOPE_CACHE", str(ms_root))
    monkeypatch.setattr(CacheManager, "is_initialized", False)

    phai_snapshots = {
        "huggingface": hf_root
        / "models--PhaiModelLab--Phai-M1"
        / "snapshots"
        / "commit",
        "modelscope": ms_root / "models" / "PhaiModelLab--Phai-M1",
    }
    base_snapshots = {
        "huggingface": hf_root
        / "models--Lightricks--LTX-2.5-Diffusers"
        / "snapshots"
        / "base",
        "modelscope": ms_root / "models" / "Lightricks--LTX-2.5-Diffusers",
    }
    state = SimpleNamespace(
        checkpoint_calls=[],
        base_calls=[],
        phai_snapshots=phai_snapshots,
        base_snapshots=base_snapshots,
        offline=False,
    )

    def download_checkpoint(hub, model_id, filename, *, revision, **kwargs):
        assert not state.offline, "Unexpected checkpoint download on cache hit"
        assert model_id == "PhaiModelLab/Phai-M1"
        assert revision == REVISIONS[hub]
        assert filename in {_remote_checkpoint(hub, variant) for variant in VARIANTS}
        state.checkpoint_calls.append((hub, filename))
        return str(_write_fixture(phai_snapshots[hub] / filename))

    def download_base(
        hub,
        model_id,
        *,
        revision,
        allow_patterns=None,
        allow_file_pattern=None,
        **kwargs,
    ):
        assert not state.offline, "Unexpected base component download on cache hit"
        assert (model_id, revision) == ltx25_checkpoint.LTX25_MODEL_SOURCES[hub]
        assert set(allow_patterns or allow_file_pattern) == set(
            ltx25_checkpoint.LTX25_BASE_FILES
        )
        state.base_calls.append(hub)
        snapshot = base_snapshots[hub]
        for filename in (*ltx25_checkpoint.LTX25_BASE_FILES, "unused.safetensors"):
            _write_fixture(snapshot / filename)
        return str(snapshot)

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", partial(download_checkpoint, "huggingface")
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", partial(download_base, "huggingface")
    )
    monkeypatch.setitem(
        sys.modules,
        "modelscope.hub.file_download",
        SimpleNamespace(model_file_download=partial(download_checkpoint, "modelscope")),
    )
    monkeypatch.setitem(
        sys.modules,
        "modelscope.hub.snapshot_download",
        SimpleNamespace(snapshot_download=partial(download_base, "modelscope")),
    )
    return state


@pytest.mark.parametrize("hub", HUBS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_phai_m1_cache_selects_only_required_files(catalog, hub_cache, hub, variant):
    manager = VideoCacheManager(catalog[hub, variant])
    assert not manager.get_cache_status()

    checkpoint, base = map(Path, manager.cache_phai_m1())

    assert manager.get_cache_status()
    assert checkpoint == Path(manager.get_cache_dir()) / _checkpoint_name(variant)
    assert checkpoint.resolve() == hub_cache.phai_snapshots[hub] / _remote_checkpoint(
        hub, variant
    )
    assert set(
        str(path.relative_to(base)) for path in base.rglob("*") if path.is_file()
    ) == set(ltx25_checkpoint.LTX25_BASE_FILES)
    assert hub_cache.checkpoint_calls == [(hub, _remote_checkpoint(hub, variant))]
    assert hub_cache.base_calls == [hub]

    scheduler_config = base / "scheduler" / "scheduler_config.json"
    scheduler_config.unlink()
    assert not manager.get_cache_status()
    assert tuple(map(Path, manager.cache_phai_m1())) == (checkpoint, base)
    assert scheduler_config.is_file()
    assert manager.get_cache_status()
    assert hub_cache.checkpoint_calls == [(hub, _remote_checkpoint(hub, variant))]
    assert hub_cache.base_calls == [hub, hub]


@pytest.mark.parametrize(
    "hub, prefix", [("huggingface", "diffusion_models"), ("modelscope", "")]
)
def test_phai_m1_reuses_existing_checkpoint_with_other_layout(
    catalog, hub_cache, hub, prefix
):
    manager = VideoCacheManager(catalog[hub, "dev"])
    legacy_checkpoint = _write_fixture(
        Path(manager.get_cache_dir()) / prefix / _checkpoint_name("dev")
    )
    other_weight = _write_fixture(
        hub_cache.phai_snapshots[hub] / _remote_checkpoint(hub, "distilled")
    )
    stale_sibling = legacy_checkpoint.with_name(other_weight.name)
    stale_sibling.symlink_to(other_weight)
    assert not manager.get_cache_status()  # The old checkpoint alone is incomplete.

    checkpoint, base = manager.cache_phai_m1()

    assert Path(checkpoint).resolve() == legacy_checkpoint
    assert Path(base, "model_index.json").is_file()
    assert manager.get_cache_status()
    assert hub_cache.checkpoint_calls == []
    assert hub_cache.base_calls == [hub]
    assert not stale_sibling.is_symlink()
    assert other_weight.is_file()


def test_phai_m1_local_paths_do_not_download(catalog, hub_cache, tmp_path):
    manager = VideoCacheManager(catalog["huggingface", "dev"])
    checkpoint = _write_fixture(tmp_path / "custom" / "phai.safetensors")
    base = tmp_path / "custom" / "base"
    _write_fixture(base / "model_index.json")
    hub_cache.offline = True

    assert manager.cache_phai_m1(str(checkpoint), str(base)) == (
        str(checkpoint),
        str(base),
    )
    assert not manager.get_cache_status()
    assert not Path(manager.get_cache_dir()).exists()


@pytest.mark.parametrize("variant", VARIANTS)
def test_phai_m1_factory_rejects_device_map_before_downloading(
    registered_catalog, hub_cache, variant
):
    with pytest.raises(ValueError, match="device_map"):
        create_video_model_instance(
            "phai",
            registered_catalog["huggingface", variant].model_name,
            download_hub="huggingface",
            model_engine="diffusers",
            enable_virtual_env=False,
            device_map="balanced",
        )

    assert hub_cache.checkpoint_calls == []
    assert hub_cache.base_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("hub", HUBS)
async def test_worker_phai_m1_pre_download_supports_offline_launch(
    registered_catalog, hub_cache, monkeypatch, hub
):
    model_name = registered_catalog[hub, "distilled"].model_name
    monkeypatch.setattr(worker_module, "XINFERENCE_LOG_CONSOLE", True)
    worker = SimpleNamespace(
        address="127.0.0.1:0",
        _check_model_is_valid=Mock(),
        _download_artifact_cleanup_lock=asyncio.Lock(),
        _cache_uid_to_download_info={},
        _launch_semaphore=asyncio.Semaphore(1),
        _get_progressor=AsyncMock(return_value=nullcontext()),
        _upload_download_progress=Mock(),
        update_cache_status=AsyncMock(),
    )
    worker._download_model_files = partial(WorkerActor._download_model_files, worker)

    transformer, connectors = Mock(), Mock()
    converter = Mock(return_value=(transformer, connectors))
    monkeypatch.setattr(
        ltx25_checkpoint, "load_ltx25_transformer_and_connectors", converter
    )
    cache_builtin_model = inspect.unwrap(WorkerActor.cache_builtin_model)
    result = await cache_builtin_model(
        worker,
        cache_uid="phai-cache",
        model_type="video",
        model_name=model_name,
        model_engine="diffusers",
        model_format="diffusers",
        model_size_in_billions=None,
        quantization=None,
        download_hub=hub,
        enable_virtual_env=False,
    )
    assert result["version_info"]["cache_status"] is True
    worker.update_cache_status.assert_awaited_once_with(
        model_name, result["version_info"]
    )
    converter.assert_not_called()
    assert hub_cache.checkpoint_calls == [(hub, _remote_checkpoint(hub, "distilled"))]
    assert hub_cache.base_calls == [hub]

    hub_cache.offline = True
    pipeline_loader = Mock(return_value=Mock())
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            LTX2ConditionPipeline=SimpleNamespace(from_pretrained=pipeline_loader),
            LTX2VideoTransformer3DModel=Mock(),
            FlowMatchEulerDiscreteScheduler=Mock(),
        ),
    )
    monkeypatch.setattr(ltx2, "get_available_device", lambda: "cpu")
    model = create_video_model_instance(
        "phai",
        model_name,
        download_hub=hub,
        model_engine="diffusers",
        enable_virtual_env=False,
    )
    model.load()

    (checkpoint, base, _), _ = converter.call_args
    assert Path(checkpoint).is_file()
    assert all(Path(base, name).is_file() for name in ltx25_checkpoint.LTX25_BASE_FILES)
    assert pipeline_loader.call_args.args == (base,)
    kwargs = pipeline_loader.call_args.kwargs
    assert kwargs["local_files_only"] is True
    assert kwargs["transformer"] is transformer
    assert kwargs["connectors"] is connectors


@pytest.mark.asyncio
@pytest.mark.parametrize("hub", HUBS)
@pytest.mark.parametrize("first", VARIANTS)
@pytest.mark.parametrize("removed", VARIANTS)
async def test_worker_phai_m1_deletes_only_selected_variant(
    catalog, hub_cache, hub, first, removed
):
    second = "distilled" if first == "dev" else "dev"
    managers = {
        variant: VideoCacheManager(catalog[hub, variant]) for variant in (first, second)
    }
    cached_paths = {
        variant: tuple(map(Path, managers[variant].cache_phai_m1()))
        for variant in managers
    }
    survivor = "distilled" if removed == "dev" else "dev"
    checkpoint, base = cached_paths[removed]
    other_checkpoint, other_base = cached_paths[survivor]
    assert base == other_base
    assert hub_cache.base_calls == [hub]
    source = checkpoint.resolve()
    sibling_source = other_checkpoint.resolve()
    assert source != sibling_source

    cache_dir = managers[removed].get_cache_dir()
    tracker = SimpleNamespace(
        list_deletable_models=AsyncMock(return_value=cache_dir),
        confirm_and_remove_model=AsyncMock(),
    )
    worker = SimpleNamespace(address="127.0.0.1:0", _cache_tracker_ref=tracker)
    worker.list_deletable_models = partial(WorkerActor.list_deletable_models, worker)
    model_version = f"Phai-M1-{removed}"

    assert set(await worker.list_deletable_models(model_version)) == {
        cache_dir,
        str(source),
    }
    assert await WorkerActor.confirm_and_remove_model(worker, model_version)
    tracker.confirm_and_remove_model.assert_awaited_once_with(
        model_version, worker.address
    )
    assert not source.exists()
    assert not Path(cache_dir).exists()
    assert not managers[removed].get_cache_status()
    assert sibling_source.is_file()
    assert other_checkpoint.is_file()
    assert managers[survivor].get_cache_status()
    assert all((base / name).is_file() for name in ltx25_checkpoint.LTX25_BASE_FILES)
