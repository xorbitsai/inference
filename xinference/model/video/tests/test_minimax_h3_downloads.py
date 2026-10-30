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

import fnmatch
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ... import utils as model_utils
from ...cache_manager import CacheManager
from .. import load_model_family_from_json
from ..cache_manager import VideoCacheManager

# Representative files from the pinned H3 repository, including both layouts
# and nested native VAE code. No model weights are downloaded by these tests.
DIFFUSERS_FILES = {
    "model_index.json",
    "modular_model_index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00014.safetensors",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00014.safetensors",
    "tokenizer/tokenizer.json",
    "processor/preprocessor_config.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors.index.json",
    "vae/diffusion_pytorch_model-00001-of-00003.safetensors",
    "audio_vae/config.json",
    "audio_vae/diffusion_pytorch_model.safetensors",
    "scheduler/scheduler_config.json",
    "audio_scheduler/scheduler_config.json",
}
NATIVE_FILES = {
    "FL2VA/model_index.json",
    "FL2VA/transformer/model-00001-of-00013.safetensors",
    "FL2VA/text_encoder/model-00001-of-00014.safetensors",
    "FL2VA/tokenizer/tokenizer.json",
    "FL2VA/processor/preprocessor_config.json",
    "FL2VA/video_vae/source/model.safetensors",
    "FL2VA/video_vae/source/minimax_h3_video_vae.py",
    "FL2VA/audio_vae/model.safetensors",
}
UNUSED_FILES = {
    "Ref2VA/transformer/model-00001-of-00013.safetensors",
    "Ref2VA/text_encoder/model-00001-of-00014.safetensors",
    "transformer_ref/diffusion_pytorch_model-00001-of-00014.safetensors",
    "assets/demo.mp4",
}
REPO_FILES = DIFFUSERS_FILES | NATIVE_FILES | UNUSED_FILES | {"LICENSE", "README.md"}


@pytest.fixture
def catalog():
    families = {}
    load_model_family_from_json("models", families)
    return families["MiniMax-H3"]


@pytest.fixture
def downloads(monkeypatch, tmp_path):
    from .... import constants

    monkeypatch.setattr(constants, "XINFERENCE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(constants, "XINFERENCE_MODEL_DIR", str(tmp_path / "custom"))
    monkeypatch.setattr(CacheManager, "is_initialized", False)
    calls = []
    snapshot = tmp_path / "snapshot"

    def download(model_id, *, revision, allow_patterns, local_dir=None, **kwargs):
        calls.append((model_id, revision, allow_patterns, kwargs))
        destination = Path(local_dir) if local_dir else snapshot
        for name in REPO_FILES:
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in allow_patterns):
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"checkpoint")
        return str(destination)

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    monkeypatch.setitem(
        sys.modules,
        "modelscope.hub.snapshot_download",
        SimpleNamespace(snapshot_download=download),
    )
    monkeypatch.setattr(
        model_utils,
        "create_symlink",
        lambda source, target: Path(target).symlink_to(
            source, target_is_directory=True
        ),
    )
    return calls


@pytest.mark.parametrize(
    "hub,new_hf", [("huggingface", True), ("huggingface", False), ("modelscope", True)]
)
@pytest.mark.parametrize("engine", ["diffusers", "vLLM", "SGLang"])
def test_h3_download_excludes_unused_layouts(
    monkeypatch, catalog, downloads, hub, new_hf, engine
):
    monkeypatch.setattr(model_utils, "IS_NEW_HUGGINGFACE_HUB", new_hf)
    spec = next(s for s in catalog if s.engine == engine and s.model_hub == hub)
    path = Path(VideoCacheManager(spec).cache())
    files = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
    needed = (
        DIFFUSERS_FILES
        if engine == "diffusers"
        else NATIVE_FILES | {"model_index.json"}
    )
    assert files == needed | {"LICENSE", "README.md"}
    assert not files & UNUSED_FILES
    assert downloads[0][2] == spec.cache_config["allow_patterns"]
    if hub == "modelscope":
        assert downloads[0][3]["ignore_file_pattern"] == [".gitkeep"]


@pytest.mark.parametrize("hub", ["huggingface", "modelscope"])
def test_h3_native_layout_is_cached_after_diffusers(
    catalog, downloads, monkeypatch, hub
):
    monkeypatch.setattr(model_utils, "IS_NEW_HUGGINGFACE_HUB", True)
    specs = {s.engine: s for s in catalog if s.model_hub == hub}
    diffusers_path = VideoCacheManager(specs["diffusers"]).cache()
    native_path = VideoCacheManager(specs["SGLang"]).cache()
    assert native_path != diffusers_path
    assert len(downloads) == 2  # Existing Diffusers cache must not skip native files.
    assert (Path(native_path) / "FL2VA/video_vae/source/model.safetensors").is_file()
    assert VideoCacheManager(specs["vLLM"]).cache() == native_path
    assert len(downloads) == 2  # Both native engines can share the same checkpoint.
