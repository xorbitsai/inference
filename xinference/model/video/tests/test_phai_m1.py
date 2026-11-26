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
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from ....core.exceptions import InvalidModelRequestError
from .. import load_model_family_from_json, ltx2, ltx25_checkpoint
from ..ltx2 import LTX2VideoModel


@pytest.fixture(scope="module")
def catalog():
    families = {}
    load_model_family_from_json("models", families)
    return families


@pytest.mark.parametrize("variant", ["dev", "distilled"])
@pytest.mark.parametrize(
    "hub, revision", [("huggingface", "main"), ("modelscope", "master")]
)
def test_phai_m1_catalog(catalog, variant, hub, revision):
    model_name = f"Phai-M1-{variant}"
    checkpoint = f"diffusion_models/Phai-M1-22b-{variant}-transformer-bf16.safetensors"
    spec = next(spec for spec in catalog[model_name] if spec.model_hub == hub)

    assert spec.model_id == "PhaiModelLab/Phai-M1"
    assert spec.model_revision == revision
    assert spec.model_family == "Phai-M1"
    assert spec.engine == "diffusers"
    assert set(spec.model_ability) == {"text2video", "image2video"}
    assert spec.default_model_config["model_variant"] == variant
    assert spec.cache_config["allow_patterns"] == [checkpoint]


@pytest.mark.parametrize(
    "hub, revision, keyword",
    [
        ("huggingface", "a6de4b5354f078db24d9cf4778c14846788aea3d", "allow_patterns"),
        ("modelscope", "master", "allow_file_pattern"),
    ],
)
def test_phai_m1_base_snapshot_excludes_unused_weights(
    monkeypatch, hub, revision, keyword
):
    import huggingface_hub

    download = Mock(return_value="/cache/ltx25")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setitem(
        sys.modules,
        "modelscope.hub.snapshot_download",
        SimpleNamespace(snapshot_download=download),
    )

    assert ltx25_checkpoint._download_ltx25_base(hub) == "/cache/ltx25"
    args, kwargs = download.call_args
    assert args == ("Lightricks/LTX-2.5-Diffusers",)
    assert kwargs["revision"] == revision
    patterns = kwargs[keyword]
    for filename in (
        "model_index.json",
        "transformer/config.json",
        "connectors/config.json",
        "connectors/diffusion_pytorch_model-00001-of-00002.safetensors",
        "connectors/diffusion_pytorch_model-00002-of-00002.safetensors",
        "connectors/diffusion_pytorch_model.safetensors.index.json",
        "text_encoder/model-00001-of-00005.safetensors",
        "audio_vae/diffusion_pytorch_model.safetensors",
        "vocoder/diffusion_pytorch_model.safetensors",
    ):
        assert any(fnmatch.fnmatchcase(filename, pattern) for pattern in patterns)
    for filename in (
        "connectors/diffusion_pytorch_model.safetensors",
        "transformer/diffusion_pytorch_model-00001-of-00004.safetensors",
        "transformer_full/diffusion_pytorch_model-00001-of-00004.safetensors",
        "prompt_enhancer/model-00001-of-00003.safetensors",
    ):
        assert not any(fnmatch.fnmatchcase(filename, pattern) for pattern in patterns)


@pytest.mark.parametrize("variant", ["dev", "distilled"])
def test_phai_m1_loads_checkpoint_and_shared_components(
    catalog, monkeypatch, tmp_path, variant
):
    import diffusers
    import torch

    spec = next(
        spec for spec in catalog[f"Phai-M1-{variant}"] if spec.model_hub == "modelscope"
    )
    checkpoint = tmp_path / spec.default_model_config["transformer_filename"]
    checkpoint.parent.mkdir()
    checkpoint.touch()
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "model_index.json").touch()
    base_path = str(base_dir)
    transformer, connectors = object(), object()
    load_components = Mock(return_value=(transformer, connectors))
    download_base = Mock(return_value=base_path)
    pipeline = Mock(scheduler=diffusers.FlowMatchEulerDiscreteScheduler())
    pipeline_loader = Mock(return_value=pipeline)
    monkeypatch.setattr(
        ltx25_checkpoint, "load_ltx25_transformer_and_connectors", load_components
    )
    monkeypatch.setattr(ltx25_checkpoint, "_download_ltx25_base", download_base)
    monkeypatch.setattr(ltx2, "get_available_device", lambda: "cuda")
    monkeypatch.setattr(
        diffusers.LTX2ConditionPipeline, "from_pretrained", pipeline_loader
    )
    if variant == "distilled":
        model = LTX2VideoModel("phai", str(checkpoint), spec, base_model_path=base_path)
    else:
        model = LTX2VideoModel("phai", str(tmp_path), spec)
    model.load()

    load_components.assert_called_once_with(str(checkpoint), base_path, torch.bfloat16)
    pipeline_loader.assert_called_once_with(
        base_path,
        transformer=transformer,
        connectors=connectors,
        processor=None,
        prompt_enhancer=None,
        dtype=torch.bfloat16,
        local_files_only=True,
        use_safetensors=True,
    )
    pipeline.vae.enable_tiling.assert_called_once()
    pipeline.enable_model_cpu_offload.assert_called_once_with(device="cuda")
    assert model._model is pipeline
    if variant == "distilled":
        download_base.assert_not_called()
    else:
        download_base.assert_called_once_with("modelscope")
    assert pipeline.scheduler.config.use_dynamic_shifting == (variant == "dev")


def test_phai_m1_distilled_accepts_legacy_sampling_inputs(
    catalog, monkeypatch, tmp_path
):
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

    model = LTX2VideoModel("phai", str(tmp_path), catalog["Phai-M1-distilled"][0])
    pipeline = Mock()
    model._model = pipeline
    monkeypatch.setattr(model, "_output_to_audio_video", Mock())

    model.text_to_video(
        "A red fox in the snow",
        num_frames=8,
        num_inference_steps=25,
        guidance_scale=7.5,
        audio_guidance_scale=7.5,
        stg_scale=2.0,
        sigmas=[0.8, 0.2],
        timesteps=[80, 20],
    )

    kwargs = pipeline.call_args.kwargs
    assert kwargs["num_frames"] == 9
    assert list(kwargs["sigmas"]) == list(DISTILLED_SIGMA_VALUES)
    assert kwargs["guidance_scale"] == kwargs["audio_guidance_scale"] == 1.0
    assert kwargs["stg_scale"] == 0.0
    assert "num_inference_steps" not in kwargs
    assert "timesteps" not in kwargs


def test_phai_m1_rejects_ltx_only_generation_modes(catalog, tmp_path):
    from PIL import Image

    model = LTX2VideoModel("phai", str(tmp_path), catalog["Phai-M1-distilled"][0])
    pipeline = Mock()
    model._model = pipeline

    with pytest.raises(InvalidModelRequestError, match="single-stage"):
        model.text_to_video("A still frame", two_stage=True)
    image = Image.new("RGB", (32, 32))
    with pytest.raises(InvalidModelRequestError, match="first/last-frame"):
        model.firstlastframe_to_video(image, image, "A still frame")

    pipeline.assert_not_called()


def test_phai_m1_muxes_audio_and_video(catalog, monkeypatch, tmp_path):
    import torch

    av = pytest.importorskip("av")
    fps, frames, sample_rate = 30, 9, 48000
    pipeline = Mock(
        return_value=SimpleNamespace(
            frames=np.zeros((1, frames, 32, 32, 3), dtype=np.float32),
            audio=torch.zeros(1, 2, sample_rate * frames // fps),
        )
    )
    pipeline.vocoder.config.output_sampling_rate = sample_rate
    monkeypatch.setattr(ltx2, "XINFERENCE_VIDEO_DIR", str(tmp_path))
    model = LTX2VideoModel("phai", str(tmp_path), catalog["Phai-M1-distilled"][0])
    model._model = pipeline

    result = model.text_to_video(
        "A still frame",
        fps=fps,
        width=32,
        height=32,
        num_frames=frames,
        response_format="url",
    )

    with av.open(result["data"][0]["url"]) as video:
        assert video.streams.video[0].average_rate == fps
        assert video.streams.video[0].frames == frames
        assert video.streams.audio[0].sample_rate == sample_rate
        assert video.streams.audio[0].channels == 2
