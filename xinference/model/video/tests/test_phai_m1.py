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
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from ....core.exceptions import InvalidModelRequestError
from .. import load_model_family_from_json, ltx2
from ..ltx2 import LTX2VideoModel


@pytest.fixture(scope="module")
def catalog():
    families = {}
    load_model_family_from_json("models", families)
    return families


@pytest.fixture
def ltx2_pipeline_loader(monkeypatch):
    pipeline = Mock(scheduler=SimpleNamespace(config={"use_dynamic_shifting": False}))
    pipeline_loader = Mock(return_value=pipeline)

    def from_scheduler_config(config, **kwargs):
        return SimpleNamespace(config={**config, **kwargs})

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            LTX2ConditionPipeline=SimpleNamespace(from_pretrained=pipeline_loader),
            LTX2VideoTransformer3DModel=SimpleNamespace(
                from_pretrained=Mock(
                    side_effect=AssertionError("Phai-M1 supplies its own transformer")
                )
            ),
            FlowMatchEulerDiscreteScheduler=SimpleNamespace(
                from_config=from_scheduler_config
            ),
        ),
    )
    return pipeline_loader


@pytest.fixture
def ltx2_sampling(monkeypatch):
    # Linux CI's shared Diffusers predates the LTX-2 pipeline.
    sigmas = (1.0, 0.5)
    monkeypatch.setitem(
        sys.modules,
        "diffusers.pipelines.ltx2.utils",
        SimpleNamespace(
            DEFAULT_NEGATIVE_PROMPT="default negative prompt",
            DISTILLED_SIGMA_VALUES=sigmas,
        ),
    )
    return sigmas


@pytest.mark.parametrize("variant", ["dev", "distilled"])
@pytest.mark.parametrize(
    "hub, revision", [("huggingface", "main"), ("modelscope", "master")]
)
def test_phai_m1_catalog(catalog, variant, hub, revision):
    model_name = f"Phai-M1-{variant}"
    checkpoint = f"Phai-M1-22b-{variant}-transformer-bf16.safetensors"
    spec = next(spec for spec in catalog[model_name] if spec.model_hub == hub)

    assert spec.model_id == "PhaiModelLab/Phai-M1"
    assert spec.model_revision == revision
    assert spec.model_family == "Phai-M1"
    assert spec.engine == "diffusers"
    assert set(spec.model_ability) == {"text2video", "image2video"}
    assert spec.default_model_config["model_variant"] == variant
    assert spec.default_model_config["transformer_filename"] == checkpoint


@pytest.mark.parametrize("variant", ["dev", "distilled"])
def test_phai_m1_loads_checkpoint_and_shared_components(
    catalog, monkeypatch, tmp_path, variant, ltx2_pipeline_loader
):
    import torch

    spec = next(
        spec for spec in catalog[f"Phai-M1-{variant}"] if spec.model_hub == "modelscope"
    )
    checkpoint = str(tmp_path / "phai.safetensors")
    base_path = str(tmp_path / "base")
    transformer, connectors = object(), object()
    load_components = Mock(return_value=(base_path, transformer, connectors))
    monkeypatch.setattr(ltx2, "load_phai_m1_components", load_components)
    monkeypatch.setattr(ltx2, "get_available_device", lambda: "cuda")
    model = LTX2VideoModel("phai", checkpoint, spec, base_model_path=base_path)
    model.load()

    load_components.assert_called_once_with(
        checkpoint,
        spec.default_model_config["transformer_filename"],
        base_path,
        torch.bfloat16,
    )
    ltx2_pipeline_loader.assert_called_once_with(
        base_path,
        transformer=transformer,
        connectors=connectors,
        processor=None,
        prompt_enhancer=None,
        dtype=torch.bfloat16,
        local_files_only=True,
        use_safetensors=True,
    )
    pipeline = ltx2_pipeline_loader.return_value
    pipeline.vae.enable_tiling.assert_called_once()
    pipeline.enable_model_cpu_offload.assert_called_once_with(device="cuda")
    assert model._model is pipeline
    assert pipeline.scheduler.config["use_dynamic_shifting"] == (variant == "dev")


@pytest.mark.parametrize("variant", ["dev", "distilled"])
def test_phai_m1_rejects_device_map_before_loading_components(
    catalog, monkeypatch, tmp_path, variant, ltx2_pipeline_loader
):
    load_components = Mock()
    monkeypatch.setattr(ltx2, "load_phai_m1_components", load_components)
    model = LTX2VideoModel(
        "phai", str(tmp_path), catalog[f"Phai-M1-{variant}"][0], device_map="balanced"
    )

    with pytest.raises(ValueError, match="device_map"):
        model.load()

    load_components.assert_not_called()
    ltx2_pipeline_loader.assert_not_called()


def test_phai_m1_distilled_accepts_legacy_sampling_inputs(
    catalog, monkeypatch, tmp_path, ltx2_sampling
):
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
    assert list(kwargs["sigmas"]) == list(ltx2_sampling)
    assert kwargs["guidance_scale"] == kwargs["audio_guidance_scale"] == 1.0
    assert kwargs["stg_scale"] == 0.0
    assert "num_inference_steps" not in kwargs
    assert "timesteps" not in kwargs


def test_phai_m1_rejects_ltx_only_generation_modes(catalog, tmp_path, ltx2_sampling):
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


def test_phai_m1_muxes_audio_and_video(catalog, monkeypatch, tmp_path, ltx2_sampling):
    import torch

    av = pytest.importorskip("av")
    diffusers_utils = pytest.importorskip("diffusers.utils")
    if not hasattr(diffusers_utils, "encode_video"):
        pytest.skip("Diffusers does not provide the audio-video encoder")
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
