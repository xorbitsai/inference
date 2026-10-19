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

import base64
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from .. import load_model_family_from_json
from ..core import create_image_model_instance
from ..engine import DiffusersImageModel, Flux3ImageModel
from ..flux3 import Flux3Model


@pytest.fixture(params=["huggingface", "modelscope"])
def spec(request):
    families = {}
    load_model_family_from_json(
        str(Path(__file__).parents[1] / "models" / "flux-3-image-klein.json"),
        families,
    )
    return next(
        item
        for item in families["flux-3-image-klein"]
        if item.model_hub == request.param
    )


def test_catalog_and_dispatch(spec, tmp_path):
    from ..engine_family import IMAGE_ENGINES

    assert spec.model_id == "black-forest-labs/flux-3-image-klein"
    assert spec.model_revision == (
        "main" if spec.model_hub == "huggingface" else "master"
    )
    assert spec.model_ability == ["text2image", "image2image"]
    assert spec.default_generate_config == {"num_inference_steps": 16}
    assert Flux3ImageModel.match(spec)
    assert not DiffusersImageModel.match(spec)
    assert set(IMAGE_ENGINES[spec.model_name]) == {"diffusers"}
    assert "transformers>=5.8.1,<6" in spec.virtualenv.packages
    model = create_image_model_instance(
        "custom-flux3-uid",
        spec.model_name,
        model_path=str(tmp_path),
        download_hub=spec.model_hub,
        enable_virtual_env=False,
    )
    assert isinstance(model, Flux3ImageModel)
    assert model.model_family.model_hub == spec.model_hub
    assert model.model_family.model_engine == "diffusers"


@pytest.fixture
def loaded_model(monkeypatch, tmp_path, spec):
    from ...cache_manager import CacheManager

    calls = SimpleNamespace(
        downloads=[], loads=[], generated=[], offloads=[], devices=[]
    )
    model_path = tmp_path / "model"
    model_path.mkdir()
    pup = "black-forest-labs/flux-3-image-klein-pup"

    class Pipeline:
        pretrained_component_names = [
            "transformer",
            "vae",
            "prompt_upsampler",
            "prompt_upsampler_processor",
        ]
        blocks = SimpleNamespace(
            input_names=[
                "prompt",
                "image",
                "width",
                "height",
                "num_inference_steps",
                "num_images_per_prompt",
                "generator",
                "boxes",
                "max_new_tokens",
            ]
        )

        @classmethod
        def from_pretrained(cls, path, components_manager):
            assert path == str(model_path)
            return cls()

        def get_component_spec(self, name):
            # Local components must not trigger a second snapshot download.
            repo = str(model_path) if name in ("transformer", "vae") else pup
            return SimpleNamespace(pretrained_model_name_or_path=repo, revision=None)

        def load_components(self, **kwargs):
            calls.loads.append(kwargs)
            for name in self.pretrained_component_names:
                setattr(self, name, object())

        def to(self, device):
            calls.devices.append(device)

        def __call__(self, *, output, **kwargs):
            assert output == "images"
            calls.generated.append(kwargs)
            return [Image.new("RGB", (32, 48))] * kwargs["num_images_per_prompt"]

    class Manager:
        def enable_auto_cpu_offload(self, **kwargs):
            calls.offloads.append(kwargs)

    def cache(manager):
        calls.downloads.append(manager._model_family)
        return str(tmp_path / "pup")

    monkeypatch.setattr(CacheManager, "cache", cache)
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            ComponentsManager=Manager,
            Flux3DistilledModularPipeline=Pipeline,
        ),
    )
    model = Flux3Model("flux3", str(model_path), spec, device="cpu")
    model.load()
    return model, calls


def test_load_keeps_auxiliary_components_on_selected_hub(loaded_model):
    model, calls = loaded_model
    assert len(calls.downloads) == 1
    for download in calls.downloads:
        assert download.model_id == "black-forest-labs/flux-3-image-klein-pup"
        assert download.model_hub == model.model_family.model_hub
        assert download.model_revision == model.model_family.model_revision
    load = calls.loads[0]
    assert load["dtype"] == {"vae": torch.float32, "default": torch.bfloat16}
    assert load["local_files_only"] is True
    assert load["revision"] is None
    assert set(load["pretrained_model_name_or_path"]) == set(
        model._model.pretrained_component_names
    )
    assert calls.devices == ["cpu"]
    assert not calls.offloads


def test_cpu_offload(loaded_model):
    model, calls = loaded_model
    model._kwargs["cpu_offload"] = True
    model.load()
    assert calls.offloads == [{"device": "cpu"}]
    assert calls.devices == ["cpu"]


def test_component_failure_surfaces_at_load(monkeypatch, loaded_model):
    model, _ = loaded_model
    pipeline_class = type(model._model)
    monkeypatch.setattr(pipeline_class, "load_components", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="Failed to load FLUX 3 components"):
        model.load()


def test_text_to_image_parameters_and_output(loaded_model):
    model, calls = loaded_model
    progress = []
    boxes = [{"id": "fox", "desc": "a red fox", "bbox": [0.1, 0.2, 0.8, 0.9]}]
    result = model.text_to_image(
        "a fox",
        n=2,
        size="768x512",
        seed=[7, 11],
        boxes=boxes,
        progressor=SimpleNamespace(set_progress=progress.append),
        guidance_scale=3.5,
        negative_prompt="blurry",
        request_id="request-id",
        response_format="b64_json",
    )
    call = calls.generated[0]
    assert (call["width"], call["height"]) == (768, 512)
    assert call["num_inference_steps"] == 16
    assert call["boxes"] is boxes
    assert [generator.initial_seed() for generator in call["generator"]] == [7, 11]
    assert not set(call) & {"guidance_scale", "negative_prompt", "request_id"}
    assert len(result["data"]) == 2
    assert Image.open(
        BytesIO(base64.b64decode(result["data"][0]["b64_json"]))
    ).size == (32, 48)
    assert progress == [0.0, 1.0]


@pytest.mark.parametrize("size", [None, "512*768"])
def test_edit_merges_reference_images(loaded_model, size):
    model, calls = loaded_model
    primary = Image.new("RGB", (64, 48))
    reference = Image.new("RGB", (32, 32))
    images = model.image_to_image(
        primary,
        prompt="combine",
        reference_images=[reference],
        n=2,
        size=size,
        seed=42,
        num_inference_steps=8,
        max_new_tokens=128,
        _return_images=True,
    )
    call = calls.generated[0]
    assert call["image"] == [primary, reference]
    assert call["num_images_per_prompt"] == 2
    assert call["num_inference_steps"] == 8
    assert call["max_new_tokens"] == 128
    assert call["generator"].initial_seed() == 42
    if size:
        assert (call["width"], call["height"]) == (512, 768)
    else:
        assert "width" not in call and "height" not in call
    assert len(images) == 2


@pytest.mark.parametrize("count", [0, 11])
def test_reference_limits(loaded_model, count):
    model, calls = loaded_model
    with pytest.raises(ValueError, match="1 to 10"):
        model.image_to_image([Image.new("RGB", (16, 16))] * count, prompt="edit")
    assert not calls.generated


def test_url_output_and_ten_references(monkeypatch, loaded_model, tmp_path):
    from .. import utils

    monkeypatch.setattr(utils, "XINFERENCE_IMAGE_DIR", str(tmp_path))
    model, _ = loaded_model
    result = model.image_to_image(
        [Image.new("RGB", (16, 16))] * 10, prompt="combine", response_format="url"
    )
    image_path = Path(result["data"][0]["url"])
    assert image_path.is_file()
    assert Image.open(image_path).size == (32, 48)


def test_existing_flux_models_keep_generic_engine():
    from .. import BUILTIN_IMAGE_MODELS

    for name in ("FLUX.1-dev", "FLUX.2-klein-4B"):
        family = BUILTIN_IMAGE_MODELS[name][0]
        assert DiffusersImageModel.match(family)
        assert not Flux3ImageModel.match(family)


@pytest.mark.parametrize(
    "options",
    [
        {"n": 0},
        {"n": True},
        {"prompt": ["fox"]},
        {"seed": -2},
        {"seed": True},
        {"size": "bad"},
        {"size": "0*1024"},
    ],
)
def test_invalid_requests(loaded_model, options):
    model, calls = loaded_model
    with pytest.raises(ValueError):
        model.text_to_image(**{"prompt": "fox", **options})
    assert not calls.generated


def test_old_diffusers_has_actionable_error(monkeypatch, spec):
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace())
    model = Flux3Model("flux3", "/unused", spec)
    with pytest.raises(
        ImportError, match="git\\+https://github.com/huggingface/diffusers"
    ):
        model.load()
