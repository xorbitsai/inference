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

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from .. import load_model_family_from_json
from ..core import ImageModelFamilyV2
from ..stable_diffusion.core import DiffusionModel


@pytest.fixture
def joyai_families():
    families = {}
    spec_path = Path(__file__).parents[1] / "model_spec.json"
    load_model_family_from_json(str(spec_path), families)
    return families


def test_joyai_image_model_metadata(joyai_families):
    model_ids = {
        "joyai-image-edit": "JoyAI-Image-Edit-Diffusers",
        "joyai-image-edit-plus": "JoyAI-Image-Edit-Plus-Diffusers",
    }

    assert set(joyai_families) >= set(model_ids)
    for model_name, model_id in model_ids.items():
        specs = {spec.model_hub: spec for spec in joyai_families[model_name]}
        assert specs["huggingface"].model_id == f"jdopensource/{model_id}"
        assert specs["huggingface"].model_revision == "main"
        assert specs["modelscope"].model_id == f"jd-opensource/{model_id}"
        assert specs["modelscope"].model_revision == "master"

        for spec in specs.values():
            assert spec.model_family == "stable_diffusion"
            assert spec.model_ability == ["image2image"]
            assert spec.default_model_config == {"torch_dtype": "bfloat16"}
            assert spec.virtualenv.no_build_isolation is False
            assert (
                'diffusers>=0.40.0 ; #engine# == "diffusers"'
                in spec.virtualenv.packages
            )
            assert (
                'huggingface-hub>=1.23.0,<2.0 ; #engine# == "diffusers"'
                in spec.virtualenv.packages
            )
            assert (
                '#system_torchvision# ; #engine# == "diffusers"'
                in spec.virtualenv.packages
            )

    assert joyai_families["joyai-image-edit"][0].default_generate_config == {
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
    }
    assert joyai_families["joyai-image-edit-plus"][0].default_generate_config == {
        "num_inference_steps": 30,
        "guidance_scale": 4.0,
    }


@pytest.mark.parametrize(
    "model_name",
    ["joyai-image-edit", "joyai-image-edit-plus"],
)
def test_joyai_image_loads_with_diffusion_pipeline(monkeypatch, model_name):
    loaded = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            loaded.update(model_path=model_path, kwargs=kwargs)
            return cls()

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(DiffusionPipeline=FakePipeline),
    )

    model = DiffusionModel(
        "joyai",
        "/models/joyai",
        model_spec=ImageModelFamilyV2(
            model_family="stable_diffusion",
            model_name=model_name,
            model_id="test/joyai",
            model_revision="main",
            model_ability=["image2image"],
        ),
        torch_dtype="bfloat16",
        device_map="balanced",
    )

    model.load()

    assert isinstance(model._model, FakePipeline)
    assert loaded == {
        "model_path": "/models/joyai",
        "kwargs": {
            "torch_dtype": torch.bfloat16,
            "device_map": "balanced",
        },
    }


def test_joyai_image_edit_plus_uses_multi_image_input_and_honors_n(monkeypatch):
    calls = []
    pipeline_seeds = []

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(DiffusionPipeline=object),
    )

    class FakeJoyImageEditPlusPipeline:
        def __call__(
            self,
            images=None,
            prompt=None,
            height=None,
            width=None,
            num_inference_steps=30,
            guidance_scale=4.0,
            negative_prompt=None,
            generator=None,
        ):
            if generator is not None:
                pipeline_seeds.append(generator.initial_seed())
            calls.append(
                {
                    "images": images,
                    "prompt": prompt,
                    "height": height,
                    "width": width,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                }
            )
            return SimpleNamespace(images=[Image.new("RGB", (32, 32))])

    model = DiffusionModel(
        "joyai-plus",
        model_spec=ImageModelFamilyV2(
            model_family="stable_diffusion",
            model_name="joyai-image-edit-plus",
            model_id="test/joyai-plus",
            model_revision="main",
            model_ability=["image2image"],
            default_generate_config={
                "num_inference_steps": 30,
                "guidance_scale": 4.0,
            },
        ),
    )
    model._model = FakeJoyImageEditPlusPipeline()

    primary_image = Image.new("RGB", (64, 32))
    reference_image = Image.new("RGB", (96, 48))
    single_result = model.image_to_image(
        primary_image,
        prompt="Edit one image",
        _return_images=True,
    )
    multi_result = model.image_to_image(
        primary_image,
        prompt="Combine both images",
        reference_images=[reference_image],
        _return_images=True,
    )
    repeated_result = model.image_to_image(
        primary_image,
        prompt="Create two images",
        n=2,
        seed=[11, 22],
        response_format="b64_json",
    )

    assert len(single_result) == 1
    assert len(multi_result) == 1
    assert len(repeated_result["data"]) == 2
    assert pipeline_seeds == [11, 22]
    assert calls == [
        {
            "images": [primary_image],
            "prompt": "Edit one image",
            "height": None,
            "width": None,
            "num_inference_steps": 30,
            "guidance_scale": 4.0,
        },
        {
            "images": [primary_image, reference_image],
            "prompt": "Combine both images",
            "height": None,
            "width": None,
            "num_inference_steps": 30,
            "guidance_scale": 4.0,
        },
        {
            "images": [primary_image],
            "prompt": "Create two images",
            "height": None,
            "width": None,
            "num_inference_steps": 30,
            "guidance_scale": 4.0,
        },
        {
            "images": [primary_image],
            "prompt": "Create two images",
            "height": None,
            "width": None,
            "num_inference_steps": 30,
            "guidance_scale": 4.0,
        },
    ]
