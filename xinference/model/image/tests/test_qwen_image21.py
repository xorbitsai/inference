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

import asyncio
import base64
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from .. import load_model_family_from_json
from ..stable_diffusion.core import DiffusionModel


@pytest.fixture
def families():
    result = {}
    load_model_family_from_json(
        str(Path(__file__).parents[1] / "models" / "Qwen-Image-2.1.json"), result
    )
    return result["Qwen-Image-2.1"]


def test_metadata(families):
    specs = {spec.model_hub: spec for spec in families}
    assert set(specs) == {"huggingface", "modelscope"}
    for hub, revision in (("huggingface", "main"), ("modelscope", "master")):
        spec = specs[hub]
        assert spec.model_id == "Qwen/Qwen-Image-2.1"
        assert spec.model_revision == revision
        assert spec.model_ability == ["text2image", "image2image"]
        assert spec.default_model_config == {"torch_dtype": "bfloat16"}
        assert spec.default_generate_config == {"num_inference_steps": 40}
        assert "transformers>=5.17.0,<6" in spec.virtualenv.packages


@pytest.fixture
def loaded_model(monkeypatch, families):
    calls = []

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert path == "/models/qwen-image-2.1"
            assert kwargs == {
                "torch_dtype": torch.bfloat16,
                "device_map": "balanced",
            }
            return cls()

        def __call__(
            self,
            prompt=None,
            image=None,
            width=None,
            height=None,
            num_inference_steps=40,
            num_images_per_prompt=1,
            generator=None,
        ):
            calls.append(
                dict(
                    prompt=prompt,
                    image=image,
                    width=width,
                    height=height,
                    steps=num_inference_steps,
                    n=num_images_per_prompt,
                    seed=generator.initial_seed() if generator is not None else None,
                )
            )
            return SimpleNamespace(
                images=[Image.new("RGBA", (32, 32), (1, 2, 3, 42))]
                * num_images_per_prompt
            )

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            QwenImage21Pipeline=FakePipeline,
            DiffusionPipeline=FakePipeline,
            AutoPipelineForImage2Image=SimpleNamespace(
                from_pipe=lambda *args, **kwargs: pytest.fail("must reuse pipeline")
            ),
        ),
    )
    model = DiffusionModel(
        "qwen21",
        "/models/qwen-image-2.1",
        model_spec=families[0],
        torch_dtype="bfloat16",
        device_map="balanced",
    )
    model.load()
    return model, calls


def test_text_to_image(loaded_model):
    model, calls = loaded_model
    images = asyncio.run(
        model.text_to_image(
            "A transparent sticker",
            n=2,
            size="2048*2048",
            seed=42,
            _return_images=True,
        )
    )
    assert len(images) == 2
    assert images[0].mode == "RGBA"
    assert calls == [
        dict(
            prompt="A transparent sticker",
            image=None,
            width=2048,
            height=2048,
            steps=40,
            n=2,
            seed=42,
        )
    ]


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("size", [None, "2400*1792"])
def test_edit_preserves_references_and_alpha(loaded_model, multiple, size):
    model, calls = loaded_model
    primary = Image.new("RGBA", (64, 48), (1, 2, 3, 42))
    reference = Image.new("RGB", (32, 48))
    kwargs = {"reference_images": [reference]} if multiple else {}
    result = model.image_to_image(
        primary,
        prompt="Edit",
        size=size,
        num_inference_steps=12,
        response_format="b64_json",
        **kwargs,
    )
    assert model._ability_to_models["image2image", None] is model._model
    assert calls == [
        dict(
            prompt="Edit",
            image=[primary, reference] if multiple else primary,
            width=2400 if size else None,
            height=1792 if size else None,
            steps=12,
            n=1,
            seed=None,
        )
    ]
    output = Image.open(io.BytesIO(base64.b64decode(result["data"][0]["b64_json"])))
    assert output.mode == "RGBA"
    assert output.getpixel((0, 0)) == (1, 2, 3, 42)
