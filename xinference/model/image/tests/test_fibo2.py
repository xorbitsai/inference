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
import base64
import io
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from .. import load_model_family_from_json
from ..engine import DiffusersImageModel, Fibo2DiffusersImageModel
from ..engine_family import generate_engine_config_by_model_name
from ..fibo2 import Fibo2ImageModel


@pytest.fixture(params=["Fibo-2.0-dev", "Fibo-2.0-edit-turbo"])
def families(request):
    result = {}
    load_model_family_from_json(
        str(Path(__file__).parents[1] / "models" / f"{request.param}.json"), result
    )
    return {spec.model_hub: spec for spec in result[request.param]}


def test_metadata_and_engine(families):
    assert set(families) == {"huggingface", "modelscope"}
    for hub, revision in (("huggingface", "main"), ("modelscope", "master")):
        spec = families[hub]
        assert (spec.model_id, spec.model_revision) == (
            f"briaai/{spec.model_name}",
            revision,
        )
        turbo = spec.model_name.endswith("-turbo")
        assert spec.model_ability == (
            ["image2image", "inpainting"]
            if turbo
            else ["text2image", "image2image", "inpainting"]
        )
        assert spec.default_model_config == {"torch_dtype": "bfloat16"}
        assert spec.default_generate_config == {
            "num_inference_steps": 4 if turbo else 50,
            "guidance_scale": 1.0 if turbo else 5.0,
        }
        assert (
            "diffusers @ https://github.com/huggingface/diffusers/archive/"
            '1ddaf8f4e7c33afaa7c6c2c3be9e56f2c2a634c8.zip ; #engine# == "diffusers"'
            in spec.virtualenv.packages
        )
        assert "transformers>=5.20.1,<6" in spec.virtualenv.packages
        assert all(
            package in spec.virtualenv.packages
            for package in ("#system_torch#", "#system_torchvision#", "#system_numpy#")
        )
        engines = {}
        generate_engine_config_by_model_name(spec, engines)
        assert set(engines[spec.model_name]) == {"diffusers"}
        assert (
            engines[spec.model_name]["diffusers"][0]["image_class"]
            is Fibo2DiffusersImageModel
        )
        assert not DiffusersImageModel.match(spec)


@pytest.fixture(params=["huggingface", "modelscope"])
def loaded_model(request, families, monkeypatch, tmp_path):
    class FakePipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert path == str(tmp_path)
            assert kwargs == {"torch_dtype": torch.bfloat16, "local_files_only": True}
            instance = cls()
            instance.calls = []
            return instance

        def to(self, device):
            assert device == "cpu"
            return self

        def __call__(
            self,
            prompt,
            image=None,
            mask=None,
            width=None,
            height=None,
            num_images_per_prompt=1,
            num_inference_steps=30,
            guidance_scale=5.0,
            negative_prompt="trained-null",
            fonts=None,
            generator=None,
            callback_on_step_end=None,
        ):
            if mask is not None and len(image) != 1:
                raise ValueError("`mask` needs exactly one image in `image`.")
            self.calls.append(
                dict(
                    prompt=prompt,
                    image=image,
                    mask=mask,
                    width=width,
                    height=height,
                    n=num_images_per_prompt,
                    steps=num_inference_steps,
                    guidance=guidance_scale,
                    negative_prompt=negative_prompt,
                    fonts=fonts,
                    seeds=(
                        [g.initial_seed() for g in generator]
                        if isinstance(generator, list)
                        else generator.initial_seed() if generator is not None else None
                    ),
                )
            )
            self.num_timesteps = num_inference_steps
            for step in range(num_inference_steps):
                if callback_on_step_end:
                    assert callback_on_step_end(self, step, step, {}) == {}
            return SimpleNamespace(
                images=[Image.new("RGB", (32, 32))] * num_images_per_prompt
            )

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(BriaFibo2Pipeline=FakePipeline, DiffusionPipeline=FakePipeline),
    )
    monkeypatch.setattr(
        "xinference.model.image.stable_diffusion.core.get_available_device",
        lambda: "cpu",
    )
    (tmp_path / "fonts").mkdir()
    for name in ("Aileron-Regular.otf", "DejaVuSans.ttf"):
        (tmp_path / "fonts" / name).touch()
    model = Fibo2ImageModel(
        "fibo2",
        str(tmp_path),
        device="cpu",
        model_spec=families[request.param],
        torch_dtype="bfloat16",
        local_files_only=True,
    )
    model.load()
    return model


def test_text_to_image_and_edit_only_guard(loaded_model):
    model = loaded_model
    prompt = json.dumps(
        {"short_description": "A bakery", "text_render": [{"text": "BREAD"}]}
    )
    if "text2image" not in model.model_ability:
        with pytest.raises(RuntimeError, match="does not support text2image"):
            asyncio.run(model.text_to_image(prompt))
        assert model._model.calls == []
        return
    result = asyncio.run(
        model.text_to_image(
            prompt, n=2, size="1024*768", seed=[11, 22], _return_images=True
        )
    )
    assert len(result) == 2
    call = model._model.calls[0]
    assert call["prompt"] == prompt
    assert (call["width"], call["height"], call["n"], call["seeds"]) == (
        1024,
        768,
        2,
        [11, 22],
    )
    assert (call["steps"], call["guidance"]) == (50, 5.0)
    assert [Path(path).name for path in call["fonts"]] == [
        "Aileron-Regular.otf",
        "DejaVuSans.ttf",
    ]


@pytest.mark.parametrize("size", [None, "1024*768"])
def test_edit_defaults_references_and_output(loaded_model, size):
    model = loaded_model
    primary = Image.new("RGBA", (64, 48))
    reference = Image.new("L", (32, 48))
    prompt = (
        '{"structured_caption":"A red jacket","edit_instruction":"Change the jacket"}'
    )
    result = model.image_to_image(
        primary,
        prompt,
        n=2,
        size=size,
        response_format="b64_json",
        reference_images=[reference],
        seed=42,
        negative_prompt=" ",
        denoising_strength=0.75,
    )
    call = model._model.calls[0]
    assert call["prompt"] == prompt
    assert [im.mode for im in call["image"]] == ["RGB", "RGB"]
    assert [im.size for im in call["image"]] == [primary.size, reference.size]
    assert (primary.mode, reference.mode) == ("RGBA", "L")
    assert (call["width"], call["height"]) == ((1024, 768) if size else (None, None))
    assert (call["steps"], call["guidance"]) == (
        model._model_spec.default_generate_config["num_inference_steps"],
        model._model_spec.default_generate_config["guidance_scale"],
    )
    assert call["negative_prompt"] == "trained-null"
    assert call["seeds"] == 42
    assert len(result["data"]) == 2
    assert (
        Image.open(io.BytesIO(base64.b64decode(result["data"][0]["b64_json"]))).mode
        == "RGB"
    )


def test_mask_and_sampling_overrides(loaded_model):
    model = loaded_model
    image = Image.new("RGB", (64, 48))
    mask = Image.new("L", image.size, 255)
    result = model.inpainting(
        image,
        mask,
        "Add a balloon",
        n=2,
        size="1024*768",
        seed=[1, 2],
        num_inference_steps=8,
        guidance_scale=2.0,
        negative_prompt="blur",
        fonts=[],
        _return_images=True,
    )
    assert len(result) == 2
    call = model._model.calls[0]
    assert call["mask"] is mask
    assert call["image"] == [image]
    assert (call["steps"], call["guidance"], call["seeds"]) == (8, 2.0, [1, 2])
    assert (call["width"], call["height"]) == (1024, 768)
    assert call["negative_prompt"] == "blur"
    assert call["fonts"] == []
    assert not model._need_set_scheduler(object())


def test_progress_and_cancellation(loaded_model):
    model = loaded_model
    progressor = Mock(request_id="request")
    model.image_to_image(
        Image.new("RGB", (32, 32)),
        "Edit",
        num_inference_steps=4,
        progressor=progressor,
        _return_images=True,
    )
    assert [call.args[0] for call in progressor.set_progress.call_args_list] == [
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    cancel_event = threading.Event()
    cancel_event.set()
    with pytest.raises(RuntimeError, match="Image generation cancelled"):
        model.inpainting(
            Image.new("RGB", (32, 32)),
            Image.new("L", (32, 32)),
            "Edit",
            _cancel_event=cancel_event,
            _return_images=True,
        )


@pytest.mark.parametrize(
    "option", ["lora_model", "gguf_model_path", "lightning_model_path"]
)
def test_reject_unsupported_load_options(loaded_model, option):
    model = loaded_model
    setattr(model, f"_{option}", "unsupported")
    with pytest.raises(ValueError, match="does not support LoRA, GGUF or Lightning"):
        model.load()
