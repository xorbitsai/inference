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
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from .. import load_model_family_from_json
from ..engine import DiffusersImageModel, MOSSDiffusersImageModel
from ..engine_family import generate_engine_config_by_model_name
from ..moss_image import MOSSImageModel


@pytest.fixture
def families():
    result = {}
    load_model_family_from_json(
        str(Path(__file__).parents[1] / "models" / "MOSS-Image-SFT.json"), result
    )
    return {spec.model_hub: spec for spec in result["MOSS-Image-SFT"]}


def test_metadata_and_engine(families):
    assert set(families) == {"huggingface", "modelscope"}
    for hub, model_id, revision in (
        ("huggingface", "OpenMOSS-Team/MOSS-Image-SFT", "main"),
        ("modelscope", "openmoss/MOSS-Image-SFT", "master"),
    ):
        spec = families[hub]
        assert (spec.model_id, spec.model_revision) == (model_id, revision)
        assert spec.model_ability == ["text2image"]
        assert spec.default_generate_config == {
            "num_inference_steps": 50,
            "guidance_scale": 3.0,
            "uncond_skip_layers": [22],
        }
        engines = {}
        generate_engine_config_by_model_name(spec, engines)
        assert set(engines[spec.model_name]) == {"diffusers"}
        assert (
            engines[spec.model_name]["diffusers"][0]["image_class"]
            is MOSSDiffusersImageModel
        )
        assert not DiffusersImageModel.match(spec)


@pytest.fixture
def pipeline(monkeypatch):
    class FakeScheduler:
        config = {"shift": 5.0}

        @classmethod
        def from_config(cls, config):
            assert config == {"shift": 5.0}
            return cls()

    class FakePipeline:
        def __init__(self):
            self.scheduler = FakeScheduler()
            self.calls = []
            self.offload = False

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            instance = cls()
            instance.path = path
            instance.load_kwargs = kwargs
            return instance

        def to(self, device):
            self.device = device
            return self

        def enable_model_cpu_offload(self):
            self.offload = True

        def progress_bar(self, timesteps):
            return iter(timesteps)

        def __call__(
            self,
            prompt,
            width,
            height,
            num_images_per_prompt=1,
            num_inference_steps=50,
            guidance_scale=3.0,
            uncond_skip_layers=(22,),
            negative_prompt="",
            generator=None,
        ):
            self.calls.append(
                dict(
                    prompt=prompt,
                    width=width,
                    height=height,
                    steps=num_inference_steps,
                    guidance=guidance_scale,
                    skip=uncond_skip_layers,
                    negative_prompt=negative_prompt,
                    seeds=(
                        [g.initial_seed() for g in generator]
                        if isinstance(generator, list)
                        else generator.initial_seed() if generator is not None else None
                    ),
                )
            )
            for _ in self.progress_bar(range(num_inference_steps)):
                pass
            return SimpleNamespace(
                images=[Image.new("RGB", (width, height))] * num_images_per_prompt
            )

    monkeypatch.setitem(
        sys.modules, "diffusers", SimpleNamespace(DiffusionPipeline=FakePipeline)
    )
    monkeypatch.setattr(
        "xinference.model.image.stable_diffusion.core.get_available_device",
        lambda: "cpu",
    )
    return FakePipeline


@pytest.mark.parametrize("hub", ["huggingface", "modelscope"])
@pytest.mark.parametrize(
    "dtype,expected",
    [
        ("bfloat16", {"default": torch.bfloat16, "vae": torch.float32}),
        ("torch.float32", {"default": torch.float32, "vae": torch.float32}),
        (
            {"transformer": "float32"},
            {
                "default": torch.bfloat16,
                "transformer": torch.float32,
                "vae": torch.float32,
            },
        ),
    ],
)
def test_load_local_custom_pipeline(families, pipeline, hub, dtype, expected):
    model = MOSSImageModel(
        "moss",
        "/models/moss",
        model_spec=families[hub],
        torch_dtype=dtype,
        cpu_offload=True,
        local_files_only=True,
    )
    model.load()
    assert model._model.path == "/models/moss"
    assert model._model.load_kwargs == {
        "torch_dtype": expected,
        "custom_pipeline": "pipeline_moss_image",
        "trust_remote_code": True,
        "local_files_only": True,
    }
    assert model._model.offload
    assert not model._need_set_scheduler(object())


@pytest.fixture
def loaded_model(families, pipeline):
    model = MOSSImageModel(
        "moss", "/models/moss", device="cpu", model_spec=families["modelscope"]
    )
    model.load()
    return model


def test_text_to_image_defaults_and_batch(loaded_model):
    result = asyncio.run(
        loaded_model.text_to_image(
            '{"summary":{"caption":"一只猫"},"items":[]}',
            n=2,
            size="64*32",
            seed=[17, 23],
            response_format="b64_json",
        )
    )
    assert len(result["data"]) == 2
    assert loaded_model._model.calls == [
        dict(
            prompt='{"summary":{"caption":"一只猫"},"items":[]}',
            width=64,
            height=32,
            steps=50,
            guidance=3.0,
            skip=[22],
            negative_prompt="",
            seeds=[17, 23],
        )
    ]
    image = Image.open(io.BytesIO(base64.b64decode(result["data"][0]["b64_json"])))
    assert image.size == (64, 32)


def test_sampling_overrides_and_progress(loaded_model):
    pipe = loaded_model._model
    original_progress_bar = pipe.progress_bar
    progressor = SimpleNamespace(request_id="test", set_progress=Mock())
    asyncio.run(
        loaded_model.text_to_image(
            "A cat",
            size="32*32",
            seed=42,
            num_inference_steps=2,
            guidance_scale=4.0,
            uncond_skip_layers=[],
            negative_prompt="blurry",
            progressor=progressor,
            _return_images=True,
        )
    )
    assert pipe.calls[0] == dict(
        prompt="A cat",
        width=32,
        height=32,
        steps=2,
        guidance=4.0,
        skip=[],
        negative_prompt="blurry",
        seeds=42,
    )
    assert [call.args[0] for call in progressor.set_progress.call_args_list] == [
        0.5,
        1.0,
    ]
    assert pipe.progress_bar == original_progress_bar


def test_cancel_restores_progress_bar(loaded_model):
    pipe = loaded_model._model
    original_progress_bar = pipe.progress_bar
    event = threading.Event()
    event.set()
    with pytest.raises(RuntimeError, match="Image generation cancelled"):
        asyncio.run(loaded_model.text_to_image("A cat", _cancel_event=event))
    assert pipe.progress_bar == original_progress_bar


def test_progress_failure_restores_progress_bar(loaded_model):
    pipe = loaded_model._model
    original_progress_bar = pipe.progress_bar
    progressor = SimpleNamespace(
        request_id="test",
        set_progress=Mock(side_effect=RuntimeError("progress failed")),
    )
    with pytest.raises(RuntimeError, match="progress failed"):
        asyncio.run(loaded_model.text_to_image("A cat", progressor=progressor))
    assert pipe.progress_bar == original_progress_bar


@pytest.mark.parametrize(
    "option", ["lora_model", "gguf_model_path", "lightning_model_path"]
)
def test_reject_unsupported_load_options(families, pipeline, option):
    model = MOSSImageModel(
        "moss", "/models/moss", model_spec=families["huggingface"], **{option: "test"}
    )
    with pytest.raises(ValueError, match="does not support"):
        model.load()
