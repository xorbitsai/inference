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

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ...cache_manager import CacheManager
from .. import BUILTIN_VIDEO_MODELS
from ..diffusers import DiffusersVideoModel

logger = logging.getLogger(__name__)


def test_process_seed_builds_generator_or_ignores_it(monkeypatch):
    import torch

    from .. import diffusers as diffusers_module

    model = DiffusersVideoModel(
        "mock",
        "/tmp/mock",
        SimpleNamespace(model_ability=[], default_generate_config={}),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(diffusers_module, "get_available_device", lambda: "cpu")

    class AcceptsGenerator:
        def __call__(self, prompt, generator=None):
            return prompt, generator

    model._model = AcceptsGenerator()
    kwargs = {"seed": 42}
    model._process_seed(kwargs)
    assert "seed" not in kwargs
    assert isinstance(kwargs["generator"], torch.Generator)
    assert kwargs["generator"].initial_seed() == 42

    class RejectsGenerator:
        def __call__(self, prompt):
            return prompt

    model._model = RejectsGenerator()
    kwargs = {"seed": 42}
    model._process_seed(kwargs)
    assert kwargs == {}


@pytest.mark.parametrize(
    "last_size, bars",
    [
        ((200, 100), [(0, 0, 20, 100), (180, 0, 200, 100)]),
        ((100, 200), [(0, 0, 100, 20), (0, 180, 100, 200)]),
    ],
)
def test_center_crop_resize_matches_first_frame(last_size, bars):
    from PIL import Image

    last_frame = Image.new("RGB", last_size, (0, 0, 255))
    for bar in bars:
        last_frame.paste((255, 0, 0), bar)

    resized = DiffusersVideoModel._center_crop_resize(last_frame, height=48, width=64)

    assert resized.size == (64, 48)
    assert resized.getpixel((0, 0)) == (0, 0, 255)
    assert resized.getpixel((63, 47)) == (0, 0, 255)


class _FakeWanImageToVideoPipeline:
    vae_scale_factor_spatial = 8
    transformer = SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2)))

    def __call__(
        self,
        image,
        prompt,
        last_image=None,
        height=480,
        width=832,
        num_videos_per_prompt=1,
        generator=None,
        **kwargs,
    ):
        return {
            "image": image,
            "last_image": last_image,
            "height": height,
            "width": width,
        }


def _wan_i2v_model(monkeypatch, default_generate_config):
    model = DiffusersVideoModel(
        "mock",
        "/tmp/mock",
        SimpleNamespace(  # type: ignore[arg-type]
            model_family="Wan",
            model_ability=["image2video"],
            default_generate_config=default_generate_config,
        ),
    )
    model._model = _FakeWanImageToVideoPipeline()
    monkeypatch.setattr(
        DiffusersVideoModel,
        "_output_to_video",
        lambda self, output, fps, response_format: output,
    )
    return model


@pytest.mark.parametrize(
    "default_generate_config, kwargs, expected_size",
    [
        # Wan2.2-i2v-A14B (diffusers) has no max_area in its spec
        ({}, {}, (832, 480)),
        ({}, {"width": 416, "height": 240}, (416, 240)),
        ({"max_area": [480, 832]}, {}, (832, 480)),
        ({"max_area": [480, 832]}, {"max_area": "240*416"}, (416, 240)),
        ({}, {"max_area": 240 * 416}, (416, 240)),
    ],
)
def test_image_to_video_resolves_max_area(
    monkeypatch, default_generate_config, kwargs, expected_size
):
    from PIL import Image

    model = _wan_i2v_model(monkeypatch, default_generate_config)
    output = model.image_to_video(Image.new("RGB", (1664, 960)), "a cat", **kwargs)

    assert output["image"].size == expected_size
    assert (output["width"], output["height"]) == expected_size


def test_firstlastframe_to_video_without_max_area(monkeypatch):
    from PIL import Image

    model = _wan_i2v_model(monkeypatch, {})
    output = model.firstlastframe_to_video(
        Image.new("RGB", (1664, 960)), Image.new("RGB", (960, 1664)), "a cat"
    )

    assert output["image"].size == (832, 480)
    assert output["last_image"].size == (832, 480)


@pytest.mark.parametrize("has_image_encoder", [True, False])
def test_load_wan_image_to_video_without_image_encoder(
    monkeypatch, tmp_path, has_image_encoder
):
    import diffusers
    import transformers

    from .. import diffusers as diffusers_module

    (tmp_path / "vae").mkdir()
    if has_image_encoder:
        (tmp_path / "image_encoder").mkdir()

    calls = {}

    class FakeCLIPVisionModel:
        @classmethod
        def from_pretrained(cls, path, subfolder=None, **kwargs):
            assert subfolder == "image_encoder"
            calls["image_encoder"] = True
            return "clip"

    class FakeVAE:
        @classmethod
        def from_pretrained(cls, path, subfolder=None, **kwargs):
            return "vae"

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["pipeline_kwargs"] = kwargs
            return MagicMock()

    monkeypatch.setattr(transformers, "CLIPVisionModel", FakeCLIPVisionModel)
    monkeypatch.setattr(diffusers, "AutoencoderKLWan", FakeVAE)
    monkeypatch.setattr(diffusers, "WanImageToVideoPipeline", FakePipeline)
    monkeypatch.setattr(diffusers_module, "gpu_count", lambda: 0)
    monkeypatch.setattr(
        diffusers_module, "move_model_to_available_device", lambda pipeline: pipeline
    )

    model = DiffusersVideoModel(
        "mock",
        str(tmp_path),
        SimpleNamespace(  # type: ignore[arg-type]
            model_family="Wan",
            model_ability=["image2video"],
            default_model_config={},
            default_generate_config={},
        ),
    )
    model.load()

    assert calls.get("image_encoder", False) == has_image_encoder
    assert calls["pipeline_kwargs"]["vae"] == "vae"
    if has_image_encoder:
        assert calls["pipeline_kwargs"]["image_encoder"] == "clip"
    else:
        assert "image_encoder" not in calls["pipeline_kwargs"]


@pytest.mark.skip(reason="Video model requires too many GRAM.")
def test_model():
    test_model_spec = next(iter(BUILTIN_VIDEO_MODELS.values()))
    model_path = CacheManager(test_model_spec).cache()
    model = DiffusersVideoModel("mock", model_path, test_model_spec)
    # input is a string
    input_text = "an apple"
    model.load()
    r = model.text_to_image(input_text)
    assert r


@pytest.mark.skip(reason="Video model requires too many GRAM.")
def test_client(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_uid="my_video_model",
        model_name="CogVideoX-2b",
        model_type="video",
    )
    model = client.get_model(model_uid)
    assert model

    r = model.text_to_video(
        prompt="A panda, dressed in a small, red jacket and a tiny hat, "
        "sits on a wooden stool in a serene bamboo forest. "
        "The panda's fluffy paws strum a miniature acoustic guitar, "
        "producing soft, melodic tunes. Nearby, a few other pandas gather, "
        "watching curiously and some clapping in rhythm. "
        "Sunlight filters through the tall bamboo, casting a gentle glow on the scene. "
        "The panda's face is expressive, showing concentration and joy as it plays. "
        "The background includes a small, flowing stream and vibrant green foliage, "
        "enhancing the peaceful and magical atmosphere of this unique musical performance."
    )
    assert r
