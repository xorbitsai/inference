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
