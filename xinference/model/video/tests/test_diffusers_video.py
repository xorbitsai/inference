# Copyright 2022-2023 XProbe Inc.
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
import logging

import pytest

from .. import BUILTIN_VIDEO_MODELS
from ..core import cache
from ..diffusers import DiffUsersVideoModel

logger = logging.getLogger(__name__)


@pytest.mark.skip(reason="Video model requires too many GRAM.")
def test_model():
    test_model_spec = next(iter(BUILTIN_VIDEO_MODELS.values()))
    model_path = cache(test_model_spec)
    model = DiffUsersVideoModel("mock", model_path, test_model_spec)
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
