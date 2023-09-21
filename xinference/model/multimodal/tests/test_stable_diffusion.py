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
import os.path

from ..core import MultimodalModelSpec, cache
from ..stable_diffusion.core import DiffusionModel

TEST_MODEL_SPEC = MultimodalModelSpec(
    model_family="stable_diffusion",
    model_name="small-stable-diffusion-v0",
    model_id="OFA-Sys/small-stable-diffusion-v0",
    model_revision="38e10e5e71e8fbf717a47a81e7543cd01c1a8140",
)


def test_model():
    model_path = cache(TEST_MODEL_SPEC)
    model = DiffusionModel("mock", model_path)
    # input is a string
    input_text = "an apple"
    model.load()
    r = model.text_to_image(input_text)
    assert len(r["data"]) == 1
    assert os.path.exists(r["data"][0]["url"])
