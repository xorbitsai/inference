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
import base64
import os.path
import requests
from io import BytesIO

from PIL import Image

from ..core import MultimodalModelFamilyV1, cache
from ..stable_diffusion.core import DiffusionModel
from .. import BUILTIN_MULTIMODAL_MODELS

TEST_MODEL_SPEC = MultimodalModelFamilyV1(
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
    r = model.text_to_image(input_text, size="256*256")
    assert len(r["data"]) == 1
    assert os.path.exists(r["data"][0]["url"])
    r = model.text_to_image(input_text, size="256*256", response_format="b64_json")
    assert len(r["data"]) == 1
    b64_json = r["data"][0]["b64_json"]
    image_bytes = base64.decodebytes(b64_json)
    img = Image.open(BytesIO(image_bytes))
    assert img.size == (256, 256)


def test_restful_api_for_multimodal(setup):
    model_name = "stable-diffusion-v1-5"
    model_spec = BUILTIN_MULTIMODAL_MODELS[model_name]

    endpoint, _ = setup
    url = f"{endpoint}/v1/models"

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data) == 0

    # launch
    payload = {
        "model_uid": "test_stable_diffusion",
        "model_name": model_name,
        "model_type": "multimodal",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_stable_diffusion"

    response = requests.get(url)
    response_data = response.json()
    assert len(response_data) == 1

    # test embedding
    url = f"{endpoint}/v1/images/generations"
    payload = {
        "model": "test_stable_diffusion",
        "prompt": "an apple",
        "size": "256*256",
    }
    response = requests.post(url, json=payload)
    r = response.json()
    assert len(r["data"]) == 1
    assert os.path.exists(r["data"][0]["url"])
