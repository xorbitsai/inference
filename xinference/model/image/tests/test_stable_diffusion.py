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
import io
import os.path
from io import BytesIO

import requests
from PIL import Image

from ..core import ImageModelFamilyV1, cache
from ..stable_diffusion.core import DiffusionModel

TEST_MODEL_SPEC = ImageModelFamilyV1(
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


def test_restful_api_for_image(setup):
    endpoint, _ = setup
    from ....client import Client
    client = Client(endpoint)

    model_uid = client.launch_model(model_uid="my_controlnet", model_name="stable-diffusion-xl-base-1.0",
                                    model_type="image",
                                    controlnet="canny")
    model = client.get_model(model_uid)


    import cv2
    import numpy as np
    from diffusers.utils import load_image
    from PIL import Image

    image = load_image("/Users/po/Work/inference/hf-logo.png")
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
    negative_prompt = 'low quality, bad quality, sketches'
    bio = io.BytesIO()
    image.save(bio, format="png")
    model.image_to_image(image=bio.getvalue(), prompt=prompt, negative_prompt=negative_prompt,
                         controlnet_conditioning_scale=0.5)
