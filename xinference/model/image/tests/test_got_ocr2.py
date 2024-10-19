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

import io

from diffusers.utils import load_image


def test_got_ocr2(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_uid="ocr_test",
        model_name="GOT-OCR2_0",
        model_type="image",
    )
    model = client.get_model(model_uid)

    url = "https://huggingface.co/stepfun-ai/GOT-OCR2_0/resolve/main/assets/train_sample.jpg"
    image = load_image(url)
    bio = io.BytesIO()
    image.save(bio, format="JPEG")
    r = model.ocr(
        image=bio.getvalue(),
        ocr_type="ocr",
    )
    assert "Jesuits Estate" in r
