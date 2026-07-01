# Copyright 2022-2026 XProbe Inc.
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
from pathlib import Path

import pytest

from .....client import Client


def test_gguf(setup):
    endpoint, _ = setup
    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="tiny-llama",
        model_engine="llama.cpp",
        model_size_in_billions=1,
        model_format="ggufv2",
        quantization="q2_K",
    )
    assert len(client.list_models()) == 1
    model = client.get_model(model_uid)
    completion = model.generate("AI is going to", generate_config={"max_tokens": 5})
    assert "id" in completion
    assert "text" in completion["choices"][0]
    assert len(completion["choices"][0]["text"]) > 0


@pytest.fixture(scope="session")
def surprise_image_base64():
    img_path = Path(__file__).parent / "assets" / "surprise.png"

    with open(img_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    return "data:image/png;base64," + img_base64


def test_gguf_multimodal(setup, surprise_image_base64):
    endpoint, _ = setup
    client = Client(endpoint)

    r = client.query_engine_by_model_name("gemma-3-it")
    assert (
        "mmproj-google_gemma-3-4b-it-f16.gguf"
        in r["llama.cpp"][0]["multimodal_projectors"]
    )

    model_uid = client.launch_model(
        model_name="gemma-3-it",
        model_engine="llama.cpp",
        model_size_in_billions=4,
        model_format="ggufv2",
        quantization="IQ4_XS",
        multimodal_projector="mmproj-google_gemma-3-4b-it-f16.gguf",
        n_ctx=512,
        n_batch=512,
        n_parallel=1,
    )
    assert len(client.list_models()) == 1
    model = client.get_model(model_uid)

    completion = model.chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this:\n"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": surprise_image_base64,
                        },
                    },
                ],
            }
        ],
        generate_config={"max_tokens": 128},
    )
    content = completion["choices"][0]["message"]["content"]
    assert "id" in completion
    assert isinstance(content, str)
    assert len(content) > 0
