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
import os
import platform
import sys

import pytest

from .....client import Client


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.processor() != "arm",
    reason="MLX only works for Apple silicon chip",
)
def test_load_mlx(setup):
    endpoint, _ = setup
    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="qwen2-instruct",
        model_engine="MLX",
        model_size_in_billions="0_5",
        model_format="mlx",
        quantization="4bit",
    )
    assert len(client.list_models()) == 1
    model = client.get_model(model_uid)
    messages = [{"role": "user", "content": "write a poem."}]
    completion = model.chat(messages)
    assert "content" in completion["choices"][0]["message"]
    assert len(completion["choices"][0]["message"]["content"]) != 0
    content = completion["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": content})
    messages.append({"role": "user", "content": "explain it"})
    completion = model.chat(messages)
    assert "content" in completion["choices"][0]["message"]
    assert len(completion["choices"][0]["message"]["content"]) != 0


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.processor() != "arm",
    reason="MLX only works for Apple silicon chip",
)
def test_load_mlx_vision(setup):
    endpoint, _ = setup
    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="qwen2-vl-instruct",
        model_engine="MLX",
        model_size_in_billions=2,
        model_format="mlx",
        quantization="4bit",
    )
    assert len(client.list_models()) == 1
    model = client.get_model(model_uid)

    path = os.path.join(os.path.dirname(__file__), "fish.png")
    with open(path, "rb") as f:
        content = f.read()
    b64_img = base64.b64encode(content).decode("utf-8")

    completion = model.chat(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "图中有几条鱼？"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_img}",
                        },
                    },
                ],
            }
        ],
    )
    assert "图中" in completion["choices"][0]["message"]["content"]
    assert "鱼" in completion["choices"][0]["message"]["content"]

    # test no image
    messages = [{"role": "user", "content": "write a poem."}]
    completion = model.chat(messages)
    assert "content" in completion["choices"][0]["message"]
    assert "content" in completion["choices"][0]["message"]
    assert len(completion["choices"][0]["message"]["content"]) != 0
