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
from io import BytesIO

import numpy as np
import requests
from PIL import Image

from ....client import Client


def test_sparse_embedding(setup):
    endpoint, _ = setup
    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="bge-m3", model_type="embedding", model_engine="flag"
    )
    assert len(client.list_models()) == 1

    model = client.get_model(model_uid)

    result = model.create_embedding("What is BGE M3?", return_sparse=True)
    emb = result["data"][0]["embedding"]
    token_ids = []
    values = []
    for token_id, v in emb.items():
        token_ids.append(token_id)
        values.append(v)
    words = model.convert_ids_to_tokens(token_ids)
    assert len(words) == len(token_ids)
    assert isinstance(words[0], str)


def test_clip_embedding(setup):
    endpoint, _ = setup
    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="jina-clip-v2", model_type="embedding", torch_dtype="float16"
    )
    assert len(client.list_models()) == 1

    model = client.get_model(model_uid)

    def image_to_base64(image: Image.Image, fmt="png") -> str:
        output_buffer = BytesIO()
        image.save(output_buffer, format=fmt)
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return f"data:image/{fmt};base64," + base64_str

    image_str = "https://i.ibb.co/r5w8hG8/beach2.jpg"
    image_str_base64 = image_to_base64(
        Image.open(BytesIO(requests.get(image_str).content))
    )
    input = [
        {"text": "This is a picture of diagram"},
        {"image": image_str_base64},
        {"text": "a dog"},
        {"image": image_str},
        {"text": "海滩上美丽的日落。"},
    ]
    response = model.create_embedding(input)
    for i in range(len(response["data"])):
        embedding = np.array([item for item in response["data"][i]["embedding"]])
        assert embedding.shape == (1024,)


def test_llama_cpp_embedding(setup):
    endpoint, _ = setup
    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="Qwen3-Embedding-0.6B",
        model_type="embedding",
        model_engine="llama.cpp",
        model_format="ggufv2",
        quantization="Q8_0",
        download_hub="huggingface",
    )
    assert len(client.list_models()) == 1

    model = client.get_model(model_uid)

    result = model.create_embedding("What is BGE M3?")
    emb = result["data"][0]["embedding"]
    assert len(emb) == 1024
