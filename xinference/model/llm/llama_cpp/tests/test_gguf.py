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
from dataclasses import dataclass

import requests

from .....client import Client
from ..memory import estimate_gpu_layers

TEST_GGUF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dummy.gguf")


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


def test_gguf_multimodal(setup):
    IMG_URL_0 = "https://github.com/bebechien/gemma/blob/main/surprise.png?raw=true"

    response = requests.get(IMG_URL_0)
    response.raise_for_status()  # Raise an exception for bad status codes
    IMG_BASE64_0 = "data:image/png;base64," + base64.b64encode(response.content).decode(
        "utf-8"
    )

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
                            "url": IMG_BASE64_0,
                        },
                    },
                ],
            }
        ],
        generate_config={"max_tokens": 128},
    )
    content = completion["choices"][0]["message"]["content"]
    assert "id" in completion
    assert "black" in content
    assert "white" in content
    assert "cat" in content


def test_estimate_gpu_layers():
    estimate = estimate_gpu_layers(
        [{"name": "CPU", "memory_free": 0}],
        TEST_GGUF,
        [],
        context_length=2048,
        batch_size=512,
        num_parallel=1,
        kv_cache_type="",
    )
    assert estimate.layers == 0
    assert estimate.graph == 0

    graph_partial_offload = 202377216
    graph_full_offload = 171968512
    layer_size = 33554436
    projector_size = 0
    memory_layer_output = 4
    gpu_minimum_memory = 2048

    gpus = [
        {"name": "cuda", "memory_min": gpu_minimum_memory},
        {"name": "cuda", "memory_min": gpu_minimum_memory},
    ]

    @dataclass
    class _TestInfo:
        layer0: int  # type: ignore
        layer1: int  # type: ignore
        expect0: int  # type: ignore
        expect1: int  # type: ignore

    test_data = [
        _TestInfo(*v)
        for v in [
            [1, 1, 1, 1],
            [2, 1, 2, 1],
            [2, 2, 2, 2],
            [1, 2, 1, 2],
            [3, 3, 3, 3],
            [4, 4, 3, 3],
            [6, 6, 3, 3],
            [0, 3, 0, 3],
        ]
    ]
    for i, s in enumerate(test_data):
        gpus[0]["memory_free"] = 0
        gpus[1]["memory_free"] = 0
        gpus[0]["memory_free"] += projector_size
        if s.layer0 > 0:
            gpus[0]["memory_free"] += memory_layer_output
        else:
            gpus[1]["memory_free"] += memory_layer_output
        gpus[0]["memory_free"] += (
            gpu_minimum_memory + layer_size + s.layer0 * layer_size + 1
        )
        gpus[1]["memory_free"] += (
            gpu_minimum_memory + layer_size + s.layer1 * layer_size + 1
        )
        gpus[0]["memory_free"] += max(graph_full_offload, graph_partial_offload)
        gpus[1]["memory_free"] += max(graph_full_offload, graph_partial_offload)
        estimate = estimate_gpu_layers(
            gpus,
            TEST_GGUF,
            [],
            context_length=2048,
            batch_size=512,
            num_parallel=1,
            kv_cache_type="",
        )
        assert s.expect0 + s.expect1 == estimate.layers
        assert f"{s.expect0},{s.expect1}" == estimate.tensor_split
        layer_sums = sum(estimate.gpu_sizes)
        if estimate.layers < 6:
            assert estimate.vram_size < estimate.total_size
            assert estimate.vram_size == layer_sums
        else:
            assert estimate.vram_size == estimate.total_size
            assert estimate.total_size == layer_sums
