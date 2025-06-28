# Copyright 2022-2025 XProbe Inc.
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
from dataclasses import dataclass

from ..memory import estimate_gpu_layers

TEST_GGUF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dummy.gguf")


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
