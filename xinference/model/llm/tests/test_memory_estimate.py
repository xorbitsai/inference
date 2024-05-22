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

from ..memory import estimate_llm_gpu_memory


def test_llm_estimate_memory():
    # TODO GGML is not tested yet

    # without model_name, use default ModelLayersInfo
    mem_info = estimate_llm_gpu_memory(1.8, None, 2048, "pytorch", kv_cache_dtype=32)
    assert mem_info.total == 5162
    mem_info = estimate_llm_gpu_memory(
        72, "8-bit", 1024 + 2048, "pytorch", kv_cache_dtype=32
    )
    assert mem_info.total == 92943
    mem_info = estimate_llm_gpu_memory(
        72, "4-bit", 1024 + 2048, "pytorch", kv_cache_dtype=32
    )
    assert mem_info.total == 58611
    mem_info = estimate_llm_gpu_memory(7, "Int4", 32768, "gptq", kv_cache_dtype=32)
    assert mem_info.total == 100550

    # with model_name to match_llm
    mem_info = estimate_llm_gpu_memory(
        72, "Int4", 32768, "gptq", kv_cache_dtype=16, model_name="qwen1.5-chat"
    )
    assert mem_info.total == 258775

    # model_size_in_billions use int
    mem_info = estimate_llm_gpu_memory(
        32, "Int4", 32768, "gptq", kv_cache_dtype=8, model_name="qwen1.5-chat"
    )
    # model_size_in_billions use str
    mem_info = estimate_llm_gpu_memory(
        "32", "Int4", 32768, "gptq", kv_cache_dtype=8, model_name="qwen1.5-chat"
    )
    assert mem_info.total == 58035

    # model_size_in_billions use float
    mem_info = estimate_llm_gpu_memory(
        "1.8", None, 2048, "pytorch", kv_cache_dtype=32, model_name="qwen1.5-chat"
    )
    assert mem_info.total == 5020
