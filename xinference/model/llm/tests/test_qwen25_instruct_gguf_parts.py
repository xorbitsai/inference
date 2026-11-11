# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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

# CI: linux
import pytest

# Shard counts in Qwen/Qwen2.5-7B-Instruct-GGUF (q8_0, both hubs) and
# Qwen/Qwen2.5-72B-Instruct-GGUF (fp16, offered from Hugging Face only).
CASES = [
    (7, "q8_0", "qwen2.5-7b-instruct-q8_0", 3, {"huggingface", "modelscope"}),
    (72, "fp16", "qwen2.5-72b-instruct-fp16", 42, {"huggingface"}),
]


@pytest.mark.parametrize("size, quantization, stem, parts, hubs", CASES)
def test_qwen25_instruct_gguf_parts(size, quantization, stem, parts, hubs):
    from ...utils import generate_model_file_names_with_quantization_parts
    from ..llm_family import get_llm_family_by_name

    specs = [
        s
        for s in get_llm_family_by_name("qwen2.5-instruct").model_specs
        if s.model_format == "ggufv2"
        and s.model_size_in_billions == size
        and s.quantization == quantization
    ]
    assert {s.model_hub for s in specs} == hubs
    for spec in specs:
        names, _, _ = generate_model_file_names_with_quantization_parts(spec)
        assert names == [
            f"{stem}-{i:05d}-of-{parts:05d}.gguf" for i in range(1, parts + 1)
        ]
