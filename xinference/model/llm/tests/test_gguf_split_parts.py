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

# These quantizations are uploaded as two shards in a folder (same names on
# Hugging Face and ModelScope), not as one file.
CASES = [
    (
        "deepseek-r1-distill-qwen",
        "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "F16",
        [
            "DeepSeek-R1-Distill-Qwen-32B-F16/DeepSeek-R1-Distill-Qwen-32B-F16-00001-of-00002.gguf",
            "DeepSeek-R1-Distill-Qwen-32B-F16/DeepSeek-R1-Distill-Qwen-32B-F16-00002-of-00002.gguf",
        ],
    ),
    (
        "gemma-3-it",
        "bartowski/google_gemma-3-27b-it-GGUF",
        "bf16",
        [
            "google_gemma-3-27b-it-bf16/google_gemma-3-27b-it-bf16-00001-of-00002.gguf",
            "google_gemma-3-27b-it-bf16/google_gemma-3-27b-it-bf16-00002-of-00002.gguf",
        ],
    ),
]


@pytest.mark.parametrize("model_name, model_id, quantization, files", CASES)
def test_split_gguf_quantization_resolves_to_its_shards(
    model_name, model_id, quantization, files
):
    from ...utils import generate_model_file_names_with_quantization_parts
    from ..llm_family import get_llm_family_by_name

    specs = [
        s
        for s in get_llm_family_by_name(model_name).model_specs
        if s.model_format == "ggufv2"
        and s.model_id == model_id
        and s.quantization == quantization
    ]
    assert len(specs) == 2  # Hugging Face and ModelScope
    for spec in specs:
        names, _, need_merge = generate_model_file_names_with_quantization_parts(spec)
        assert (names, need_merge) == (files, True)
