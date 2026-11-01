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

# unsloth/Qwen3-235B-A22B-Thinking-2507-GGUF at Hugging Face commit
# 3eba50b4e3ea2c0df1c43e8c8f5ce2db22645e81 (same names on ModelScope).
THINKING_235B_PARTS = {
    "BF16": 10, "IQ4_XS": 3, "Q2_K": 2, "Q2_K_L": 2, "Q3_K_M": 3, "Q3_K_S": 3,
    "Q4_0": 3, "Q4_1": 3, "Q4_K_M": 3, "Q4_K_S": 3, "Q5_K_M": 4, "Q5_K_S": 4,
    "Q6_K": 4, "Q8_0": 6, "UD-Q2_K_XL": 2, "UD-Q3_K_XL": 3, "UD-Q4_K_XL": 3,
    "UD-Q5_K_XL": 4, "UD-Q6_K_XL": 4, "UD-Q8_K_XL": 6,
}  # fmt: skip
THINKING_235B_FILES = {
    f"{q}/Qwen3-235B-A22B-Thinking-2507-{q}-{i:05d}-of-{n:05d}.gguf"
    for q, n in THINKING_235B_PARTS.items()
    for i in range(1, n + 1)
}

# unsloth/Qwen3-4B-{Thinking,Instruct}-2507-GGUF at commits
# f40adb104d4d44aee52f398b60597c5866a973a3 / a06e946bb6b655725eafa393f4a9745d460374c9:
# single files; the half-precision file is F16, there is no BF16.
QWEN3_4B_QUANTS = (
    "F16", "IQ4_NL", "IQ4_XS", "Q2_K", "Q2_K_L", "Q3_K_M", "Q3_K_S", "Q4_0", "Q4_1",
    "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "UD-IQ1_M", "UD-IQ1_S",
    "UD-IQ2_M", "UD-IQ2_XXS", "UD-IQ3_XXS", "UD-Q2_K_XL", "UD-Q3_K_XL",
    "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "UD-Q8_K_XL",
)  # fmt: skip


def _files_4b(kind):
    return {f"Qwen3-4B-{kind}-2507-{q}.gguf" for q in QWEN3_4B_QUANTS}


@pytest.mark.parametrize(
    "model_name, model_id, files",
    [
        (
            "Qwen3-Thinking",
            "unsloth/Qwen3-235B-A22B-Thinking-2507-GGUF",
            THINKING_235B_FILES,
        ),
        (
            "Qwen3-Thinking",
            "unsloth/Qwen3-4B-Thinking-2507-GGUF",
            _files_4b("Thinking"),
        ),
        (
            "Qwen3-Instruct",
            "unsloth/Qwen3-4B-Instruct-2507-GGUF",
            _files_4b("Instruct"),
        ),
    ],
)
def test_qwen3_2507_gguf_files_exist(model_name, model_id, files):
    from ...utils import generate_model_file_names_with_quantization_parts
    from ..llm_family import get_llm_family_by_name

    family = get_llm_family_by_name(model_name)
    specs = [
        s
        for s in family.model_specs
        if s.model_format == "ggufv2" and s.model_id == model_id
    ]
    assert specs
    missing = []
    for spec in specs:
        names, _, _ = generate_model_file_names_with_quantization_parts(spec)
        missing += [(spec.quantization, n) for n in names if n not in files]
    assert missing == []
