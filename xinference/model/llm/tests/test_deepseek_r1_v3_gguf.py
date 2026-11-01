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


def _shards(folder, stem, total):
    return {
        f"{folder}/{stem}-{part:05d}-of-{total:05d}.gguf"
        for part in range(1, total + 1)
    }


# GGUF shards in unsloth/DeepSeek-R1-GGUF at Hugging Face commit
# 4dab99fdb3fea560e5a402b49c6ef11e025321a8 (the ModelScope mirror uses the same
# names). Q8_0 and BF16 put a dot between the model name and the quantization.
R1_PARTS = {
    "UD-IQ1_S": 3,
    "UD-IQ1_M": 4,
    "UD-IQ2_XXS": 4,
    "UD-Q2_K_XL": 5,
    "Q2_K": 5,
    "Q2_K_L": 5,
    "Q2_K_XS": 5,
    "Q3_K_M": 7,
    "Q4_K_M": 9,
    "Q5_K_M": 10,
    "Q6_K": 12,
}
R1_FILES = set().union(
    *(_shards(f"DeepSeek-R1-{q}", f"DeepSeek-R1-{q}", n) for q, n in R1_PARTS.items()),
    _shards("DeepSeek-R1-Q8_0", "DeepSeek-R1.Q8_0", 15),
    _shards("DeepSeek-R1-BF16", "DeepSeek-R1.BF16", 30),
)

# GGUF shards in unsloth/DeepSeek-V3-GGUF at Hugging Face commit
# 6b9a45d8a30b48660644560df90e5ded5c0cb0e9 (same names on ModelScope).
V3_PARTS = {
    "Q2_K_L": 5,
    "Q2_K_XS": 5,
    "Q3_K_M": 7,
    "Q4_K_M": 9,
    "Q5_K_M": 10,
    "Q6_K": 12,
}
V3_FILES = set().union(
    *(_shards(f"DeepSeek-V3-{q}", f"DeepSeek-V3-{q}", n) for q, n in V3_PARTS.items()),
    _shards("DeepSeek-V3-Q8_0", "DeepSeek-V3-BF16-256x20B-Q8_0", 16),
)


@pytest.mark.parametrize(
    "model_name, files",
    [("deepseek-r1", R1_FILES), ("deepseek-v3", V3_FILES)],
)
def test_deepseek_gguf_files_exist(model_name, files):
    from ...utils import generate_model_file_names_with_quantization_parts
    from ..llm_family import get_llm_family_by_name

    family = get_llm_family_by_name(model_name)
    specs = [s for s in family.model_specs if s.model_format == "ggufv2"]
    assert specs
    missing = []
    for spec in specs:
        file_names, _, _ = generate_model_file_names_with_quantization_parts(spec)
        missing += [(spec.quantization, n) for n in file_names if n not in files]
    assert missing == []
