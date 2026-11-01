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

# Shard counts in unsloth/Qwen3.5-397B-A17B-GGUF (Hugging Face and the
# ModelScope mirror). Every quantization is split and stored in a folder named
# after it, e.g. Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M-00001-of-00006.gguf.
SHARDS = {
    "BF16": 17, "MXFP4_MOE": 6, "Q3_K_M": 5, "Q3_K_S": 5, "Q4_K_M": 6,
    "Q4_K_S": 6, "Q5_K_M": 8, "Q5_K_S": 7, "Q6_K": 8, "Q8_0": 10,
    "UD-IQ1_M": 4, "UD-IQ2_M": 4, "UD-IQ2_XXS": 4, "UD-IQ3_S": 4,
    "UD-IQ3_XXS": 4, "UD-IQ4_NL": 5, "UD-IQ4_XS": 5, "UD-Q3_K_M": 5,
    "UD-Q3_K_S": 5, "UD-Q3_K_XL": 5, "UD-Q4_K_M": 6, "UD-Q4_K_S": 6,
    "UD-Q4_K_XL": 6, "UD-Q5_K_M": 8, "UD-Q5_K_S": 7, "UD-Q5_K_XL": 8,
    "UD-Q6_K": 8, "UD-Q6_K_XL": 9, "UD-Q8_K_XL": 10,
}  # fmt: skip


def test_qwen35_397b_gguf_files_exist():
    from ...utils import generate_model_file_names_with_quantization_parts
    from ..llm_family import get_llm_family_by_name

    family = get_llm_family_by_name("qwen3.5")
    specs = [
        s
        for s in family.model_specs
        if s.model_format == "ggufv2" and s.model_size_in_billions == 397
    ]
    assert {(s.model_hub, s.quantization) for s in specs} == {
        (hub, q) for hub in ("huggingface", "modelscope") for q in SHARDS
    }
    for spec in specs:
        q, n = spec.quantization, SHARDS[spec.quantization]
        names, _, need_merge = generate_model_file_names_with_quantization_parts(spec)
        assert need_merge
        assert names == [
            f"{q}/Qwen3.5-397B-A17B-{q}-{i:05d}-of-{n:05d}.gguf"
            for i in range(1, n + 1)
        ]
