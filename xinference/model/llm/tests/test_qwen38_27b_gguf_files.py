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

# Model GGUF files in unsloth/Qwen3.8-27B-GGUF (Hugging Face and the ModelScope
# mirror) since its 2026-08-19 re-upload.
QWEN38_27B_FILES = {
    f"Qwen3.8-27B-{q}.gguf"
    for q in (
        "Q4_0", "Q4_1", "Q8_0", "UD-IQ1_M", "UD-IQ1_S", "UD-IQ2_S",
        "UD-IQ2_XXS", "UD-IQ3_S", "UD-IQ3_XXS", "UD-IQ4_XS", "UD-Q2_K_XL",
        "UD-Q3_K_XL", "UD-Q4_K_M", "UD-Q4_K_S", "UD-Q4_K_XL", "UD-Q5_K_M",
        "UD-Q5_K_S", "UD-Q5_K_XL", "UD-Q6_K", "UD-Q6_K_L", "UD-Q6_K_M",
        "UD-Q6_K_XL", "UD-Q8_K_L", "UD-Q8_K_XL",
    )
} | {
    "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf",
    "BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf",
}  # fmt: skip


def test_qwen38_27b_gguf_files_exist():
    from ...utils import generate_model_file_names_with_quantization_parts
    from ..llm_family import get_llm_family_by_name

    family = get_llm_family_by_name("qwen3.8")
    specs = [
        s
        for s in family.model_specs
        if s.model_format == "ggufv2" and s.model_id == "unsloth/Qwen3.8-27B-GGUF"
    ]
    assert specs
    missing = []
    for spec in specs:
        names, _, _ = generate_model_file_names_with_quantization_parts(spec)
        missing += [(spec.quantization, n) for n in names if n not in QWEN38_27B_FILES]
    assert missing == []
