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

# GGUF files in unsloth/QwQ-32B-GGUF (Hugging Face and the ModelScope mirror).
# Q8_0 is the only single file with a dot before the quantization.
QWQ_32B_FILES = {
    f"QwQ-32B-{q}.gguf"
    for q in (
        "IQ4_NL", "IQ4_XS", "Q2_K", "Q2_K_L", "Q3_K_M", "Q3_K_S", "Q4_0", "Q4_1",
        "Q4_K_M", "Q5_K_M", "Q6_K", "UD-IQ1_M", "UD-IQ1_S", "UD-IQ2_M",
        "UD-IQ2_XXS", "UD-IQ3_XXS", "UD-Q2_K_XL", "UD-Q3_K_XL", "UD-Q4_K_XL",
        "UD-Q5_K_XL", "UD-Q6_K_XL", "UD-Q8_K_XL",
    )
} | {
    "QwQ-32B.Q8_0.gguf",
    "BF16/QwQ-32B-BF16-00001-of-00002.gguf",
    "BF16/QwQ-32B-BF16-00002-of-00002.gguf",
}  # fmt: skip


def test_qwq_32b_gguf_files_exist():
    from ...utils import generate_model_file_names_with_quantization_parts
    from ..llm_family import get_llm_family_by_name

    family = get_llm_family_by_name("QwQ-32B")
    specs = [s for s in family.model_specs if s.model_format == "ggufv2"]
    assert "Q8_0" in {s.quantization for s in specs}
    missing = []
    for spec in specs:
        names, _, _ = generate_model_file_names_with_quantization_parts(spec)
        missing += [(spec.quantization, n) for n in names if n not in QWQ_32B_FILES]
    assert missing == []
