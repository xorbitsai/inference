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

# Single-file GGUFs in bartowski/THUDM_GLM-4-32B-0414-GGUF (Hugging Face and
# the ModelScope mirror).
GLM_4_32B_FILES = {
    f"THUDM_GLM-4-32B-0414-{q}.gguf"
    for q in (
        "IQ2_M", "IQ2_S", "IQ2_XS", "IQ3_M", "IQ3_XS", "IQ3_XXS", "IQ4_NL",
        "IQ4_XS", "Q2_K", "Q2_K_L", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q3_K_XL",
        "Q4_0", "Q4_1", "Q4_K_L", "Q4_K_M", "Q4_K_S", "Q5_K_L", "Q5_K_M",
        "Q5_K_S", "Q6_K", "Q6_K_L", "Q8_0",
    )
}  # fmt: skip


def test_glm4_0414_32b_gguf_downloads_the_32b_model():
    from ...utils import generate_model_file_names_with_quantization_parts
    from ..llm_family import get_llm_family_by_name

    family = get_llm_family_by_name("glm4-0414")
    specs = [
        s
        for s in family.model_specs
        if s.model_format == "ggufv2" and s.model_size_in_billions == 32
    ]
    pytorch = next(
        s
        for s in family.model_specs
        if s.model_format == "pytorch" and s.model_size_in_billions == 32
    )
    assert specs
    for spec in specs:
        assert spec.model_id == "bartowski/THUDM_GLM-4-32B-0414-GGUF"
        names, _, _ = generate_model_file_names_with_quantization_parts(spec)
        assert set(names) <= GLM_4_32B_FILES
        for field in ("num_hidden_layers", "hidden_size", "num_attention_heads"):
            assert getattr(spec.model_metadata, field) == getattr(
                pytorch.model_metadata, field
            )
