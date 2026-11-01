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

# GGUF files in tencent/Hy-MT2-7B-GGUF on Hugging Face and
# Tencent-Hunyuan/Hy-MT2-7B-GGUF on ModelScope. File names are case-sensitive,
# and Q6_K/Q8_0 are spelled "HY-".
HY_MT2_7B_FILES = {
    "Hy-MT2-7B-Q4_K_M.gguf",
    "HY-MT2-7B-Q6_K.gguf",
    "HY-MT2-7B-Q8_0.gguf",
}


def test_hy_mt2_7b_gguf_files_exist():
    from ...utils import generate_model_file_names_with_quantization_parts
    from ..llm_family import get_llm_family_by_name

    family = get_llm_family_by_name("Hy-MT2-7B")
    specs = [s for s in family.model_specs if s.model_format == "ggufv2"]
    assert {s.quantization for s in specs} == {"Q4_K_M", "Q6_K", "Q8_0"}
    missing = []
    for spec in specs:
        names, _, _ = generate_model_file_names_with_quantization_parts(spec)
        missing += [(spec.quantization, n) for n in names if n not in HY_MT2_7B_FILES]
    assert missing == []
