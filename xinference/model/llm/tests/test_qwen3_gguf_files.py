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

# unsloth/Qwen3-235B-A22B-GGUF at Hugging Face commit
# 09e11417ffdc30c1c63d0296a40fd8fde0abb180: every quantization is split, in
# "<quant>/Qwen3-235B-A22B-<quant>-0000N-of-0000M.gguf"; there is no Q4_0 or
# IQ4_NL. The ModelScope mirror has the same files.
QWEN3_235B_PARTS = {
    "BF16": 10, "IQ4_XS": 3, "Q2_K": 2, "Q2_K_L": 2, "Q3_K_M": 3, "Q3_K_S": 3,
    "Q4_1": 3, "Q4_K_M": 3, "Q4_K_S": 3, "Q5_K_M": 4, "Q5_K_S": 4, "Q6_K": 4,
    "Q8_0": 6, "UD-Q2_K_XL": 2, "UD-Q3_K_XL": 3, "UD-Q4_K_XL": 3,
    "UD-Q5_K_XL": 4, "UD-Q6_K_XL": 4, "UD-Q8_K_XL": 6,
}  # fmt: skip
QWEN3_235B_FILES = {
    f"{q}/Qwen3-235B-A22B-{q}-{i:05d}-of-{n:05d}.gguf"
    for q, n in QWEN3_235B_PARTS.items()
    for i in range(1, n + 1)
}

# unsloth/Qwen3-8B-GGUF at commit a6adef130ffb23ddaf1a62fec9dced968c9bc482:
# single files, no Q4_0.
QWEN3_8B_FILES = {
    f"Qwen3-8B-{q}.gguf"
    for q in (
        "BF16", "IQ4_NL", "IQ4_XS", "Q2_K", "Q2_K_L", "Q3_K_M", "Q3_K_S", "Q4_1",
        "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "UD-IQ1_M",
        "UD-IQ1_S", "UD-IQ2_M", "UD-IQ2_XXS", "UD-IQ3_XXS", "UD-Q2_K_XL",
        "UD-Q3_K_XL", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "UD-Q8_K_XL",
    )
}  # fmt: skip


@pytest.mark.parametrize(
    "model_id, files",
    [
        ("unsloth/Qwen3-235B-A22B-GGUF", QWEN3_235B_FILES),
        ("unsloth/Qwen3-8B-GGUF", QWEN3_8B_FILES),
    ],
)
def test_qwen3_gguf_files_exist(model_id, files):
    from ...utils import generate_model_file_names_with_quantization_parts
    from ..llm_family import get_llm_family_by_name

    family = get_llm_family_by_name("qwen3")
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
