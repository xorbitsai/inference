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
# GGUF files for the registered quantizations in
# MaziyarPanahi/Mistral-Large-Instruct-2407-GGUF at Hugging Face commit
# b33370a735c0cde1776fd0bd244a511555e4fa79 (its only upload, 2024-07-26).
# Q2_K is a single file; the others are gguf-split shards named
# "<model>.<quantization>.gguf-0000N-of-00007.gguf".
GGUF_FILES = {"Mistral-Large-Instruct-2407.Q2_K.gguf"} | {
    f"Mistral-Large-Instruct-2407.{quantization}.gguf-{part:05d}-of-00007.gguf"
    for quantization in ("Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M")
    for part in range(1, 8)
}


def test_mistral_large_instruct_gguf_files_exist():
    from ...utils import generate_model_file_names_with_quantization_parts
    from ..llm_family import get_llm_family_by_name

    family = get_llm_family_by_name("mistral-large-instruct")
    specs = [s for s in family.model_specs if s.model_format == "ggufv2"]
    assert specs
    missing = []
    for spec in specs:
        file_names, _, _ = generate_model_file_names_with_quantization_parts(spec)
        missing += [
            (spec.quantization, name) for name in file_names if name not in GGUF_FILES
        ]
    assert missing == []
