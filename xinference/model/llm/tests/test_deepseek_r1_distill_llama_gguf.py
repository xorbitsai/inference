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
# GGUF files for the registered quantizations, as listed in
# unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF at Hugging Face commit
# 732dd974083ea5877d7b6d788b36fe7c2e5eab36 and
# unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF at commit
# 615f8936e16dfde29dcc00be71145d4d5ce8ed53. The ModelScope mirrors have the
# same file names.
GGUF_FILES = {
    "unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF": {
        "DeepSeek-R1-Distill-Llama-70B-Q2_K.gguf",
        "DeepSeek-R1-Distill-Llama-70B-Q2_K_L.gguf",
        "DeepSeek-R1-Distill-Llama-70B-Q3_K_M.gguf",
        "DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf",
        "DeepSeek-R1-Distill-Llama-70B-Q5_K_M.gguf",
        "DeepSeek-R1-Distill-Llama-70B-Q6_K/DeepSeek-R1-Distill-Llama-70B-Q6_K-00001-of-00002.gguf",
        "DeepSeek-R1-Distill-Llama-70B-Q6_K/DeepSeek-R1-Distill-Llama-70B-Q6_K-00002-of-00002.gguf",
        "DeepSeek-R1-Distill-Llama-70B-Q8_0/DeepSeek-R1-Distill-Llama-70B-Q8_0-00001-of-00002.gguf",
        "DeepSeek-R1-Distill-Llama-70B-Q8_0/DeepSeek-R1-Distill-Llama-70B-Q8_0-00002-of-00002.gguf",
        "DeepSeek-R1-Distill-Llama-70B-F16/DeepSeek-R1-Distill-Llama-70B-F16-00001-of-00003.gguf",
        "DeepSeek-R1-Distill-Llama-70B-F16/DeepSeek-R1-Distill-Llama-70B-F16-00002-of-00003.gguf",
        "DeepSeek-R1-Distill-Llama-70B-F16/DeepSeek-R1-Distill-Llama-70B-F16-00003-of-00003.gguf",
    },
    "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF": {
        "DeepSeek-R1-Distill-Llama-8B-Q2_K.gguf",
        "DeepSeek-R1-Distill-Llama-8B-Q2_K_L.gguf",
        "DeepSeek-R1-Distill-Llama-8B-Q3_K_M.gguf",
        "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
        "DeepSeek-R1-Distill-Llama-8B-Q5_K_M.gguf",
        "DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf",
        "DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf",
        "DeepSeek-R1-Distill-Llama-8B-F16.gguf",
    },
}


def test_deepseek_r1_distill_llama_gguf_files_exist():
    from ...utils import generate_model_file_names_with_quantization_parts
    from ..llm_family import get_llm_family_by_name

    family = get_llm_family_by_name("deepseek-r1-distill-llama")
    specs = [s for s in family.model_specs if s.model_format == "ggufv2"]
    assert specs
    missing = []
    for spec in specs:
        file_names, _, _ = generate_model_file_names_with_quantization_parts(spec)
        missing += [
            (spec.model_hub, spec.quantization, name)
            for name in file_names
            if name not in GGUF_FILES[spec.model_id]
        ]
    assert missing == []


def test_deepseek_r1_distill_llama_8b_gguf_matches_its_size():
    from ..llm_family import match_llm

    for hub in ("huggingface", "modelscope"):
        family = match_llm(
            "deepseek-r1-distill-llama",
            model_format="ggufv2",
            model_size_in_billions=8,
            download_hub=hub,
        )
        assert family is not None
        assert (
            family.model_specs[0].model_id
            == "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF"
        )
