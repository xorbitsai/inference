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

# Model GGUF files in the SeaLLMs repos on Hugging Face. The v2.5 repo also has
# a ChatML variant (seallm-7b-v2.5-chatml.Q4_K_M.gguf); the plain file uses the
# SeaLLM-7B-v2.5 prompt format.
REPO_FILES = {
    "seallm_v2": {"SeaLLM-7B-v2.q4_0.gguf", "SeaLLM-7B-v2.q8_0.gguf"},
    "seallm_v2.5": {"seallm-7b-v2.5.Q4_K_M.gguf"},
}


@pytest.mark.parametrize("name", sorted(REPO_FILES))
def test_seallm_gguf_files_exist(name):
    from ...utils import generate_model_file_names_with_quantization_parts
    from ..llm_family import get_llm_family_by_name

    family = get_llm_family_by_name(name)
    specs = [s for s in family.model_specs if s.model_format == "ggufv2"]
    assert specs
    for spec in specs:
        names, _, _ = generate_model_file_names_with_quantization_parts(spec)
        assert set(names) <= REPO_FILES[name], spec.quantization
