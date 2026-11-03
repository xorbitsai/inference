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

# Quantizations with no file in the GGUF repo on Hugging Face or ModelScope.
ABSENT = [
    ("qwen3.5", 122, "UD-IQ1_S"),
    ("qwen3.5", 35, "UD-IQ1_M"),
    ("gemma-4", 26, "UD-Q3_K_S"),
    ("skywork-or1-preview", 7, "IQ2_S"),
    ("skywork-or1-preview", 7, "IQ2_XS"),
]


@pytest.mark.parametrize("name, size, quantization", ABSENT)
def test_gguf_spec_does_not_list_absent_quantization(name, size, quantization):
    from ..llm_family import get_llm_family_by_name

    family = get_llm_family_by_name(name)
    listed = {
        s.quantization
        for s in family.model_specs
        if s.model_format == "ggufv2" and s.model_size_in_billions == size
    }
    assert listed
    assert quantization not in listed
