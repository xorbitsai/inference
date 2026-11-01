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

# Single-file quantizations in each Hugging Face repo. TheBloke uploaded the
# 67B Q6_K and Q8_0 files as raw byte chunks (``*.gguf-split-a``/``-b``) that
# must be concatenated by hand, so they are not loadable downloads.
REPO_QUANTIZATIONS = {
    ("deepseek", 67): {
        "Q2_K", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q4_0", "Q4_K_M", "Q4_K_S",
        "Q5_0", "Q5_K_M", "Q5_K_S",
    },
    ("deepseek-chat", 67): {
        "Q2_K", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q4_0", "Q4_K_M", "Q4_K_S",
        "Q5_0", "Q5_K_M", "Q5_K_S",
    },
    ("deepseek-coder-instruct", 7): {
        "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S",
        "Q6_K", "Q8_0",
    },
    ("deepseek-coder", 7): {
        "Q2_K", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q4_K_M", "Q4_K_S", "Q5_K_M",
        "Q5_K_S", "Q6_K", "Q8_0",
    },
}  # fmt: skip


@pytest.mark.parametrize("name, size", list(REPO_QUANTIZATIONS))
def test_deepseek_gguf_quantizations_exist(name, size):
    from ..llm_family import get_llm_family_by_name

    family = get_llm_family_by_name(name)
    listed = {
        s.quantization
        for s in family.model_specs
        if s.model_format == "ggufv2" and s.model_size_in_billions == size
    }
    assert listed == REPO_QUANTIZATIONS[(name, size)]
