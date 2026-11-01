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


def test_qwen25_instruct_72b_mlx_specs_use_72b_repos():
    from ..llm_family import get_llm_family_by_name

    family = get_llm_family_by_name("qwen2.5-instruct")
    specs = [
        s
        for s in family.model_specs
        if s.model_format == "mlx" and s.model_size_in_billions == 72
    ]
    assert {(s.model_hub, s.quantization) for s in specs} >= {("modelscope", "2bit")}
    for spec in specs:
        assert "72B" in spec.model_id, (spec.model_hub, spec.model_id)
        assert spec.model_metadata.num_hidden_layers == 80
