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


@pytest.mark.parametrize("model_name", ["gemma-3-it", "gemma-3-1b-it"])
def test_gemma3_mlx_quantizations_name_existing_repos(model_name):
    # mlx-community publishes gemma-3-{1b,4b,12b,27b}-it-{4bit,6bit,8bit,bf16}
    # on both hubs; there are no -fp16 repos.
    from ..llm_family import BUILTIN_LLM_FAMILIES

    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == model_name)
    mlx = [s for s in family.model_specs if s.model_format == "mlx"]
    assert mlx
    for spec in mlx:
        assert spec.quantization in {"4bit", "6bit", "8bit", "bf16"}
        assert spec.model_id.endswith(f"-it-{spec.quantization}")
