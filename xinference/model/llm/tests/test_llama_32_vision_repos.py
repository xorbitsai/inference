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


def test_llama_32_vision_points_at_meta_repos():
    # Meta publishes the base Llama 3.2 Vision models as
    # meta-llama/Llama-3.2-{11,90}B-Vision; there is no "Meta-Llama-3.2-" repo.
    from ..llm_family import BUILTIN_LLM_FAMILIES

    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "llama-3.2-vision")
    hf_ids = {s.model_id for s in family.model_specs if s.model_hub == "huggingface"}
    assert hf_ids == {
        "meta-llama/Llama-3.2-11B-Vision",
        "meta-llama/Llama-3.2-90B-Vision",
    }
