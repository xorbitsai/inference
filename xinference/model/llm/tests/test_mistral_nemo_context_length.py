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


def test_mistral_nemo_context_length_matches_its_config():
    # mistralai/Mistral-Nemo-Instruct-2407 config.json: max_position_embeddings
    # 131072 since commit 8aedd450 ("fix position embeddings"); it was 1024000
    # before. The llama.cpp engine uses context_length as the default n_ctx.
    from ..llm_family import BUILTIN_LLM_FAMILIES

    family = next(
        f for f in BUILTIN_LLM_FAMILIES if f.model_name == "mistral-nemo-instruct"
    )
    assert family.context_length == 131072
