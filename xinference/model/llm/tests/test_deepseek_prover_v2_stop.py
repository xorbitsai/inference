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


def test_deepseek_prover_v2_has_no_family_wide_stop_ids():
    # The family's stop_token_ids are added to the eos ids of whichever size is
    # loaded (transformers and MLX engines). The 671B and 7B tokenizers differ:
    # id 1 is <｜end▁of▁sentence｜> for the 671B but the `"` character for the
    # 7B, whose eos is 100001. No id is an end token in both, so the family
    # lists none and each size stops on its own eos token.
    from ..llm_family import BUILTIN_LLM_FAMILIES

    family = next(
        f for f in BUILTIN_LLM_FAMILIES if f.model_name == "deepseek-prover-v2"
    )
    assert not family.stop_token_ids
    assert family.stop == ["<｜end▁of▁sentence｜>"]
