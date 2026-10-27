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
from ..llm_family import BUILTIN_LLM_FAMILIES


def test_hy_mt2_7b_stop_token_ids_are_generation_config_eos():
    # eos_token_id of tencent/Hy-MT2-7B generation_config.json (<|eos|>,
    # <|extra_5|>). 120020 is the 1.8B eos; in the 7B vocabulary it is the
    # ordinary token for "胸口".
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "Hy-MT2-7B")
    assert family.stop_token_ids == [127960, 127967]
