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
from types import SimpleNamespace

import pytest

from ..llm_family import BUILTIN_LLM_FAMILIES
from ..utils import ChatModelMixin

MESSAGES = [
    {"role": "system", "content": "Translate into English."},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "Hello"},
    {"role": "user", "content": "谢谢"},
]


@pytest.mark.parametrize(
    "name, expected",
    [
        # Official chat_template.jinja of tencent/Hy-MT2-7B (HunYuanDenseV1).
        (
            "Hy-MT2-7B",
            "<|startoftext|>Translate into English.<|extra_4|>你好<|extra_0|>"
            "Hello<|eos|><|startoftext|>谢谢<|extra_0|>",
        ),
        # Official chat_template.jinja of tencent/Hy-MT2-30B-A3B (HYV3).
        (
            "Hy-MT2-30B-A3B",
            "<｜hy_begin▁of▁sentence｜>Translate into English."
            "<｜reasoning_mode｜>reasoning_effort:no_think"
            "<｜hy_User｜>你好<｜hy_Assistant｜><think></think>Hello<eos:6124c78e>"
            "<｜hy_User｜>谢谢<｜hy_Assistant｜><think></think>",
        ),
    ],
)
def test_prompt_uses_the_model_own_special_tokens(name, expected):
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == name)
    mixin = ChatModelMixin()
    mixin.tool_parser = None
    mixin.model_family = SimpleNamespace(
        model_name=name, model_family=None, model_ability=family.model_ability
    )

    assert mixin.get_full_context(MESSAGES, family.chat_template) == expected
