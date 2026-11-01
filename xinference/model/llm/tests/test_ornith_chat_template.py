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

ORNITH_FAMILIES = ["Ornith-1.0-35B", "Ornith-1.5-35B-A3B", "Ornith-1.5-397B"]


def _render(model_name, messages, tools=None):
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == model_name)
    mixin = ChatModelMixin()
    mixin.tool_parser = None
    mixin.model_family = SimpleNamespace(
        model_name=model_name, model_family=None, model_ability=family.model_ability
    )
    return mixin.get_full_context(messages, family.chat_template, tools=tools)


@pytest.mark.parametrize("model_name", ORNITH_FAMILIES)
def test_ornith_keeps_reasoning_of_earlier_turns(model_name):
    # The official Ornith chat templates render reasoning_content on every
    # assistant turn, not only on the turns after the last user query.
    prompt = _render(
        model_name,
        [
            {"role": "user", "content": "first question"},
            {
                "role": "assistant",
                "content": "first answer",
                "reasoning_content": "EARLY_REASONING",
            },
            {"role": "user", "content": "second question"},
        ],
    )

    assert (
        "<|im_start|>assistant\n<think>\nEARLY_REASONING\n</think>\n\n"
        "first answer<|im_end|>\n<|im_start|>user\nsecond question"
    ) in prompt


@pytest.mark.parametrize("model_name", ["Ornith-1.0-35B", "Ornith-1.5-35B-A3B"])
def test_ornith_35b_renders_non_string_arguments_as_json(model_name):
    # Their official templates pass every non-string argument through tojson.
    prompt = _render(
        model_name,
        [
            {"role": "user", "content": "search"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"exact": true, "limit": 5}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call", "content": "NO_RESULTS"},
        ],
    )

    assert "<parameter=exact>\ntrue\n</parameter>" in prompt
    assert "<parameter=limit>\n5\n</parameter>" in prompt
