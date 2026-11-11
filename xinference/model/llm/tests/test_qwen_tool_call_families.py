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
import json
from types import SimpleNamespace

import pytest

from ..llm_family import BUILTIN_LLM_FAMILIES
from ..tool_parsers import TOOL_PARSERS
from ..utils import ChatModelMixin

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}


def _family(name):
    return next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == name)


@pytest.mark.parametrize("name", ["qwen2.5-instruct-1m", "QwQ-32B-Preview"])
def test_family_with_qwen_tool_markup_accepts_tool_calls(name):
    from ..utils import QWEN_TOOL_CALL_FAMILY

    family = _family(name)
    assert "tools" in family.model_ability
    assert family.model_name in QWEN_TOOL_CALL_FAMILY

    mixin = ChatModelMixin()
    mixin.model_family = SimpleNamespace(
        model_name=family.model_name,
        model_family=None,
        model_ability=family.model_ability,
    )
    prompt = mixin.get_full_context(
        [{"role": "user", "content": "Weather in Beijing?"}],
        family.chat_template,
        tools=[TOOL],
    )
    assert json.dumps(TOOL["function"]["name"]) in prompt
    assert "<tool_call>" in prompt

    output = (
        '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Beijing"}}'
        "\n</tool_call>"
    )
    assert TOOL_PARSERS[family.tool_parser]().extract_tool_calls(output) == [
        (None, "get_weather", {"location": "Beijing"})
    ]
