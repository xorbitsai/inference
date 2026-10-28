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


@pytest.mark.parametrize("name", ["qwen2.5-coder-instruct", "XiYanSQL-QwenCoder-2504"])
def test_tools_prompt_has_single_brace_call_example(name):
    # Same markup as the official chat_template of Qwen/Qwen2.5-Coder-7B-Instruct.
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == name)
    mixin = ChatModelMixin()
    mixin.tool_parser = None
    mixin.model_family = SimpleNamespace(
        model_name=name, model_family=None, model_ability=family.model_ability
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "weather in Beijing?"}]

    prompt = mixin.get_full_context(messages, family.chat_template, tools=tools)

    assert "{{" not in prompt
    assert (
        '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call><|im_end|>\n"
    ) in prompt
