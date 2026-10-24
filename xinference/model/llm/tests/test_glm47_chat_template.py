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


@pytest.mark.parametrize("name", ["GLM-4.7", "GLM-4.7-Flash"])
def test_tool_history_uses_closing_and_response_tags(name):
    # Same markup as the official chat_template.jinja of zai-org/GLM-4.7.
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
    messages = [
        {"role": "user", "content": "weather in Beijing?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Beijing"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call", "content": "SUNNY_25C"},
    ]

    prompt = mixin.get_full_context(messages, family.chat_template, tools=tools)

    assert (
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>Beijing</arg_value></tool_call>"
        "<|observation|><tool_response>SUNNY_25C</tool_response><|assistant|>"
    ) in prompt
    assert prompt.startswith("[gMASK]<sop><|system|>\n# Tools\n\n")
