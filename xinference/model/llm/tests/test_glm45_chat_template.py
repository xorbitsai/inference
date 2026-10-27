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

from ..llm_family import BUILTIN_LLM_FAMILIES
from ..utils import ChatModelMixin


def test_tool_history_keeps_official_newlines():
    # Same markup as the official chat_template.jinja of zai-org/GLM-4.5.
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "glm-4.5")
    mixin = ChatModelMixin()
    mixin.tool_parser = None
    mixin.model_family = SimpleNamespace(
        model_name="glm-4.5", model_family=None, model_ability=family.model_ability
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

    assert prompt.startswith(
        "[gMASK]<sop><|system|>\n# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
    )
    assert (
        "<|user|>\nweather in Beijing?<|assistant|>\n<think></think>\n"
        "<tool_call>get_weather\n<arg_key>city</arg_key>\n"
        "<arg_value>Beijing</arg_value>\n</tool_call>"
        "<|observation|>\n<tool_response>\nSUNNY_25C\n</tool_response><|assistant|>"
    ) in prompt
