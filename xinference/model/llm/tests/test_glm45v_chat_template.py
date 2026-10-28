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


def test_tools_prompt_keeps_official_blank_lines():
    # Same markup as the official chat_template.jinja of zai-org/GLM-4.5V.
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "glm-4.5v")
    mixin = ChatModelMixin()
    mixin.tool_parser = None
    mixin.model_family = SimpleNamespace(
        model_name="glm-4.5v", model_family=None, model_ability=family.model_ability
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

    assert prompt.startswith(
        "[gMASK]<sop><|system|>\n# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:"
    )
    assert (
        "</tools>\n\nFor each function call, output the function name and "
        "arguments within the following XML format:"
    ) in prompt
