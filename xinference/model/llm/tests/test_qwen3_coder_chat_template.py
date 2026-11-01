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

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {"type": "array", "items": {"type": "string"}},
                    "mode": {"type": "string", "enum": ["fast", "full"]},
                },
                "required": ["paths"],
            },
        },
    }
]


def test_qwen3_coder_renders_lists_as_json():
    # Same markup as the chat_template.jinja of the Qwen/Qwen3-Coder-* repos:
    # list values in tool schemas and tool-call arguments go through tojson.
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "Qwen3-Coder")
    mixin = ChatModelMixin()
    mixin.tool_parser = None
    mixin.model_family = SimpleNamespace(
        model_name=family.model_name,
        model_family=None,
        model_ability=family.model_ability,
    )
    messages = [
        {"role": "user", "content": "find it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"paths": ["src", "tests"], "mode": "fast"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call", "content": "NO_MATCH"},
    ]

    prompt = mixin.get_full_context(messages, family.chat_template, tools=TOOLS)

    assert "\n\n# Tools\n\nYou have access to the following functions:\n\n" in prompt
    assert '<enum>["fast", "full"]</enum>' in prompt
    assert '<required>["paths"]</required>' in prompt
    assert '<parameter=paths>\n["src", "tests"]\n</parameter>' in prompt
    assert "['src', 'tests']" not in prompt
