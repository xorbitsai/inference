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


def _render(messages):
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "deepseek-v3")
    mixin = ChatModelMixin()
    mixin.model_family = SimpleNamespace(
        model_name=family.model_name,
        model_ability=family.model_ability,
        chat_template=family.chat_template,
    )
    return mixin.get_full_context(
        messages, chat_template=family.chat_template, tokenizer=None
    )


def test_deepseek_v3_keeps_system_prompt_and_official_markup():
    # Same markup as the chat template in deepseek-ai/DeepSeek-V3's
    # tokenizer_config.json: no spaces around the special tokens.
    prompt = _render(
        [
            {"role": "system", "content": "SYS_PROMPT"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
        ]
    )

    assert prompt == (
        "SYS_PROMPT<｜User｜>hi<｜Assistant｜>hello<｜end▁of▁sentence｜>"
        "<｜User｜>again<｜Assistant｜>"
    )


def test_deepseek_v3_renders_tool_calls_for_its_tool_parser():
    prompt = _render(
        [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "SUNNY"},
        ]
    )

    # The call format the deepseek_v3 tool parser reads.
    assert (
        "<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function"
        '<｜tool▁sep｜>get_weather\n```json\n{"city": "Paris"}\n```<｜tool▁call▁end｜>'
    ) in prompt
    assert (
        "<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>SUNNY"
        "<｜tool▁output▁end｜><｜tool▁outputs▁end｜>"
    ) in prompt
