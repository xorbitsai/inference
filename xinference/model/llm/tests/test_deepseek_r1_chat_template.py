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

TOOL_CALL = {
    "id": "call",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
}


def _render(model_name, messages):
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == model_name)
    mixin = ChatModelMixin()
    mixin.tool_parser = None
    mixin.model_family = SimpleNamespace(
        model_name=model_name, model_family=None, model_ability=family.model_ability
    )
    return mixin.get_full_context(messages, family.chat_template)


def test_deepseek_r1_uses_the_deepseek_r1_template():
    # Markup of the chat template in deepseek-ai/DeepSeek-R1's tokenizer_config.json:
    # every system message is kept, and an assistant turn with both content and
    # tool calls renders both.
    prompt = _render(
        "deepseek-r1",
        [
            {"role": "system", "content": "SYS_ONE"},
            {"role": "system", "content": "SYS_TWO"},
            {"role": "user", "content": "weather in Beijing?"},
            {"role": "assistant", "content": "LOOKING", "tool_calls": [TOOL_CALL]},
            {"role": "tool", "tool_call_id": "call", "content": "SUNNY_25C"},
        ],
    )

    assert "SYS_ONE\n\nSYS_TWO<｜User｜>weather in Beijing?" in prompt
    assert (
        "<｜Assistant｜>LOOKING<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function"
        '<｜tool▁sep｜>get_weather\n```json\n{"city": "Beijing"}\n```'
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"
    ) in prompt


def test_deepseek_r1_distill_qwen_uses_the_distill_template():
    # Markup of the chat template in the deepseek-ai/DeepSeek-R1-Distill-*
    # tokenizer_config.json files: the last system message is used, and every
    # tool output after the first starts on a new line.
    prompt = _render(
        "deepseek-r1-distill-qwen",
        [
            {"role": "system", "content": "SYS_ONE"},
            {"role": "system", "content": "SYS_TWO"},
            {"role": "user", "content": "weather in Beijing and Shanghai?"},
            {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
            {"role": "tool", "tool_call_id": "call", "content": "SUNNY_25C"},
            {"role": "tool", "tool_call_id": "call", "content": "RAIN_18C"},
        ],
    )

    assert "SYS_ONE" not in prompt
    assert "SYS_TWO<｜User｜>weather in Beijing and Shanghai?" in prompt
    assert (
        "<｜tool▁output▁begin｜>SUNNY_25C<｜tool▁output▁end｜>"
        "\n<｜tool▁output▁begin｜>RAIN_18C<｜tool▁output▁end｜>"
    ) in prompt
