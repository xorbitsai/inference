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

# The default system prompt of the chat template in the
# FreedomIntelligence/HuatuoGPT-o1-7B and -72B tokenizer configs.
HUATUO_SYSTEM = (
    "You are HuatuoGPT-o1, created by The Chinese University of Hong Kong, "
    "Shenzhen. You are a helpful assistant."
)


def _render(messages, tools=None):
    family = next(
        f for f in BUILTIN_LLM_FAMILIES if f.model_name == "HuatuoGPT-o1-Qwen2.5"
    )
    mixin = ChatModelMixin()
    mixin.tool_parser = None
    mixin.model_family = SimpleNamespace(
        model_name=family.model_name,
        model_family=None,
        model_ability=family.model_ability,
    )
    return mixin.get_full_context(messages, family.chat_template, tools=tools)


def test_huatuogpt_o1_default_system_prompt():
    prompt = _render([{"role": "user", "content": "headache?"}])

    assert prompt.startswith(f"<|im_start|>system\n{HUATUO_SYSTEM}<|im_end|>\n")


def test_huatuogpt_o1_default_system_prompt_with_tools():
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    prompt = _render([{"role": "user", "content": "headache?"}], tools=tools)

    assert prompt.startswith(f"<|im_start|>system\n{HUATUO_SYSTEM}\n\n# Tools\n\n")
