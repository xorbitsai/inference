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

# The default system prompt of the chat template in Qwen/QwQ-32B-Preview.
QWQ_SYSTEM = (
    "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. "
    "You should think step-by-step."
)


def _render(messages, tools=None):
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "QwQ-32B-Preview")
    mixin = ChatModelMixin()
    mixin.tool_parser = None
    mixin.model_family = SimpleNamespace(
        model_name=family.model_name,
        model_family=None,
        model_ability=family.model_ability,
    )
    return mixin.get_full_context(messages, family.chat_template, tools=tools)


def test_qwq_32b_preview_default_system_prompt():
    prompt = _render([{"role": "user", "content": "1+1?"}])

    assert prompt.startswith(f"<|im_start|>system\n{QWQ_SYSTEM}<|im_end|>\n")
    assert "Alibaba Cloud" not in prompt


def test_qwq_32b_preview_default_system_prompt_with_tools():
    tools = [{"type": "function", "function": {"name": "calc", "parameters": {}}}]
    prompt = _render([{"role": "user", "content": "1+1?"}], tools=tools)

    assert prompt.startswith(f"<|im_start|>system\n{QWQ_SYSTEM}\n\n# Tools\n\n")
