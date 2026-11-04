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
import re
from types import SimpleNamespace

import pytest

from ..llm_family import BUILTIN_LLM_FAMILIES
from ..utils import ChatModelMixin


@pytest.mark.parametrize(
    "model_name",
    ["qwen1.5-chat", "qwen1.5-moe-chat", "qwen2-instruct", "qwen2-moe-instruct"],
)
@pytest.mark.parametrize("calls", [1, 2])
def test_tool_call_history_is_one_json_object_per_tag(model_name, calls):
    # The system prompt these templates write asks for one
    # <tool_call>{"name": ..., "arguments": ...}</tool_call> block per call, so
    # history calls should be rendered the same way.
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == model_name)
    mixin = ChatModelMixin()
    mixin.tool_parser = None
    mixin.model_family = SimpleNamespace(
        model_name=model_name, model_family=None, model_ability=family.model_ability
    )
    tool_calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": f"tool_{i}", "arguments": {"n": i}},
        }
        for i in range(1, calls + 1)
    ]
    messages = [
        {"role": "user", "content": "run the tools"},
        {"role": "assistant", "content": None, "tool_calls": tool_calls},
    ] + [
        {
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "name": f"tool_{i}",
            "content": "ok",
        }
        for i in range(1, calls + 1)
    ]
    prompt = mixin.get_full_context(messages, family.chat_template)

    expected = "".join(
        f'\n<tool_call>\n{{"name": "tool_{i}", "arguments": {{"n": {i}}}}}\n</tool_call>'
        for i in range(1, calls + 1)
    )
    assert f"<|im_start|>assistant{expected}<|im_end|>\n" in prompt
    blocks = re.findall(
        r"<tool_call>\n(.*?)\n</tool_call>", prompt.split("<|im_start|>assistant")[-2]
    )
    assert [json.loads(b) for b in blocks] == [
        {"name": f"tool_{i}", "arguments": {"n": i}} for i in range(1, calls + 1)
    ]
