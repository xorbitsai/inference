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


def test_role_markers_are_followed_by_newlines():
    # Same output as the official chat_template.jinja of
    # zai-org/GLM-4.1V-9B-Thinking.
    name = "glm-4.1v-thinking"
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == name)
    mixin = ChatModelMixin()
    mixin.tool_parser = None
    mixin.model_family = SimpleNamespace(
        model_name=name, model_family=None, model_ability=family.model_ability
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is in the image?"},
            ],
        },
        {"role": "assistant", "content": "A cat."},
        {"role": "user", "content": "What color is it?"},
    ]

    prompt = mixin.get_full_context(messages, family.chat_template)

    assert prompt == (
        "[gMASK]<sop><|system|>\nYou are a helpful assistant."
        "<|user|>\n<|begin_of_image|><|image|><|end_of_image|>What is in the image?"
        "<|assistant|>\nA cat.<|user|>\nWhat color is it?<|assistant|>\n"
    )
