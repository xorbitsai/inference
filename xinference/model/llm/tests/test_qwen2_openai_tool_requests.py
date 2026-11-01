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

FAMILIES = ["qwen1.5-chat", "qwen1.5-moe-chat", "qwen2-instruct", "qwen2-moe-instruct"]

# OpenAI tool definitions: description and parameters are optional.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    },
    {"type": "function", "function": {"name": "get_time"}},
]

CALLS = [
    {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
    },
    {
        "id": "call_2",
        "type": "function",
        "function": {"name": "get_time", "arguments": {}},
    },
]


def _stub_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")


def _render(render, model_name, messages):
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == model_name)
    if render == "raw":
        mixin = ChatModelMixin()
        mixin.tool_parser = None
        mixin.model_family = SimpleNamespace(
            model_name=model_name,
            model_family=None,
            model_ability=family.model_ability,
        )
        return mixin.get_full_context(messages, family.chat_template, tools=TOOLS)
    # Call apply_chat_template directly: get_full_context falls back to the raw
    # template when tokenizer rendering raises, which would hide a failure.
    return _stub_tokenizer().apply_chat_template(
        messages,
        tools=TOOLS,
        chat_template=family.chat_template,
        tokenize=False,
        add_generation_prompt=True,
    )


@pytest.mark.parametrize("model_name", FAMILIES)
@pytest.mark.parametrize("render", ["raw", "tokenizer"])
def test_openai_tool_messages_take_the_name_of_their_call(model_name, render):
    # OpenAI tool messages carry tool_call_id but no name.
    prompt = _render(
        render,
        model_name,
        [
            {"role": "user", "content": "weather and time?"},
            {"role": "assistant", "content": None, "tool_calls": CALLS},
            {"role": "tool", "tool_call_id": "call_1", "content": "SUNNY"},
            {"role": "tool", "tool_call_id": "call_2", "content": "NOON"},
        ],
    )

    assert prompt.endswith(
        "<|im_start|>user\nweather and time?<|im_end|>\n"
        "<|im_start|>assistant\n"
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call><|im_end|>\n'
        "<|im_start|>user\n<tool_response>\n"
        '{"name": "get_weather", "content": "SUNNY"}\n</tool_response><|im_end|>\n'
        "<|im_start|>user\n<tool_response>\n"
        '{"name": "get_time", "content": "NOON"}\n</tool_response><|im_end|>\n'
        "<|im_start|>assistant\n"
    )


@pytest.mark.parametrize("model_name", FAMILIES)
@pytest.mark.parametrize("render", ["raw", "tokenizer"])
def test_tool_message_without_name_or_matching_call_still_raises(model_name, render):
    with pytest.raises(Exception, match="tool_call_id"):
        _render(
            render,
            model_name,
            [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": None, "tool_calls": CALLS[:1]},
                {"role": "tool", "tool_call_id": "call_9", "content": "SUNNY"},
            ],
        )


@pytest.mark.parametrize("model_name", FAMILIES)
@pytest.mark.parametrize("render", ["raw", "tokenizer"])
def test_tools_without_description_or_parameters_are_listed(model_name, render):
    prompt = _render(render, model_name, [{"role": "user", "content": "hi"}])

    assert (
        '<tools> {"type": "function", "function": {"name": "get_weather", '
        '"description": "get_weather(city: str)\n\n'
    ) in prompt
    assert (
        '{"type": "function", "function": {"name": "get_time", '
        '"description": "get_time()\n\n", "parameters": {}} </tools>'
    ) in prompt
