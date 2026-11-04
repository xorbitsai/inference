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


def _render(render, model_name, messages, tools=TOOLS):
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == model_name)
    if render == "raw":
        mixin = ChatModelMixin()
        mixin.tool_parser = None
        mixin.model_family = SimpleNamespace(
            model_name=model_name,
            model_family=None,
            model_ability=family.model_ability,
        )
        return mixin.get_full_context(messages, family.chat_template, tools=tools)
    # Call apply_chat_template directly: get_full_context falls back to the raw
    # template when tokenizer rendering raises, which would hide a failure.
    return _stub_tokenizer().apply_chat_template(
        messages,
        tools=tools,
        chat_template=family.chat_template,
        tokenize=False,
        add_generation_prompt=True,
    )


def _listed_tools(render, model_name, tools):
    prompt = _render(render, model_name, [{"role": "user", "content": "hi"}], tools)
    listed = prompt.split("<tools> ", 1)[1].split(" </tools>", 1)[0]
    return [json.loads(entry) for entry in listed.split("\n")]


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
    assert _listed_tools(render, model_name, TOOLS) == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "get_weather(city: str)\n\n    Args:\n        city(str): ",
                "parameters": TOOLS[0]["function"]["parameters"],
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "get_time()\n\n",
                "parameters": {},
            },
        },
    ]


@pytest.mark.parametrize("model_name", FAMILIES)
@pytest.mark.parametrize("render", ["raw", "tokenizer"])
def test_a_null_description_is_left_out(model_name, render):
    tools = [{"type": "function", "function": {"name": "f", "description": None}}]

    listed = _listed_tools(render, model_name, tools)

    assert listed[0]["function"]["description"] == "f()\n\n"


@pytest.mark.parametrize("model_name", FAMILIES)
@pytest.mark.parametrize("render", ["raw", "tokenizer"])
def test_a_tool_without_properties_lists_empty_parameters(model_name, render):
    parameters = {"type": "object", "properties": {}}
    tools = [{"type": "function", "function": {"name": "f", "parameters": parameters}}]

    assert _listed_tools(render, model_name, tools)[0]["function"]["parameters"] == {}


@pytest.mark.parametrize("model_name", FAMILIES)
@pytest.mark.parametrize("render", ["raw", "tokenizer"])
def test_quotes_in_a_tool_definition_stay_inside_its_entry(model_name, render):
    description = 'Say "hi"\nand leave'
    parameters = {
        "type": "object",
        "properties": {"q": {"type": "string", "description": 'a "quoted" value'}},
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "say",
                "description": description,
                "parameters": parameters,
            },
        }
    ]

    assert _listed_tools(render, model_name, tools) == [
        {
            "type": "function",
            "function": {
                "name": "say",
                "description": (
                    f"say(q: str) - {description}\n\n"
                    '    Args:\n        q(str): a "quoted" value'
                ),
                "parameters": parameters,
            },
        }
    ]


@pytest.mark.parametrize("model_name", FAMILIES)
@pytest.mark.parametrize("render", ["raw", "tokenizer"])
def test_parameter_schemas_are_named_in_the_signature(model_name, render):
    schemas = {
        "nullable": ({"type": ["string", "null"]}, "Union[str,None]"),
        "null": ({"type": "null"}, "None"),
        "unknown": ({"type": "file"}, "Any"),
        "strings": ({"type": "array", "items": {"type": "string"}}, "list[str]"),
        "nullable_items": (
            {"type": "array", "items": {"type": ["integer", "null"]}},
            "list[Union[int,None]]",
        ),
        "no_items": ({"type": "array"}, "list[Any]"),
        "no_type": ({"enum": ["a", "b"]}, "Any"),
        "any_value": (
            {"type": "object", "additionalProperties": True},
            "dict[str, Any]",
        ),
    }
    parameters = {
        "type": "object",
        "properties": {name: schema for name, (schema, _) in schemas.items()},
    }
    tools = [{"type": "function", "function": {"name": "f", "parameters": parameters}}]

    function = _listed_tools(render, model_name, tools)[0]["function"]

    signature = ", ".join(f"{name}: {named}" for name, (_, named) in schemas.items())
    assert function["description"].startswith(f"f({signature})\n\n")
    assert function["parameters"] == parameters
