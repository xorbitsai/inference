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

from ...llm_family import BUILTIN_LLM_FAMILIES
from ...utils import QWEN_TOOL_CALL_FAMILY, ChatModelMixin
from .. import TOOL_PARSERS
from ..abstract_tool_parser import request_tools_var
from ..seed_oss_tool_parser import SeedOssToolParser

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "days": {"type": "integer"},
                },
            },
        },
    }
]


def _call(name, **args):
    params = "".join(f"<parameter={k}>{v}</parameter>\n" for k, v in args.items())
    return (
        f"\n<seed:tool_call>\n<function={name}>\n{params}</function>\n</seed:tool_call>"
    )


def _family():
    return next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "seed-oss")


def test_seed_oss_family_accepts_tool_calls():
    family = _family()
    assert "tools" in family.model_ability
    assert isinstance(TOOL_PARSERS[family.tool_parser](), SeedOssToolParser)
    assert family.model_name in QWEN_TOOL_CALL_FAMILY


def test_parses_text_and_every_function():
    token = request_tools_var.set(TOOLS)
    try:
        result = SeedOssToolParser().extract_tool_calls(
            "Checking."
            + _call("get_weather", location="404", days=3)
            + _call("get_weather", location="Line one\nline two")
        )
    finally:
        request_tools_var.reset(token)
    assert result[0][0] == "Checking.\n"
    assert [r[1:] for r in result[1:] if r[1]] == [
        ("get_weather", {"location": "404", "days": 3}),
        ("get_weather", {"location": "Line one\nline two"}),
    ]


def test_output_without_a_tool_call_stays_text():
    assert SeedOssToolParser().extract_tool_calls("It is sunny.") == [
        ("It is sunny.", None, None)
    ]


def test_unfinished_tool_call_stays_text():
    text = "<seed:tool_call>\n<function=get_weather>\n<parameter=location>Bei"
    assert SeedOssToolParser().extract_tool_calls(text) == [(text, None, None)]


def test_template_format_round_trips_through_the_parser():
    family = _family()
    mixin = ChatModelMixin()
    mixin.model_family = SimpleNamespace(
        model_name=family.model_name,
        model_family=None,
        model_ability=family.model_ability,
    )
    prompt = mixin.get_full_context(
        [
            {"role": "user", "content": "Weather in Beijing?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"location": "Beijing"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "Sunny"},
        ],
        family.chat_template,
        tools=TOOLS,
    )
    assert "get_weather" in prompt
    assert "Sunny" in prompt

    start = prompt.index("<seed:tool_call>\n<function=get_weather>")
    end = prompt.index("</seed:tool_call>", start) + len("</seed:tool_call>")
    assert SeedOssToolParser().extract_tool_calls(prompt[start:end]) == [
        (None, "get_weather", {"location": "Beijing"})
    ]
