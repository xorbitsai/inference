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
import asyncio
import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ...core import LLM
from ...llm_family import BUILTIN_LLM_FAMILIES
from ...utils import QWEN_TOOL_CALL_FAMILY, ChatModelMixin
from .. import TOOL_PARSERS
from ..abstract_tool_parser import request_tools_var
from ..seed_oss_tool_parser import SeedOssToolParser
from .test_streaming_partial_markers import _splits, _stream

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


def _model(reasoning_content):
    model = ChatModelMixin()
    model.tool_parser = SeedOssToolParser()
    model.reasoning_parser = None
    if reasoning_content is not None:
        family = SimpleNamespace(model_family=_family())
        LLM.prepare_parse_reasoning_content(family, reasoning_content)
        model.reasoning_parser = family.reasoning_parser
    model.model_family = None
    model.model_uid = "test-model"
    return model


def _completion_fields(choice):
    return {
        "id": "test",
        "object": "text_completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, **choice}],
    }


async def _complete(model, text, finish_reason):
    result = model._post_process_completion(
        None,
        "test-model",
        _completion_fields({"text": text, "finish_reason": finish_reason}),
        tools=TOOLS,
    )
    choice = result["choices"][0]
    calls = [
        (call["function"]["name"], json.loads(call["function"]["arguments"]))
        for call in choice["message"]["tool_calls"]
    ]
    return choice["message"]["content"], calls, choice["finish_reason"]


async def _stream_response(model, text, finish_reason):
    async def chunks():
        # The reasoning tags are single tokens, so they arrive whole.
        for part in re.split(r"(</?seed:think>)", text):
            size = len(part) if "seed:think" in part else 7
            for start in range(0, len(part), size or 1):
                piece = part[start : start + size]
                yield _completion_fields({"text": piece, "finish_reason": None})
        yield _completion_fields({"text": "", "finish_reason": finish_reason})

    content, calls, reason = "", [], None
    async for chunk in model._async_to_tool_completion_chunks(chunks(), tools=TOOLS):
        for choice in chunk["choices"]:
            content += choice["delta"].get("content") or ""
            calls += [
                (call["function"]["name"], json.loads(call["function"]["arguments"]))
                for call in choice["delta"].get("tool_calls") or []
            ]
            reason = choice["finish_reason"] or reason
    return content, calls, reason


RESPONSE_MODES = [_complete, _stream_response]
QUOTED_CALL = "<seed:think>Maybe" + _call("get_weather", location="Paris")
TRUNCATED_CALL = (
    "<seed:tool_call>\n<function=get_weather>\n"
    "<parameter=location>Beijing</parameter>\n</function>\n"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_content", [None, False, True])
@pytest.mark.parametrize("respond", RESPONSE_MODES)
async def test_a_call_quoted_in_reasoning_is_not_emitted(respond, reasoning_content):
    text = QUOTED_CALL + "</seed:think>It is sunny."
    content, calls, finish_reason = await respond(
        _model(reasoning_content), text, "stop"
    )
    assert calls == []
    assert finish_reason == "stop"
    assert content.endswith("It is sunny.")


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_content", [None, False, True])
@pytest.mark.parametrize("respond", RESPONSE_MODES)
async def test_only_a_call_after_reasoning_is_emitted(respond, reasoning_content):
    text = QUOTED_CALL + "</seed:think>" + _call("get_weather", location="Beijing")
    _, calls, finish_reason = await respond(_model(reasoning_content), text, "stop")
    assert calls == [("get_weather", {"location": "Beijing"})]
    assert finish_reason == "tool_calls"


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_content", [None, False, True])
@pytest.mark.parametrize("respond", RESPONSE_MODES)
async def test_output_cut_off_in_reasoning_keeps_length(respond, reasoning_content):
    _, calls, finish_reason = await respond(
        _model(reasoning_content), QUOTED_CALL, "length"
    )
    assert calls == []
    assert finish_reason == "length"


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_content", [None, False, True])
@pytest.mark.parametrize("respond", RESPONSE_MODES)
@pytest.mark.parametrize(
    "text",
    [
        TRUNCATED_CALL,
        "<seed:think>Checking.</seed:think>" + TRUNCATED_CALL,
    ],
    ids=["no-reasoning", "after-reasoning"],
)
async def test_call_cut_off_before_its_closing_tag_keeps_length(
    text, respond, reasoning_content
):
    _, calls, finish_reason = await respond(_model(reasoning_content), text, "length")
    assert calls == []
    assert finish_reason == "length"


@pytest.mark.asyncio
@pytest.mark.parametrize("respond", RESPONSE_MODES)
async def test_a_call_from_output_without_reasoning_is_emitted(respond):
    # thinking_budget=0 closes reasoning in the prompt, so the output has no tag.
    text = _call("get_weather", location="Beijing")
    content, calls, finish_reason = await respond(_model(False), text, "stop")
    assert calls == [("get_weather", {"location": "Beijing"})]
    assert finish_reason == "tool_calls"
    assert "seed:think" not in content


def test_every_chunk_split_keeps_reasoning_out_of_tool_calls():
    reasoning = QUOTED_CALL + "</seed:think>"
    text = reasoning + "Now." + _call("get_weather", location="Beijing")
    for chunks in _splits(text):
        assert _stream(SeedOssToolParser(), chunks) == (
            reasoning + "Now.\n",
            [("get_weather", {"location": "Beijing"})],
        )


def test_a_call_may_quote_the_reasoning_tag_in_an_argument():
    token = request_tools_var.set(TOOLS)
    try:
        text = _call("get_weather", location="<seed:think>") + "Done."
        assert SeedOssToolParser().extract_tool_calls(text) == [
            ("\n", None, None),
            (None, "get_weather", {"location": "<seed:think>"}),
            ("Done.", None, None),
        ]
        for chunks in _splits(text):
            assert _stream(SeedOssToolParser(), chunks) == (
                "\nDone.",
                [("get_weather", {"location": "<seed:think>"})],
            )
    finally:
        request_tools_var.reset(token)


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


def _stub_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")


def _prompt(render, tools):
    messages = [{"role": "user", "content": "hi"}]
    chat_template = _family().chat_template
    if render == "vllm":
        from ...vllm.core import VLLMChatModel

        model = VLLMChatModel.__new__(VLLMChatModel)
        model.model_family = _family()
        model.reasoning_parser = None
        model.lora_requests = []
        model._get_tokenizer = AsyncMock(return_value=_stub_tokenizer())
        return asyncio.run(model._prepare_chat_inputs(messages, {"tools": tools}))[0]
    if render == "raw":
        mixin = ChatModelMixin()
        mixin.tool_parser = None
        mixin.model_family = _family()
        return mixin.get_full_context(messages, chat_template, tools=tools)
    # Call apply_chat_template directly: get_full_context falls back to the raw
    # template when tokenizer rendering raises, which would hide a failure.
    return _stub_tokenizer().apply_chat_template(
        messages,
        tools=tools,
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=True,
    )


@pytest.mark.parametrize("render", ["vllm", "raw", "tokenizer"])
@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"parameters": {"type": "object"}},
        {"parameters": None},
        {"parameters": {"type": "object", "properties": None}},
        {"description": None},
    ],
)
def test_a_tool_without_arguments_is_listed(render, fields):
    tools = [{"type": "function", "function": {"name": "get_time", **fields}}]

    assert 'Function:def get_time():    """        """' in _prompt(render, tools)


@pytest.mark.parametrize("render", ["vllm", "raw", "tokenizer"])
@pytest.mark.parametrize("required,marker", [(["city"], "必填"), (None, "选填")])
def test_a_tool_with_arguments_marks_the_required_ones(render, required, marker):
    parameters = {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "A city"}},
        "required": required,
    }
    tools = [{"type": "function", "function": {"name": "f", "parameters": parameters}}]

    prompt = _prompt(render, tools)

    assert "Function:def f(city: str):" in prompt
    assert f"- city (str) [{marker}]: A city" in prompt
