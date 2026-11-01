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

FAMILIES = ["deepseek-r1", "deepseek-r1-distill-qwen"]

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


def _stub_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")


def _render_with(render, model_name, messages):
    if render == "raw":
        return _render(model_name, messages)
    # Call apply_chat_template directly: get_full_context falls back to the raw
    # template when tokenizer rendering raises, which would hide a failure.
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == model_name)
    return _stub_tokenizer().apply_chat_template(
        messages,
        chat_template=family.chat_template,
        tokenize=False,
        add_generation_prompt=True,
    )


def _call(i):
    return {
        "id": f"call_{i}",
        "type": "function",
        "function": {"name": f"tool_{i}", "arguments": f'{{"n": {i}}}'},
    }


def _call_markup(i):
    return (
        f"<｜tool▁call▁begin｜>function<｜tool▁sep｜>tool_{i}\n"
        f'```json\n{{"n": {i}}}\n```<｜tool▁call▁end｜>'
    )


@pytest.mark.parametrize("model_name", FAMILIES)
@pytest.mark.parametrize("render", ["raw", "tokenizer"])
@pytest.mark.parametrize("tool_calls", ["omitted", None, []])
def test_assistant_history_without_tool_calls_keeps_its_text(
    model_name, render, tool_calls
):
    assistant = {"role": "assistant", "content": "DONE"}
    if tool_calls != "omitted":
        assistant["tool_calls"] = tool_calls
    prompt = _render_with(
        render,
        model_name,
        [
            {"role": "user", "content": "hi"},
            assistant,
            {"role": "user", "content": "again"},
        ],
    )

    assert "<｜Assistant｜>DONE<｜end▁of▁sentence｜><｜User｜>again" in prompt
    assert "<｜tool▁calls▁begin｜>" not in prompt


@pytest.mark.parametrize("model_name", FAMILIES)
@pytest.mark.parametrize("render", ["raw", "tokenizer"])
@pytest.mark.parametrize("content", [None, "", "LOOKING"])
@pytest.mark.parametrize("calls", [1, 2, 3])
def test_tool_call_history_keeps_calls_and_text(model_name, render, content, calls):
    messages = [
        {"role": "user", "content": "run the tools"},
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [_call(i) for i in range(1, calls + 1)],
        },
    ] + [
        {"role": "tool", "tool_call_id": f"call_{i}", "content": f"OUT_{i}"}
        for i in range(1, calls + 1)
    ]
    prompt = _render_with(render, model_name, messages)

    assert (
        "<｜Assistant｜>"
        + (content or "")
        + "<｜tool▁calls▁begin｜>"
        + "\n".join(_call_markup(i) for i in range(1, calls + 1))
        + "<｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁outputs▁begin｜>"
    ) in prompt
    assert prompt.count("<｜tool▁calls▁end｜>") == 1
    assert prompt.endswith("<｜tool▁outputs▁end｜>")


@pytest.mark.parametrize("model_name", FAMILIES)
@pytest.mark.parametrize("render", ["raw", "tokenizer"])
def test_anthropic_text_and_tool_use_history_keeps_the_call(model_name, render):
    from ....api.protocols import parse_anthropic_request

    request = parse_anthropic_request(
        {
            "model": "test",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "read a"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "LOOKING"},
                        {
                            "type": "tool_use",
                            "id": "call_read",
                            "name": "Read",
                            "input": {"path": "a"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_read",
                            "content": "RESULT_42",
                        }
                    ],
                },
            ],
        }
    )
    prompt = _render_with(render, model_name, request.messages)

    assert (
        "<｜Assistant｜>LOOKING<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function"
        '<｜tool▁sep｜>Read\n```json\n{"path": "a"}\n```<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜><｜end▁of▁sentence｜>"
    ) in prompt
    assert "RESULT_42" in prompt
