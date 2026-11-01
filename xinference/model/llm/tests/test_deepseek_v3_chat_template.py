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


def _family():
    return next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "deepseek-v3")


def _stub_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")


def _render(render, messages):
    family = _family()
    if render == "raw":
        mixin = ChatModelMixin()
        mixin.tool_parser = None
        mixin.model_family = SimpleNamespace(
            model_name=family.model_name,
            model_family=None,
            model_ability=family.model_ability,
        )
        return mixin.get_full_context(messages, family.chat_template)
    # Call apply_chat_template directly: get_full_context falls back to the raw
    # template when tokenizer rendering raises, which would hide a failure.
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


def _output_markup(i):
    return f"<｜tool▁output▁begin｜>RESULT_{i}<｜tool▁output▁end｜>"


RENDERS = ["raw", "tokenizer"]


@pytest.mark.parametrize("render", RENDERS)
def test_deepseek_v3_keeps_system_prompt_and_official_markup(render):
    # Same markup as the chat template in deepseek-ai/DeepSeek-V3's
    # tokenizer_config.json: no spaces around the special tokens.
    prompt = _render(
        render,
        [
            {"role": "system", "content": "SYS_PROMPT"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
        ],
    )

    assert prompt == (
        "SYS_PROMPT<｜User｜>hi<｜Assistant｜>hello<｜end▁of▁sentence｜>"
        "<｜User｜>again<｜Assistant｜>"
    )


@pytest.mark.parametrize("render", RENDERS)
@pytest.mark.parametrize("tool_calls", ["omitted", None, []])
def test_assistant_history_without_tool_calls_keeps_its_text(render, tool_calls):
    assistant = {"role": "assistant", "content": "DONE"}
    if tool_calls != "omitted":
        assistant["tool_calls"] = tool_calls
    prompt = _render(
        render,
        [
            {"role": "user", "content": "hi"},
            assistant,
            {"role": "user", "content": "again"},
        ],
    )

    assert prompt == (
        "<｜User｜>hi<｜Assistant｜>DONE<｜end▁of▁sentence｜>"
        "<｜User｜>again<｜Assistant｜>"
    )


@pytest.mark.parametrize("render", RENDERS)
@pytest.mark.parametrize("content", [None, "", "Checking."])
@pytest.mark.parametrize("n_calls", [1, 2, 3])
def test_parallel_tool_calls_render_one_balanced_group(render, content, n_calls):
    # /v1/responses input combines assistant text and the following function
    # calls into one message, so the text must be kept along with the calls.
    ids = range(1, n_calls + 1)
    messages = [
        {"role": "user", "content": "Q"},
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [_call(i) for i in ids],
        },
    ]
    messages += [
        {"role": "tool", "tool_call_id": f"call_{i}", "content": f"RESULT_{i}"}
        for i in ids
    ]

    prompt = _render(render, messages)

    assert prompt == (
        "<｜User｜>Q<｜Assistant｜>"
        + (content or "")
        + "<｜tool▁calls▁begin｜>"
        + "\n".join(_call_markup(i) for i in ids)
        + "<｜tool▁calls▁end｜><｜end▁of▁sentence｜>"
        + "<｜tool▁outputs▁begin｜>"
        + "\n".join(_output_markup(i) for i in ids)
        + "<｜tool▁outputs▁end｜>"
    )


@pytest.mark.parametrize("render", RENDERS)
def test_sequential_tool_rounds_each_open_and_close_their_groups(render):
    prompt = _render(
        render,
        [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "Checking.", "tool_calls": [_call(1)]},
            {"role": "tool", "tool_call_id": "call_1", "content": "RESULT_1"},
            {"role": "assistant", "content": None, "tool_calls": [_call(2), _call(3)]},
            {"role": "tool", "tool_call_id": "call_2", "content": "RESULT_2"},
            {"role": "tool", "tool_call_id": "call_3", "content": "RESULT_3"},
            {"role": "assistant", "content": "", "tool_calls": [_call(4)]},
            {"role": "tool", "tool_call_id": "call_4", "content": "RESULT_4"},
            {"role": "assistant", "content": "DONE"},
            {"role": "user", "content": "again"},
        ],
    )

    assert prompt == (
        "SYS<｜User｜>Q<｜Assistant｜>"
        "Checking.<｜tool▁calls▁begin｜>"
        + _call_markup(1)
        + "<｜tool▁calls▁end｜><｜end▁of▁sentence｜>"
        "<｜tool▁outputs▁begin｜>" + _output_markup(1) + "<｜tool▁outputs▁end｜>"
        "<｜tool▁calls▁begin｜>"
        + _call_markup(2)
        + "\n"
        + _call_markup(3)
        + "<｜tool▁calls▁end｜><｜end▁of▁sentence｜>"
        "<｜tool▁outputs▁begin｜>"
        + _output_markup(2)
        + "\n"
        + _output_markup(3)
        + "<｜tool▁outputs▁end｜>"
        "<｜tool▁calls▁begin｜>"
        + _call_markup(4)
        + "<｜tool▁calls▁end｜><｜end▁of▁sentence｜>"
        "<｜tool▁outputs▁begin｜>" + _output_markup(4) + "<｜tool▁outputs▁end｜>"
        "DONE<｜end▁of▁sentence｜><｜User｜>again<｜Assistant｜>"
    )


@pytest.mark.parametrize("render", RENDERS)
def test_user_turn_after_tool_results_closes_the_output_group(render):
    prompt = _render(
        render,
        [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": None, "tool_calls": [_call(1)]},
            {"role": "tool", "tool_call_id": "call_1", "content": "RESULT_1"},
            {"role": "user", "content": "again"},
        ],
    )

    assert prompt == (
        "<｜User｜>Q<｜Assistant｜><｜tool▁calls▁begin｜>"
        + _call_markup(1)
        + "<｜tool▁calls▁end｜><｜end▁of▁sentence｜>"
        "<｜tool▁outputs▁begin｜>"
        + _output_markup(1)
        + "<｜tool▁outputs▁end｜><｜User｜>again<｜Assistant｜>"
    )
