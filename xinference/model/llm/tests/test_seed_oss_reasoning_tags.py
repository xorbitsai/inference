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
from types import SimpleNamespace

import pytest

from ..core import LLM, chat_context_var
from ..llm_family import BUILTIN_LLM_FAMILIES
from ..reasoning_parser import ReasoningParser
from ..utils import ChatModelMixin

# The format in ByteDance-Seed/Seed-OSS-36B-Instruct's model card. The tags are
# added tokens with "special": false, so decoding keeps them in the text.
OUTPUT = ["<seed:think>", "Two", " plus two.", "</seed:think>", "It", " is 4."]


def _parser():
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "seed-oss")
    return ReasoningParser(True, family.reasoning_start_tag, family.reasoning_end_tag)


def test_seed_oss_reasoning_is_split_from_the_answer():
    assert _parser().extract_reasoning_content("".join(OUTPUT)) == (
        "Two plus two.",
        "It is 4.",
    )


def test_seed_oss_streamed_reasoning_is_split_from_the_answer():
    parser = _parser()
    previous = ""
    reasoning = content = ""
    for delta in OUTPUT:
        current = previous + delta
        chunk = parser.extract_reasoning_content_streaming(previous, current, delta)
        reasoning += chunk.get("reasoning_content") or ""
        content += chunk.get("content") or ""
        previous = current
    assert (reasoning, content) == ("Two plus two.", "It is 4.")


def _completion(text, finish_reason):
    return {
        "id": "test",
        "object": "text_completion",
        "created": 0,
        "model": "seed-oss",
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
        "usage": None,
    }


def _family_parser():
    family = SimpleNamespace(
        model_family=next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "seed-oss")
    )
    LLM.prepare_parse_reasoning_content(family, True)
    return family.reasoning_parser


def _respond(parser, output, stream, context):
    """Return (reasoning, content) from the chat response or the stream."""
    token = chat_context_var.set(context)
    try:
        if not stream:
            message = ChatModelMixin._to_chat_completion(
                _completion("".join(output), "stop"), parser
            )["choices"][0]["message"]
            return message.get("reasoning_content") or "", message["content"] or ""

        async def chunks():
            for delta in output:
                yield _completion(delta, None)
            yield _completion("", "stop")

        async def collect():
            return [
                c
                async for c in ChatModelMixin._async_to_chat_completion_chunks(
                    chunks(), parser
                )
            ]

        deltas = [c["choices"][0]["delta"] for c in asyncio.run(collect())]
        return (
            "".join(d.get("reasoning_content") or "" for d in deltas),
            "".join(d.get("content") or "" for d in deltas),
        )
    finally:
        chat_context_var.reset(token)


@pytest.mark.parametrize("stream", [False, True])
def test_seed_oss_answer_without_reasoning_is_content(stream):
    # With thinking_budget=0 the chat template closes reasoning in the prompt,
    # so the answer has no reasoning tag.
    assert _respond(
        _family_parser(), ["It", " is", " 4."], stream, {"thinking_budget": 0}
    ) == ("", "It is 4.")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "context", [{}, {"enable_thinking": True, "thinking_budget": 4}]
)
def test_seed_oss_start_tag_from_the_prompt_keeps_reasoning(stream, context):
    # An engine can append <seed:think> to the prompt (MLX thinking budgets,
    # structured output with thinking), so only the end tag is generated.
    output = ["Two", " plus two.", "</seed:think>", "It", " is 4."]
    assert _respond(_family_parser(), output, stream, context) == (
        "Two plus two.",
        "It is 4.",
    )


@pytest.mark.parametrize("stream", [False, True])
def test_seed_oss_budget_zero_with_a_prompt_start_tag_keeps_the_split(stream):
    # No engine appends the start tag for thinking_budget=0, but if a prompt
    # still opens reasoning, the generated end tag must keep the split.
    output = ["</seed:think>", "It", " is 4."]
    assert _respond(
        _family_parser(),
        output,
        stream,
        {"enable_thinking": True, "thinking_budget": 0},
    ) == ("", "It is 4.")
