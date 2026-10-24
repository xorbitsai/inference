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

from typing import List, Tuple

import pytest

from ....types import Completion, CompletionChunk
from ..core import chat_context_var
from ..llm_family import match_llm
from ..transformers.core import PytorchChatModel
from ..utils import ChatModelMixin


def _parse_builtin_output(
    model_name: str, texts: List[str], streaming: bool
) -> Tuple[str, str]:
    family = match_llm(
        model_name,
        model_format="pytorch",
        quantization="none",
        download_hub="huggingface",
    )
    assert family is not None
    assert "reasoning" in family.model_ability
    model = PytorchChatModel("reasoning-tags", family, "unused")
    model.prepare_parse_reasoning_content(reasoning_content=True, enable_thinking=True)
    parser = model.reasoning_parser
    assert parser is not None
    ctx = {"enable_thinking": True, "reasoning_content": True}
    token = chat_context_var.set(ctx)
    try:
        assert parser.check_content_parser()
        if streaming:
            chunks: List[CompletionChunk] = [
                {
                    "id": "test",
                    "object": "text_completion",
                    "created": 1,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "text": text,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
                for text in texts
            ]
            chunks.append(
                {
                    "id": "test",
                    "object": "text_completion",
                    "created": 1,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "text": "",
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
            deltas = [
                chunk["choices"][0]["delta"]
                for chunk in ChatModelMixin._to_chat_completion_chunks(
                    iter(chunks), parser, ctx
                )
            ]
            return (
                "".join(delta.get("reasoning_content") or "" for delta in deltas),
                "".join(delta.get("content") or "" for delta in deltas),
            )
        completion = Completion(
            id="test",
            object="text_completion",
            created=1,
            model=model_name,
            choices=[
                {
                    "index": 0,
                    "text": "".join(texts),
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
        message = ChatModelMixin._to_chat_completion(completion, parser)["choices"][0][
            "message"
        ]
        return message.get("reasoning_content") or "", message.get("content") or ""
    finally:
        chat_context_var.reset(token)


@pytest.mark.parametrize("model_name", ["DeepSeek-V3.2", "DeepSeek-V3.2-Exp"])
@pytest.mark.parametrize("streaming", [False, True], ids=["non-streaming", "streaming"])
@pytest.mark.parametrize(
    "start_tag", ["", "<think>"], ids=["prompt-tag", "generated-tag"]
)
def test_builtin_reasoning_extraction(
    model_name: str, streaming: bool, start_tag: str
) -> None:
    # The template can supply the opening marker in the generation prompt.
    reasoning, content = _parse_builtin_output(
        model_name, [start_tag + "analysis", "</think>", "answer"], streaming
    )
    assert reasoning.strip() == "analysis"
    assert content == "answer"
