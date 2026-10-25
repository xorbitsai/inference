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
import torch
from transformers import BatchEncoding

from ....scheduler.request import InferenceRequest
from ..chatglm import ChatglmPytorchChatModel


def _completion_chunk(text, finish_reason=None, usage=None):
    chunk = {
        "id": "cmpl-test",
        "object": "text_completion",
        "created": 1,
        "model": "chatglm-test",
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        chunk["choices"] = []
        chunk["usage"] = usage
    return chunk


def test_chatglm_stream_include_usage_ends_with_usage_only_chunk():
    request = InferenceRequest(
        [{"role": "user", "content": "hi"}],
        object(),
        False,
        "chat",
        {"stream": True},
    )
    request.sanitized_generate_config = {"stream_options": {"include_usage": True}}
    request.stopped = True
    request.finish_reason = "length"
    usage = {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }
    request.completion = [
        "<bos_stream>",
        _completion_chunk("answer"),
        _completion_chunk("", finish_reason="length"),
        "<eos_stream>",
        _completion_chunk(None, usage=usage),
    ]
    request.outputs = ["answer", "<eos_stream>"]

    model = object.__new__(ChatglmPytorchChatModel)
    model.handle_chat_result_streaming(request)

    assert request.completion[-2]["choices"][0]["finish_reason"] == "length"
    assert request.completion[-1]["choices"] == []
    assert request.completion[-1]["usage"] == usage


@pytest.mark.parametrize("stream_options", [{}, {"include_obfuscation": False}])
def test_chatglm_chat_stream_options_without_include_usage(stream_options):
    model = object.__new__(ChatglmPytorchChatModel)
    model.model_uid = "chatglm-test"
    model.model_family = SimpleNamespace(model_name="glm4-chat", chat_template=None)
    model._model = SimpleNamespace(device="cpu")
    model._tokenizer = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: BatchEncoding(
            {"input_ids": torch.tensor([[1, 2, 3]])}
        )
    )
    model.reasoning_parser = model.tool_parser = None
    model._stream_chat = lambda *args, **kwargs: iter(["hello"])

    chunks = list(
        model.chat(
            [{"role": "user", "content": "hi"}],
            {"stream": True, "stream_options": stream_options},
        )
    )

    assert all(chunk["choices"] for chunk in chunks)
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert text == "hello"
