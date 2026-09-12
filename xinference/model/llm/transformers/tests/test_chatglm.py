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
