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

from types import SimpleNamespace

import pytest
import torch

from ....scheduler.request import InferenceRequest
from .. import utils as batch_utils
from ..utils import _get_token_from_logits


def _make_request() -> InferenceRequest:
    request = InferenceRequest("prompt", object(), True, "generate", {})
    request.prompt_tokens = [1, 2, 3]
    return request


@pytest.mark.parametrize(
    "temperature,repetition_penalty,top_p,top_k",
    [
        (0.0, 1.0, 1.0, -1),
        (0.0, 1.0, 1.0, 2),
    ],
)
def test_get_token_from_batched_logits(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
):
    logits = torch.zeros((2, 1, 4))
    logits[1, -1, 3] = 10

    token = _get_token_from_logits(
        _make_request(),
        1,
        logits,
        temperature,
        repetition_penalty,
        top_p,
        top_k,
    )

    assert token == 3


def _make_legacy_cache(batch_size: int, seq_len: int):
    return (
        (
            torch.zeros((batch_size, 2, seq_len, 4)),
            torch.zeros((batch_size, 2, seq_len, 4)),
        ),
    )


class _TokenStreamModel:
    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch_size, seq_len = input_ids.shape
        token = (seq_len + 1) % 10
        logits = torch.full((batch_size, seq_len, 12), -100.0)
        logits[:, :, token] = 100.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=_make_legacy_cache(batch_size, seq_len + 1),
        )


class _TokenStreamTokenizer:
    eos_token_id = 11

    def decode(self, tokens, **kwargs):
        return "".join(str(token % 10) for token in tokens)


class _TokenStreamRuntime:
    model_uid = "length-limit-model"

    def __init__(self, context_length: int = 32):
        self.context_length = context_length

    def get_context_len(self) -> int:
        return self.context_length

    def get_builtin_stop_token_ids(self):
        return ()

    @staticmethod
    def get_batch_size_and_seq_len_indexes_from_kv():
        return 0, 2

    def build_prefill_kwargs(self, prompts, req_list):
        for request in req_list:
            if request.prompt_tokens is None:
                request.prompt_tokens = list(range(14))
        token_count = len(req_list[0].prompt_tokens)
        return {"input_ids": torch.ones((len(req_list), token_count), dtype=torch.long)}

    def build_decode_kwargs(self, prompts, req_list, batch_size, seq_len):
        return {"input_ids": torch.ones((len(prompts), 1), dtype=torch.long)}


def _make_token_limit_request(
    max_tokens,
    *,
    stream: bool,
    include_usage: bool = False,
    set_prompt_tokens: bool = True,
    stop=None,
):
    generate_config = {
        "max_tokens": max_tokens,
        "stream": stream,
        "stream_interval": 1,
        "stop": stop,
    }
    if include_usage:
        generate_config["stream_options"] = {"include_usage": True}
    request = InferenceRequest("prompt", object(), True, "generate", generate_config)
    request.sanitized_generate_config = {
        "max_tokens": max_tokens,
        "stream_interval": 1,
        "stream_options": generate_config.get("stream_options"),
        "stop": stop,
        "temperature": 0.0,
    }
    if set_prompt_tokens:
        request.prompt_tokens = list(range(14))
    return request


def _run_step(monkeypatch, request, *, decode_round: int):
    monkeypatch.setattr(
        batch_utils,
        "_get_token_from_logits",
        lambda req, i, logits, *args: int(torch.argmax(logits[i, -1]).item()),
    )
    runtime = _TokenStreamRuntime()
    batch_utils._batch_inference_one_step_internal(
        runtime,
        [request],
        runtime.model_uid,
        _TokenStreamModel(),
        _TokenStreamTokenizer(),
        decode_round=decode_round,
    )


def test_none_max_tokens_persists_across_decode_steps(monkeypatch):
    request = _make_token_limit_request(None, stream=False)

    _run_step(monkeypatch, request, decode_round=15)

    assert request.effective_max_new_tokens == 18
    assert request.visible_new_tokens_count == 16
    assert not request.stopped

    _run_step(monkeypatch, request, decode_round=15)

    assert request.stopped
    assert request.finish_reason == "length"
    assert request.visible_new_tokens_count == 18
    assert len(request.new_tokens) == 31
    assert request.completion[0]["choices"][0]["text"] == "5" + "2" * 17
    assert request.completion[0]["usage"] == {
        "prompt_tokens": 14,
        "completion_tokens": 18,
        "total_tokens": 32,
    }


def test_stream_output_and_usage_do_not_exceed_max_tokens(monkeypatch):
    request = _make_token_limit_request(3, stream=True, include_usage=True)

    _run_step(monkeypatch, request, decode_round=5)

    assert request.stopped
    assert request.finish_reason == "length"
    assert request.visible_new_tokens_count == 3
    assert len(request.new_tokens) == 6
    text_chunks = [
        chunk["choices"][0]["text"]
        for chunk in request.completion
        if isinstance(chunk, dict) and chunk.get("choices")
    ]
    assert "".join(text_chunks) == "522"
    assert request.completion[-1]["choices"] == []
    assert request.completion[-1]["usage"] == {
        "prompt_tokens": 14,
        "completion_tokens": 3,
        "total_tokens": 17,
    }


def test_none_max_tokens_is_resolved_after_fresh_request_prefill(monkeypatch):
    request = _make_token_limit_request(None, stream=False, set_prompt_tokens=False)

    _run_step(monkeypatch, request, decode_round=1)

    assert request.prompt_tokens == list(range(14))
    assert request.effective_max_new_tokens == 18
    assert request.visible_new_tokens_count == 2
    assert not request.stopped


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("stop", ["ZZ", "2"])
def test_stop_strings_cannot_observe_tokens_beyond_max_tokens(
    monkeypatch, stream, stop
):
    request = _make_token_limit_request(
        1, stream=stream, include_usage=stream, stop=stop
    )

    _run_step(monkeypatch, request, decode_round=2)

    if stream:
        text = "".join(
            chunk["choices"][0]["text"]
            for chunk in request.completion
            if isinstance(chunk, dict) and chunk.get("choices")
        )
        usage = request.completion[-1]["usage"]
    else:
        text = request.completion[0]["choices"][0]["text"]
        usage = request.completion[0]["usage"]
    assert text == "5"
    assert request.finish_reason == "length"
    assert usage == {
        "prompt_tokens": 14,
        "completion_tokens": 1,
        "total_tokens": 15,
    }


def test_zero_elapsed_time_does_not_fail(monkeypatch):
    request = _make_token_limit_request(1, stream=False)
    monkeypatch.setattr(batch_utils.time, "time", lambda: 1.0)

    _run_step(monkeypatch, request, decode_round=1)

    assert request.stopped
    assert request.finish_reason == "length"
