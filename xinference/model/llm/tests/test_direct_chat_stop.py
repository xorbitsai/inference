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
from copy import deepcopy
from types import SimpleNamespace

import pytest

from ..transformers.multimodal.minicpmv47 import MiniCPMV47Model


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "parts,stop,expected",
    [
        (["hello<END>discarded"], "<END>", "hello"),
        (["hello<", "EN", "D>discarded", "more"], "<END>", "hello"),
        (["helloAB"], ["B", "A"], "hello"),
        (["hello\n\n"], "\n", "hello"),
        (["<END>discarded"], "<END>", ""),
        (["hello", "<END>"], ["missing", "<END>"], "hello"),
        (["hello<EN"], "<END>", "hello<EN"),
        (["hello<EN", "ough"], "<END>", "hello<ENough"),
        (["hello", " world"], None, "hello world"),
        (["hello", " world"], [], "hello world"),
        (["hello", " world"], "absent", "hello world"),
    ],
)
def test_direct_chat_stop_strings(monkeypatch, stream, parts, stop, expected):
    model = object.__new__(MiniCPMV47Model)
    model.model_uid = "stop-test"
    model.reasoning_parser = model.tool_parser = None
    model._tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text))
    consumed = []

    def backend():
        for part in parts:
            consumed.append(part)
            yield part

    monkeypatch.setattr(model, "build_streaming_iter", lambda *args: (backend(), 3))
    config = {
        "stream": stream,
        "stop": stop,
        "stream_options": {"include_usage": True},
    }
    original = deepcopy(config)
    result = model.chat([{"role": "user", "content": "hello"}], config)
    if stream:
        chunks = list(result)
        text = "".join(
            choice["delta"].get("content") or ""
            for chunk in chunks
            for choice in chunk["choices"]
        )
        assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
        assert chunks[-1]["choices"] == []
        usage = chunks[-1]["usage"]
    else:
        text = result["choices"][0]["message"]["content"]
        assert result["choices"][0]["finish_reason"] == "stop"
        usage = result["usage"]
    assert text == expected
    assert usage == {
        "prompt_tokens": 3,
        "completion_tokens": len(expected),
        "total_tokens": 3 + len(expected),
    }
    assert config == original
    assert consumed == parts
