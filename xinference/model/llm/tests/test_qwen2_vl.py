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
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ..transformers.multimodal.qwen2_vl import Qwen2VLChatModel


@pytest.fixture
def adapter():
    from transformers import BatchFeature

    model = object.__new__(Qwen2VLChatModel)
    model.model_uid = "qwen2-vl-test"
    model.reasoning_parser = None
    model._tokenizer = Mock(encode=lambda text, **kwargs: text.split())
    model.build_inputs_from_messages = Mock(
        return_value=BatchFeature({"input_ids": [[1, 2, 3]]})
    )
    model._model = SimpleNamespace(
        generate=Mock(side_effect=lambda **kw: kw["streamer"].end())
    )
    return model


def complete(adapter, stream, config=None):
    result = adapter.build_direct_chat_result([], {"stream": stream, **(config or {})})
    return list(result) if stream else result


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("temperature", [0, 1e-6, 1e-5, 0.7, None])
def test_temperature_reaches_generate(adapter, stream, temperature):
    def generate(**kwargs):
        from transformers import TemperatureLogitsWarper

        if kwargs.get("do_sample", True):
            TemperatureLogitsWarper(float(kwargs["temperature"]))
        kwargs["streamer"].on_finalized_text("answer", stream_end=True)

    adapter._model.generate.side_effect = generate
    config = {} if temperature is None else {"temperature": temperature}
    result = complete(adapter, stream, config)
    text = (
        "".join(c["choices"][0]["delta"].get("content", "") for c in result)
        if stream
        else result["choices"][0]["message"]["content"]
    )
    assert text == "answer"
    kwargs = adapter._model.generate.call_args.kwargs
    assert kwargs["do_sample"] is (temperature != 0)
    if temperature != 0:
        assert kwargs["temperature"] == (1 if temperature is None else temperature)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("stage", ["before", "partial", "ended"])
def test_producer_errors_reach_caller(adapter, monkeypatch, stream, stage):
    from transformers import TextIteratorStreamer

    stopped = Event()
    original_next = TextIteratorStreamer.__next__

    def next_text(self):
        try:
            return original_next(self)
        except StopIteration:
            stopped.set()
            raise

    monkeypatch.setattr(TextIteratorStreamer, "__next__", next_text)
    error = RuntimeError("generation failed")

    def generate(**kwargs):
        if stage == "partial":
            kwargs["streamer"].on_finalized_text("partial ")
        if stage == "ended":
            kwargs["streamer"].end()
            assert stopped.wait(5)
        raise error

    adapter._model.generate.side_effect = generate
    chunks = []
    with pytest.raises(RuntimeError) as caught:
        if stream:
            for chunk in adapter.build_direct_chat_result([], {"stream": True}):
                chunks.append(chunk)
        else:
            complete(adapter, False)
    assert caught.value is error
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks)
    if stream and stage == "partial":
        assert chunks[0]["choices"][0]["delta"]["content"] == "partial "


@pytest.mark.parametrize("stream", [False, True])
def test_empty_eos_completes_normally(adapter, stream):
    result = complete(adapter, stream)
    assert (result[-1] if stream else result)["choices"][0]["finish_reason"] == "stop"
    assert (result[-1] if stream else result)["usage"]["completion_tokens"] == 0


def test_request_errors_are_isolated(adapter):
    def generate(**kwargs):
        if kwargs["max_new_tokens"] == 1:
            raise RuntimeError("first request")
        kwargs["streamer"].end()

    adapter._model.generate.side_effect = generate
    failed, _ = adapter.build_streaming_iter([], {"max_tokens": 1})
    succeeded, _ = adapter.build_streaming_iter([], {})
    assert list(succeeded) == [""]
    with pytest.raises(RuntimeError, match="first request"):
        list(failed)


@pytest.mark.parametrize("close", [False, True])
@pytest.mark.timeout(10)
def test_iterator_does_not_wait_forever(adapter, monkeypatch, close):
    import transformers

    streamer_cls = transformers.TextIteratorStreamer
    monkeypatch.setattr(
        transformers,
        "TextIteratorStreamer",
        lambda *a, **kw: streamer_cls(*a, **{**kw, "timeout": 0.01}),
    )
    release, finished = Event(), Event()

    def generate(**kwargs):
        kwargs["streamer"].on_finalized_text("partial", stream_end=not close)
        release.wait(5)
        finished.set()

    adapter._model.generate.side_effect = generate
    iterator, _ = adapter.build_streaming_iter([], {})
    try:
        assert next(iterator) == "partial"
        if close:
            iterator.close()
            assert not finished.is_set()
        else:
            with pytest.raises(TimeoutError):
                list(iterator)
    finally:
        release.set()
        assert finished.wait(5)
